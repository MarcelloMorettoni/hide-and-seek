"""Fast CPU tests for the hide-and-seek pipeline (no training required)."""

import numpy as np
import pytest

from hideseek.config import Config, SEEKER, HIDER
from hideseek.env import HideAndSeekEnv


# ---------------------------------------------------------------- config
def test_config_roundtrip(tmp_path):
    cfg = Config()
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    assert Config.from_yaml(p).to_dict() == cfg.to_dict()


def test_config_unknown_key_raises():
    with pytest.raises(ValueError):
        Config.from_dict({"match": {"nope": 1}})


# ---------------------------------------------------------------- scene
@pytest.mark.parametrize("preset", ["room", "maze"])
def test_preset_compiles(preset):
    import mujoco
    from hideseek.arena_builder import build_scene
    m = mujoco.MjModel.from_xml_string(build_scene(preset, size=10.0))
    assert m.nu == 16 and m.nq == 16   # 8 joints/actuators per agent x2


# ---------------------------------------------------------------- env
def test_env_spaces_and_reset():
    env = HideAndSeekEnv(Config(), role=SEEKER)
    obs, _ = env.reset(seed=0)
    assert env.action_space.shape == (3,)
    assert obs.shape == env.observation_space.shape == (32,)
    assert np.all(np.isfinite(obs))
    env.close()


def test_env_determinism():
    env = HideAndSeekEnv(Config(), role=SEEKER)
    env.reset(seed=5)
    o1 = env.step(np.zeros(3, np.float32))[0]
    env.reset(seed=5)
    o2 = env.step(np.zeros(3, np.float32))[0]
    assert np.allclose(o1, o2)
    env.close()


def test_env_runs_stable():
    env = HideAndSeekEnv(Config(), role=HIDER)
    env.reset(seed=1)
    for _ in range(120):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r) and np.all(np.isfinite(obs))
        if term or trunc:
            env.reset()
    env.close()


def test_vision_discriminates():
    # over a rollout the seeker should sometimes see the hider and sometimes not
    env = HideAndSeekEnv(Config(), role=SEEKER)
    env.reset(seed=2)
    seen = set()
    for _ in range(300):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        seen.add(bool(info["seeker_sees"]))
        if term or trunc:
            env.reset()
    assert seen == {True, False}
    env.close()


def test_crouch_lowers_eye():
    env = HideAndSeekEnv(Config(), role=HIDER)
    env.reset(seed=3)
    eye = env._ids[HIDER]["eye"]
    for _ in range(40):        # full crouch = action[2] = +1
        env.step(np.array([0, 0, 1], np.float32))
    crouched = float(env.data.site_xpos[eye][2])
    env.reset(seed=3)
    for _ in range(40):        # stand = action[2] = -1
        env.step(np.array([0, 0, -1], np.float32))
    standing = float(env.data.site_xpos[eye][2])
    assert standing - crouched > 0.4      # crouching noticeably lowers the eye
    assert crouched < 1.1                 # below the low-box top
    env.close()


def test_prep_phase_freezes_seeker():
    cfg = Config()
    cfg.match.prep_time = 100.0           # whole episode is prep
    env = HideAndSeekEnv(cfg, role=SEEKER)
    env.reset(seed=4)
    _, _, _, _, info = env.step(np.array([1, 0, 0], np.float32))   # try to charge forward
    act = env._ids[SEEKER]["act"]
    # the freeze zeroes the seeker's velocity command, and it's blind during prep
    assert abs(env.data.ctrl[act["vx"]]) < 1e-6
    assert abs(env.data.ctrl[act["vy"]]) < 1e-6
    assert info["seeker_sees"] is False
    env.close()


def test_crouch_occlusion_mechanic():
    # standing hider behind a low box is visible; crouched is hidden
    import mujoco
    from hideseek.arena_builder import build_scene
    from hideseek import vision
    m = mujoco.MjModel.from_xml_string(build_scene("room", size=10.0))
    d = mujoco.MjData(m)
    jadr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
    sbody = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "seeker")
    # seeker in the open, hider just behind box_0 at (-3,-2.8). Pose is set
    # kinematically (mj_forward, no stepping) so there is no drift.
    d.qpos[jadr("seeker_x")] = 0.0 + 3.8; d.qpos[jadr("seeker_y")] = -2.8 + 3.8
    d.qpos[jadr("hider_x")] = -3.7 - 3.8; d.qpos[jadr("hider_y")] = -2.8 - 3.8

    def visible():
        mujoco.mj_forward(m, d)
        return vision.can_see(mujoco, m, d, d.site_xpos[sid("seeker_eye")],
                              d.site_xpos[sid("hider_eye")], sbody, "hider",
                              np.pi, 2.0, 20.0)[0]

    assert visible() is True                       # standing -> seen over the low box
    d.qpos[jadr("hider_z")] = -0.63                # crouch: lower the head/eye
    assert visible() is False                      # crouched behind the box -> hidden


def test_option_a_tag_reaches_over_low_cover_but_not_walls():
    # Option A: a proximity tag ignores low cover but is blocked by a tall wall.
    env = HideAndSeekEnv(Config(), role=SEEKER)   # default: catch_needs_line_of_sight=False
    env.reset(seed=0)

    def setpos(name, x, y):
        j = env._ids[name]["jq"]; sp = env.model.body(env._ids[name]["body"]).pos
        env.data.qpos[j["_x"]] = x - sp[0]; env.data.qpos[j["_y"]] = y - sp[1]
        env.data.qpos[j["_yaw"]] = 0.0

    def reachable(sx, sy, hx, hy):
        setpos(SEEKER, sx, sy); setpos(HIDER, hx, hy)
        env._mj.mj_forward(env.model, env.data)
        return env._reachable()

    assert reachable(0, 0, 1.0, 0) is True                 # open + close -> tag
    assert reachable(-2.2, -2.8, -3.8, -2.8) is True       # across a LOW box (box_0) -> still tag
    assert reachable(-2.8, 1.0, -0.8, 1.0) is False        # across a TALL divider (div_1) -> no tag
    env.close()
