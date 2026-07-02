"""Hide-and-seek environment (MuJoCo + Gymnasium).

One sim holds both humanoids. The env is single-agent from SB3's point of view:
it controls ``role`` (seeker or hider) while the opponent is driven by an
injected policy (``set_opponent``) — random until one is trained. This lets us
train with plain PPO and run self-play by swapping the frozen opponent
(see selfplay.py). The same env, seeker-view + a trained hider opponent, *is* a
match (see match.py).

Match structure: a preparation phase (seeker frozen while the hider hides), then
the seek phase. The seeker wins by getting within ``catch_radius`` with line of
sight; the hider wins by surviving until ``match_time``.
"""

from __future__ import annotations

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from hideseek.arena_builder import build_scene
from hideseek.config import Config, SEEKER, HIDER
from hideseek.humanoid import CROUCH_DROP
from hideseek import vision

JOINTS = ["_x", "_y", "_z", "_yaw", "_lhip", "_lknee", "_rhip", "_rknee"]


def _rand_policy(obs, n=3, rng=np.random):
    return rng.uniform(-1, 1, size=n).astype(np.float32)


class HideAndSeekEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config: Config | None = None, role: str = SEEKER,
                 opponent=None, render_mode=None):
        import mujoco
        self._mj = mujoco
        self.cfg = config or Config()
        self.role = role
        self.opp_role = HIDER if role == SEEKER else SEEKER
        self.opponent = opponent            # callable obs->action, or None (random)
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_string(build_scene(self.cfg.arena.preset,
                                                                self.cfg.arena.size))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = max(1, round((1.0 / self.cfg.agent.control_freq) / self.model.opt.timestep))
        self.dt = self.frame_skip * self.model.opt.timestep

        self._ids = {r: self._agent_ids(r) for r in (SEEKER, HIDER)}
        self.n_rays = self.cfg.vision.n_rays

        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        obs_dim = 7 + self.n_rays + 4 + 3 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)

        self._phase = {SEEKER: 0.0, HIDER: 0.0}      # leg-swing animation phase
        self._speed = {SEEKER: 0.0, HIDER: 0.0}      # commanded forward speed
        self._mem = {SEEKER: [0.0, 0.0, 1.0], HIDER: [0.0, 0.0, 1.0]}
        self._t = 0.0
        self._prev_dist = None
        self._renderer = None
        self._render_cam = -1        # -1 = free/overhead; or a POV camera id

    # -- id bookkeeping -----------------------------------------------------
    def _agent_ids(self, name):
        mj, m = self._mj, self.model
        jq = {j: m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name + j)] for j in JOINTS}
        jv = {j: m.jnt_dofadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name + j)] for j in JOINTS}
        act = {a: mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, name + "_" + a)
               for a in ["vx", "vy", "wyaw", "pz", "plhip", "plknee", "prhip", "prknee"]}
        return dict(name=name, jq=jq, jv=jv, act=act,
                    eye=mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, name + "_eye"),
                    body=mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name))

    def _pos(self, role):
        return self.data.xpos[self._ids[role]["body"]][:2].copy()

    def _yaw(self, role):
        return float(self.data.qpos[self._ids[role]["jq"]["_yaw"]])

    # -- reset --------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mj = self._mj
        mj.mj_resetData(self.model, self.data)
        half = self.cfg.arena.size * 0.42
        spawn = {SEEKER: self.model.body(self._ids[SEEKER]["body"]).pos[:2].copy(),
                 HIDER: self.model.body(self._ids[HIDER]["body"]).pos[:2].copy()}
        for _ in range(30):
            tgt = {r: self.np_random.uniform(-half, half, 2) for r in (SEEKER, HIDER)}
            if np.linalg.norm(tgt[SEEKER] - tgt[HIDER]) < self.cfg.arena.size * 0.4:
                continue
            for r in (SEEKER, HIDER):
                jq = self._ids[r]["jq"]
                self.data.qpos[jq["_x"]] = tgt[r][0] - spawn[r][0]
                self.data.qpos[jq["_y"]] = tgt[r][1] - spawn[r][1]
                self.data.qpos[jq["_yaw"]] = self.np_random.uniform(-math.pi, math.pi)
            mj.mj_forward(self.model, self.data)
            if not self._deep_penetration():
                break
        for _ in range(3):
            mj.mj_step(self.model, self.data)      # settle
        self._t = 0.0
        self._phase = {SEEKER: 0.0, HIDER: 0.0}
        self._mem = {SEEKER: [0.0, 0.0, 1.0], HIDER: [0.0, 0.0, 1.0]}
        self._prev_dist = float(np.linalg.norm(self._pos(SEEKER) - self._pos(HIDER)))
        return self._observe(self.role), {}

    def _deep_penetration(self):
        for c in self.data.contact[: self.data.ncon]:
            if c.dist < -0.08:
                return True
        return False

    # -- control ------------------------------------------------------------
    def _apply(self, role, action, frozen=False):
        """Set turn/crouch/legs once per control step and store the forward speed.

        The world-frame velocity is (re)computed every sim substep in
        `_update_velocity`, so the agent always moves along its *current* heading
        — otherwise a turning agent's momentum lags its facing and it appears to
        slide/walk backwards, blinding its view cone.
        """
        a = np.clip(action, -1, 1)
        ids, c = self._ids[role], self.cfg.agent
        # forward speed: full range [-1,1] -> [0, max] unless reverse is allowed,
        # so the agent faces (and sees) its direction of travel
        fwd_cmd = float(a[0]) if c.allow_reverse else (float(a[0]) + 1.0) * 0.5
        cap = c.max_speed * (c.hider_speed_frac if role == HIDER else 1.0)
        self._speed[role] = 0.0 if frozen else fwd_cmd * cap
        self.data.ctrl[ids["act"]["wyaw"]] = 0.0 if frozen else float(a[1]) * c.max_turn
        crouch = (float(a[2]) + 1.0) * 0.5
        self.data.ctrl[ids["act"]["pz"]] = -crouch * CROUCH_DROP
        # walk animation: opposite-leg swing scaled by speed
        self._phase[role] += abs(self._speed[role]) * self.dt * 3.0
        swing = 0.5 * math.sin(self._phase[role]) * min(abs(self._speed[role]) / c.max_speed, 1.0)
        self.data.ctrl[ids["act"]["plhip"]] = -crouch + swing
        self.data.ctrl[ids["act"]["prhip"]] = -crouch - swing
        self.data.ctrl[ids["act"]["plknee"]] = crouch * 1.6
        self.data.ctrl[ids["act"]["prknee"]] = crouch * 1.6

    def _update_velocity(self, role):
        yaw, s, ids = self._yaw(role), self._speed[role], self._ids[role]
        self.data.ctrl[ids["act"]["vx"]] = s * math.cos(yaw)
        self.data.ctrl[ids["act"]["vy"]] = s * math.sin(yaw)

    # -- step ---------------------------------------------------------------
    def step(self, action):
        prep = self._t < self.cfg.match.prep_time
        # opponent acts from its own view
        opp_obs = self._observe(self.opp_role)
        opp_action = self.opponent(opp_obs) if self.opponent else _rand_policy(opp_obs, rng=self.np_random)
        # seeker is frozen during prep
        self._apply(self.role, action, frozen=(prep and self.role == SEEKER))
        self._apply(self.opp_role, opp_action, frozen=(prep and self.opp_role == SEEKER))
        for _ in range(self.frame_skip):
            self._update_velocity(self.role)          # velocity tracks current heading
            self._update_velocity(self.opp_role)
            self._mj.mj_step(self.model, self.data)
        self._t += self.dt

        seeker_sees, _, _, dist = self._see(SEEKER, HIDER)
        seeker_sees = seeker_sees and not prep
        close = dist < self.cfg.vision.catch_radius
        if self.cfg.vision.catch_needs_line_of_sight:
            caught = seeker_sees and close
        else:  # Option A: tag by proximity, blocked only by a tall wall
            caught = close and not prep and self._reachable()
        timeout = self._t >= self.cfg.match.match_time
        terminated = bool(caught)
        truncated = bool(timeout and not caught)

        reward = self._reward(self.role, seeker_sees, dist, caught, prep, action)
        self._prev_dist = dist
        obs = self._observe(self.role)
        info = {"caught": caught, "seeker_sees": seeker_sees, "dist": dist,
                "t": self._t, "is_success": self._is_success(caught, timeout)}
        if terminated or truncated:
            info["result"] = SEEKER if caught else HIDER
        return obs, float(reward), terminated, truncated, info

    def _is_success(self, caught, timeout):
        # "success" from the controlled role's perspective (for logging)
        if self.role == SEEKER:
            return bool(caught)
        return bool(timeout and not caught)

    def _reward(self, role, seeker_sees, dist, caught, prep, action):
        m = self.cfg.match
        effort = float(np.sum(np.square(action))) * m.move_penalty
        closing = self._prev_dist - dist          # >0 when the gap shrank this step
        if role == SEEKER:
            # dense distance shaping (uses true positions) gives a gradient even
            # when the hider isn't in view -> the seeker learns to search, not freeze
            r = -m.seeker_time_penalty - effort + m.seeker_pursue_weight * closing
            if seeker_sees:
                r += m.seeker_see_reward
            if caught:
                r += m.catch_bonus
            return r
        # hider
        r = -effort
        if prep:
            return r
        r += m.hider_flee_weight * (-closing)     # reward opening the gap
        r += m.hider_hidden_reward if not seeker_sees else -m.hider_seen_penalty
        if caught:
            r -= m.caught_penalty
        return r

    # -- perception ---------------------------------------------------------
    def _see(self, watcher, target):
        v = self.cfg.vision
        return vision.can_see(self._mj, self.model, self.data,
                              self.data.site_xpos[self._ids[watcher]["eye"]],
                              self.data.site_xpos[self._ids[target]["eye"]],
                              self._ids[watcher]["body"], target,
                              self._yaw(watcher), v.fov, v.view_range)

    def _reachable(self) -> bool:
        """True if no TALL WALL sits between seeker and hider (for the tag).

        The ray is cast at 1.4 m — above the 1.1 m low-cover boxes, below the
        2.0 m walls — so ducking behind low cover doesn't block a tag, but a
        solid wall does. FOV and crouch are irrelevant here; this is 'can I reach
        you', not 'can I see your eyes'.
        """
        sp, hp = self._pos(SEEKER), self._pos(HIDER)
        a = np.array([sp[0], sp[1], 1.4]); b = np.array([hp[0], hp[1], 1.4])
        vec = b - a; d = float(np.linalg.norm(vec))
        if d < 1e-6:
            return True
        gid = np.zeros(1, np.int32)
        hit = self._mj.mj_ray(self.model, self.data, a, vec / d, None, 1,
                              self._ids[SEEKER]["body"], gid)
        if hit >= 0 and hit < d - 0.05:
            name = self._mj.mj_id2name(self.model, self._mj.mjtObj.mjOBJ_BODY,
                                       self.model.geom_bodyid[int(gid[0])]) or ""
            if not name.startswith(HIDER):
                return False          # a wall (or box tall enough at 1.4 m) blocks the tag
        return True

    def _observe(self, role):
        ids, m, c = self._ids[role], self.model, self.cfg
        half = c.arena.size / 2.0
        pos, yaw = self._pos(role), self._yaw(role)
        vx = float(self.data.qvel[ids["jv"]["_x"]]); vy = float(self.data.qvel[ids["jv"]["_y"]])
        fwd_v = math.cos(yaw) * vx + math.sin(yaw) * vy
        left_v = -math.sin(yaw) * vx + math.cos(yaw) * vy
        crouch = -float(self.data.qpos[ids["jq"]["_z"]]) / CROUCH_DROP
        own = [pos[0] / half, pos[1] / half, math.sin(yaw), math.cos(yaw),
               crouch, fwd_v / c.agent.max_speed, left_v / c.agent.max_speed]

        rays = vision.lidar(self._mj, m, self.data,
                            [pos[0], pos[1], c.vision.lidar_height], yaw,
                            self.n_rays, c.vision.lidar_range, ids["body"])

        other = self.opp_role if role == self.role else self.role
        visible, rf, lf, dist = self._see(role, other)
        mem = self._mem[role]
        if visible:
            mem[0], mem[1], mem[2] = rf, lf, 0.0
        else:
            mem[2] = min(mem[2] + self.dt / c.match.match_time, 1.0)
        opp = [1.0 if visible else 0.0, rf / half if visible else 0.0,
               lf / half if visible else 0.0, dist / c.vision.view_range if visible else 1.0]
        memo = [mem[0] / half, mem[1] / half, mem[2]]
        clock = [max(0.0, 1.0 - self._t / c.match.match_time),
                 0.0 if self._t < c.match.prep_time else 1.0]
        return np.concatenate([own, rays, opp, memo, clock]).astype(np.float32)

    # -- rendering ----------------------------------------------------------
    def set_opponent(self, policy):
        self.opponent = policy

    def set_render_camera(self, view: str = "free"):
        """Select the render camera: 'free' (overhead), 'seeker', or 'hider' (POV)."""
        if view in (None, "free"):
            self._render_cam = -1
        else:
            name = view if view.endswith("_pov") else f"{view}_pov"
            cid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                raise ValueError(f"unknown view {view!r} (use free/seeker/hider)")
            self._render_cam = cid

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = self._mj.Renderer(self.model, height=720, width=960)
        self._renderer.update_scene(self.data, camera=self._render_cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
