"""Run a hide-and-seek match between the two trained policies.

The seeker (blue) is driven by the trained seeker policy; the hider (red) is the
env's opponent, driven by the trained hider policy. Reports who wins (seeker =
caught the hider, hider = survived to the time limit) over N matches, plus the
average time-to-catch. Match/prep durations are configurable at the CLI.

    python -m hideseek.match --run outputs/<run> --matches 20
    python -m hideseek.match --run outputs/<run> --match-time 45 --prep-time 8
    MUJOCO_GL=egl python -m hideseek.match --run outputs/<run> --video match.mp4
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from hideseek.config import Config, REPO_ROOT, SEEKER, HIDER
from hideseek.env import HideAndSeekEnv


def _normfn(pkl: Path):
    vn = pickle.load(open(pkl, "rb"))
    rms, clip, eps = vn.obs_rms, vn.clip_obs, vn.epsilon
    return lambda o: np.clip((o - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)


def load_policies(run_dir: Path):
    from stable_baselines3 import PPO
    out = {}
    for r in (SEEKER, HIDER):
        model = PPO.load(str(run_dir / f"{r}_model"), device="cpu")
        norm = _normfn(run_dir / f"{r}_vecnormalize.pkl")
        out[r] = (model, norm)
    return out


def build_match_env(cfg: Config, policies, render_mode=None):
    hmodel, hnorm = policies[HIDER]
    opp = lambda o: hmodel.predict(hnorm(o), deterministic=True)[0]
    return HideAndSeekEnv(cfg, role=SEEKER, opponent=opp, render_mode=render_mode)


def run_match(env, seeker, seed=None, on_step=None):
    smodel, snorm = seeker
    obs, _ = env.reset(seed=seed)
    done = False
    info = {}
    while not done:
        a = smodel.predict(snorm(obs), deterministic=True)[0]
        obs, r, term, trunc, info = env.step(a)
        if on_step:
            on_step(info)
        done = term or trunc
    return info.get("result", HIDER), info.get("t", 0.0)


def main():
    p = argparse.ArgumentParser(description="Run hide-and-seek matches")
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--matches", type=int, default=20)
    p.add_argument("--match-time", type=float, default=None)
    p.add_argument("--prep-time", type=float, default=None)
    p.add_argument("--preset", type=str, default=None, choices=["room", "maze"])
    p.add_argument("--view", type=str, default="free", choices=["free", "seeker", "hider"],
                   help="camera for --video: overhead, or an agent's first-person POV")
    p.add_argument("--video", type=str, default=None)
    args = p.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    cfg = Config.from_yaml(run_dir / "config.yaml")
    if args.match_time is not None:
        cfg.match.match_time = args.match_time
    if args.prep_time is not None:
        cfg.match.prep_time = args.prep_time
    if args.preset is not None:
        cfg.arena.preset = args.preset

    policies = load_policies(run_dir)
    env = build_match_env(cfg, policies, render_mode="rgb_array" if args.video else None)
    if args.video:
        env.set_render_camera(args.view)

    frames = []
    wins = {SEEKER: 0, HIDER: 0}
    catch_times = []
    for i in range(args.matches):
        cb = (lambda info: frames.append(env.render())) if (args.video and i == 0) else None
        result, t = run_match(env, policies[SEEKER], seed=1000 + i, on_step=cb)
        wins[result] += 1
        if result == SEEKER:
            catch_times.append(t)
        print(f"  match {i:2d}: winner={result:6s}  t={t:5.1f}s")

    n = args.matches
    print(f"\n[match] seeker(blue) wins {wins[SEEKER]}/{n} ({100*wins[SEEKER]/n:.0f}%)  |  "
          f"hider(red) wins {wins[HIDER]}/{n} ({100*wins[HIDER]/n:.0f}%)")
    if catch_times:
        print(f"[match] avg time-to-catch: {np.mean(catch_times):.1f}s")

    if args.video and frames:
        import imageio
        out = Path(args.video)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, frames, fps=int(cfg.agent.control_freq))
        print(f"[match] saved {len(frames)}-frame video -> {out}")
    env.close()


if __name__ == "__main__":
    main()
