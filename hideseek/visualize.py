"""Watch a hide-and-seek match in the interactive MuJoCo viewer.

    python -m hideseek.visualize --run outputs/<run>
    python -m hideseek.visualize --run outputs/<run> --match-time 45 --preset maze
    MUJOCO_GL=egl python -m hideseek.visualize --run outputs/<run> --video match.mp4

Blue seeks, red hides. The viewer runs match after match until you close the
window; each match prints its winner. Use --seed for reproducible matches.

Press **V** in the window to toggle the camera: overhead -> seeker first-person
POV -> hider POV (or start on one with --view seeker|hider). The POV camera is a
natural ~90 deg lens; the agent's game-vision cone is wider, so "seen" can mean
"just off-camera".
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from hideseek.config import Config, REPO_ROOT, SEEKER
from hideseek.match import build_match_env, load_policies, run_match


VIEW_ORDER = ["free", "seeker", "hider"]   # overhead, then first-person POVs


def _set_cam(viewer, model, view):
    import mujoco
    if view == "free":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{view}_pov")


def watch(run_dir: Path, cfg: Config, matches: int, seed, realtime: bool, view: str = "free"):
    import mujoco.viewer
    policies = load_policies(run_dir)
    env = build_match_env(cfg, policies)
    smodel, snorm = policies[SEEKER]
    dt = env.dt
    print("[watch] blue seeks, red hides | press V to toggle view (overhead / seeker / hider) | close window to stop")

    # V cycles the camera; a flag tells the loop to re-apply it on the viewer.
    state = {"i": VIEW_ORDER.index(view), "dirty": True}

    def key_cb(keycode):
        if keycode == ord("V"):
            state["i"] = (state["i"] + 1) % len(VIEW_ORDER)
            state["dirty"] = True

    m = 0
    obs, _ = env.reset(seed=(seed if seed is not None else None))
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_cb) as viewer:
        while viewer.is_running() and (matches == 0 or m < matches):
            t0 = time.time()
            if state["dirty"]:
                _set_cam(viewer, env.model, VIEW_ORDER[state["i"]])
                print(f"  [view: {VIEW_ORDER[state['i']]}]")
                state["dirty"] = False
            a = smodel.predict(snorm(obs), deterministic=True)[0]
            obs, r, term, trunc, info = env.step(a)
            viewer.sync()
            if term or trunc:
                print(f"  match {m}: winner={info.get('result')}  t={info.get('t',0):.1f}s")
                m += 1
                s = (seed + m) if seed is not None else None
                obs, _ = env.reset(seed=s)
            if realtime:
                sleep = dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    env.close()


def main():
    p = argparse.ArgumentParser(description="Watch a hide-and-seek match")
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--matches", type=int, default=0, help="0 = until you close the window")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--match-time", type=float, default=None)
    p.add_argument("--prep-time", type=float, default=None)
    p.add_argument("--preset", type=str, default=None, choices=["room", "maze"])
    p.add_argument("--view", type=str, default="free", choices=["free", "seeker", "hider"],
                   help="starting camera (press V in the window to toggle)")
    p.add_argument("--video", type=str, default=None, help="headless mp4 instead of a window")
    p.add_argument("--no-realtime", action="store_true")
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

    if args.video:
        from hideseek.match import main as match_main  # reuse recording path
        sys.argv = ["match", "--run", str(run_dir), "--matches", "1",
                    "--video", args.video, "--view", args.view]
        if args.preset:
            sys.argv += ["--preset", args.preset]
        match_main()
        return

    watch(run_dir, cfg, args.matches, args.seed, realtime=not args.no_realtime, view=args.view)
    # avoid the GLFW/MuJoCo native-teardown segfault on exit
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
