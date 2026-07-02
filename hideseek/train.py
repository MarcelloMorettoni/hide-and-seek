"""Self-play training for hide-and-seek (league-lite).

Improvements over naive self-play:
  * Opponent POOL — each round we snapshot both policies and keep the last few.
    During a role's training phase, every parallel env faces a *different*
    sampled historical opponent (recent-weighted). This stops the two agents
    from cycling/forgetting by overfitting only the latest opponent.
  * Dense distance shaping (in env.py) gives a gradient even when the opponent
    isn't in view, so the seeker learns to search instead of freezing.
  * Per-round WIN-RATE evaluation — the only real gauge of self-play progress —
    printed and logged to TensorBoard (eval/seeker_winrate).

Opponents run in-process (DummyVecEnv), so a snapshot is just a deep-copied
policy + its normalization stats wrapped in a callable — no cross-process
pickling of torch models.

    python -m hideseek.train --config configs/default.yaml
    python -m hideseek.train --smoke
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np

from hideseek.config import Config, REPO_ROOT, SEEKER, HIDER
from hideseek.env import HideAndSeekEnv


def build_venv(cfg: Config, role: str):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    fns = [(lambda: HideAndSeekEnv(cfg, role=role)) for _ in range(cfg.train.n_envs)]
    venv = VecMonitor(DummyVecEnv(fns), info_keywords=("is_success",))
    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg.train.gamma)


def make_ppo(cfg: Config, venv, tb: str):
    from stable_baselines3 import PPO
    return PPO("MlpPolicy", venv, verbose=0, seed=cfg.train.seed, device=cfg.train.device,
               n_steps=cfg.train.n_steps, batch_size=cfg.train.batch_size,
               n_epochs=cfg.train.n_epochs, gamma=cfg.train.gamma, gae_lambda=cfg.train.gae_lambda,
               ent_coef=cfg.train.ent_coef, learning_rate=cfg.train.learning_rate,
               tensorboard_log=tb, policy_kwargs=dict(net_arch=list(cfg.train.policy_hidden)))


def frozen_snapshot(model, venv):
    """Self-contained, independent opponent snapshot.

    SB3 policies can't be ``deepcopy``d (non-leaf tensors), so we round-trip the
    model through its own save/load into an in-memory buffer — that yields a fully
    independent frozen model. Normalization stats are copied alongside.
    """
    import io
    from stable_baselines3 import PPO

    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    frozen = PPO.load(buf, device="cpu")
    rms, clip, eps = copy.deepcopy(venv.obs_rms), venv.clip_obs, venv.epsilon

    def snap(obs):
        o = np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)
        return frozen.predict(o, deterministic=True)[0]

    return snap


class PolicyPool:
    """Keeps the last N opponent snapshots; samples recent ones more often."""

    def __init__(self, size: int):
        self.size = size
        self.snaps: list = []

    def add(self, snap):
        self.snaps.append(snap)
        self.snaps = self.snaps[-self.size:]

    def sample(self, rng):
        w = np.arange(1, len(self.snaps) + 1, dtype=float)   # recent-weighted
        return self.snaps[rng.choice(len(self.snaps), p=w / w.sum())]


def assign_opponents(venv, pool: PolicyPool, rng, n_envs: int):
    """Give each parallel env its own sampled opponent (random until the pool fills)."""
    for i in range(n_envs):
        pol = pool.sample(rng) if pool.snaps else (lambda o: rng.uniform(-1, 1, 3).astype(np.float32))
        venv.env_method("set_opponent", pol, indices=[i])


def eval_winrate(cfg: Config, seeker_snap, hider_snap, n: int, seed0: int = 9000):
    env = HideAndSeekEnv(cfg, role=SEEKER, opponent=hider_snap)
    wins, times = 0, []
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        done, info = False, {}
        while not done:
            obs, r, term, trunc, info = env.step(seeker_snap(obs))
            done = term or trunc
        if info.get("result") == SEEKER:
            wins += 1
            times.append(info["t"])
    env.close()
    return wins / n, (float(np.mean(times)) if times else float("nan"))


def train(cfg: Config) -> Path:
    run = cfg.train.run_name or f"selfplay_{cfg.arena.preset}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = (REPO_ROOT / cfg.train.log_dir / run).resolve()
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")
    print(f"[train] {run_dir}")

    rng = np.random.default_rng(cfg.train.seed)
    venv = {r: build_venv(cfg, r) for r in (SEEKER, HIDER)}
    model = {r: make_ppo(cfg, venv[r], str(run_dir / "tb")) for r in (SEEKER, HIDER)}
    pool = {r: PolicyPool(cfg.train.pool_size) for r in (SEEKER, HIDER)}

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir / "tb" / "eval"))
    except Exception:
        pass

    for rnd in range(cfg.train.rounds):
        for role, opp in ((SEEKER, HIDER), (HIDER, SEEKER)):
            assign_opponents(venv[role], pool[opp], rng, cfg.train.n_envs)
            model[role].learn(cfg.train.steps_per_phase, reset_num_timesteps=(rnd == 0),
                              tb_log_name=role, progress_bar=False)
        snaps = {r: frozen_snapshot(model[r], venv[r]) for r in (SEEKER, HIDER)}
        for r in (SEEKER, HIDER):
            pool[r].add(snaps[r])
            model[r].save(run_dir / f"{r}_model")
            venv[r].save(str(run_dir / f"{r}_vecnormalize.pkl"))

        wr, tcatch = eval_winrate(cfg, snaps[SEEKER], snaps[HIDER], cfg.train.eval_matches)
        tc = f"{tcatch:.1f}s" if not np.isnan(tcatch) else "n/a"
        print(f"[train] round {rnd + 1}/{cfg.train.rounds}  seeker_winrate={wr*100:4.0f}%  avg_catch={tc}")
        if writer:
            writer.add_scalar("eval/seeker_winrate", wr, rnd)
            if not np.isnan(tcatch):
                writer.add_scalar("eval/avg_catch_time", tcatch, rnd)

    if writer:
        writer.close()
    for r in (SEEKER, HIDER):
        venv[r].close()
    print(f"[train] artifacts in {run_dir}")
    return run_dir


def main():
    p = argparse.ArgumentParser(description="Self-play hide-and-seek training (league-lite)")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--steps-per-phase", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    path = Path(args.config)
    if not path.is_absolute():
        path = REPO_ROOT / path
    cfg = Config.from_yaml(path) if path.exists() else Config()
    if args.smoke:
        cfg.train.rounds = 2
        cfg.train.steps_per_phase = 3000
        cfg.train.n_envs = 2
        cfg.train.eval_matches = 4
        cfg.train.run_name = "smoke"
    for a, v in (("rounds", args.rounds), ("steps_per_phase", args.steps_per_phase),
                 ("n_envs", args.n_envs), ("run_name", args.run_name)):
        if v is not None:
            setattr(cfg.train, a, v)
    train(cfg)


if __name__ == "__main__":
    main()
