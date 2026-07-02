# Hide & Seek — self-play humanoids

Two articulated humanoids learn **hide-and-seek** by playing each other
(self-play PPO) in a walled arena. The **blue** agent always seeks; the **red**
agent always hides. It can **crouch** to duck below low cover, and vision is
**line-of-sight** (walls and boxes block the view), so hiding is real.

```
  MuJoCo arena (walls + low cover)  ─▶  self-play PPO (seeker vs hider)  ─▶  a match
     articulated humanoids                alternating training rounds        blue finds red?
     ray-cast vision + crouch
```

- **Engine:** MuJoCo 3 (fast physics + built-in ray-casting for line of sight)
- **Agents:** humanoid bodies with articulated legs; **kinematic/planar
  locomotion** (move / turn / crouch) so RL learns *strategy*, not how to walk
- **Perception:** a lidar ring (walls/obstacles) + line-of-sight opponent
  detection within a view cone; crouching lowers the eye below low boxes
- **Training:** self-play — seeker and hider improve by chasing each other up a
  difficulty ladder
- **Match:** load both policies, run timed rounds, see who wins

---

## Setup

Requires **Python 3.10+**. No GPU needed — the small MLP policies train on CPU.

```bash
git clone <your-repo-url> hide-and-seek
cd hide-and-seek
python -m venv .venv && source .venv/bin/activate    # or: conda create -n hideseek python=3.11
pip install -e .                                      # mujoco, gymnasium, stable-baselines3, ...
pytest                                                # optional: fast CPU sanity check
```

## Quickstart

```bash
make smoke                               # ~1 min: verify the whole pipeline

make train CONFIG=configs/default.yaml   # full self-play (room preset)
make tb                                  # watch win-rates / rewards climb

# run a match between the trained policies (blue seeks, red hides)
python -m hideseek.match --run outputs/<run> --matches 20
python -m hideseek.match --run outputs/<run> --match-time 45 --prep-time 8

# watch it live (needs a desktop display). Press V in the window to toggle the
# camera: overhead -> seeker first-person POV -> hider POV.
make watch RUN=outputs/<run>
python -m hideseek.visualize --run outputs/<run> --view seeker   # start in blue's eyes
# ...or headless mp4 from an agent's POV:
MUJOCO_GL=egl python -m hideseek.visualize --run outputs/<run> --video match.mp4 --view hider
```

Everything is configurable in `configs/*.yaml` (or via CLI): match/prep time,
arena preset & size, agent speed, vision cone & range, catch radius, and all
reward weights.

---

## How it works

**The arena** (`arena_builder.py`) is procedurally generated per preset (`room`,
`maze`): perimeter walls, tall interior dividers (block everyone), and
crouch-height cover boxes (block a *crouched* eye, not a standing one).

**The agents** (`humanoid.py`) are humanoid bodies on a planar root (slide-x,
slide-y, yaw) driven by velocity actuators, plus a crouch DoF + articulated
hips/knees driven by position actuators. Action = `[forward, turn, crouch]`.
They can't fall over (no roll/pitch), so training is about strategy.

**Vision** (`vision.py`) uses MuJoCo's `mj_ray`: a lidar ring for navigation and
a line-of-sight test to the opponent (within a view cone + range, not occluded).
Crouching drops the eye from 1.57 m to 0.94 m — below the 1.1 m box tops.

**The match** has a *prep phase* (seeker frozen + blind while the hider hides),
then the *seek phase*. The seeker wins by **tagging** the hider — getting within
`catch_radius` with no *tall wall* between them (low cover doesn't save you, so
ducking at the last second no longer dodges the tag). The hider wins by surviving
to `match_time`. Set `vision.catch_needs_line_of_sight: true` for the stricter
rule where the seeker must also *see* the hider at the moment of the tag.

**Self-play** (`train.py`): two PPO policies trained in alternating rounds
(seeker, then hider). Rather than always facing the *latest* opponent (which
makes agents cycle and forget), each round snapshots both policies into an
**opponent pool** (league-lite) and every parallel env faces a *different*
sampled past opponent. A **dense distance reward** gives a gradient even when the
opponent is out of view, and each round logs a **win-rate** (`eval/seeker_winrate`
in TensorBoard) — the real gauge of progress.

> Like all adversarial self-play, good behavior takes real training time (many
> rounds). The smoke run only proves the plumbing; expect early policies to look
> aimless before strategy emerges. Healthy self-play keeps the win-rate roughly
> *balanced* (neither side dominating).

---

## Project structure

```
hide-and-seek/
├── hideseek/
│   ├── config.py          # typed dataclass config (arena / agent / vision / match / train)
│   ├── humanoid.py         # procedural articulated humanoid (MJCF fragments)
│   ├── arena_builder.py    # procedural scene: walls + cover + 2 agents (presets)
│   ├── vision.py           # ray-cast lidar + line-of-sight opponent detection
│   ├── env.py              # HideAndSeekEnv: 2 agents, prep/seek phases, rewards
│   ├── train.py            # self-play PPO (opponent pool + win-rate eval)
│   ├── match.py            # run matches, report blue/red win rates, record video
│   └── visualize.py        # interactive viewer (overhead / seeker POV / hider POV)
├── configs/                # default.yaml (room), maze.yaml, quick.yaml
├── tests/                  # fast CPU tests (env, presets, crouch-occlusion mechanic)
└── Makefile
```

## Configuration highlights

| Group | Keys |
|---|---|
| `arena` | `preset` (room/maze), `size` |
| `agent` | `max_speed`, `hider_speed_frac`, `max_turn`, `control_freq`, `allow_reverse` |
| `vision` | `n_rays`, `lidar_range`, `fov`, `view_range`, `catch_radius`, `catch_needs_line_of_sight` |
| `match` | `match_time`, `prep_time`, all reward weights (incl. `hider_flee_weight`) |
| `train` | `rounds`, `steps_per_phase`, `pool_size`, `eval_matches`, `n_envs`, PPO hyper-params |

### Watching a match

`make watch RUN=<run>` opens the interactive viewer; press **V** in the window to
toggle the camera **overhead → seeker first-person POV → hider POV**. Start on a
specific one with `--view seeker|hider`, and render a headless mp4 from any of
them with `python -m hideseek.visualize --run <run> --video out.mp4 --view seeker`
(set `MUJOCO_GL=egl`). The POV camera is a natural ~90° lens, so an agent can
"see" the opponent for the game while it's just off the edge of the POV frame.

## Roadmap

- [x] Articulated humanoids, crouch-to-hide, ray-cast line-of-sight
- [x] Self-play PPO, match runner, interactive viewer, room/maze presets
- [ ] Randomized wall layouts per episode (curriculum)
- [ ] Team play (multiple seekers/hiders)
- [ ] Physics-based gait (pretrained low-level walking controller)
- [ ] Pixel observations
