"""Typed configuration for the hide-and-seek project (one YAML == one run)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEKER, HIDER = "seeker", "hider"


@dataclass
class ArenaConfig:
    preset: str = "room"           # "room" | "maze"
    size: float = 10.0


@dataclass
class AgentConfig:
    max_speed: float = 3.0         # m/s forward (the seeker's top speed)
    hider_speed_frac: float = 0.65 # hider is slower, so a pursuing seeker CAN
                                   # close the gap — an equal-speed evader is uncatchable
    max_turn: float = 1.5          # rad/s (lower = less frantic spinning)
    control_freq: float = 20.0     # Hz
    allow_reverse: bool = False    # if False, agents can't walk backward — they
                                   # must turn to face where they go, so the view
                                   # cone always scans the direction of travel


@dataclass
class VisionConfig:
    n_rays: int = 16               # lidar rays (walls/obstacles)
    lidar_range: float = 8.0
    lidar_height: float = 0.6      # senses low boxes (top 1.1) and tall walls
    fov: float = 1.9               # field-of-view half-angle (rad) for seeing the opponent
    view_range: float = 20.0       # max sight distance; > arena diagonal, so in
                                   # practice only WALLS block sight, never range
    catch_radius: float = 1.6      # seeker within this => caught (see below)
    catch_needs_line_of_sight: bool = False
    # Catch rule. False (Option A, default): tag by proximity — within
    # catch_radius and no TALL WALL between the two (low cover doesn't save you,
    # so ducking at the last second no longer dodges the tag). True: also require
    # an unobstructed eye-level line of sight (harder; the hider can slip away).


@dataclass
class MatchConfig:
    match_time: float = 20.0       # seconds; hider wins if it survives this long
    prep_time: float = 1.0         # seeker frozen while the hider hides
    # seeker rewards
    seeker_see_reward: float = 1.0
    seeker_time_penalty: float = 0.02
    seeker_pursue_weight: float = 0.15
    catch_bonus: float = 50.0
    # hider rewards
    hider_hidden_reward: float = 0.6
    hider_seen_penalty: float = 1.0
    hider_flee_weight: float = 0.05   # dense: reward opening distance from the seeker
    caught_penalty: float = 50.0
    move_penalty: float = 0.005    # tiny cost on effort (both agents)


@dataclass
class TrainConfig:
    algo: str = "ppo"
    rounds: int = 12               # self-play alternations (seeker<->hider)
    steps_per_phase: int = 150_000
    pool_size: int = 5             # past opponent snapshots kept per role (league-lite)
    eval_matches: int = 12         # matches per round to measure win-rate
    n_envs: int = 8
    seed: int = 0
    device: str = "cpu"            # MLP PPO is faster on CPU than GPU
    policy_hidden: list[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 512
    batch_size: int = 2048
    n_epochs: int = 8
    ent_coef: float = 0.005
    log_dir: str = "outputs"
    run_name: str | None = None


@dataclass
class Config:
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path) -> "Config":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return _from_dict(cls, data)

    def to_dict(self) -> dict:
        return _to_plain(dataclasses.asdict(self))

    def to_yaml(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _from_dict(dc_type, data):
    if not is_dataclass(dc_type):
        return data
    hints = get_type_hints(dc_type)
    names = {f.name for f in fields(dc_type)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Unknown config keys for {dc_type.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name in names:
        if name not in data:
            continue
        val, hint = data[name], hints.get(name)
        if is_dataclass(hint) and isinstance(val, dict):
            kwargs[name] = _from_dict(hint, val)
        elif isinstance(val, list) and _is_tuple_hint(hint):
            kwargs[name] = tuple(val)
        else:
            kwargs[name] = val
    return dc_type(**kwargs)


def _to_plain(obj):
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _is_tuple_hint(hint) -> bool:
    if get_origin(hint) is tuple or hint is tuple:
        return True
    if get_origin(hint) is Union:
        return any(_is_tuple_hint(a) for a in get_args(hint))
    return False
