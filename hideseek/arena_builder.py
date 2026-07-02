"""Procedurally build the hide-and-seek scene (MJCF).

A bounded arena with perimeter walls plus a layout of interior obstacles:
  * TALL walls  (block line of sight for everyone),
  * LOW boxes   (block a crouched agent's view but not a standing one — this is
                 what makes the crouch mechanic matter).

Two presets ship: ``room`` (open arena + scattered cover) and ``maze``
(corridors). Walls are fixed per preset (compiled once); variety across episodes
comes from randomized spawns, not new geometry.
"""

from __future__ import annotations

from hideseek.humanoid import agent_body, agent_actuators

TALL_H = 2.0      # tall wall top (2.0) blocks even a standing eye (1.57)
LOW_H = 1.1       # low box top (1.1): hides a crouched eye (0.94), not a standing one (1.57)
WALL_T = 0.1      # wall half-thickness


def _box(name, cx, cy, hx, hy, hz, material):
    return (f'<geom name="{name}" type="box" pos="{cx:.3f} {cy:.3f} {hz:.3f}" '
            f'size="{hx:.3f} {hy:.3f} {hz:.3f}" material="{material}" '
            f'contype="1" conaffinity="1"/>')


def _perimeter(size):
    h = size / 2.0
    t, hz = WALL_T, TALL_H / 2
    return [
        _box("wall_n", 0,  h, h + t, t, hz, "wall"),
        _box("wall_s", 0, -h, h + t, t, hz, "wall"),
        _box("wall_e",  h, 0, t, h + t, hz, "wall"),
        _box("wall_w", -h, 0, t, h + t, hz, "wall"),
    ]


def _room_layout(size):
    """Open arena with a few tall dividers and several low boxes for cover."""
    hz_t, hz_l = TALL_H / 2, LOW_H / 2
    g = []
    # a couple of tall dividers
    g.append(_box("div_1", -size * 0.18, size * 0.1, WALL_T, size * 0.22, hz_t, "wall"))
    g.append(_box("div_2", size * 0.22, -size * 0.05, size * 0.20, WALL_T, hz_t, "wall"))
    # low cover boxes (crouch-height)
    lows = [(-size*0.30, -size*0.28), (size*0.05, size*0.30), (size*0.32, size*0.28),
            (-size*0.05, -size*0.12), (size*0.34, -size*0.30), (-size*0.34, size*0.30)]
    for i, (cx, cy) in enumerate(lows):
        g.append(_box(f"box_{i}", cx, cy, 0.35, 0.35, hz_l, "cover"))
    return g


def _maze_layout(size):
    """Corridor-style layout from axis-aligned tall wall segments."""
    hz = TALL_H / 2
    s = size
    segs = [  # (cx, cy, hx, hy)
        (-s*0.20, s*0.30, WALL_T, s*0.18),
        (-s*0.20, -s*0.02, s*0.22, WALL_T),
        (s*0.10, -s*0.20, WALL_T, s*0.28),
        (s*0.28, s*0.12, s*0.20, WALL_T),
        (-s*0.34, -s*0.22, s*0.14, WALL_T),
        (s*0.02, s*0.20, s*0.16, WALL_T),
    ]
    g = [_box(f"maze_{i}", cx, cy, hx, hy, hz, "wall") for i, (cx, cy, hx, hy) in enumerate(segs)]
    # a couple of low boxes tucked in corners
    g.append(_box("box_0", -s*0.36, s*0.34, 0.3, 0.3, LOW_H/2, "cover"))
    g.append(_box("box_1", s*0.36, -s*0.34, 0.3, 0.3, LOW_H/2, "cover"))
    return g


LAYOUTS = {"room": _room_layout, "maze": _maze_layout}


def build_scene(preset: str = "room", size: float = 10.0,
                seeker_rgba: str = "0.2 0.4 0.95 1", hider_rgba: str = "0.95 0.25 0.2 1",
                timestep: float = 0.01) -> str:
    """Return the full scene MJCF string for the given preset."""
    if preset not in LAYOUTS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(LAYOUTS)}")
    walls = _perimeter(size) + LAYOUTS[preset](size)
    walls_xml = "\n    ".join(walls)
    # spawn in opposite corners by default (env randomizes on reset)
    q = size * 0.38
    seeker = agent_body("seeker", seeker_rgba, pos_xy=(-q, -q))
    hider = agent_body("hider", hider_rgba, pos_xy=(q, q))
    return f"""<mujoco model="hide_and_seek">
  <option timestep="{timestep}" integrator="implicitfast"/>
  <compiler angle="radian"/>

  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-35" offwidth="1280" offheight="960"/>
    <map znear="0.05"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.22 0.28 0.34"
             rgb2="0.15 0.2 0.25" markrgb="0.7 0.7 0.7" width="300" height="300"/>
    <material name="floor" texture="grid" texuniform="true" texrepeat="8 8" reflectance="0.1"/>
    <material name="wall" rgba="0.55 0.57 0.62 1"/>
    <material name="cover" rgba="0.55 0.4 0.25 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor" contype="1" conaffinity="1"/>
    {walls_xml}
    {seeker}
    {hider}
  </worldbody>

  <actuator>{agent_actuators("seeker")}{agent_actuators("hider")}
  </actuator>
</mujoco>"""
