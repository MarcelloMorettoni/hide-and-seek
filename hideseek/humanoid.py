"""Procedural articulated humanoid agent (MJCF fragments).

Each agent is a human-shaped body with:
  * a planar root — slide x, slide y, hinge yaw — driven by *velocity* actuators
    (RL commands forward speed + turn rate). No roll/pitch DoF, so it can't fall
    over: locomotion is kinematic, RL learns strategy not balance.
  * a crouch DoF — slide z — plus articulated hips/knees that flex, driven by
    *position* actuators. Crouching lowers the head (eye) below low cover.
  * an "eye" site at head height used for line-of-sight vision (see vision.py).

Only a single body capsule collides with walls; limbs are visual, which keeps
wall contacts clean. Everything is generated so the two agents (blue seeker, red
hider) are identical except colour and spawn.
"""

from __future__ import annotations

# Standing geometry (metres). Crouch interpolates toward the crouched pose.
PELVIS_H = 0.95          # pelvis height when standing
EYE_STAND = 1.57         # eye/head height standing  (pelvis + 0.62)
EYE_CROUCH = 0.94        # eye/head height fully crouched (< low-box top 1.1 -> hidden)
CROUCH_DROP = 0.63       # pelvis drop when crouched (keeps collision just off the floor)


def agent_body(name: str, rgba: str, pos_xy=(0.0, 0.0)) -> str:
    """Return the <body> MJCF for one humanoid agent.

    No spawn rotation: heading comes solely from the ``_yaw`` joint, which keeps
    the ``_x``/``_y`` slide axes world-aligned (so "move along world x" means
    exactly that). The env sets the initial heading via the yaw joint on reset.
    """
    x, y = pos_xy
    # Visual geoms are non-colliding (contype/conaffinity 0): only the invisible
    # `_col` capsule collides. Otherwise the legs/feet drag on the floor and
    # self-collide, pinning the agent in place.
    limb = f'rgba="{rgba}" contype="0" conaffinity="0"'
    dark = 'rgba="0.15 0.15 0.18 1" contype="0" conaffinity="0"'
    return f"""
    <body name="{name}" pos="{x} {y} {PELVIS_H}">
      <joint name="{name}_x"   type="slide" axis="1 0 0" limited="false"/>
      <joint name="{name}_y"   type="slide" axis="0 1 0" limited="false"/>
      <joint name="{name}_z"   type="slide" axis="0 0 1" range="-{CROUCH_DROP} 0.05"/>
      <joint name="{name}_yaw" type="hinge" axis="0 0 1" limited="false"/>
      <!-- collision proxy: one capsule; kept short below the pelvis so crouching
           doesn't drive it into the floor -->
      <geom name="{name}_col" type="capsule" fromto="0 0 -0.15 0 0 0.4" size="0.15"
            rgba="0 0 0 0" contype="1" conaffinity="1" mass="20"/>
      <!-- visual torso / head / eye -->
      <geom name="{name}_torso" type="capsule" fromto="0 0 0.05 0 0 0.5" size="0.11" {limb}/>
      <geom name="{name}_head"  type="sphere" pos="0 0 0.62" size="0.11" {limb}/>
      <geom name="{name}_face"  type="box" pos="0.09 0 0.62" size="0.03 0.06 0.05" {dark}/>
      <site name="{name}_eye"   pos="0.10 0 0.62" size="0.02" rgba="1 1 0 1"/>
      <!-- first-person camera at the eye, looking along the agent's heading.
           Rides with crouch (drops with the body) and turns with the yaw joint. -->
      <camera name="{name}_pov" pos="0.14 0 0.62" xyaxes="0 -1 0 0 0 1" fovy="90"/>
      <!-- arms (visual) -->
      <geom name="{name}_arm_l" type="capsule" fromto="0 0.13 0.45 0 0.20 0.15" size="0.045" {limb}/>
      <geom name="{name}_arm_r" type="capsule" fromto="0 -0.13 0.45 0 -0.20 0.15" size="0.045" {limb}/>
      <!-- left leg: thigh (hip) + shin (knee) -->
      <body name="{name}_lthigh" pos="0 0.09 0.0">
        <joint name="{name}_lhip"  type="hinge" axis="0 1 0" range="-1.8 0.3"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.45" size="0.06" {limb}/>
        <body name="{name}_lshin" pos="0 0 -0.45">
          <joint name="{name}_lknee" type="hinge" axis="0 1 0" range="0 2.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.45" size="0.05" {limb}/>
          <geom type="box" pos="0.06 0 -0.46" size="0.10 0.05 0.03" {dark}/>
        </body>
      </body>
      <!-- right leg -->
      <body name="{name}_rthigh" pos="0 -0.09 0.0">
        <joint name="{name}_rhip"  type="hinge" axis="0 1 0" range="-1.8 0.3"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.45" size="0.06" {limb}/>
        <body name="{name}_rshin" pos="0 0 -0.45">
          <joint name="{name}_rknee" type="hinge" axis="0 1 0" range="0 2.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.45" size="0.05" {limb}/>
          <geom type="box" pos="0.06 0 -0.46" size="0.10 0.05 0.03" {dark}/>
        </body>
      </body>
    </body>"""


def agent_actuators(name: str) -> str:
    """Velocity actuators for planar motion; position actuators for the pose."""
    return f"""
    <velocity name="{name}_vx"   joint="{name}_x"   kv="80" ctrlrange="-4 4"/>
    <velocity name="{name}_vy"   joint="{name}_y"   kv="80" ctrlrange="-4 4"/>
    <velocity name="{name}_wyaw" joint="{name}_yaw" kv="15" ctrlrange="-4 4"/>
    <position name="{name}_pz"    joint="{name}_z"    kp="6000" kv="200" ctrlrange="-{CROUCH_DROP} 0.05"/>
    <position name="{name}_plhip" joint="{name}_lhip" kp="200" kv="10" ctrlrange="-1.8 0.3"/>
    <position name="{name}_plknee" joint="{name}_lknee" kp="200" kv="10" ctrlrange="0 2.2"/>
    <position name="{name}_prhip" joint="{name}_rhip" kp="200" kv="10" ctrlrange="-1.8 0.3"/>
    <position name="{name}_prknee" joint="{name}_rknee" kp="200" kv="10" ctrlrange="0 2.2"/>"""


# joint / actuator name helpers used by the env
def planar_joints(name): return [f"{name}_x", f"{name}_y", f"{name}_yaw"]
def crouch_joint(name):  return f"{name}_z"
def leg_joints(name):    return [f"{name}_lhip", f"{name}_lknee", f"{name}_rhip", f"{name}_rknee"]
def vel_actuators(name): return [f"{name}_vx", f"{name}_vy", f"{name}_wyaw"]
def pose_actuators(name): return [f"{name}_pz", f"{name}_plhip", f"{name}_plknee",
                                  f"{name}_prhip", f"{name}_prknee"]
