"""Ray-cast perception for the agents.

Two sensors, both built on MuJoCo's ``mj_ray`` (ray vs. all scene geometry):

* ``lidar`` — a ring of range readings at a fixed low height, so it senses both
  tall walls and low cover boxes (for navigation).
* ``can_see`` — line-of-sight to the opponent: within a view cone, within range,
  and not occluded by a wall/box. This is what makes hiding (and crouching
  behind low cover) actually work.
"""

from __future__ import annotations

import math
import numpy as np


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def lidar(mj, model, data, origin, heading: float, n_rays: int, max_range: float,
          exclude_body: int) -> np.ndarray:
    """Ring of normalized distances in [0, 1] (1 = nothing within max_range)."""
    out = np.ones(n_rays, dtype=np.float32)
    gid = np.zeros(1, np.int32)
    pnt = np.asarray(origin, dtype=np.float64)
    for i in range(n_rays):
        a = heading + (2 * math.pi * i) / n_rays
        vec = np.array([math.cos(a), math.sin(a), 0.0])
        dist = mj.mj_ray(model, data, pnt, vec, None, 1, exclude_body, gid)
        if dist >= 0:
            out[i] = min(dist, max_range) / max_range
    return out


def can_see(mj, model, data, eye_from, eye_to, from_body: int, opp_prefix: str,
            heading: float, fov: float, view_range: float):
    """Return (visible, rel_forward, rel_left, distance).

    rel_forward/left are the opponent's position in the viewer's heading frame
    (metres). When not visible they're returned anyway (useful only as the raw
    geometry); callers gate on the ``visible`` flag.
    """
    vec = np.asarray(eye_to, dtype=np.float64) - np.asarray(eye_from, dtype=np.float64)
    dist = float(np.linalg.norm(vec))
    fwd = math.cos(heading) * vec[0] + math.sin(heading) * vec[1]
    left = -math.sin(heading) * vec[0] + math.cos(heading) * vec[1]

    if dist < 1e-6 or dist > view_range:
        return False, fwd, left, dist
    # view cone
    if abs(_wrap(math.atan2(vec[1], vec[0]) - heading)) > fov:
        return False, fwd, left, dist
    # occlusion: first thing the ray hits should be the opponent, not a wall
    gid = np.zeros(1, np.int32)
    hit = mj.mj_ray(model, data, np.asarray(eye_from, np.float64), vec / dist,
                    None, 1, from_body, gid)
    if hit >= 0 and hit < dist - 0.05:
        body = model.geom_bodyid[int(gid[0])]
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body) or ""
        if not name.startswith(opp_prefix):
            return False, fwd, left, dist       # blocked by a wall/box
    return True, fwd, left, dist
