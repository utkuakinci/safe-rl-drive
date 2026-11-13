"""Safety observation channel.

The shield does **not** consume the (possibly noisy, normalised, world-frame) policy observation.
It gets its own ego-frame, metric, ground-truth-ish channel - mirroring a production stack where
the safety monitor runs on a separately certified perception path.

Flat float32 layout (also used on the ROS2 topic ``/rl/safety_observation``)::

    [ ego_speed, left_lane_available, right_lane_available,
      (presence, s, d, vs, vd) * SAFETY_K ]

s / d: longitudinal / lateral offset of the other vehicle in the ego frame [m] (d > 0 = right of ego,
highway-env convention). vs / vd: relative velocity (other - ego) in the ego frame [m/s].
"""

from __future__ import annotations

import math

import numpy as np

SAFETY_K = 8
SAFETY_HEADER = 3
SAFETY_FEATS = 5
SAFETY_OBS_SIZE = SAFETY_HEADER + SAFETY_K * SAFETY_FEATS


def _side_lanes(road, ego) -> tuple[bool, bool]:
    try:
        lane_index = ego.lane_index
        side = road.network.side_lanes(lane_index)
        ids = {s[2] for s in side}
        return (lane_index[2] - 1) in ids, (lane_index[2] + 1) in ids
    except Exception:  # pragma: no cover - roads without lane topology
        return False, False


def _close_vehicles(road, ego, radius: float, k: int):
    fn = getattr(road, "close_vehicles_to", None) or road.close_objects_to
    try:
        return fn(ego, radius, count=k, see_behind=True, sort=True)
    except TypeError:
        return fn(ego, radius, count=k, see_behind=True)


def safety_observation(base_env, k: int = SAFETY_K, radius: float = 100.0) -> np.ndarray:
    """Build the flat safety observation from an (unwrapped) highway-env environment."""
    ego = base_env.vehicle
    road = base_env.road
    out = np.zeros(SAFETY_HEADER + k * SAFETY_FEATS, dtype=np.float32)
    left, right = _side_lanes(road, ego)
    out[0], out[1], out[2] = float(ego.speed), float(left), float(right)

    c, s = math.cos(ego.heading), math.sin(ego.heading)
    ego_pos = np.asarray(ego.position, dtype=np.float64)
    ego_vel = np.asarray(ego.velocity, dtype=np.float64)
    for i, v in enumerate(_close_vehicles(road, ego, radius, k)[:k]):
        dp = np.asarray(v.position, dtype=np.float64) - ego_pos
        dv = np.asarray(v.velocity, dtype=np.float64) - ego_vel
        base = SAFETY_HEADER + i * SAFETY_FEATS
        out[base : base + 5] = (
            1.0,
            c * dp[0] + s * dp[1],
            -s * dp[0] + c * dp[1],
            c * dv[0] + s * dv[1],
            -s * dv[0] + c * dv[1],
        )
    return out
