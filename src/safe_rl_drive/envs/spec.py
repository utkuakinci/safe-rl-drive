"""Single source of truth for the observation / action interface.

Everything that touches observations outside of highway-env (the safety shield, the explainer,
the ROS2 bridge, the CARLA demo and the C++ node) must agree with these constants. They are also
written into the ONNX metadata on export, so a deployed model carries its own interface contract.
"""

from __future__ import annotations

import numpy as np

# Kinematics observation: V vehicles x F features, ego vehicle in row 0.
VEHICLES_COUNT = 8
FEATURES: tuple[str, ...] = ("presence", "x", "y", "vx", "vy", "cos_h", "sin_h")
# Fixed ranges (instead of highway-env's lane-dependent defaults) so normalisation is identical
# across scenarios and can be reproduced outside the simulator (CARLA, C++).
FEATURES_RANGE: dict[str, tuple[float, float]] = {
    "x": (-100.0, 100.0),
    "y": (-100.0, 100.0),
    "vx": (-40.0, 40.0),
    "vy": (-40.0, 40.0),
}
OBS_SHAPE = (VEHICLES_COUNT, len(FEATURES))

# Unified discrete meta-action space used by every scenario (lateral + longitudinal).
ACTIONS: dict[int, str] = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
ACTION_INDEX: dict[str, int] = {v: k for k, v in ACTIONS.items()}


def observation_config() -> dict:
    return {
        "type": "Kinematics",
        "vehicles_count": VEHICLES_COUNT,
        "features": list(FEATURES),
        "features_range": {k: list(v) for k, v in FEATURES_RANGE.items()},
        "absolute": False,
        "normalize": True,
        "clip": True,
        "order": "sorted",
        "see_behind": True,
        "observe_intentions": False,
    }


def normalize(raw: np.ndarray) -> np.ndarray:
    """Map raw (V, F) kinematics to [-1, 1] exactly like highway-env does (lmap + clip)."""
    out = np.array(raw, dtype=np.float32, copy=True)
    for j, name in enumerate(FEATURES):
        if name in FEATURES_RANGE:
            lo, hi = FEATURES_RANGE[name]
            out[:, j] = np.clip(-1.0 + 2.0 * (out[:, j] - lo) / (hi - lo), -1.0, 1.0)
    return out


def denormalize(obs: np.ndarray) -> np.ndarray:
    """Inverse of :func:`normalize` (up to clipping)."""
    out = np.array(obs, dtype=np.float32, copy=True).reshape(OBS_SHAPE)
    for j, name in enumerate(FEATURES):
        if name in FEATURES_RANGE:
            lo, hi = FEATURES_RANGE[name]
            out[:, j] = lo + (out[:, j] + 1.0) * (hi - lo) / 2.0
    return out
