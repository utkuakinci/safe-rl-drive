"""Scenario registry: lane change (highway), merge, intersection + held-out scenarios for robustness."""

from __future__ import annotations

import copy
from typing import Any

from safe_rl_drive.envs.spec import observation_config


def _action(target_speeds: list[float] | None = None) -> dict:
    cfg: dict[str, Any] = {"type": "DiscreteMetaAction", "longitudinal": True, "lateral": True}
    if target_speeds is not None:
        cfg["target_speeds"] = target_speeds
    return cfg


# Lower speeds are included so that SLOWER (and the shield's minimal-risk manoeuvre) has real
# braking authority behind slow leaders; reward_speed_range still rewards 20-30 m/s.
HIGHWAY_SPEEDS = [10.0, 15.0, 20.0, 25.0, 30.0]

SCENARIOS: dict[str, dict[str, Any]] = {
    # Multi-lane highway: overtaking / lane-change behaviour.
    "highway": {
        "id": "highway-fast-v0",
        "config": {
            "lanes_count": 4,
            "vehicles_count": 30,
            "vehicles_density": 1.0,
            "duration": 40,
            "collision_reward": -1.0,
            "high_speed_reward": 0.4,
            "right_lane_reward": 0.1,
            "lane_change_reward": 0.0,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "action": _action(HIGHWAY_SPEEDS),
        },
    },
    # Highway on-ramp merge: negotiate with merging traffic.
    "merge": {
        "id": "merge-v1",
        "config": {"action": _action(HIGHWAY_SPEEDS)},
    },
    # Unsignalised intersection: yield to crossing traffic.
    "intersection": {
        "id": "intersection-v2",
        "config": {
            "duration": 13,
            "initial_vehicle_count": 10,
            "spawn_probability": 0.6,
            "action": _action([0.0, 4.5, 9.0]),
        },
    },
    # ---- held-out scenarios (never used for training) ----
    "roundabout": {
        "id": "roundabout-v1",
        "config": {"action": _action([0.0, 8.0, 16.0])},
    },
    "highway-dense": {
        "id": "highway-fast-v0",
        "config": {
            "lanes_count": 4,
            "vehicles_count": 50,
            "vehicles_density": 2.0,
            "duration": 40,
            "action": _action(HIGHWAY_SPEEDS),
        },
    },
}

TRAIN_SCENARIOS = ("highway", "merge", "intersection")

# Common timing for every scenario: 2 Hz decisions (fine enough for TTC-based shielding),
# 6 Hz physics -> 3 simulation sub-steps per decision (highway-fast itself uses 5 Hz; ~40% faster
# than 10 Hz on CPU while keeping the decision period at exactly 0.5 s).
TIMING = {"policy_frequency": 2, "simulation_frequency": 6}


def scenario_config(name: str, overrides: dict | None = None) -> tuple[str, dict]:
    """Return ``(gym_id, env_config)`` with the shared observation spec injected."""
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {sorted(SCENARIOS)}")
    spec = copy.deepcopy(SCENARIOS[name])
    cfg = spec["config"]
    cfg["observation"] = observation_config()
    for k, v in TIMING.items():
        cfg.setdefault(k, v)
    cfg.setdefault("action", _action())
    cfg.setdefault("offscreen_rendering", True)
    if overrides:
        cfg.update(copy.deepcopy(overrides))
    return spec["id"], cfg
