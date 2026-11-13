"""Evaluation conditions: nominal, sensor noise, missed detections, traffic density, latency, unseen."""

from __future__ import annotations

from safe_rl_drive.envs.factory import EnvCondition
from safe_rl_drive.envs.scenarios import SCENARIOS

DENSITY_OVERRIDES: dict[str, dict[str, dict]] = {
    "highway": {
        "density_x1.5": {"vehicles_density": 1.5, "vehicles_count": 45},
        "density_x2": {"vehicles_density": 2.0, "vehicles_count": 60},
    },
    "intersection": {
        "density_x1.5": {"spawn_probability": 0.8, "initial_vehicle_count": 15},
        "density_x2": {"spawn_probability": 1.0, "initial_vehicle_count": 20},
    },
}


def nominal_suite(scenario: str, mask_features=()) -> list[EnvCondition]:
    return [EnvCondition("nominal", scenario, mask_features=tuple(mask_features))]


def robustness_suite(scenario: str, mask_features=(), include_unseen: bool = True) -> list[EnvCondition]:
    m = tuple(mask_features)
    conds = [
        EnvCondition("nominal", scenario, mask_features=m),
        EnvCondition("noise_0.02", scenario, obs_noise_std=0.02, mask_features=m),
        EnvCondition("noise_0.05", scenario, obs_noise_std=0.05, mask_features=m),
        EnvCondition("noise_0.10", scenario, obs_noise_std=0.10, mask_features=m),
        EnvCondition("dropout_0.2", scenario, obs_dropout=0.2, mask_features=m),
        EnvCondition("delay_1step", scenario, action_delay=1, mask_features=m),
    ]
    for name, ov in DENSITY_OVERRIDES.get(scenario, {}).items():
        conds.append(EnvCondition(name, scenario, env_overrides=ov, mask_features=m))
    if include_unseen:
        for other in SCENARIOS:
            if other != scenario:
                conds.append(EnvCondition(f"unseen:{other}", other, mask_features=m))
    return conds


SUITES = {"nominal": nominal_suite, "robustness": robustness_suite}
