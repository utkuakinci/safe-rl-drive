from pathlib import Path

import numpy as np
import pytest

from safe_rl_drive.envs import make_env
from safe_rl_drive.envs.spec import ACTION_INDEX as A
from safe_rl_drive.safety import SafetyShield, ShieldConfig
from safe_rl_drive.safety.observer import SAFETY_FEATS, SAFETY_HEADER, SAFETY_OBS_SIZE

VECTORS = Path(__file__).parents[1] / "ros2_ws/src/rl_policy_node/test/shield_vectors.csv"


def road(speed=25.0, left=True, right=True, cars=()):
    x = np.zeros(SAFETY_OBS_SIZE, dtype=np.float32)
    x[:3] = speed, left, right
    for i, (s, d, vs, *vd) in enumerate(cars):
        b = SAFETY_HEADER + i * SAFETY_FEATS
        x[b : b + 5] = 1, s, d, vs, (vd[0] if vd else 0)
    return x


def test_empty_road_is_transparent():
    sh = SafetyShield()
    for a in range(5):
        assert sh.filter(a, road()).final == a


def test_closing_leader_blocks_faster_and_idle():
    d = SafetyShield().filter(A["FASTER"], road(cars=[(20, 0, -8)]))
    assert d.intervened and d.final not in (A["FASTER"], A["IDLE"])
    assert "front_ttc" in d.unsafe["FASTER"]


def test_aeb_on_imminent_collision():
    d = SafetyShield().filter(A["IDLE"], road(cars=[(12, 0, -10)]))
    assert d.aeb and d.final == A["SLOWER"]


def test_occupied_target_lane():
    sh = SafetyShield()
    x = road(cars=[(2, -4, 0)])
    assert sh.filter(A["LANE_LEFT"], x).final == A["IDLE"]
    assert not sh.filter(A["LANE_RIGHT"], x).intervened


def test_lane_change_without_target_lane_inherits_idle_status():
    x = road(left=False, cars=[(8, 0, -2)])  # leader too close -> IDLE unsafe
    assert "LANE_LEFT" in SafetyShield().assess(x)[0]


def test_probabilities_choose_most_likely_safe_action():
    probs = np.array([0.05, 0.1, 0.3, 0.5, 0.05])
    assert SafetyShield().filter(A["FASTER"], road(cars=[(20, 0, -8)]), probs).final == A["LANE_RIGHT"]


def test_disabled_shield():
    assert (
        SafetyShield(ShieldConfig(enabled=False)).filter(A["FASTER"], road(cars=[(6, 0, -10)])).final
        == A["FASTER"]
    )


def test_stats_count_interventions():
    sh = SafetyShield()
    sh.filter(A["FASTER"], road(cars=[(20, 0, -8)]))
    sh.filter(A["IDLE"], road())
    s = sh.stats.to_dict()
    assert s["steps"] == 2 and s["interventions"] == 1 and s["rule/front_ttc"] == 1


@pytest.mark.skipif(not VECTORS.exists(), reason="vectors not generated")
def test_python_matches_cpp_reference_vectors():
    """The CSV is consumed by the C++ gtest; if this fails, regenerate it (scripts/gen_shield_vectors.py)."""
    sh = SafetyShield()
    rows = [line for line in VECTORS.read_text().splitlines() if line and not line.startswith("#")]
    assert len(rows) > 100
    for line in rows:
        v = np.array(line.split(","), dtype=np.float32)
        d = sh.filter(int(v[0]), v[3:])
        assert (d.final, int(d.aeb)) == (int(v[1]), int(v[2]))


@pytest.mark.slow
def test_shield_prevents_crashes_of_reckless_policy():
    """An always-FASTER policy crashes on the highway; with the shield it must not."""
    for seed in range(3):
        env = make_env("highway", seed=seed, shield=SafetyShield())
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, te, tr, info = env.step(A["FASTER"])
            done = te or tr
        assert info["episode_metrics"]["collision"] == 0.0
