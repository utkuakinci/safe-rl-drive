import numpy as np
import pandas as pd

from safe_rl_drive.envs import make_env
from safe_rl_drive.evaluation.metrics import aggregate, mean_ci, to_markdown


def _episode(env, action=1):
    env.reset(seed=0)
    done = False
    while not done:
        _, _, te, tr, info = env.step(action)
        done = te or tr
    return info


def test_metrics_wrapper_reports_kpis():
    m = _episode(make_env("merge", seed=0))["episode_metrics"]
    for k in ("success", "collision", "mean_speed", "jerk_rms", "lane_changes", "length"):
        assert k in m
    assert m["success"] + m["collision"] <= 1.0


def test_mask_wrapper_hides_velocities():
    obs, _ = make_env("highway", seed=0, mask_features=("vx", "vy")).reset(seed=0)
    assert np.all(obs[:, 3:5] == 0)


def test_noise_only_touches_present_vehicles():
    a, _ = make_env("highway", seed=3).reset(seed=3)
    b, _ = make_env("highway", seed=3, obs_noise_std=0.1).reset(seed=3)
    absent = a[:, 0] < 0.5
    np.testing.assert_array_equal(a[absent], b[absent])
    assert not np.allclose(a[~absent], b[~absent])


def test_action_delay_executes_previous_action():
    env = make_env("highway", seed=0, action_delay=1)
    env.reset(seed=0)
    lane0 = env.unwrapped.vehicle.lane_index
    env.step(0)  # LANE_LEFT is queued, IDLE executed
    assert env.unwrapped.vehicle.target_lane_index == lane0


def test_mean_ci_and_aggregate():
    m, lo, hi = mean_ci(np.array([0.8, 0.9, 1.0]))
    assert lo < m < hi and abs(m - 0.9) < 1e-9
    df = pd.DataFrame(
        {
            "run": ["a"] * 6,
            "seed": [0, 0, 1, 1, 2, 2],
            "success": [1, 1, 1, 0, 0, 1.0],
            "collision": [0, 0, 0, 1, 1, 0.0],
        }
    )
    s = aggregate(df, ["run"])
    assert s.loc[0, "n_seeds"] == 3
    assert "success" in to_markdown(s, ["run"])
