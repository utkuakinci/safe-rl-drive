import numpy as np
import pytest

from safe_rl_drive.envs import SCENARIOS, make_env
from safe_rl_drive.envs.spec import OBS_SHAPE, denormalize, normalize
from safe_rl_drive.safety.observer import SAFETY_OBS_SIZE, safety_observation


def test_normalize_roundtrip():
    rng = np.random.default_rng(0)
    raw = rng.uniform(-30, 30, size=OBS_SHAPE).astype(np.float32)
    raw[:, 0] = 1.0
    raw[:, 5:] = rng.uniform(-1, 1, size=(OBS_SHAPE[0], 2))
    np.testing.assert_allclose(denormalize(normalize(raw)), raw, atol=1e-4)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_every_scenario_shares_the_interface(scenario):
    env = make_env(scenario, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == OBS_SHAPE
    assert env.action_space.n == 5
    assert env.unwrapped.config["policy_frequency"] == 2
    assert np.all(np.abs(obs) <= 1.0)
    assert safety_observation(env.unwrapped).shape == (SAFETY_OBS_SIZE,)


def test_highway_env_normalisation_matches_spec():
    """Our normalize() must reproduce highway-env's own normalisation (CARLA demo relies on it)."""
    env = make_env("highway", seed=1)
    env.reset(seed=1)
    ob_type = env.unwrapped.observation_type
    normalized = ob_type.observe()
    ob_type.normalize = False
    raw = ob_type.observe()
    ob_type.normalize = True
    np.testing.assert_allclose(normalize(raw), normalized, atol=1e-5)
