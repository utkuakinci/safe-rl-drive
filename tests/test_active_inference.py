import numpy as np
import pytest

pytest.importorskip("pymdp")

from safe_rl_drive.active_inference.aif_agent import ActiveInferenceDriver, generative_model  # noqa: E402
from safe_rl_drive.active_inference.lane_pomdp import LaneObstacleEnv  # noqa: E402


def test_generative_model_is_normalised():
    A, B, _, D = generative_model()
    for a in A:
        np.testing.assert_allclose(a.sum(axis=0), 1.0)
    for b in B:
        np.testing.assert_allclose(b.sum(axis=0), 1.0)
    for d in D:
        assert abs(d.sum() - 1) < 1e-9


def test_active_inference_beats_chance():
    env, agent = LaneObstacleEnv(), ActiveInferenceDriver()
    crashes = 0
    for ep in range(40):
        _, info = env.reset(seed=ep)
        agent.reset()
        done = False
        while not done:
            _, _, done, _, info = env.step(agent.act(info["raw"]))
        crashes += info["outcome"] == "crash"
    assert crashes / 40 < 0.15  # chance level is ~1/3
