import json

import numpy as np
import pytest
import torch

from safe_rl_drive.deploy.export_onnx import build_actor, export
from safe_rl_drive.envs import make_env
from safe_rl_drive.explain.attribution import PolicyExplainer, _baseline
from safe_rl_drive.policy import PolicyRunner, load_model


@pytest.mark.parametrize("algo", ["ppo", "rppo"])
def test_onnx_export_parity(untrained_runs, tmp_path, algo):
    meta = export(untrained_runs[algo], tmp_path / f"{algo}.onnx")
    assert meta["parity"]["action_agreement"] == 1.0
    assert meta["recurrent"] == (algo == "rppo")
    side = json.loads((tmp_path / f"{algo}.json").read_text())
    assert side["actions"][0] == "LANE_LEFT"


def test_exported_lstm_step_matches_sb3_rollout(untrained_runs):
    """Explicit-state actor (what ONNX / C++ run) == SB3 RecurrentPPO over an episode."""
    model = load_model(untrained_runs["rppo"])
    actor, info = build_actor(model)
    runner = PolicyRunner(model)
    env = make_env("highway", seed=5)
    obs, _ = env.reset(seed=5)
    h = torch.zeros(info["lstm_layers"], 1, info["lstm_hidden"])
    c = torch.zeros_like(h)
    for _ in range(15):
        a_ref, probs = runner.act(obs)
        with torch.no_grad():
            logits, h, c = actor(torch.as_tensor(obs[None]), h, c)
        np.testing.assert_allclose(torch.softmax(logits, -1)[0].numpy(), probs, atol=1e-5)
        obs, _, te, tr, _ = env.step(a_ref)
        if te or tr:
            break


@pytest.mark.parametrize("algo", ["ppo", "rppo"])
def test_integrated_gradients_completeness(untrained_runs, algo):
    runner = PolicyRunner(load_model(untrained_runs[algo]))
    obs, _ = make_env("highway", seed=0).reset(seed=0)
    action = 1
    exp = PolicyExplainer(runner, ig_steps=128)
    attr = exp.attribution(obs, action, runner.state, True)
    with torch.no_grad():

        def f(x):
            return exp._logit(torch.as_tensor(x[None]), action, runner.state, True).item()

        delta = f(obs) - f(_baseline())
    assert abs(attr.sum() - delta) < 0.05 * max(1.0, abs(delta))
    e = exp.explain(obs, action, np.full(5, 0.2))
    assert e.text.startswith("IDLE") and e.attribution.shape == obs.shape
