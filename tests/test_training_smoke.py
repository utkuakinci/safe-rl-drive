import json

import pytest

from safe_rl_drive.training.train import load_config, train


@pytest.mark.slow
def test_smoke_training_writes_artifacts(tmp_path):
    cfg = load_config("configs/train/smoke.yaml", ["total_timesteps=256", "eval_freq=256"])
    out = train(cfg, seed=0, out_root=tmp_path, wandb_mode="disabled")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["algo"] == "ppo" and (out / "model.zip").exists()


def test_config_inheritance():
    cfg = load_config("configs/train/merge_rppo.yaml")
    assert cfg["algo"] == "rppo" and cfg["scenario"] == "merge"
    assert cfg["policy_kwargs"]["lstm_hidden_size"] == 128
    assert cfg["hyperparams"]["gamma"] == 0.85  # inherited from highway_ppo.yaml
    assert cfg["hyperparams"]["n_steps"] == 128  # overridden by highway_rppo.yaml
