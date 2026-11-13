import json
import warnings

import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _save(model, tmp_path, algo, name):
    d = tmp_path / name
    d.mkdir()
    model.save(d / "model.zip")
    (d / "meta.json").write_text(json.dumps({"name": name, "algo": algo, "scenario": "highway", "seed": 0}))
    return d


@pytest.fixture(scope="session")
def untrained_runs(tmp_path_factory):
    """Untrained PPO + RecurrentPPO run dirs (architecture/plumbing tests, no learning)."""
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    from safe_rl_drive.envs import make_env

    tmp = tmp_path_factory.mktemp("runs")
    env = make_env("highway", seed=0)
    ppo = PPO("MlpPolicy", env, n_steps=64, batch_size=32, seed=0, policy_kwargs={"net_arch": [32, 32]})
    rppo = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=64,
        batch_size=32,
        seed=0,
        policy_kwargs={"lstm_hidden_size": 16, "net_arch": [16]},
    )
    return {"ppo": _save(ppo, tmp, "ppo", "ppo"), "rppo": _save(rppo, tmp, "rppo", "rppo")}
