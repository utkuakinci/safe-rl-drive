"""Train PPO / RecurrentPPO on a highway-env scenario with W&B tracking.

Examples::

    # one seed, tracked in W&B
    python -m safe_rl_drive.training.train --config configs/train/highway_ppo.yaml --seed 0

    # all seeds listed in the config, offline W&B, override hyper-parameters
    python -m safe_rl_drive.training.train --config configs/train/highway_rppo.yaml --all-seeds \
        --wandb-mode offline --set hyperparams.learning_rate=3e-4 total_timesteps=300000

    # inside a W&B sweep agent (see configs/sweeps/)
    python -m safe_rl_drive.training.train --config configs/train/highway_ppo.yaml --sweep
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import platform
import time
from pathlib import Path

import yaml

POLICY_KWARGS_KEYS = {"lstm_hidden_size", "n_lstm_layers", "net_arch", "shared_lstm", "enable_critic_lstm"}


def _deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config; ``extends: other.yaml`` (relative path) is deep-merged first."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "extends" in cfg:
        cfg = _deep_merge(load_config(path.parent / cfg.pop("extends")), cfg)
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def _apply_sweep_params(cfg: dict, params: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for k, v in params.items():
        if k in POLICY_KWARGS_KEYS:
            cfg.setdefault("policy_kwargs", {})[k] = v
        elif k in ("total_timesteps", "n_envs"):
            cfg[k] = v
        else:
            cfg.setdefault("hyperparams", {})[k] = v
    return cfg


def train(
    cfg: dict, seed: int, out_root: Path = Path("runs"), wandb_mode: str = "online", sweep: bool = False
) -> Path:
    import torch
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from safe_rl_drive.envs.factory import EnvCondition, make_env
    from safe_rl_drive.policy import ALGOS
    from safe_rl_drive.safety.shield import SafetyShield, ShieldConfig
    from safe_rl_drive.training.callbacks import DrivingEvalCallback, ShieldTrainingLogger

    run = None
    if wandb_mode != "disabled" or sweep:
        import wandb

        run = wandb.init(
            project=cfg.get("wandb", {}).get("project", "safe-rl-drive"),
            entity=cfg.get("wandb", {}).get("entity"),
            group=f"{cfg['scenario']}/{cfg['name']}",
            name=f"{cfg['name']}-s{seed}",
            tags=[cfg["scenario"], cfg["algo"], *cfg.get("wandb", {}).get("tags", [])],
            config=cfg | {"seed": seed},
            mode="online" if sweep else wandb_mode,
            sync_tensorboard=True,
            save_code=True,
        )
        if sweep:
            cfg = _apply_sweep_params(cfg, dict(wandb.config))

    set_random_seed(seed)
    torch.set_num_threads(max(1, int(cfg.get("torch_threads", 1))))
    out_dir = out_root / cfg["scenario"] / cfg["name"] / f"seed_{seed}"
    if run is not None and sweep:
        out_dir = out_root / "sweeps" / run.id
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = cfg.get("env", {})
    mask = tuple(env_cfg.get("mask_features", ()))
    shield_train = cfg.get("shield_in_training")
    env_kwargs = dict(
        scenario=cfg["scenario"],
        env_overrides=env_cfg.get("overrides", {}),
        obs_noise_std=env_cfg.get("obs_noise_std", 0.0),
        obs_dropout=env_cfg.get("obs_dropout", 0.0),
        mask_features=mask,
    )

    def _make(rank: int):
        def _init():
            shield = SafetyShield(ShieldConfig.from_dict(shield_train)) if shield_train is not None else None
            return make_env(seed=seed * 1000 + rank, shield=shield, **env_kwargs)

        return _init

    n_envs = int(cfg.get("n_envs", 8))
    vec_cls = SubprocVecEnv if n_envs > 1 and cfg.get("subproc", True) else DummyVecEnv
    venv = make_vec_env(_make(0), n_envs=1) if n_envs == 1 else vec_cls([_make(i) for i in range(n_envs)])
    if n_envs > 1:
        from stable_baselines3.common.vec_env import VecMonitor

        venv = VecMonitor(venv)

    algo_cls = ALGOS[cfg["algo"]]
    policy = "MlpLstmPolicy" if cfg["algo"] == "rppo" else "MlpPolicy"
    # TensorBoard comes with the optional "tracking" extra; without it, train without curves.
    tb_log = str(out_dir / "tb") if importlib.util.find_spec("tensorboard") else None
    if tb_log is None:
        print("tensorboard not installed: training without TensorBoard logs (pip install '.[tracking]')")
    model = algo_cls(
        policy,
        venv,
        seed=seed,
        verbose=int(cfg.get("verbose", 1)),
        device=cfg.get("device", "cpu"),
        tensorboard_log=tb_log,
        policy_kwargs=cfg.get("policy_kwargs", {}),
        **cfg.get("hyperparams", {}),
    )

    callbacks = [
        DrivingEvalCallback(
            EnvCondition(
                "nominal", cfg["scenario"], env_overrides=env_kwargs["env_overrides"], mask_features=mask
            ),
            eval_freq=int(cfg.get("eval_freq", 20_000)),
            n_episodes=int(cfg.get("eval_episodes", 10)),
            save_dir=out_dir,
            shield_config=cfg.get("shield_eval"),
        ),
        CheckpointCallback(
            save_freq=max(int(cfg.get("checkpoint_freq", 100_000)) // n_envs, 1),
            save_path=str(out_dir / "ckpt"),
        ),
    ]
    if shield_train is not None:
        callbacks.append(ShieldTrainingLogger())
    if run is not None:
        from wandb.integration.sb3 import WandbCallback

        callbacks.append(WandbCallback(verbose=0))

    t0 = time.time()
    model.learn(
        total_timesteps=int(cfg["total_timesteps"]), callback=CallbackList(callbacks), progress_bar=False
    )
    model.save(out_dir / "model.zip")
    venv.close()

    meta = {
        "name": cfg["name"],
        "algo": cfg["algo"],
        "scenario": cfg["scenario"],
        "seed": seed,
        "mask_features": list(mask),
        "total_timesteps": int(cfg["total_timesteps"]),
        "train_seconds": round(time.time() - t0, 1),
        "python": platform.python_version(),
        "wandb_run": run.url if run is not None and wandb_mode == "online" else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if run is not None:
        run.finish()
    print(f"saved {out_dir}")
    return out_dir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--all-seeds", action="store_true", help="train every seed listed in the config")
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )
    p.add_argument(
        "--sweep", action="store_true", help="read hyper-parameters from wandb.config (sweep agent)"
    )
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. hyperparams.gamma=0.9")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.set)
    seeds = (
        cfg.get("seeds", [0])
        if args.all_seeds
        else [args.seed if args.seed is not None else cfg.get("seeds", [0])[0]]
    )
    for s in seeds:
        train(cfg, s, args.out, args.wandb_mode, args.sweep)


if __name__ == "__main__":
    main()
