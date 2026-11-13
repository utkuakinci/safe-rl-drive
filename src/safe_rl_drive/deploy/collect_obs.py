"""Record on-policy observations (+ LSTM states) for INT8 calibration and accuracy benchmarking.

    python -m safe_rl_drive.deploy.collect_obs --model runs/highway/rppo/seed_0 --scenario highway \
        --episodes 30 --out data/obs_highway_rppo.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def collect(
    model_path, scenario: str, episodes: int, seed: int = 20_000, mask_features=()
) -> dict[str, np.ndarray]:
    from safe_rl_drive.envs.factory import make_env
    from safe_rl_drive.policy import PolicyRunner, load_model

    runner = PolicyRunner(load_model(model_path), deterministic=False, seed=seed)
    env = make_env(scenario, seed=seed, mask_features=mask_features)
    obs_l, h_l, c_l, start_l = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        runner.reset()
        done = False
        while not done:
            obs_l.append(obs)
            start_l.append(runner.episode_start)
            if runner.recurrent:
                h, c = runner.state
                h_l.append(h[:, 0].numpy())
                c_l.append(c[:, 0].numpy())
            action, _ = runner.act(obs)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
    out = {"obs": np.asarray(obs_l, dtype=np.float32), "episode_start": np.asarray(start_l)}
    if h_l:  # stored as [N, L, H]
        out["h"] = np.asarray(h_l, dtype=np.float32)
        out["c"] = np.asarray(c_l, dtype=np.float32)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--scenario", default="highway")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--mask-features", nargs="*", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    data = collect(a.model, a.scenario, a.episodes, mask_features=a.mask_features)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **data)
    print(f"saved {len(data['obs'])} samples -> {a.out}")


if __name__ == "__main__":
    main()
