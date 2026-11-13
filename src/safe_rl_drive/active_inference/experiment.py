"""Active inference vs PPO vs RecurrentPPO on the lane-obstacle POMDP.

Questions:
1. Sample efficiency - how many environment steps do PPO / RecurrentPPO need to match an
   active-inference agent that is handed the generative model (zero samples)?
2. Robustness - what happens when the detector is worse at test time than assumed / trained on?
3. Cost - decision latency per step (EFE planning vs a forward pass).

    python -m safe_rl_drive.active_inference.experiment --budgets 2000 10000 50000 --seeds 0 1 2 \
        --episodes 500 --out results/active_inference
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from safe_rl_drive.active_inference.lane_pomdp import DEFAULT_ACCURACY, LaneObstacleEnv

DEGRADED_ACCURACY = (1.0, 0.6, 0.55, 0.5, 0.45, 0.4)
CONDITIONS = {"nominal": DEFAULT_ACCURACY, "degraded_sensor": DEGRADED_ACCURACY}


def evaluate(act_fn, reset_fn, accuracy, episodes: int, seed: int = 100_000) -> dict:
    env = LaneObstacleEnv(accuracy=accuracy)
    crashes, changes, t, n = [], [], 0.0, 0
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        reset_fn()
        done = False
        while not done:
            t0 = time.perf_counter()
            a = act_fn(obs, info["raw"])
            t += time.perf_counter() - t0
            n += 1
            obs, _, done, _, info = env.step(a)
        crashes.append(info["outcome"] == "crash")
        changes.append(info["lane_changes"])
    return {
        "crash_rate": float(np.mean(crashes)),
        "lane_changes": float(np.mean(changes)),
        "ms_per_step": 1000 * t / n,
    }


def run_aif(episodes: int) -> list[dict]:
    from safe_rl_drive.active_inference.aif_agent import ActiveInferenceDriver

    rows = []
    agent = ActiveInferenceDriver(accuracy=DEFAULT_ACCURACY)  # model assumes the nominal detector
    for cond, acc in CONDITIONS.items():
        r = evaluate(lambda _o, raw: agent.act(raw), agent.reset, acc, episodes)
        rows.append({"agent": "active_inference", "train_steps": 0, "seed": 0, "condition": cond, **r})
        print(rows[-1])
    return rows


def run_rl(algo: str, budget: int, seed: int, episodes: int) -> list[dict]:
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    env = LaneObstacleEnv(accuracy=DEFAULT_ACCURACY)
    if algo == "ppo":
        model = PPO(
            "MlpPolicy", env, n_steps=256, batch_size=64, gamma=0.99, ent_coef=0.01, seed=seed, verbose=0
        )
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            n_steps=128,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.01,
            seed=seed,
            verbose=0,
            policy_kwargs={"lstm_hidden_size": 32, "net_arch": [64]},
        )
    model.learn(budget)

    state = {"s": None, "start": np.ones(1, dtype=bool)}

    def reset():
        state["s"], state["start"] = None, np.ones(1, dtype=bool)

    def act(obs, _raw):
        a, state["s"] = model.predict(obs, state=state["s"], episode_start=state["start"], deterministic=True)
        state["start"] = np.zeros(1, dtype=bool)
        return int(a)

    rows = []
    for cond, acc in CONDITIONS.items():
        r = evaluate(act, reset, acc, episodes)
        rows.append({"agent": algo, "train_steps": budget, "seed": seed, "condition": cond, **r})
        print(rows[-1])
    return rows


def plot(df: pd.DataFrame, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, cond in zip(axes, CONDITIONS, strict=True):
        d = df[df.condition == cond]
        for algo, color in (("ppo", "#3b6fd8"), ("rppo", "#d8853b")):
            g = d[d.agent == algo].groupby("train_steps")["crash_rate"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                g.train_steps,
                g["mean"],
                yerr=g["std"],
                marker="o",
                capsize=3,
                label=algo.upper(),
                color=color,
            )
        aif = d[d.agent == "active_inference"]["crash_rate"].mean()
        ax.axhline(aif, color="#2a9d5c", ls="--", label="Active inference (0 samples)")
        ax.set_xscale("log")
        ax.set_title(cond.replace("_", " "))
        ax.set_xlabel("training environment steps")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("crash rate")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budgets", type=int, nargs="+", default=[2000, 10000, 50000])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--algos", nargs="+", default=["ppo", "rppo"])
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--out", type=Path, default=Path("results/active_inference"))
    a = p.parse_args(argv)

    rows = run_aif(a.episodes)
    for algo in a.algos:
        for budget in a.budgets:
            for seed in a.seeds:
                rows += run_rl(algo, budget, seed, a.episodes)
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "results.csv", index=False)
    summary = (
        df.groupby(["agent", "train_steps", "condition"])[["crash_rate", "lane_changes", "ms_per_step"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.to_csv(a.out / "summary.csv")
    (a.out / "summary.md").write_text(
        summary.to_markdown() if hasattr(summary, "to_markdown") else summary.to_string()
    )
    (a.out / "config.json").write_text(json.dumps(vars(a), default=str, indent=2))
    plot(df, a.out / "crash_vs_samples.png")
    print(summary)


if __name__ == "__main__":
    main()
