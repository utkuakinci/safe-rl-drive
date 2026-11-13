"""Offline evaluation: all runs x conditions x {shield off, shield on}, with CIs across seeds.

Example::

    python -m safe_rl_drive.evaluation.evaluate --runs runs/highway --suite robustness \
        --episodes 20 --shield both --workers 8 --out results/highway
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from safe_rl_drive.envs.factory import EnvCondition
from safe_rl_drive.evaluation.metrics import HEADLINE_METRICS, aggregate, to_markdown
from safe_rl_drive.evaluation.robustness import SUITES

_MODEL_CACHE: dict = {}


def discover_runs(root: Path) -> list[dict]:
    runs = []
    for meta_file in sorted(root.rglob("meta.json")):
        meta = json.loads(meta_file.read_text())
        meta["dir"] = str(meta_file.parent)
        runs.append(meta)
    return runs


def _job(run: dict, cond: dict, shield: bool, episodes: int, shield_cfg: dict | None) -> list[dict]:
    import torch

    torch.set_num_threads(1)
    from safe_rl_drive.evaluation.rollout import run_episodes
    from safe_rl_drive.policy import load_model

    if run["dir"] not in _MODEL_CACHE:
        _MODEL_CACHE[run["dir"]] = load_model(run["dir"])
    model = _MODEL_CACHE[run["dir"]]
    eps, _ = run_episodes(
        model, EnvCondition(**cond), episodes, shield_config=(shield_cfg or {}) if shield else None
    )
    for e in eps:
        e.update(
            run=run["name"],
            algo=run["algo"],
            seed=run["seed"],
            condition=cond["name"],
            shield="on" if shield else "off",
        )
    return eps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, required=True, help="directory searched recursively for meta.json")
    p.add_argument("--suite", choices=sorted(SUITES), default="nominal")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--shield", choices=["off", "on", "both"], default="both")
    p.add_argument("--shield-config", type=Path, default=Path("configs/shield.yaml"))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", type=Path, default=Path("results/eval"))
    p.add_argument("--wandb", action="store_true", help="log the summary table to W&B")
    args = p.parse_args(argv)

    runs = discover_runs(args.runs)
    if not runs:
        raise SystemExit(f"No runs (meta.json) found under {args.runs}")
    shield_cfg = yaml.safe_load(args.shield_config.read_text()) if args.shield_config.exists() else {}
    shields = {"off": [False], "on": [True], "both": [False, True]}[args.shield]

    jobs = []
    for run in runs:
        for cond in SUITES[args.suite](run["scenario"], mask_features=run.get("mask_features", ())):
            for sh in shields:
                jobs.append((run, asdict(cond), sh, args.episodes, shield_cfg))
    print(f"{len(runs)} runs x conditions x shield -> {len(jobs)} jobs on {args.workers} workers")

    rows: list[dict] = []
    if args.workers <= 1:
        for j in jobs:
            rows += _job(*j)
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(_job, *j) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                rows += f.result()
                print(f"  [{i}/{len(futs)}] done", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "episodes.csv", index=False)
    group = ["run", "condition", "shield"]
    summary = aggregate(df, group)
    summary.to_csv(args.out / "summary.csv", index=False)
    md = to_markdown(summary, group, HEADLINE_METRICS)
    (args.out / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)

    if args.wandb:
        import wandb

        with wandb.init(
            project="safe-rl-drive", job_type="evaluation", config=vars(args) | {"runs": str(args.runs)}
        ):
            wandb.log(
                {"eval/summary": wandb.Table(dataframe=summary), "eval/episodes": wandb.Table(dataframe=df)}
            )


if __name__ == "__main__":
    main()
