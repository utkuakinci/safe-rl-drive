"""Aggregation of episode KPIs with confidence intervals across training seeds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

HEADLINE_METRICS = ("success", "collision", "mean_speed", "jerk_rms", "shield_rate")
PERCENT_METRICS = {"success", "collision", "offroad", "shield_rate"}


def mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and two-sided Student-t confidence interval."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    m = float(values.mean())
    if len(values) < 2:
        return m, np.nan, np.nan
    half = float(stats.sem(values) * stats.t.ppf((1 + confidence) / 2.0, len(values) - 1))
    return m, m - half, m + half


def aggregate(
    episodes: pd.DataFrame,
    group_cols: list[str],
    metrics: tuple[str, ...] = HEADLINE_METRICS,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Two-level aggregation: episodes -> per-seed means -> mean and CI across seeds.

    The CI therefore reflects *training* variance (what a reviewer cares about when comparing
    algorithms), not just evaluation noise. With a single seed it falls back to a CI over episodes.
    """
    metrics = tuple(m for m in metrics if m in episodes.columns)
    rows = []
    for key, grp in episodes.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key, strict=True))
        per_seed = grp.groupby("seed")[list(metrics)].mean()
        row["n_seeds"] = len(per_seed)
        row["n_episodes"] = len(grp)
        for m in metrics:
            sample = per_seed[m].to_numpy() if len(per_seed) > 1 else grp[m].to_numpy()
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = mean_ci(sample, confidence)
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(m: str, mean: float, lo: float, hi: float) -> str:
    if np.isnan(mean):
        return "-"
    scale, unit = (100.0, "%") if m in PERCENT_METRICS else (1.0, "")
    if np.isnan(lo):
        return f"{mean * scale:.1f}{unit}"
    if m in PERCENT_METRICS:  # rates live in [0, 1]
        lo, hi = max(lo, 0.0), min(hi, 1.0)
    return f"{mean * scale:.1f}{unit} [{lo * scale:.1f}, {hi * scale:.1f}]"


def to_markdown(
    summary: pd.DataFrame, group_cols: list[str], metrics: tuple[str, ...] = HEADLINE_METRICS
) -> str:
    metrics = tuple(m for m in metrics if m in summary.columns)
    header = "| " + " | ".join(group_cols + list(metrics)) + " |"
    sep = "|" + "---|" * (len(group_cols) + len(metrics))
    lines = [header, sep]
    for _, r in summary.iterrows():
        cells = [str(r[c]) for c in group_cols] + [_fmt(m, r[m], r[f"{m}_lo"], r[f"{m}_hi"]) for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
