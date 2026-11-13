"""Render the README result tables from ``results/<scenario>/summary.csv``.

The tables are written between ``<!-- results:<name>:start -->`` / ``<!-- results:<name>:end -->`` markers
in README.md, so numbers in the README are never typed in by hand:

    python scripts/readme_tables.py            # rewrite README.md in place
    python scripts/readme_tables.py --print    # only print the tables
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safe_rl_drive.evaluation.metrics import _fmt  # noqa: E402

SCENARIOS = ("highway", "merge", "intersection")
RUN_LABELS = {
    "ppo": "PPO",
    "rppo": "RecurrentPPO",
    "ppo_pomdp": "PPO, velocities hidden",
    "rppo_pomdp": "RecurrentPPO, velocities hidden",
}
ROBUSTNESS_RUNS = ("ppo", "rppo")


def _cell(row: pd.Series, m: str) -> str:
    return _fmt(m, row[m], row[f"{m}_lo"], row[f"{m}_hi"])


def _pct(row: pd.Series, m: str) -> str:
    return f"{100 * row[m]:.0f} %"


def load(results: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for s in SCENARIOS:
        f = results / s / "summary.csv"
        if f.exists():
            out[s] = pd.read_csv(f)
    return out


def headline(data: dict[str, pd.DataFrame]) -> str:
    lines = [
        "| Scenario | Agent | Seeds × episodes | Success, shield off | Success, shield on "
        "| Collision, shield off | Collision, shield on | Speed off → on [m/s] | Shield interventions |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s, df in data.items():
        nom = df[df.condition == "nominal"]
        for run in [r for r in RUN_LABELS if r in set(nom.run)]:
            off = nom[(nom.run == run) & (nom.shield == "off")].iloc[0]
            on = nom[(nom.run == run) & (nom.shield == "on")].iloc[0]
            lines.append(
                f"| {s} | {RUN_LABELS[run]} | {int(off.n_seeds)} × {int(off.n_episodes / off.n_seeds)} "
                f"| {_cell(off, 'success')} | {_cell(on, 'success')} "
                f"| {_cell(off, 'collision')} | {_cell(on, 'collision')} "
                f"| {off.mean_speed:.1f} → {on.mean_speed:.1f} | {_pct(on, 'shield_rate')} |"
            )
    return "\n".join(lines)


def _cond_order(conds) -> list[str]:
    def key(c: str):
        group = 0 if c == "nominal" else 2 if c.startswith("unseen:") else 1
        return group, c

    return sorted(conds, key=key)


def robustness(data: dict[str, pd.DataFrame]) -> str:
    blocks = []
    for s, df in data.items():
        runs = [r for r in ROBUSTNESS_RUNS if r in set(df.run)]
        head = "| Condition | " + " | ".join(f"{RUN_LABELS[r]}: off → on" for r in runs) + " |"
        lines = [f"**{s}** — collision rate, shield off → on", "", head, "|---|" + "---|" * len(runs)]
        for c in _cond_order(df.condition.unique()):
            cells = []
            for r in runs:
                sel = df[(df.run == r) & (df.condition == c)].set_index("shield")
                cells.append(f"{_pct(sel.loc['off'], 'collision')} → {_pct(sel.loc['on'], 'collision')}")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def benchmark(results: Path, name: str = "benchmark_rppo_highway") -> str | None:
    f = results / name / "benchmark.md"
    if not f.exists():
        return None
    lines = f.read_text(encoding="utf-8").splitlines()
    table = [ln for ln in lines if ln.startswith("|")]
    hw = next((ln for ln in lines if ln.startswith("Hardware:")), "")
    return f"`{f.as_posix()}` — {hw}\n\n" + "\n".join(table)


def metadrive(results: Path) -> str | None:
    import json

    rows = []
    for run in ("ppo", "rppo"):
        f = results / "metadrive" / run / "summary.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        for mode in ("off", "on"):
            r = m.get(f"shield_{mode}")
            if r is None:
                continue
            rows.append(
                f"| {RUN_LABELS[run]} | {mode} | {m['episodes']} | {100 * r['success']:.0f} % "
                f"| {100 * r['collision']:.0f} % | {100 * r['collision_after_traffic_pileup']:.0f} % "
                f"| {100 * r['offroad']:.0f} % | {r['mean_speed']:.1f} | {100 * r['shield_rate']:.0f} % |"
            )
    if not rows:
        return None
    head = [
        "| Agent (highway-env, 200k steps) | Shield | Episodes | Success | Collision "
        "| ↳ after a traffic pile-up | Off-road | Speed [m/s] | Shield interventions |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    return "\n".join(head + rows)


def replace_block(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- results:{name}:start -->\n).*?(<!-- results:{name}:end -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"README is missing the results:{name} markers")
    return pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--print", action="store_true", dest="only_print")
    a = p.parse_args(argv)

    data = load(a.results)
    if not data:
        raise SystemExit(f"no results/<scenario>/summary.csv under {a.results}")
    tables = {"headline": headline(data), "robustness": robustness(data)}
    for name, fn in (("benchmark", benchmark), ("metadrive", metadrive)):
        block = fn(a.results)
        if block:
            tables[name] = block
    if a.only_print:
        print("\n\n".join(tables.values()))
        return
    text = a.readme.read_text(encoding="utf-8")
    for name, body in tables.items():
        text = replace_block(text, name, body)
    a.readme.write_text(text, encoding="utf-8")
    print(f"updated {a.readme} ({', '.join(data)})")


if __name__ == "__main__":
    main()
