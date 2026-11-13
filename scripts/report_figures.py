"""Charts for the HTML report (``results/index.html``), drawn from the saved results only.

    python scripts/report_figures.py            # -> results/figures/*.png

Colour encoding is the same in every chart: shield off = orange, shield on = blue
(palette checked with the dataviz validator: CVD ΔE 24.7, contrast >= 3:1 on the surface).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path("results")
OUT = RESULTS / "figures"

SURFACE, INK, SECONDARY, MUTED, HAIRLINE, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
OFF, ON = "#eb6834", "#2a78d6"
LABELS = {"ppo": "PPO", "rppo": "RecurrentPPO", "ppo_pomdp": "PPO, velocities hidden",
          "rppo_pomdp": "RecurrentPPO, velocities hidden"}  # fmt: skip


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "xtick.color": SECONDARY,
            "ytick.color": SECONDARY,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": HAIRLINE,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.labelcolor": SECONDARY,
        }
    )


def _paired_hbars(ax, labels, off, on, off_err=None, on_err=None, hatch_on=None, hatch_off=None):
    """Horizontal off/on bar pairs with value labels; optional CI whiskers and hatched sub-segments."""
    y = np.arange(len(labels))[::-1]
    h = 0.36
    for vals, err, dy, color, name, hatch in (
        (off, off_err, h / 2 + 0.02, OFF, "Shield off", hatch_off),
        (on, on_err, -h / 2 - 0.02, ON, "Shield on", hatch_on),
    ):
        vals = 100 * np.asarray(vals)
        ax.barh(y + dy, vals, height=h, color=color, label=name, zorder=3)
        if hatch is not None:
            ax.barh(y + dy, 100 * np.asarray(hatch), height=h, color="none", edgecolor=SURFACE,
                    hatch="////", linewidth=0, zorder=4)  # fmt: skip
        label_x = vals
        if err is not None:
            lo, hi = np.clip(100 * np.asarray(err[0]), 0, 100), np.clip(100 * np.asarray(err[1]), 0, 100)
            ax.hlines(y + dy, lo, hi, color=SECONDARY, linewidth=1, zorder=5)
            label_x = np.maximum(vals, hi)  # label after the whisker, never on top of it
        for yy, v, x in zip(y + dy, vals, label_x, strict=True):
            ax.annotate(f"{v:.0f}%", (x, yy), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=9, color=SECONDARY, zorder=6)  # fmt: skip
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def shield_effect():
    rows, off, on, off_e, on_e = [], [], [], ([], []), ([], [])
    for scen in ("highway", "merge", "intersection"):
        df = pd.read_csv(RESULTS / scen / "summary.csv")
        nom = df[df.condition == "nominal"]
        for run in LABELS:
            if run not in set(nom.run):
                continue
            o = nom[(nom.run == run) & (nom.shield == "off")].iloc[0]
            n = nom[(nom.run == run) & (nom.shield == "on")].iloc[0]
            rows.append(f"{scen} · {LABELS[run]}")
            off.append(o.collision)
            on.append(n.collision)
            off_e[0].append(o.collision_lo)
            off_e[1].append(o.collision_hi)
            on_e[0].append(n.collision_lo)
            on_e[1].append(n.collision_hi)
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    _paired_hbars(ax, rows, off, on, off_e, on_e)
    ax.set_title("Collision rate, nominal conditions")
    ax.set_xlabel("Episodes ending in a collision (lines: 95% interval across 3 training seeds)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "shield_effect.png", dpi=160)
    plt.close(fig)


def robustness():
    df = pd.read_csv(RESULTS / "highway" / "summary.csv")
    order = ["nominal", "noise_0.02", "noise_0.05", "noise_0.10", "dropout_0.2", "delay_1step",
             "density_x1.5", "density_x2", "unseen:highway-dense", "unseen:merge", "unseen:roundabout",
             "unseen:intersection"]  # fmt: skip
    names = {"nominal": "Nominal", "noise_0.02": "Sensor noise σ 0.02", "noise_0.05": "Sensor noise σ 0.05",
             "noise_0.10": "Sensor noise σ 0.10", "dropout_0.2": "20% missed detections",
             "delay_1step": "1-step actuation delay", "density_x1.5": "Traffic ×1.5", "density_x2": "Traffic ×2",
             "unseen:highway-dense": "Unseen: dense highway", "unseen:merge": "Unseen: merge",
             "unseen:roundabout": "Unseen: roundabout", "unseen:intersection": "Unseen: intersection"}  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 6.0), sharey=True)
    for ax, run in zip(axes, ("ppo", "rppo"), strict=True):
        sel = df[df.run == run].set_index(["condition", "shield"])
        off = [sel.loc[(c, "off"), "collision"] for c in order]
        on = [sel.loc[(c, "on"), "collision"] for c in order]
        _paired_hbars(ax, [names[c] for c in order], off, on)
        ax.set_title(LABELS[run], fontsize=11)
    axes[0].legend(loc="lower right")
    fig.suptitle("Highway policies: collision rate under perturbations", x=0.01, ha="left",
                 fontweight="bold", fontsize=12, color=INK)  # fmt: skip
    fig.supxlabel(
        "Episodes ending in a collision (3 seeds × 20 episodes per bar)", fontsize=10, color=SECONDARY
    )
    fig.tight_layout()
    fig.savefig(OUT / "robustness_highway.png", dpi=160)
    plt.close(fig)


def latency():
    bench = json.loads((RESULTS / "benchmark_rppo_highway" / "benchmark.json").read_text())
    rows = bench["rows"] if isinstance(bench, dict) and "rows" in bench else bench
    rows = [r for r in rows if r["batch"] == 1]
    # One colour: orange / blue mean "shield off / on" everywhere else in the report.
    names = {"pytorch-cpu-fp32": "CPU · PyTorch FP32", "onnxruntime-cpu-fp32": "CPU · ONNX Runtime FP32",
             "pytorch-cuda-fp32": "GPU · PyTorch FP32", "pytorch-cuda-fp16": "GPU · PyTorch FP16",
             "tensorrt-cuda-fp32": "GPU · TensorRT FP32", "tensorrt-cuda-fp16": "GPU · TensorRT FP16",
             "tensorrt-cuda-int8": "GPU · TensorRT INT8"}  # fmt: skip
    rows.sort(key=lambda r: r["p50_us"])
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    y = np.arange(len(rows))[::-1]
    p50 = np.array([r["p50_us"] for r in rows])
    p99 = np.array([r["p99_us"] for r in rows])
    ax.barh(y, p50, height=0.55, color=SECONDARY, zorder=3)
    ax.scatter(p99, y, marker="|", s=220, color=INK, linewidths=1.5, zorder=4)
    for yy, a, b in zip(y, p50, p99, strict=True):
        ax.annotate(f"{a:.0f} µs", (b, yy), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=SECONDARY)  # fmt: skip
    ax.set_yticks(y, [names.get(r["backend"], r["backend"]) for r in rows])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, p99.max() * 1.22)
    ax.set_title("Single-decision latency, RecurrentPPO actor, batch 1")
    ax.set_xlabel("Microseconds, host to host (bar: median, tick: p99). CPU: 1 thread. GPU: RTX A2000 Laptop")
    fig.tight_layout()
    fig.savefig(OUT / "latency.png", dpi=160)
    plt.close(fig)


def metadrive():
    labels, off, on, off_p, on_p = [], [], [], [], []
    for run in ("ppo", "rppo"):
        s = json.loads((RESULTS / "metadrive" / run / "summary.json").read_text())
        labels.append(LABELS[run])
        off.append(s["shield_off"]["collision"])
        on.append(s["shield_on"]["collision"])
        off_p.append(s["shield_off"]["collision_after_traffic_pileup"])
        on_p.append(s["shield_on"]["collision_after_traffic_pileup"])
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(9.8, 2.9))
    _paired_hbars(ax, labels, off, on, hatch_off=off_p, hatch_on=on_p)
    ax.set_title("MetaDrive, no retraining: collision rate (20 episodes per bar)")
    ax.set_xlabel("Episodes ending in a collision")
    handles = [Patch(color=OFF, label="Shield off"), Patch(color=ON, label="Shield on"),
               Patch(facecolor=MUTED, edgecolor=SURFACE, hatch="////", linewidth=0,
                     label="Hatched: after a traffic\npile-up within 60 m")]  # fmt: skip
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "metadrive.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    for fn in (shield_effect, robustness, latency, metadrive):
        fn()
        print(f"wrote {fn.__name__}")


if __name__ == "__main__":
    main()
