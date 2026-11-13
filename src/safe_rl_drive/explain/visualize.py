"""Render an explained episode: simulator frame | action distribution | attribution heatmap.

    python -m safe_rl_drive.explain.visualize --model runs/highway/rppo/seed_0 --scenario highway \
        --shield --out results/explain_highway.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _figure_frame(frame, exp, final_action, actions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from safe_rl_drive.envs.spec import FEATURES

    fig = plt.figure(figsize=(12, 5.4), dpi=90)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], height_ratios=[1, 1.3])
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(frame)
    ax_img.axis("off")

    ax_p = fig.add_subplot(gs[0, 1])
    names = [actions[i] for i in range(len(exp.probs))]
    colors = ["#8a8f98"] * len(names)
    colors[exp.action] = "#3b6fd8"
    if final_action != exp.action:
        colors[final_action] = "#d8453b"
    ax_p.barh(names, exp.probs, color=colors)
    ax_p.set_xlim(0, 1)
    ax_p.invert_yaxis()
    ax_p.set_title("policy (blue) / shield override (red)", fontsize=9)
    ax_p.tick_params(labelsize=8)

    ax_h = fig.add_subplot(gs[1, 1])
    lim = max(np.abs(exp.attribution).max(), 1e-6)
    ax_h.imshow(exp.attribution, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax_h.set_xticks(range(len(FEATURES)), FEATURES, fontsize=7, rotation=30)
    ax_h.set_yticks(
        range(exp.attribution.shape[0]),
        ["ego"] + [f"#{i}" for i in range(1, exp.attribution.shape[0])],
        fontsize=7,
    )
    ax_h.set_title("integrated gradients (chosen logit)", fontsize=9)

    fig.suptitle(exp.text, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def explain_episode(model_path, scenario, out, shield=False, seed=0, max_steps=200, fps=4, mask_features=()):
    import imageio.v2 as imageio

    from safe_rl_drive.envs.factory import find_wrapper, make_env
    from safe_rl_drive.envs.spec import ACTIONS
    from safe_rl_drive.explain.attribution import PolicyExplainer
    from safe_rl_drive.policy import PolicyRunner, load_model
    from safe_rl_drive.safety.shield import SafetyShield
    from safe_rl_drive.safety.wrapper import ShieldWrapper

    runner = PolicyRunner(load_model(model_path))
    explainer = PolicyExplainer(runner)
    env = make_env(
        scenario,
        seed=seed,
        render_mode="rgb_array",
        shield=SafetyShield() if shield else None,
        mask_features=mask_features,
    )
    shield_env = find_wrapper(env, ShieldWrapper)
    obs, _ = env.reset(seed=seed)
    runner.reset()
    frames, log = [], []
    for _ in range(max_steps):
        state, start = runner.state, runner.episode_start
        action, probs = runner.act(obs)
        if shield_env is not None:
            shield_env.set_action_probs(probs)
        frame = env.render()
        obs_before = obs
        obs, _, term, trunc, info = env.step(action)
        sh = info.get("shield", {})
        exp = explainer.explain(obs_before, action, probs, state, start, sh.get("reason", ""))
        frames.append(_figure_frame(frame, exp, sh.get("final", action), ACTIONS))
        log.append(exp.text)
        if term or trunc:
            break
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps) if str(out).endswith(".gif") else imageio.mimsave(
        out, frames, fps=fps, macro_block_size=1
    )
    Path(out).with_suffix(".txt").write_text("\n".join(log), encoding="utf-8")
    print(f"wrote {out} ({len(frames)} frames); final metrics: {info.get('episode_metrics')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--scenario", default="highway")
    p.add_argument("--shield", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--mask-features", nargs="*", default=[])
    p.add_argument("--out", default="results/explain.mp4")
    a = p.parse_args(argv)
    explain_episode(a.model, a.scenario, a.out, a.shield, a.seed, a.max_steps, mask_features=a.mask_features)


if __name__ == "__main__":
    main()
