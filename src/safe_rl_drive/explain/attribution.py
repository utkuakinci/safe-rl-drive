"""Input attribution for driving policies: "why did the agent choose this action?".

Two attribution methods over the (vehicles x features) observation:

* gradient x input  - cheap, one backward pass (suitable for on-line logging);
* integrated gradients (Sundararajan et al., 2017) against an "empty road" baseline - satisfies
  completeness (attributions sum to logit(x) - logit(baseline)), so it is the one used in reports.

For recurrent policies the LSTM state is held fixed and attribution is w.r.t. the *current* frame.
Attributions are aggregated per vehicle and turned into a short natural-language explanation using
the de-normalised observation (distance, lane offset, relative speed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from safe_rl_drive.envs.spec import ACTIONS, FEATURES, OBS_SHAPE, denormalize
from safe_rl_drive.policy import PolicyRunner


@dataclass
class Explanation:
    action: int
    probs: np.ndarray
    attribution: np.ndarray  # (V, F), signed, for the chosen action's logit
    vehicle_importance: np.ndarray  # (V,), share of total |attribution|
    text: str
    shield_reason: str = ""
    top_features: list[tuple[int, str, float]] = field(default_factory=list)


def _baseline() -> np.ndarray:
    """Empty road: ego row kept zero except presence; no other vehicles."""
    b = np.zeros(OBS_SHAPE, dtype=np.float32)
    b[0, 0] = 1.0
    return b


class PolicyExplainer:
    def __init__(self, runner: PolicyRunner, method: str = "ig", ig_steps: int = 32):
        self.runner = runner
        self.method = method
        self.ig_steps = ig_steps

    def _logit(self, obs_t: torch.Tensor, action: int, state, start) -> torch.Tensor:
        if self.runner.recurrent:
            n = obs_t.shape[0]
            h, c = state if state is not None else self.runner.state
            logits, _ = self.runner.distribution(obs_t, (h.repeat(1, n, 1), c.repeat(1, n, 1)), start)
        else:
            logits, _ = self.runner.distribution(obs_t)
        return logits[:, action]

    def attribution(
        self, obs: np.ndarray, action: int, state=None, episode_start: bool = False
    ) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None])
        if self.method == "grad":
            x.requires_grad_(True)
            self._logit(x, action, state, episode_start).sum().backward()
            return (x.grad[0] * x[0]).detach().numpy()
        base = torch.as_tensor(_baseline()[None])
        alphas = torch.linspace(1.0 / self.ig_steps, 1.0, self.ig_steps).view(-1, 1, 1)
        path = (base + alphas * (x - base)).requires_grad_(True)
        self._logit(path, action, state, episode_start).sum().backward()
        return ((x - base)[0] * path.grad.mean(0)).detach().numpy()

    def explain(
        self,
        obs: np.ndarray,
        action: int,
        probs: np.ndarray,
        state=None,
        episode_start=False,
        shield_reason="",
    ) -> Explanation:
        attr = self.attribution(obs, action, state, episode_start)
        mag = np.abs(attr).sum(axis=1)
        share = mag / max(mag.sum(), 1e-9)
        flat = [
            (int(i), FEATURES[j], float(attr[i, j]))
            for i, j in zip(*np.unravel_index(np.argsort(-np.abs(attr), axis=None), attr.shape), strict=True)
        ]
        text = describe(obs, action, probs, share, flat[:3], shield_reason)
        return Explanation(action, probs, attr, share, text, shield_reason, flat[:5])


def describe(obs, action, probs, share, top, shield_reason="") -> str:
    raw = denormalize(obs)
    ranked = np.argsort(-share)
    lines = [f"{ACTIONS[action]} (p={probs[action]:.2f})"]
    for i in ranked[:2]:
        if share[i] < 0.1:
            continue
        if i == 0:
            lines.append(f"ego state ({share[i]:.0%}): v={raw[0, 3]:.1f} m/s")
        elif raw[i, 0] > 0.5:
            x, y, vx = raw[i, 1], raw[i, 2], raw[i, 3]
            where = "ahead" if x > 0 else "behind"
            lane = "same lane" if abs(y) < 2 else ("left" if y < 0 else "right")
            dist = ">100" if abs(x) >= 99 else f"{abs(x):.0f}"
            lines.append(f"car #{i} ({share[i]:.0%}): {dist} m {where}, {lane}, dv={vx:+.1f} m/s")
    if shield_reason:
        lines.append(f"SHIELD: {shield_reason}")
    return " | ".join(lines)
