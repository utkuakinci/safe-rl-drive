"""Model loading and a uniform single-step policy interface for PPO and RecurrentPPO.

Training uses SB3's own machinery; everything downstream (evaluation, shield, explainer, export,
benchmark) goes through :class:`PolicyRunner`, so feed-forward and recurrent policies are handled in
exactly one place.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

ALGOS = {"ppo": PPO, "rppo": RecurrentPPO}


def is_recurrent(model) -> bool:
    return isinstance(model, RecurrentPPO)


def load_model(path: str | Path, device: str = "cpu"):
    """Load ``model.zip`` (or a run directory containing it). The algo is read from ``meta.json``."""
    path = Path(path)
    if path.is_dir():
        path = path / "model.zip"
    meta_file = path.parent / "meta.json"
    if meta_file.exists():
        algo = json.loads(meta_file.read_text())["algo"]
        return ALGOS[algo].load(path, device=device)
    for cls in (PPO, RecurrentPPO):  # no metadata: try both
        try:
            return cls.load(path, device=device)
        except Exception:  # noqa: S112
            continue
    raise ValueError(f"Could not load model from {path}")


class PolicyRunner:
    """Stateful single-environment policy wrapper returning ``(action, probs)``."""

    def __init__(self, model, deterministic: bool = True, seed: int | None = None):
        self.model = model
        self.policy = model.policy
        self.policy.set_training_mode(False)
        self.recurrent = is_recurrent(model)
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.episode_start = True
        if self.recurrent:
            shape = self.policy.lstm_hidden_state_shape
            self.state = (torch.zeros(shape), torch.zeros(shape))
        else:
            self.state = None

    def distribution(self, obs: np.ndarray, state=None, episode_start: bool | None = None):
        """Forward pass. Returns ``(logits tensor [1, A], new_state)``. Differentiable w.r.t. ``obs``."""
        obs_t = (
            obs
            if isinstance(obs, torch.Tensor)
            else torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)
        )
        if self.recurrent:
            state = self.state if state is None else state
            start = self.episode_start if episode_start is None else episode_start
            starts = torch.full((obs_t.shape[0],), float(start))
            dist, new_state = self.policy.get_distribution(obs_t, state, starts)
            return dist.distribution.logits, new_state
        dist = self.policy.get_distribution(obs_t)
        return dist.distribution.logits, None

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[int, np.ndarray]:
        logits, new_state = self.distribution(obs)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        if self.deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(self.rng.choice(len(probs), p=probs / probs.sum()))
        self.state = new_state
        self.episode_start = False
        return action, probs
