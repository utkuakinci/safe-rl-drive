"""SB3 callbacks: periodic driving-KPI evaluation and shield-intervention logging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from safe_rl_drive.envs.factory import EnvCondition
from safe_rl_drive.evaluation.rollout import run_episodes, summarize


class DrivingEvalCallback(BaseCallback):
    """Every ``eval_freq`` env steps: evaluate success / collision / speed / jerk and keep the best model.

    Metrics go to the SB3 logger (``eval/*``), which is synced to TensorBoard and W&B.
    """

    def __init__(
        self,
        cond: EnvCondition,
        eval_freq: int,
        n_episodes: int,
        save_dir: Path,
        shield_config: dict | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.cond = cond
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.save_dir = Path(save_dir)
        self.shield_config = shield_config
        self.best_score = -np.inf
        self._last = 0

    def _score(self, s: dict) -> float:
        # success first, then speed as tie-breaker
        return s.get("success", 0.0) - s.get("collision", 0.0) + 1e-3 * s.get("mean_speed", 0.0)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.eval_freq:
            return True
        self._last = self.num_timesteps
        eps, _ = run_episodes(self.model, self.cond, self.n_episodes, seed=50_000)
        s = summarize(eps)
        for k in ("success", "collision", "mean_speed", "jerk_rms", "return", "length", "lane_changes"):
            if k in s:
                self.logger.record(f"eval/{k}", s[k])
        if self.shield_config is not None:
            eps_sh, stats = run_episodes(
                self.model, self.cond, self.n_episodes, seed=50_000, shield_config=self.shield_config
            )
            s_sh = summarize(eps_sh)
            self.logger.record("eval_shielded/success", s_sh["success"])
            self.logger.record("eval_shielded/collision", s_sh["collision"])
            self.logger.record("eval_shielded/intervention_rate", stats.get("intervention_rate", 0.0))
        score = self._score(s)
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.save_dir / "best_model.zip")
            if self.verbose:
                print(
                    f"[eval] t={self.num_timesteps} new best: success={s['success']:.2f} collision={s['collision']:.2f}"
                )
        self.logger.dump(self.num_timesteps)
        return True


class ShieldTrainingLogger(BaseCallback):
    """When training *with* the shield in the loop, log how often it intervenes."""

    def __init__(self):
        super().__init__()
        self._steps = 0
        self._interventions = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "shield" in info:
                self._steps += 1
                self._interventions += int(info["shield"]["intervened"])
        return True

    def _on_rollout_end(self) -> None:
        if self._steps:
            self.logger.record("train/shield_intervention_rate", self._interventions / self._steps)
            self._steps = self._interventions = 0
