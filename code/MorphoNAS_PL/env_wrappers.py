"""Environment wrappers for B0.6 non-stationary CartPole experiments.

Provides mid-episode physics switching to test whether Hebbian plasticity
enables genuine online adaptation vs just static weight perturbation.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType


class MidEpisodeSwitchWrapper(gym.Wrapper):
    """Switches CartPole physics parameters at a specified step within each episode.

    Steps 0 to switch_step-1 run with default physics.
    Steps switch_step onward run with modified physics.

    The wrapper tracks which phase the episode is in and exposes it via info dict.

    Args:
        env: CartPole environment to wrap.
        switch_step: Step number at which physics change (default 200).
        target_params: Dict of parameter overrides applied at switch_step.
            Supported keys: gravity, masscart, masspole, length, force_mag.
    """

    SUPPORTED_PARAMS = {"gravity", "masscart", "masspole", "length", "force_mag"}

    def __init__(
        self,
        env: gym.Env,
        switch_step: int = 200,
        target_params: Optional[dict[str, float]] = None,
    ):
        super().__init__(env)
        if target_params is None:
            target_params = {"gravity": 20.0}

        unsupported = set(target_params.keys()) - self.SUPPORTED_PARAMS
        if unsupported:
            raise ValueError(f"Unsupported params: {unsupported}. Use: {self.SUPPORTED_PARAMS}")

        self.switch_step = int(switch_step)
        self.target_params = {k: float(v) for k, v in target_params.items()}

        # Capture original values on first reset
        self._originals: dict[str, float] = {}
        self._originals_captured = False
        self._step_count = 0
        self._switched = False

    def _capture_originals(self) -> None:
        """Snapshot default physics values from the unwrapped env."""
        unwrapped = self.env.unwrapped
        for key in self.SUPPORTED_PARAMS:
            self._originals[key] = float(getattr(unwrapped, key))
        self._originals_captured = True

    def _apply_params(self, params: dict[str, float]) -> None:
        """Set physics parameters and recompute derived quantities."""
        unwrapped = self.env.unwrapped
        for key, value in params.items():
            setattr(unwrapped, key, value)
        # Recompute derived quantities used in step()
        unwrapped.total_mass = unwrapped.masspole + unwrapped.masscart
        unwrapped.polemass_length = unwrapped.masspole * unwrapped.length

    def _restore_originals(self) -> None:
        """Restore original physics for the next episode."""
        self._apply_params(self._originals)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        # Restore originals before reset so episode starts with default physics
        if self._originals_captured:
            self._restore_originals()

        obs, info = self.env.reset(seed=seed, options=options)

        if not self._originals_captured:
            self._capture_originals()

        self._step_count = 0
        self._switched = False
        info["phase"] = 1
        info["switch_step"] = self.switch_step
        info["target_params"] = self.target_params
        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        self._step_count += 1

        if not self._switched and self._step_count >= self.switch_step:
            self._apply_params(self.target_params)
            self._switched = True

        obs, reward, terminated, truncated, info = self.env.step(action)
        info["phase"] = 2 if self._switched else 1
        info["step_in_episode"] = self._step_count
        return obs, float(reward), terminated, truncated, info
