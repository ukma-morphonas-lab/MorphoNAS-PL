"""Environment wrappers for Acrobot non-stationary experiments.

Provides mid-episode physics switching for Acrobot-v1, analogous to
MidEpisodeSwitchWrapper for CartPole (env_wrappers.py).

Acrobot physics parameters that can be modified at runtime:
  LINK_MASS_1, LINK_MASS_2       mass of each link (kg)
  LINK_LENGTH_1, LINK_LENGTH_2   length of each link (m)
  LINK_COM_POS_1, LINK_COM_POS_2 center-of-mass position (fraction of length)
  LINK_MOI                       moment of inertia for both links
  MAX_VEL_1, MAX_VEL_2           angular velocity caps

These are class-level attributes on AcrobotEnv but can be overridden per-instance.
The dynamics (_dsdt) reads them from self each step, so no derived quantities
need recomputation after a change (unlike CartPole's total_mass).
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class AcrobotMidEpisodeSwitchWrapper(gym.Wrapper):
    """Switches Acrobot physics at a specified step within each episode.

    Steps 0 to switch_step-1 run with default physics.
    Steps switch_step onward run with modified physics.

    Args:
        env: Acrobot environment to wrap.
        switch_step: Step at which physics change (default 100).
        target_params: Dict of parameter overrides applied at switch_step.
            Supported keys: LINK_MASS_1, LINK_MASS_2, LINK_LENGTH_1,
            LINK_LENGTH_2, LINK_COM_POS_1, LINK_COM_POS_2, LINK_MOI,
            MAX_VEL_1, MAX_VEL_2.
    """

    SUPPORTED_PARAMS = {
        "LINK_MASS_1", "LINK_MASS_2",
        "LINK_LENGTH_1", "LINK_LENGTH_2",
        "LINK_COM_POS_1", "LINK_COM_POS_2",
        "LINK_MOI",
        "MAX_VEL_1", "MAX_VEL_2",
    }

    def __init__(
        self,
        env: gym.Env,
        switch_step: int = 100,
        target_params: Optional[dict[str, float]] = None,
    ):
        super().__init__(env)
        if target_params is None:
            target_params = {"LINK_MASS_2": 3.0}

        unsupported = set(target_params.keys()) - self.SUPPORTED_PARAMS
        if unsupported:
            raise ValueError(
                f"Unsupported params: {unsupported}. Use: {self.SUPPORTED_PARAMS}"
            )

        self.switch_step = int(switch_step)
        self.target_params = {k: float(v) for k, v in target_params.items()}

        self._originals: dict[str, float] = {}
        self._originals_captured = False
        self._step_count = 0
        self._switched = False

    def _capture_originals(self) -> None:
        unwrapped = self.env.unwrapped
        for key in self.SUPPORTED_PARAMS:
            self._originals[key] = float(getattr(unwrapped, key))
        self._originals_captured = True

    def _apply_params(self, params: dict[str, float]) -> None:
        unwrapped = self.env.unwrapped
        for key, value in params.items():
            setattr(unwrapped, key, value)

    def _restore_originals(self) -> None:
        self._apply_params(self._originals)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ObsType, dict[str, Any]]:
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
