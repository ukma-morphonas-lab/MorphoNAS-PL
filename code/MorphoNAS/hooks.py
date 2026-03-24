from __future__ import annotations

from typing import Optional, Protocol

import torch


class EdgeDynamicsHook(Protocol):
    """Hook for edge-list based synaptic dynamics.

    MorphoNAS core stays agnostic to learning rules: it only provides per-step or
    per-sequence (batched) access to presynaptic and postsynaptic activity plus the
    current edge weights.

    Implementations may keep internal state (e.g., eligibility traces) and should
    return edge-weight deltas at episode boundaries.
    """

    def reset(
        self, *, num_neurons: int, num_edges: int, device: torch.device
    ) -> None: ...

    def reset_episode(self) -> None: ...

    def on_sequence(
        self,
        *,
        pre_activities: torch.Tensor,
        post_activities: torch.Tensor,
        edge_pre_idx: torch.Tensor,
        edge_post_idx: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> None: ...

    def on_step(
        self,
        *,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        edge_pre_idx: torch.Tensor,
        edge_post_idx: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> None: ...

    def on_reward(self, reward: float) -> None: ...

    def end_episode(self, episode_reward: float) -> torch.Tensor:
        """Return edge-wise weight delta to apply."""
        ...


class WeightStabilizer(Protocol):
    """Optional post-update stabilization applied to synaptic weights."""

    def stabilize(self, *, W: torch.Tensor, edge_weights: torch.Tensor) -> None: ...


def hook_supports_sequence(hook: EdgeDynamicsHook) -> bool:
    """Runtime check: does the hook override `on_sequence`?"""

    return "on_sequence" in type(hook).__dict__
