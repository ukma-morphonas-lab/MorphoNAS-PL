from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch

from MorphoNAS.hooks import EdgeDynamicsHook, WeightStabilizer
from .plasticity_strategies.plasticity_genome import PlasticityGenome


@torch.jit.script
def _trace_loop_jit(
    pre_activities: torch.Tensor,
    post_activities: torch.Tensor,
    traces: torch.Tensor,
    edge_post_idx: torch.Tensor,
    edge_pre_idx: torch.Tensor,
    x_bar: torch.Tensor,
    tau_x: float,
    one_minus_tau_x: float,
    one_minus_decay: float,
    theta_10: float,
    theta_11: float,
    theta_01: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = pre_activities.shape[0]
    for t in range(T):
        pre_activity = pre_activities[t]
        post_activity = post_activities[t]

        x_bar = tau_x * x_bar + one_minus_tau_x * post_activity
        post_fluct = x_bar - post_activity
        post_fluct_edge = post_fluct[edge_post_idx]
        pre_edge = pre_activity[edge_pre_idx]

        # H_θ = θ₁₀·pre + θ₁₁·pre·fluct + θ₀₁·fluct
        H_theta = (
            theta_10 * pre_edge
            + theta_11 * post_fluct_edge * pre_edge
            + theta_01 * post_fluct_edge
        )
        traces = one_minus_decay * traces + H_theta * dt

    return traces, x_bar


@dataclass
class WeightClamp(WeightStabilizer):
    min_value: float = -2.0
    max_value: float = 2.0

    last_edge_clamp_frac: float = 0.0
    last_edge_clamp_max_abs: float = 0.0

    def stabilize(self, *, W: torch.Tensor, edge_weights: torch.Tensor) -> None:
        if edge_weights.numel() > 0:
            over = (edge_weights < self.min_value) | (edge_weights > self.max_value)
            self.last_edge_clamp_frac = float(over.float().mean().item())
            self.last_edge_clamp_max_abs = float(edge_weights.abs().max().item())
        else:
            self.last_edge_clamp_frac = 0.0
            self.last_edge_clamp_max_abs = 0.0

        edge_weights.clamp_(min=self.min_value, max=self.max_value)
        W.clamp_(min=self.min_value, max=self.max_value)


class ThreeFactorEdgeHook(EdgeDynamicsHook):
    def __init__(
        self,
        genome: PlasticityGenome,
        dt: float = 0.02,
        *,
        bootstrap_baseline: bool = True,
        min_abs_delta_r: float = 1.0,
    ):
        self.genome = genome
        self.dt = float(dt)
        self.bootstrap_baseline = bool(bootstrap_baseline)
        self.min_abs_delta_r = float(min_abs_delta_r)

        self.theta_10 = float(genome.theta_10)
        self.theta_11 = float(genome.theta_11)
        self.theta_01 = float(genome.theta_01)
        self.eta = float(genome.eta)
        self.tau_e = float(genome.tau_e)
        self.tau_x = float(genome.tau_x)
        self.baseline_decay = float(genome.baseline_decay)

        self._decay_factor = self.dt / self.tau_e
        self._one_minus_decay = 1.0 - self._decay_factor
        self._one_minus_tau_x = 1.0 - self.tau_x

        self._num_edges: Optional[int] = None
        self._traces: Optional[torch.Tensor] = None
        self._x_bar: Optional[torch.Tensor] = None

        self._episode_reward: float = 0.0
        self._reward_baseline: float = 0.0
        self._baseline_initialized: bool = False

    def reset(self, *, num_neurons: int, num_edges: int, device: torch.device) -> None:
        self._num_edges = int(num_edges)
        self._traces = torch.zeros((num_edges,), device=device)
        self._x_bar = None
        self._episode_reward = 0.0
        self._reward_baseline = 0.0
        self._baseline_initialized = False

    def reset_episode(self) -> None:
        self._episode_reward = 0.0
        if self._traces is not None:
            self._traces.zero_()

    def on_reward(self, reward: float) -> None:
        self._episode_reward += float(reward)

    def on_step(
        self,
        *,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        edge_pre_idx: torch.Tensor,
        edge_post_idx: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> None:
        if self._traces is None:
            raise RuntimeError("ThreeFactorEdgeHook.reset() was not called")

        if self._x_bar is None:
            self._x_bar = post_activity.detach().clone()

        self._x_bar = self.tau_x * self._x_bar + self._one_minus_tau_x * post_activity
        post_fluct = self._x_bar - post_activity
        post_fluct_edge = post_fluct[edge_post_idx]
        pre_edge = pre_activity[edge_pre_idx]

        H_theta = (
            self.theta_10 * pre_edge
            + self.theta_11 * post_fluct_edge * pre_edge
            + self.theta_01 * post_fluct_edge
        )
        self._traces.mul_(self._one_minus_decay).add_(H_theta * self.dt)

    def on_sequence(
        self,
        *,
        pre_activities: torch.Tensor,
        post_activities: torch.Tensor,
        edge_pre_idx: torch.Tensor,
        edge_post_idx: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> None:
        if self._traces is None:
            raise RuntimeError("ThreeFactorEdgeHook.reset() was not called")

        if pre_activities.shape[0] == 0:
            return

        if self._x_bar is None:
            self._x_bar = post_activities[0].detach().clone()

        self._traces, self._x_bar = _trace_loop_jit(
            pre_activities,
            post_activities,
            self._traces,
            edge_post_idx,
            edge_pre_idx,
            self._x_bar,
            self.tau_x,
            self._one_minus_tau_x,
            self._one_minus_decay,
            self.theta_10,
            self.theta_11,
            self.theta_01,
            self.dt,
        )

    def end_episode(self, episode_reward: float) -> torch.Tensor:
        if self._traces is None:
            raise RuntimeError("ThreeFactorEdgeHook.reset() was not called")

        R = float(episode_reward)

        if self.bootstrap_baseline and not self._baseline_initialized:
            self._reward_baseline = R
            self._baseline_initialized = True
            deltas = torch.zeros_like(self._traces)
            self._traces.zero_()
            self._episode_reward = 0.0
            return deltas

        self._baseline_initialized = True
        delta_r = R - self._reward_baseline

        if self.baseline_decay > 0:
            self._reward_baseline = (
                self.baseline_decay * self._reward_baseline
                + (1.0 - self.baseline_decay) * R
            )

        if self.min_abs_delta_r > 0 and abs(delta_r) < self.min_abs_delta_r:
            deltas = torch.zeros_like(self._traces)
            self._traces.zero_()
            self._episode_reward = 0.0
            return deltas

        deltas = (self.eta * delta_r) * self._traces

        self._traces.zero_()
        self._episode_reward = 0.0

        return deltas


@torch.jit.script
def _oja_trace_loop_jit(
    pre_activities: torch.Tensor,
    post_activities: torch.Tensor,
    traces: torch.Tensor,
    edge_post_idx: torch.Tensor,
    edge_pre_idx: torch.Tensor,
    edge_weights: torch.Tensor,
    trace_decay: float,
) -> torch.Tensor:
    T = pre_activities.shape[0]
    for t in range(T):
        post = post_activities[t][edge_post_idx]
        pre = pre_activities[t][edge_pre_idx]
        update = post * pre - (post * post) * edge_weights
        traces = traces * trace_decay + update
    return traces


class OjaEdgeHook(EdgeDynamicsHook):
    def __init__(self, learning_rate: float, trace_decay: float = 0.0):
        self.learning_rate = float(learning_rate)
        self.trace_decay = float(trace_decay)

        self._num_edges: Optional[int] = None
        self._traces: Optional[torch.Tensor] = None

    def reset(self, *, num_neurons: int, num_edges: int, device: torch.device) -> None:
        self._num_edges = int(num_edges)
        self._traces = torch.zeros((num_edges,), device=device)

    def reset_episode(self) -> None:
        if self._traces is not None:
            self._traces.zero_()

    def on_reward(self, reward: float) -> None:
        return

    def on_step(
        self,
        *,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        edge_pre_idx: torch.Tensor,
        edge_post_idx: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> None:
        if self._traces is None:
            raise RuntimeError("OjaEdgeHook.reset() was not called")

        post = post_activity[edge_post_idx]
        pre = pre_activity[edge_pre_idx]
        update = post * pre - (post * post) * edge_weights
        self._traces.mul_(self.trace_decay).add_(update)

    def on_sequence(
        self,
        *,
        pre_activities: torch.Tensor,
        post_activities: torch.Tensor,
        edge_pre_idx: torch.Tensor,
        edge_post_idx: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> None:
        if self._traces is None:
            raise RuntimeError("OjaEdgeHook.reset() was not called")

        if pre_activities.shape[0] == 0:
            return

        self._traces = _oja_trace_loop_jit(
            pre_activities,
            post_activities,
            self._traces,
            edge_post_idx,
            edge_pre_idx,
            edge_weights,
            float(self.trace_decay),
        )

    def end_episode(self, episode_reward: float) -> torch.Tensor:
        if self._traces is None:
            raise RuntimeError("OjaEdgeHook.reset() was not called")

        deltas = self.learning_rate * self._traces
        self._traces.zero_()
        return deltas


class HebbianEdgeHook(EdgeDynamicsHook):
    """Simple Hebbian plasticity applied per environment step.

    Accumulates pre/post activity during a single propagate() call and returns
    a per-step delta via end_step(). The caller is responsible for applying
    deltas each environment step.
    """

    def __init__(
        self,
        learning_rate: float,
        weight_decay: float = 0.0,
        edge_mask: Optional[torch.Tensor] = None,
    ):
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self._edge_mask_input = edge_mask

        self._num_edges: Optional[int] = None
        self._step_deltas: Optional[torch.Tensor] = None
        self._plastic_weights: Optional[torch.Tensor] = None
        self._edge_mask: Optional[torch.Tensor] = None

    def reset(self, *, num_neurons: int, num_edges: int, device: torch.device) -> None:
        self._num_edges = int(num_edges)
        self._step_deltas = torch.zeros((num_edges,), device=device)
        self._plastic_weights = torch.zeros((num_edges,), device=device)
        if self._edge_mask_input is not None:
            self.set_edge_mask(self._edge_mask_input)

    def set_edge_mask(self, mask: torch.Tensor) -> None:
        if self._num_edges is None or self._step_deltas is None:
            raise RuntimeError("HebbianEdgeHook.reset() must be called before set_edge_mask()")

        mask = mask.to(self._step_deltas.device).float()
        if mask.numel() != self._num_edges:
            raise ValueError(
                f"Expected edge mask of size {self._num_edges}, got {int(mask.numel())}"
            )
        self._edge_mask = mask

    def reset_episode(self) -> None:
        if self._step_deltas is not None:
            self._step_deltas.zero_()
        if self._plastic_weights is not None:
            self._plastic_weights.zero_()

    def on_reward(self, reward: float) -> None:
        return

    def on_step(
        self,
        *,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        edge_pre_idx: torch.Tensor,
        edge_post_idx: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> None:
        if self._step_deltas is None:
            raise RuntimeError("HebbianEdgeHook.reset() was not called")

        post = post_activity[edge_post_idx]
        pre = pre_activity[edge_pre_idx]
        update = post * pre
        if self._edge_mask is not None:
            update = update * self._edge_mask
        self._step_deltas.add_(update)

    def on_sequence(
        self,
        *,
        pre_activities: torch.Tensor,
        post_activities: torch.Tensor,
        edge_pre_idx: torch.Tensor,
        edge_post_idx: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> None:
        if self._step_deltas is None:
            raise RuntimeError("HebbianEdgeHook.reset() was not called")

        if pre_activities.shape[0] == 0:
            return

        post = post_activities[:, edge_post_idx]
        pre = pre_activities[:, edge_pre_idx]
        update = (post * pre).sum(dim=0)
        if self._edge_mask is not None:
            update = update * self._edge_mask
        self._step_deltas.add_(update)

    def end_step(self) -> torch.Tensor:
        if self._step_deltas is None or self._plastic_weights is None:
            raise RuntimeError("HebbianEdgeHook.reset() was not called")

        old_plastic = self._plastic_weights
        self._plastic_weights = (
            (1.0 - self.weight_decay) * self._plastic_weights
            + self.learning_rate * self._step_deltas
        )
        deltas = self._plastic_weights - old_plastic
        self._step_deltas.zero_()
        return deltas

    def end_episode(self, episode_reward: float) -> torch.Tensor:
        if self._step_deltas is None or self._plastic_weights is None:
            raise RuntimeError("HebbianEdgeHook.reset() was not called")

        self._step_deltas.zero_()
        return torch.zeros_like(self._plastic_weights)


class OjaStepEdgeHook(EdgeDynamicsHook):
    """Oja's rule applied per environment step."""

    def __init__(self, learning_rate: float, trace_decay: float = 0.0):
        self.learning_rate = float(learning_rate)
        self.trace_decay = float(trace_decay)

        self._num_edges: Optional[int] = None
        self._step_deltas: Optional[torch.Tensor] = None

    def reset(self, *, num_neurons: int, num_edges: int, device: torch.device) -> None:
        self._num_edges = int(num_edges)
        self._step_deltas = torch.zeros((num_edges,), device=device)

    def reset_episode(self) -> None:
        if self._step_deltas is not None:
            self._step_deltas.zero_()

    def on_reward(self, reward: float) -> None:
        return

    def on_step(
        self,
        *,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        edge_pre_idx: torch.Tensor,
        edge_post_idx: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> None:
        if self._step_deltas is None:
            raise RuntimeError("OjaStepEdgeHook.reset() was not called")

        post = post_activity[edge_post_idx]
        pre = pre_activity[edge_pre_idx]
        update = post * pre - (post * post) * edge_weights
        if self.trace_decay:
            self._step_deltas.mul_(self.trace_decay).add_(update)
        else:
            self._step_deltas.add_(update)

    def on_sequence(
        self,
        *,
        pre_activities: torch.Tensor,
        post_activities: torch.Tensor,
        edge_pre_idx: torch.Tensor,
        edge_post_idx: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> None:
        if self._step_deltas is None:
            raise RuntimeError("OjaStepEdgeHook.reset() was not called")

        if pre_activities.shape[0] == 0:
            return

        post = post_activities[:, edge_post_idx]
        pre = pre_activities[:, edge_pre_idx]
        update = (post * pre - (post * post) * edge_weights).sum(dim=0)
        if self.trace_decay:
            self._step_deltas.mul_(self.trace_decay).add_(update)
        else:
            self._step_deltas.add_(update)

    def end_step(self) -> torch.Tensor:
        if self._step_deltas is None:
            raise RuntimeError("OjaStepEdgeHook.reset() was not called")

        deltas = self.learning_rate * self._step_deltas
        self._step_deltas.zero_()
        return deltas

    def end_episode(self, episode_reward: float) -> torch.Tensor:
        if self._step_deltas is None:
            raise RuntimeError("OjaStepEdgeHook.reset() was not called")

        self._step_deltas.zero_()
        return torch.zeros_like(self._step_deltas)
