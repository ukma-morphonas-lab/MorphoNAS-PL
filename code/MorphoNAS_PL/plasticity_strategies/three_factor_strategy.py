"""
Three-factor plasticity strategy with eligibility traces and reward modulation.

Implements reward-modulated Hebbian learning following Maoutsa 2025 and
Gerstner et al. 2018. Synapses accumulate eligibility traces during behavior,
and weight updates occur at episode end modulated by reward prediction error.

Extended with theta_10 (pure Hebbian term) for rate-coded networks.
"""

import torch
import logging
from typing import Optional

from .base_strategy import BasePlasticityStrategy
from .plasticity_genome import PlasticityGenome

logger = logging.getLogger(__name__)


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


class ThreeFactorPlasticityStrategy(BasePlasticityStrategy):
    """
    Three-factor plasticity: eligibility traces + reward modulation.

    Eligibility dynamics (per propagation step):
        de_ij/dt = H_θ(r_j, x_i) - e_ij/τ_e

    Where H_θ = θ₁₀ · r_j + θ₁₁ · r_j · (x̄_i - x_i) + θ₀₁ · (x̄_i - x_i)

    Weight update (at episode end):
        Δw_ij = η · e_ij · (R - R̄)
    """

    def __init__(
        self,
        genome: PlasticityGenome,
        dt: float = 0.02,
    ):
        super().__init__(
            learning_rate=genome.eta,
            trace_decay=1.0 - dt / genome.tau_e,
        )

        self.genome = genome
        self.theta_10 = genome.theta_10
        self.theta_11 = genome.theta_11
        self.theta_01 = genome.theta_01
        self.eta = genome.eta
        self.tau_e = genome.tau_e
        self.tau_x = genome.tau_x
        self.baseline_decay = genome.baseline_decay
        self.dt = dt

        self._decay_factor = dt / genome.tau_e
        self._one_minus_decay = 1.0 - self._decay_factor
        self._one_minus_tau_x = 1.0 - genome.tau_x

        self.x_bar: Optional[torch.Tensor] = None
        self.reward_baseline = 0.0
        self.episode_reward = 0.0

    def reset_episode_state(self):
        self.episode_reward = 0.0

    def reset_all_state(self, n_neurons: Optional[int] = None):
        self.reward_baseline = 0.0
        self.episode_reward = 0.0
        if n_neurons is not None:
            self.x_bar = None

    def accumulate_traces(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        connection_mask: torch.Tensor,
        traces: torch.Tensor,
    ) -> torch.Tensor:
        n_neurons = len(post_activity)

        if self.x_bar is None:
            self.x_bar = post_activity.clone()

        self.x_bar = self.tau_x * self.x_bar + self._one_minus_tau_x * post_activity
        post_fluct = self.x_bar - post_activity

        # H_θ = θ₁₀·pre + θ₁₁·pre·fluct + θ₀₁·fluct
        pure_hebbian = self.theta_10 * pre_activity.unsqueeze(0).expand(
            n_neurons, n_neurons
        )
        interaction_term = self.theta_11 * torch.outer(post_fluct, pre_activity)
        fluctuation_term = self.theta_01 * post_fluct.unsqueeze(1).expand(
            n_neurons, n_neurons
        )

        H_theta = (pure_hebbian + interaction_term + fluctuation_term) * connection_mask
        updated_traces = self._one_minus_decay * traces + H_theta * self.dt

        return updated_traces

    def accumulate_traces_sparse(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        traces: torch.Tensor,
        edge_post_idx: torch.Tensor,
        edge_pre_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.x_bar is None:
            self.x_bar = post_activity.clone()

        self.x_bar = self.tau_x * self.x_bar + self._one_minus_tau_x * post_activity
        post_fluct = self.x_bar - post_activity
        post_fluct_edge = post_fluct[edge_post_idx]
        pre_edge = pre_activity[edge_pre_idx]

        # H_θ = θ₁₀·pre + θ₁₁·pre·fluct + θ₀₁·fluct
        H_theta = (
            self.theta_10 * pre_edge
            + self.theta_11 * post_fluct_edge * pre_edge
            + self.theta_01 * post_fluct_edge
        )

        traces.mul_(self._one_minus_decay).add_(H_theta * self.dt)

        return traces

    def accumulate_traces_sparse_batched(
        self,
        pre_activities: torch.Tensor,
        post_activities: torch.Tensor,
        weights: torch.Tensor,
        traces: torch.Tensor,
        edge_post_idx: torch.Tensor,
        edge_pre_idx: torch.Tensor,
    ) -> torch.Tensor:
        T = pre_activities.shape[0]
        if T == 0:
            return traces

        if self.x_bar is None:
            self.x_bar = post_activities[0].clone()

        traces, self.x_bar = _trace_loop_jit(
            pre_activities,
            post_activities,
            traces,
            edge_post_idx,
            edge_pre_idx,
            self.x_bar,
            self.tau_x,
            self._one_minus_tau_x,
            self._one_minus_decay,
            self.theta_10,
            self.theta_11,
            self.theta_01,
            self.dt,
        )

        return traces

    def reset_traces(
        self, shape: tuple, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        return torch.zeros(shape, device=device)

    def set_reward(self, reward: float):
        self.episode_reward += reward
        self.last_reward = reward

    def compute_reward_modulation(self) -> float:
        delta_r = self.episode_reward - self.reward_baseline
        if self.baseline_decay > 0:
            self.reward_baseline = (
                self.baseline_decay * self.reward_baseline
                + (1.0 - self.baseline_decay) * self.episode_reward
            )
        return delta_r

    def get_weight_update_multiplier(self) -> float:
        return self.compute_reward_modulation()

    def __repr__(self) -> str:
        return (
            f"ThreeFactorPlasticityStrategy("
            f"θ₁₀={self.theta_10:.4f}, "
            f"θ₁₁={self.theta_11:.4f}, "
            f"θ₀₁={self.theta_01:.4f}, "
            f"η={self.eta:.2e}, "
            f"τ_e={self.tau_e:.2f}, "
            f"τ_x={self.tau_x:.3f}, "
            f"baseline={self.baseline_decay:.2f})"
        )
