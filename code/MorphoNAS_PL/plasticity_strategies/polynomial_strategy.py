"""
Polynomial plasticity strategy following Maoutsa 2025.

H_theta = SUM_{k,l=0}^{d} theta[k,l] * pre^k * fluct^l

Weight update: Delta_w_ij = eta * e_ij * (R - R_bar)
"""

import torch
import logging
from typing import Optional

from .base_strategy import BasePlasticityStrategy
from .polynomial_genome import PolynomialPlasticityGenome

logger = logging.getLogger(__name__)


class PolynomialPlasticityStrategy(BasePlasticityStrategy):
    """
    Polynomial H_theta plasticity with eligibility traces and reward modulation.

    H_theta = SUM_{k,l} theta[k,l] * pre^k * fluct^l
    de_ij/dt = H_theta - e_ij/tau_e
    Delta_w_ij = eta * e_ij * (R - R_bar)
    """

    def __init__(self, genome: PolynomialPlasticityGenome, dt: float = 0.02):
        super().__init__(
            learning_rate=genome.eta,
            trace_decay=1.0 - dt / genome.tau_e,
        )

        self.genome = genome
        self.degree = genome.degree
        self.theta = torch.from_numpy(genome.theta).float()
        self.eta = genome.eta
        self.tau_e = genome.tau_e
        self.tau_x = genome.tau_x
        self.dt = dt

        self._decay_factor = dt / genome.tau_e
        self._one_minus_decay = 1.0 - self._decay_factor
        self._one_minus_tau_x = 1.0 - genome.tau_x

        self.x_bar: Optional[torch.Tensor] = None
        self.reward_baseline = 0.0
        self.episode_reward = 0.0

        self._pre_powers: Optional[torch.Tensor] = None
        self._fluct_powers: Optional[torch.Tensor] = None

    def _compute_H_theta_dense(
        self, pre_activity: torch.Tensor, fluct: torch.Tensor, n_neurons: int
    ) -> torch.Tensor:
        """
        Compute polynomial H_theta for dense (N, N) representation.

        H_theta[i,j] = SUM_{k,l} theta[k,l] * pre[j]^k * fluct[i]^l
        """
        d = self.degree

        pre_powers = torch.zeros(d + 1, n_neurons, device=pre_activity.device)
        fluct_powers = torch.zeros(d + 1, n_neurons, device=fluct.device)

        pre_powers[0] = 1.0
        fluct_powers[0] = 1.0
        for p in range(1, d + 1):
            pre_powers[p] = pre_powers[p - 1] * pre_activity
            fluct_powers[p] = fluct_powers[p - 1] * fluct

        H_theta = torch.zeros(n_neurons, n_neurons, device=pre_activity.device)

        theta_device = self.theta.to(pre_activity.device)
        for k in range(d + 1):
            for l in range(d + 1):
                if theta_device[k, l] != 0:
                    H_theta += theta_device[k, l] * torch.outer(
                        fluct_powers[l], pre_powers[k]
                    )

        return H_theta

    def _compute_H_theta_sparse(
        self,
        pre_edge: torch.Tensor,
        fluct_edge: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute polynomial H_theta for sparse edge representation.

        H_theta[e] = SUM_{k,l} theta[k,l] * pre[e]^k * fluct[e]^l
        """
        d = self.degree
        n_edges = pre_edge.shape[0]
        device = pre_edge.device

        pre_powers = torch.zeros(d + 1, n_edges, device=device)
        fluct_powers = torch.zeros(d + 1, n_edges, device=device)

        pre_powers[0] = 1.0
        fluct_powers[0] = 1.0
        for p in range(1, d + 1):
            pre_powers[p] = pre_powers[p - 1] * pre_edge
            fluct_powers[p] = fluct_powers[p - 1] * fluct_edge

        H_theta = torch.zeros(n_edges, device=device)

        theta_device = self.theta.to(device)
        for k in range(d + 1):
            for l in range(d + 1):
                if theta_device[k, l] != 0:
                    H_theta += theta_device[k, l] * pre_powers[k] * fluct_powers[l]

        return H_theta

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
        fluct = self.x_bar - post_activity

        H_theta = self._compute_H_theta_dense(pre_activity, fluct, n_neurons)
        H_theta = H_theta * connection_mask

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
        fluct = self.x_bar - post_activity

        fluct_edge = fluct[edge_post_idx]
        pre_edge = pre_activity[edge_pre_idx]

        H_theta = self._compute_H_theta_sparse(pre_edge, fluct_edge)

        traces.mul_(self._one_minus_decay).add_(H_theta * self.dt)

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
        self.reward_baseline = (
            self.tau_x * self.reward_baseline + (1.0 - self.tau_x) * self.episode_reward
        )
        return delta_r

    def get_weight_update_multiplier(self) -> float:
        return self.compute_reward_modulation()

    def __repr__(self) -> str:
        t = self.genome.theta
        return (
            f"PolynomialPlasticityStrategy("
            f"d={self.degree}, "
            f"theta[1,0]={t[1, 0]:.3f}, "
            f"theta[1,1]={t[1, 1]:.3f}, "
            f"eta={self.eta:.2e}, "
            f"tau_e={self.tau_e:.2f})"
        )
