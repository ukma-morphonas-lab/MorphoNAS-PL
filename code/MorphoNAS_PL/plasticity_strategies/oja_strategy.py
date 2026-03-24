"""
Oja's rule plasticity strategy.

Implements Oja's learning rule with automatic weight normalization to prevent
unbounded growth while preserving Hebbian strengthening.

References:
    Oja, E. (1982). Simplified neuron model as a principal component analyzer.
    Journal of Mathematical Biology, 15(3), 267-273.
"""

import torch
from .base_strategy import BasePlasticityStrategy


class OjaPlasticityStrategy(BasePlasticityStrategy):
    """
    Oja's rule with eligibility traces.

    Combines Hebbian strengthening with automatic normalization:
        Δe_ij = decay * e_ij + (x_i * y_j - y_j^2 * w_ij)
        Δw_ij = lr * e_ij  (applied at episode boundaries)

    Where:
        x_i: Pre-synaptic activity
        y_j: Post-synaptic activity
        w_ij: Current weight
        e_ij: Eligibility trace

    The y_j^2 * w_ij term provides automatic normalization, preventing
    unbounded weight growth and making neurons learn principal components
    of the input.

    NOTE: Traces accumulate during episode, weights update once at episode end.
    """

    def accumulate_traces(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        connection_mask: torch.Tensor,
        traces: torch.Tensor,
    ) -> torch.Tensor:
        """
        Accumulate eligibility traces using Oja's rule.

        Args:
            pre_activity: Pre-synaptic neuron activations (N,)
            post_activity: Post-synaptic neuron activations (N,)
            weights: Current weight matrix (N, N)
            connection_mask: Binary mask for existing connections (N, N)
            traces: Current eligibility traces (N, N)

        Returns:
            Updated eligibility traces (N, N)
        """
        # Hebbian term: correlation between pre- and post-synaptic activity
        # W[i,j] is the weight from neuron j (pre) to neuron i (post)
        # So hebbian_term[i,j] should be post[i] * pre[j]
        hebbian_term = torch.outer(post_activity, pre_activity)

        # Oja's normalization term (y_i^2 * w_ij) uses broadcasting instead of outer products for speed
        oja_term = (post_activity**2).unsqueeze(1) * weights

        # Combined update with Oja's rule (masked to existing connections only)
        hebbian_update = (hebbian_term - oja_term) * connection_mask

        # Update eligibility traces with exponential decay:
        #   e_ij(t) = decay * e_ij(t-1) + hebbian_update
        updated_traces = self.trace_decay * traces + hebbian_update

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
        """Accumulate Oja traces over an edge list (O(E) per step)."""
        post = post_activity[edge_post_idx]
        pre = pre_activity[edge_pre_idx]
        w = weights if weights.dim() == 1 else weights[edge_post_idx, edge_pre_idx]
        update = post * pre - (post * post) * w
        traces.mul_(self.trace_decay).add_(update)
        return traces

    def reset_traces(
        self, shape: tuple, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Initialize eligibility traces to zero.

        Args:
            shape: Shape of the trace tensor (same as weight matrix)
            device: Optional device for the trace tensor

        Returns:
            Zero-initialized trace tensor
        """
        return torch.zeros(shape, device=device)

    def set_reward(self, reward: float):
        """
        Oja's rule is unsupervised and does not use reward signals.

        This method exists for interface compatibility but does nothing.

        Args:
            reward: Reward value (ignored)
        """
        # Oja's rule doesn't use reward - unsupervised learning
        pass
