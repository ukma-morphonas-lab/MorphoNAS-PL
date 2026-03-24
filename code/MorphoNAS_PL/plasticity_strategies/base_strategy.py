"""
Abstract base class for plasticity strategies.

Defines the interface that all plasticity strategies must implement.
"""

from abc import ABC, abstractmethod
import torch


class BasePlasticityStrategy(ABC):
    """
    Base class for plasticity learning rules.

    All plasticity strategies must implement the apply_plasticity method
    which computes weight updates based on neural activity.
    """

    def __init__(self, learning_rate: float, trace_decay: float):
        """
        Initialize plasticity strategy.

        Args:
            learning_rate: Learning rate for weight updates
            trace_decay: Decay rate for eligibility traces (between 0 and 1)
        """
        self.learning_rate = learning_rate
        self.trace_decay = trace_decay
        self.last_reward = 0.0

    @abstractmethod
    def accumulate_traces(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        connection_mask: torch.Tensor,
        traces: torch.Tensor,
    ) -> torch.Tensor:
        """
        Accumulate eligibility traces based on neural activity.

        This method updates traces but does NOT apply weight updates.
        Weight updates are applied at episode boundaries by the NeuralPropagator.

        Args:
            pre_activity: Pre-synaptic neuron activations (N,)
            post_activity: Post-synaptic neuron activations (N,)
            weights: Current weight matrix (N, N)
            connection_mask: Binary mask indicating existing connections (N, N)
            traces: Current eligibility traces (N, N)

        Returns:
            Updated eligibility traces (N, N)
        """
        pass

    def accumulate_traces_sparse(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        traces: torch.Tensor,
        edge_post_idx: torch.Tensor,
        edge_pre_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optional sparse trace update for edge-list representations.

        If implemented by a strategy, NeuralPropagator may store traces as a 1D tensor
        over existing connections (E,) instead of a dense (N, N) matrix.

        Args:
            pre_activity: Pre-synaptic activations (N,)
            post_activity: Post-synaptic activations (N,)
            weights: Current weight matrix (N, N)
            traces: Current traces over edges (E,)
            edge_post_idx: Post-synaptic indices for each edge (E,)
            edge_pre_idx: Pre-synaptic indices for each edge (E,)

        Returns:
            Updated traces over edges (E,)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement sparse traces"
        )

    @abstractmethod
    def reset_traces(
        self, shape: tuple, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Initialize or reset eligibility traces.

        Args:
            shape: Shape of the trace tensor (typically weight matrix shape)
            device: Optional device to place the trace tensor on

        Returns:
            Initialized trace tensor
        """
        pass

    def set_reward(self, reward: float):
        """
        Set the current timestep reward for reward-modulated strategies.

        Args:
            reward: Reward value from environment
        """
        self.last_reward = reward

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"lr={self.learning_rate}, "
            f"decay={self.trace_decay})"
        )
