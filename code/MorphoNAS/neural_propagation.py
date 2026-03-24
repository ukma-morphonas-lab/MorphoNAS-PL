from __future__ import annotations

import numpy as np
import networkx as nx
import gymnasium as gym
from typing import Callable, Tuple, Union, List, Optional

import torch

from .hooks import EdgeDynamicsHook, WeightStabilizer, hook_supports_sequence


class NeuralPropagator:
    def __init__(
        self,
        G: nx.DiGraph,
        input_dim: int,
        output_dim: int,
        activation_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        extra_thinking_time: int = 0,
        additive_update: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        graph_diameter: Optional[int] = None,
        edge_hook: Optional[EdgeDynamicsHook] = None,
        weight_stabilizer: Optional[WeightStabilizer] = None,
    ):
        self.device = device

        # Create a mapping from original node indices to 0-based indices
        self.node_mapping = {node: idx for idx, node in enumerate(sorted(G.nodes()))}

        # Reorder nodes by in-degree (ascending)
        self.node_order = sorted(G.nodes(), key=lambda x: G.in_degree(x))
        self.G = nx.DiGraph()

        # Create new graph with reordered nodes using 0-based indices
        for node in self.node_order:
            self.G.add_node(self.node_mapping[node])

        # Add edges with original weights, using mapped indices
        for u, v, data in G.edges(data=True):
            self.G.add_edge(
                self.node_mapping[u], self.node_mapping[v], weight=data["weight"]
            )

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.activation_function = activation_function or self.tanh_activation

        # Calculate graph diameter and set thinking time
        if graph_diameter is not None:
            self.graph_diameter = int(graph_diameter)
        else:
            try:
                self.graph_diameter = int(nx.diameter(self.G))
            except nx.NetworkXError:
                max_length = 0
                for source in self.G.nodes():
                    lengths = nx.shortest_path_length(self.G, source)
                    max_length = max(max_length, max(lengths.values()))
                self.graph_diameter = int(max_length)

        self.network_thinking_time = int(self.graph_diameter + extra_thinking_time)
        self.additive_update = bool(additive_update)

        # Input nodes are now simply the first input_dim nodes
        self.input_nodes = [
            self.node_mapping[node] for node in self.node_order[: self.input_dim]
        ]
        self.input_nodes_tensor = torch.tensor(
            self.input_nodes, device=self.device, dtype=torch.long
        )

        # Convert graph to weight matrix
        self.W = self._get_weight_matrix()
        self.connection_mask = (self.W != 0).float()

        # Precompute edge indices for sparse math (W[post, pre])
        self.edge_post_idx, self.edge_pre_idx = self.connection_mask.nonzero(
            as_tuple=True
        )
        self.num_edges = int(self.edge_post_idx.numel())
        self.edge_weights = self.W[self.edge_post_idx, self.edge_pre_idx].clone()

        # Optional sparse forward pass (CPU): O(E) index_add instead of dense matmul O(N^2)
        num_neurons = int(self.W.shape[0])
        self.use_sparse_forward = (
            self.device == "cpu"
            and not self.additive_update
            and self.activation_function
            in (NeuralPropagator.tanh_activation, NeuralPropagator.relu_activation)
            and num_neurons >= 256
            and self.num_edges < num_neurons * 16
        )
        self._sparse_forward_buffer = torch.zeros(num_neurons, device=self.device)

        # Initialize network state
        self.network_state = torch.zeros(len(G), device=self.device)

        # Hook plumbing (external state)
        self.edge_hook = edge_hook
        self.weight_stabilizer = weight_stabilizer

        if self.edge_hook is not None:
            self.edge_hook.reset(
                num_neurons=num_neurons, num_edges=self.num_edges, device=self.W.device
            )

            if hook_supports_sequence(self.edge_hook):
                T = self.network_thinking_time
                self._activity_pre_buffer = torch.empty(
                    (T, num_neurons), device=self.device
                )
                self._activity_post_buffer = torch.empty(
                    (T, num_neurons), device=self.device
                )

    def _select_input_nodes(self) -> List[int]:
        return self.node_order[: self.input_dim]

    @staticmethod
    def tanh_activation(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def relu_activation(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    def _get_weight_matrix(self) -> torch.Tensor:
        num_neurons = len(self.G)
        W = torch.zeros((num_neurons, num_neurons), device=self.device)

        node_to_idx = {node: idx for idx, node in enumerate(sorted(self.G.nodes()))}

        for i, j, data in self.G.edges(data=True):
            pre_idx = node_to_idx[i]
            post_idx = node_to_idx[j]
            W[post_idx, pre_idx] = data["weight"]

        return W

    @torch.no_grad()
    def apply_edge_weight_deltas(self, deltas: torch.Tensor) -> None:
        if deltas.numel() != self.num_edges:
            raise ValueError(
                f"Expected {self.num_edges} edge deltas, got {int(deltas.numel())}"
            )

        if deltas.device != self.edge_weights.device:
            deltas = deltas.to(self.edge_weights.device)

        self.edge_weights.add_(deltas)
        self.W[self.edge_post_idx, self.edge_pre_idx] = self.edge_weights

        if self.weight_stabilizer is not None:
            self.weight_stabilizer.stabilize(W=self.W, edge_weights=self.edge_weights)
            self.edge_weights = self.W[self.edge_post_idx, self.edge_pre_idx].clone()

    @torch.no_grad()
    def propagate(self, input_values: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(input_values, np.ndarray):
            input_values = torch.from_numpy(input_values).to(self.device)

        if len(input_values) != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {len(input_values)}"
            )

        self.network_state[self.input_nodes_tensor] = input_values

        current_state = (
            self.network_state.clone()
            if (self.use_sparse_forward or self.additive_update)
            else self.network_state
        )

        scratch_state = self._sparse_forward_buffer

        hook = self.edge_hook
        use_sequence = hook is not None and hook_supports_sequence(hook)
        all_pre: Optional[torch.Tensor] = None
        all_post: Optional[torch.Tensor] = None
        if use_sequence:
            all_pre = self._activity_pre_buffer
            all_post = self._activity_post_buffer

        for t in range(self.network_thinking_time):
            pre_activity = (
                current_state.clone() if self.additive_update else current_state
            )

            if self.use_sparse_forward:
                scratch_state.zero_()
                scratch_state.index_add_(
                    0,
                    self.edge_post_idx,
                    self.edge_weights * current_state[self.edge_pre_idx],
                )

                if self.activation_function == NeuralPropagator.tanh_activation:
                    scratch_state.tanh_()
                else:
                    scratch_state.relu_()

                post_activity = scratch_state
            else:
                new_state = torch.matmul(self.W, current_state)
                post_activity = self.activation_function(new_state)

            if hook is not None:
                if use_sequence and all_pre is not None and all_post is not None:
                    all_pre[t] = pre_activity
                    all_post[t] = post_activity
                else:
                    hook.on_step(
                        pre_activity=pre_activity,
                        post_activity=post_activity,
                        edge_pre_idx=self.edge_pre_idx,
                        edge_post_idx=self.edge_post_idx,
                        edge_weights=self.edge_weights,
                    )

            if self.additive_update:
                current_state += post_activity
            else:
                if self.use_sparse_forward:
                    current_state, scratch_state = post_activity, current_state
                else:
                    current_state = post_activity

        self.network_state = current_state

        if (
            hook is not None
            and use_sequence
            and all_pre is not None
            and all_post is not None
        ):
            hook.on_sequence(
                pre_activities=all_pre,
                post_activities=all_post,
                edge_pre_idx=self.edge_pre_idx,
                edge_post_idx=self.edge_post_idx,
                edge_weights=self.edge_weights,
            )

        if self.use_sparse_forward:
            self._sparse_forward_buffer = scratch_state

        return current_state

    def get_output(self) -> torch.Tensor:
        return self.network_state[-self.output_dim :]

    def reset(self) -> None:
        self.network_state.zero_()
        self._sparse_forward_buffer.zero_()

        if self.edge_hook is not None:
            self.edge_hook.reset_episode()

    def get_weights(self) -> np.ndarray:
        return self.W.cpu().numpy()

    def set_weights(self, weights: np.ndarray) -> None:
        if weights.shape != self.W.shape:
            raise ValueError(
                f"Weight shape mismatch: expected {self.W.shape}, got {weights.shape}"
            )

        self.W = torch.from_numpy(weights).to(self.device).float()
        self.edge_weights = self.W[self.edge_post_idx, self.edge_pre_idx].clone()

    def get_weight_stats(self) -> dict:
        weights = self.W.cpu().numpy()
        nonzero = weights[weights != 0]
        return {
            "weight_norm": float(np.linalg.norm(weights)),
            "num_connections": int(np.count_nonzero(weights)),
            "mean_weight": float(np.mean(nonzero)) if len(nonzero) > 0 else 0.0,
            "std_weight": float(np.std(nonzero)) if len(nonzero) > 0 else 0.0,
            "min_weight": float(np.min(nonzero)) if len(nonzero) > 0 else 0.0,
            "max_weight": float(np.max(nonzero)) if len(nonzero) > 0 else 0.0,
        }

    def get_input_nodes_info(self) -> dict:
        zero_in_degree_nodes = [
            node for node in self.G.nodes() if self.G.in_degree(node) == 0
        ]
        return {
            "total_input_nodes": len(self.input_nodes),
            "zero_in_degree_nodes": len(zero_in_degree_nodes),
            "selected_nodes": self.input_nodes,
            "node_in_degrees": {
                node: self.G.in_degree(node) for node in self.input_nodes
            },
        }


class GymEnvironment:
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.input_dim, self.output_dim = self._get_network_dimensions()
        self.env = None

    def _get_network_dimensions(self) -> Tuple[int, int]:
        env = gym.make(self.env_name)

        if isinstance(env.observation_space, gym.spaces.Discrete):
            input_dim = int(env.observation_space.n)
        elif isinstance(env.observation_space, gym.spaces.Box):
            input_dim = int(np.prod(env.observation_space.shape))
        else:
            raise ValueError(
                f"Unsupported observation space type: {type(env.observation_space)}"
            )

        if isinstance(env.action_space, gym.spaces.Discrete):
            output_dim = int(env.action_space.n)
        elif isinstance(env.action_space, gym.spaces.Box):
            output_dim = int(np.prod(env.action_space.shape))
        else:
            raise ValueError(f"Unsupported action space type: {type(env.action_space)}")

        env.close()
        return input_dim, output_dim

    def rollout(
        self,
        propagator: NeuralPropagator,
        render: bool = False,
        seed: Optional[int] = None,
    ) -> float:
        self.env = gym.make(self.env_name, render_mode="human" if render else None)

        if seed is not None:
            self.env.reset(seed=seed)

        observation, _ = self.env.reset()
        total_reward: float = 0.0
        done = False

        while not done:
            if isinstance(self.env.observation_space, gym.spaces.Discrete):
                obs = torch.zeros(propagator.input_dim, device=propagator.device)
                obs[observation] = 1
            else:
                obs = torch.from_numpy(np.array(observation).flatten()).to(
                    propagator.device
                )

            propagator.propagate(obs)

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                if propagator.output_dim != self.env.action_space.n:
                    raise ValueError(
                        f"Output dimension ({propagator.output_dim}) must match number of actions ({self.env.action_space.n})"
                    )

                output_values = propagator.get_output()
                action = int(output_values.argmax().item())

                if render:
                    print(
                        f"Output values: {output_values.cpu().numpy()}, Selected action: {action}"
                    )
            else:
                action_space = self.env.action_space
                if not isinstance(action_space, gym.spaces.Box):
                    raise TypeError(
                        f"Expected Box action space, got {type(action_space)}"
                    )
                action = propagator.get_output().cpu().numpy()
                action = np.clip(action, action_space.low, action_space.high)

            observation, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)

        self.env.close()
        self.env = None
        return float(total_reward)

    def visualize_network(self, G: nx.DiGraph, propagator: NeuralPropagator) -> None:
        import matplotlib.pyplot as plt

        if len(G.nodes()) == 0:
            print("Warning: Cannot visualize empty graph")
            return

        if len(G.edges()) == 0:
            print("Warning: Graph has no edges, adding self-loops for visualization")
            for node in G.nodes():
                G.add_edge(node, node, weight=0.1)

        G_viz = nx.DiGraph()
        node_mapping = {
            old_node: new_idx for new_idx, old_node in enumerate(sorted(G.nodes()))
        }

        for old_node in G.nodes():
            G_viz.add_node(node_mapping[old_node])

        for u, v, data in G.edges(data=True):
            G_viz.add_edge(node_mapping[u], node_mapping[v], weight=data["weight"])

        try:
            pos = nx.kamada_kawai_layout(G_viz)
        except nx.NetworkXError:
            try:
                pos = nx.spring_layout(G_viz, k=1, iterations=50)
            except nx.NetworkXError:
                print(
                    "Error: Could not compute node positions. Graph may be disconnected."
                )
                return

        plt.figure(figsize=(10, 8))

        input_nodes = propagator.input_nodes
        hidden_nodes = [
            node
            for node in G_viz.nodes()
            if node not in input_nodes and node < len(G) - propagator.output_dim
        ]
        output_nodes = [
            node for node in G_viz.nodes() if node >= len(G) - propagator.output_dim
        ]

        nx.draw_networkx_edges(G_viz, pos, edge_color="gray", alpha=0.5, arrows=True)

        if input_nodes:
            nx.draw_networkx_nodes(
                G_viz, pos, nodelist=input_nodes, node_color="blue", label="Input"
            )
        if hidden_nodes:
            nx.draw_networkx_nodes(
                G_viz, pos, nodelist=hidden_nodes, node_color="green", label="Hidden"
            )
        if output_nodes:
            nx.draw_networkx_nodes(
                G_viz, pos, nodelist=output_nodes, node_color="red", label="Output"
            )

        labels = {new_idx: str(old_node) for old_node, new_idx in node_mapping.items()}
        nx.draw_networkx_labels(G_viz, pos, labels=labels)

        plt.legend()
        plt.title("Neural Network Structure")
        plt.axis("off")
        plt.show()
