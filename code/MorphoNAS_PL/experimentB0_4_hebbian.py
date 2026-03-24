from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import gymnasium as gym
import networkx as nx
import numpy as np
import torch

from MorphoNAS.genome import Genome
from MorphoNAS.grid import Grid
from MorphoNAS.neural_propagation import NeuralPropagator
from MorphoNAS_PL.plasticity_hooks import HebbianEdgeHook, OjaStepEdgeHook, WeightClamp

logger = logging.getLogger(__name__)

_shutdown_event = None


def set_shutdown_event(event: Any) -> None:
    """Set the global shutdown event for this process."""

    global _shutdown_event
    _shutdown_event = event


VERIFICATION_SEEDS = list(range(42, 62))


def load_network_from_pool(filepath: str) -> tuple[Genome, dict]:
    """Load a network from the pool."""

    with open(filepath, "r") as f:
        data = json.load(f)

    genome = Genome.from_dict(data["genome"])
    metadata = {
        "network_id": data.get("network_id"),
        "seed": data.get("seed"),
        "category": data.get("category"),
        "original_avg_reward": data.get("avg_reward"),
    }

    return genome, metadata


def create_propagator(
    grid: Grid,
    *,
    edge_hook=None,
    graph_diameter: Optional[int] = None,
    weight_stabilizer=None,
) -> NeuralPropagator:
    """Create a NeuralPropagator from a grown grid."""

    G = grid.get_graph()

    return NeuralPropagator(
        G=G,
        input_dim=4,
        output_dim=2,
        activation_function=NeuralPropagator.tanh_activation,
        extra_thinking_time=2,
        additive_update=False,
        graph_diameter=graph_diameter,
        edge_hook=edge_hook,
        weight_stabilizer=weight_stabilizer,
    )


def _apply_step_learning(propagator: NeuralPropagator) -> float:
    hook = getattr(propagator, "edge_hook", None)
    if hook is None or not hasattr(hook, "end_step"):
        return 0.0

    deltas = hook.end_step()
    propagator.apply_edge_weight_deltas(deltas)
    return float(deltas.abs().mean().item()) if deltas.numel() > 0 else 0.0


def run_rollouts(
    propagator: NeuralPropagator,
    num_rollouts: int,
    *,
    seeds: Optional[list[int]] = None,
    reset_plastic_each_episode: bool = True,
    record_plasticity: bool = False,
    env: Optional[gym.Env] = None,
) -> dict:
    """Run multiple rollouts with optional per-step plasticity updates."""

    close_env = False
    if env is None:
        env = gym.make("CartPole-v1", render_mode=None)
        close_env = True

    base_weights = propagator.get_weights().copy()
    rewards: list[float] = []
    lengths: list[int] = []
    mean_abs_deltas: list[float] = []
    max_abs_deltas: list[float] = []

    for i in range(int(num_rollouts)):
        if _shutdown_event is not None and _shutdown_event.is_set():
            logger.debug("Rollout interrupted by shutdown event.")
            break

        propagator.reset()
        if reset_plastic_each_episode:
            propagator.set_weights(base_weights.copy())

        if seeds is not None and i < len(seeds):
            observation, _ = env.reset(seed=int(seeds[i]))
        else:
            observation, _ = env.reset()

        total_reward = 0.0
        done = False
        steps = 0
        step_delta_sum = 0.0
        step_delta_max = 0.0

        hook = getattr(propagator, "edge_hook", None)

        while not done:
            obs = np.array(observation).flatten()
            propagator.propagate(obs)
            output_values = propagator.get_output()
            action = int(output_values.argmax().item())

            observation, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            reward_f = float(reward)
            total_reward += reward_f
            steps += 1

            if hook is not None:
                hook.on_reward(reward_f)

            step_delta = _apply_step_learning(propagator)
            if record_plasticity:
                step_delta_sum += step_delta
                if step_delta > step_delta_max:
                    step_delta_max = step_delta

        if hook is not None and hasattr(hook, "end_episode"):
            hook.end_episode(total_reward)

        rewards.append(float(total_reward))
        lengths.append(int(steps))

        if record_plasticity:
            mean_abs_deltas.append(step_delta_sum / max(steps, 1))
            max_abs_deltas.append(step_delta_max)

    if close_env:
        env.close()

    result = {
        "rewards": rewards,
        "lengths": lengths,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "min_reward": float(np.min(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
        "fitness": (float(np.mean(rewards)) / 500.0) if rewards else 0.0,
    }

    if record_plasticity:
        result["plasticity"] = {
            "mean_abs_delta_per_step": float(np.mean(mean_abs_deltas))
            if mean_abs_deltas
            else 0.0,
            "max_abs_delta_per_step": float(np.max(max_abs_deltas))
            if max_abs_deltas
            else 0.0,
            "per_episode_mean_abs_delta": mean_abs_deltas,
            "per_episode_max_abs_delta": max_abs_deltas,
        }

    return result


def run_experiment_on_network(
    *,
    genome: Genome,
    metadata: dict,
    rollouts: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    reset_plastic_each_episode: bool = True,
    edge_scope: str = "all",
    weight_clamp: bool = False,
    clamp_min: float = -2.0,
    clamp_max: float = 2.0,
    plasticity_rule: str = "hebbian",
) -> dict:
    """Run the simple Hebbian evaluation on a single network."""

    network_id = metadata.get("network_id")
    category = metadata.get("category")

    logger.debug(f"[{network_id}] STARTING experiment: LR={learning_rate:.0e}")

    grid = Grid(genome)
    grid.run_simulation(verbose=False)
    G = grid.get_graph()

    num_nodes = int(G.number_of_nodes())
    num_edges = int(G.number_of_edges())

    if num_nodes < 6:
        logger.warning(f"[{network_id}] SKIP: Insufficient neurons ({num_nodes})")
        return {"error": "insufficient_neurons", "num_neurons": num_nodes}

    try:
        graph_diameter = int(nx.diameter(G))
    except nx.NetworkXError:
        max_length = 0
        for source in G.nodes():
            lengths = nx.shortest_path_length(G, source)
            max_length = max(max_length, max(lengths.values()))
        graph_diameter = int(max_length)

    env = gym.make("CartPole-v1", render_mode=None)

    results = {
        "network_id": network_id,
        "category": category,
        "original_seed": metadata.get("seed"),
        "original_avg_reward": metadata.get("original_avg_reward"),
        "plasticity_strategy": f"{plasticity_rule}_per_step",
        "rollouts": int(rollouts),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "reset_plastic_each_episode": bool(reset_plastic_each_episode),
        "edge_scope": edge_scope,
        "weight_clamp": bool(weight_clamp),
        "clamp_min": float(clamp_min),
        "clamp_max": float(clamp_max),
        "network_stats": {"num_neurons": num_nodes, "num_connections": num_edges},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        baseline_propagator = create_propagator(
            grid, edge_hook=None, graph_diameter=graph_diameter
        )
        baseline_results = run_rollouts(
            baseline_propagator,
            int(rollouts),
            seeds=VERIFICATION_SEEDS[: int(rollouts)],
            reset_plastic_each_episode=True,
            env=env,
        )
        results["baseline"] = baseline_results

        if _shutdown_event is not None and _shutdown_event.is_set():
            return results

        if plasticity_rule == "hebbian":
            hook = HebbianEdgeHook(
                learning_rate=float(learning_rate), weight_decay=float(weight_decay)
            )
        elif plasticity_rule == "oja_step":
            hook = OjaStepEdgeHook(learning_rate=float(learning_rate))
        else:
            raise ValueError(f"Unknown plasticity_rule: {plasticity_rule}")
        stabilizer = None
        if weight_clamp:
            stabilizer = WeightClamp(min_value=float(clamp_min), max_value=float(clamp_max))
        plastic_propagator = create_propagator(
            grid,
            edge_hook=hook,
            graph_diameter=graph_diameter,
            weight_stabilizer=stabilizer,
        )
        if edge_scope != "all":
            if edge_scope == "output":
                num_neurons = int(plastic_propagator.W.shape[0])
                output_start = num_neurons - int(plastic_propagator.output_dim)
                edge_mask = (plastic_propagator.edge_post_idx >= output_start).float()
                hook.set_edge_mask(edge_mask)
            elif edge_scope == "input":
                input_nodes = torch.tensor(
                    plastic_propagator.input_nodes,
                    device=plastic_propagator.edge_pre_idx.device,
                    dtype=plastic_propagator.edge_pre_idx.dtype,
                )
                edge_mask = torch.isin(plastic_propagator.edge_pre_idx, input_nodes).float()
                hook.set_edge_mask(edge_mask)
            else:
                raise ValueError(f"Unknown edge_scope: {edge_scope}")
        plastic_results = run_rollouts(
            plastic_propagator,
            int(rollouts),
            seeds=VERIFICATION_SEEDS[: int(rollouts)],
            reset_plastic_each_episode=reset_plastic_each_episode,
            record_plasticity=True,
            env=env,
        )
        results["plastic"] = plastic_results

        results["summary"] = {
            "baseline_fitness": float(baseline_results["fitness"]),
            "plastic_fitness": float(plastic_results["fitness"]),
            "delta_fitness": float(plastic_results["fitness"])
            - float(baseline_results["fitness"]),
            "improved": bool(
                float(plastic_results["fitness"])
                > float(baseline_results["fitness"])
            ),
            "mean_abs_delta_per_step": float(
                plastic_results.get("plasticity", {}).get(
                    "mean_abs_delta_per_step", 0.0
                )
            ),
        }
    finally:
        env.close()

    logger.debug(f"[{network_id}] FINISHED experiment: LR={learning_rate:.0e}")
    return results
