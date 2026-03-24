from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import gymnasium as gym
import networkx as nx
import numpy as np

from MorphoNAS.genome import Genome
from MorphoNAS.grid import Grid
from MorphoNAS.neural_propagation import NeuralPropagator
from MorphoNAS_PL.plasticity_hooks import OjaEdgeHook

logger = logging.getLogger(__name__)

_shutdown_event = None


def set_shutdown_event(event: Any) -> None:
    """Set the global shutdown event for this process."""

    global _shutdown_event
    _shutdown_event = event


VERIFICATION_SEEDS = list(range(42, 62))


def _configure_cpu_bound_settings() -> None:
    """Tune CPU settings to avoid oversubscription in multiprocessing."""
    from MorphoNAS_PL.parallel_utils import configure_worker_threads
    configure_worker_threads()


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
    )


def _apply_episode_learning(
    propagator: NeuralPropagator, episode_reward: float
) -> None:
    hook = getattr(propagator, "edge_hook", None)
    if hook is None:
        return

    deltas = hook.end_episode(float(episode_reward))
    propagator.apply_edge_weight_deltas(deltas)


def run_rollouts(
    propagator: NeuralPropagator,
    num_rollouts: int,
    *,
    seeds: Optional[list[int]] = None,
    record_trajectory: bool = False,
    env: Optional[gym.Env] = None,
) -> dict:
    """Run multiple rollouts and collect statistics."""

    close_env = False
    if env is None:
        env = gym.make("CartPole-v1", render_mode=None)
        close_env = True

    rewards: list[float] = []
    lengths: list[int] = []
    weight_norms: list[float] = []

    for i in range(int(num_rollouts)):
        if _shutdown_event is not None and _shutdown_event.is_set():
            logger.debug("Rollout interrupted by shutdown event.")
            break

        propagator.reset()

        if seeds is not None and i < len(seeds):
            observation, _ = env.reset(seed=int(seeds[i]))
        else:
            observation, _ = env.reset()

        total_reward = 0.0
        done = False
        steps = 0

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

        _apply_episode_learning(propagator, total_reward)

        rewards.append(float(total_reward))
        lengths.append(int(steps))

        if record_trajectory:
            weight_norms.append(float(propagator.W.norm().item()))

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

    if record_trajectory:
        result["weight_norms"] = weight_norms

    return result


def run_experiment_on_network(
    *,
    genome: Genome,
    metadata: dict,
    plasticity_rollouts: int = 500,
    verification_rollouts: int = 20,
    learning_rate: float = 0.01,
    plasticity_strategy_name: str = "oja",
    trace_decay: float | None = None,
    checkpoint_at: list[int] | None = None,
) -> dict:
    """Run the full B0.2 experiment on a single network for one learning rate."""

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
        "plasticity_strategy": plasticity_strategy_name,
        "plasticity_rollouts": int(plasticity_rollouts),
        "verification_rollouts": int(verification_rollouts),
        "learning_rate": float(learning_rate),
        "network_stats": {"num_neurons": num_nodes, "num_connections": num_edges},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        baseline_propagator = create_propagator(
            grid, edge_hook=None, graph_diameter=graph_diameter
        )
        initial_weights = baseline_propagator.get_weights().copy()

        baseline_results = run_rollouts(
            baseline_propagator,
            int(verification_rollouts),
            seeds=VERIFICATION_SEEDS[: int(verification_rollouts)],
            env=env,
        )
        results["baseline"] = baseline_results

        if _shutdown_event is not None and _shutdown_event.is_set():
            return results

        decay = float(trace_decay) if trace_decay is not None else 1.0
        hook = OjaEdgeHook(learning_rate=float(learning_rate), trace_decay=decay)
        training_propagator = create_propagator(
            grid, edge_hook=hook, graph_diameter=graph_diameter
        )

        training_rewards: list[float] = []
        training_weight_norms: list[float] = []
        checkpoint_results: dict[int, dict] = {}

        if checkpoint_at:
            sorted_checkpoints = sorted(int(c) for c in checkpoint_at)
            current_rollout = 0
            for checkpoint in sorted_checkpoints:
                if _shutdown_event is not None and _shutdown_event.is_set():
                    break

                rollouts_needed = checkpoint - current_rollout
                for batch_start in range(0, rollouts_needed, 50):
                    if _shutdown_event is not None and _shutdown_event.is_set():
                        break

                    batch_results = run_rollouts(
                        training_propagator,
                        num_rollouts=min(50, rollouts_needed - batch_start),
                        record_trajectory=True,
                        env=env,
                    )
                    training_rewards.extend(batch_results["rewards"])
                    training_weight_norms.extend(batch_results["weight_norms"])

                checkpoint_weights = training_propagator.get_weights().copy()
                checkpoint_verification_prop = create_propagator(
                    grid, edge_hook=None, graph_diameter=graph_diameter
                )
                checkpoint_verification_prop.set_weights(checkpoint_weights)
                checkpoint_verification = run_rollouts(
                    checkpoint_verification_prop,
                    int(verification_rollouts),
                    seeds=VERIFICATION_SEEDS[: int(verification_rollouts)],
                    env=env,
                )

                checkpoint_results[checkpoint] = {
                    "fitness": float(checkpoint_verification["fitness"]),
                    "avg_reward": float(checkpoint_verification["avg_reward"]),
                    "delta_fitness": float(checkpoint_verification["fitness"])
                    - float(baseline_results["fitness"]),
                    "weight_change_norm": float(
                        np.linalg.norm(checkpoint_weights - initial_weights)
                    ),
                }
                current_rollout = checkpoint

            trained_weights = training_propagator.get_weights().copy()
        else:
            batch_results = run_rollouts(
                training_propagator,
                int(plasticity_rollouts),
                record_trajectory=True,
                env=env,
            )
            training_rewards.extend(batch_results["rewards"])
            training_weight_norms.extend(batch_results["weight_norms"])
            trained_weights = training_propagator.get_weights().copy()

        results["training"] = {
            "rewards": training_rewards,
            "avg_reward": float(np.mean(training_rewards)) if training_rewards else 0.0,
            "weight_norms": training_weight_norms,
        }
        if checkpoint_results:
            results["checkpoints"] = checkpoint_results

        if _shutdown_event is not None and _shutdown_event.is_set():
            return results

        verification_propagator = create_propagator(
            grid, edge_hook=None, graph_diameter=graph_diameter
        )
        verification_propagator.set_weights(trained_weights)
        verification_results = run_rollouts(
            verification_propagator,
            int(verification_rollouts),
            seeds=VERIFICATION_SEEDS[: int(verification_rollouts)],
            env=env,
        )
        results["verification"] = verification_results

        results["summary"] = {
            "baseline_fitness": float(baseline_results["fitness"]),
            "post_plasticity_fitness": float(verification_results["fitness"]),
            "delta_fitness": float(verification_results["fitness"])
            - float(baseline_results["fitness"]),
            "improved": bool(
                float(verification_results["fitness"])
                > float(baseline_results["fitness"])
            ),
            "weight_change_norm": float(
                np.linalg.norm(trained_weights - initial_weights)
            ),
        }
    finally:
        env.close()

    logger.debug(f"[{network_id}] FINISHED experiment: LR={learning_rate:.0e}")
    return results
