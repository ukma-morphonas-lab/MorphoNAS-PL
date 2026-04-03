"""
Acrobot Plasticity Experiment Module

Adapts the B0.5 CartPole infrastructure for Acrobot-v1:
- 6D observations (cos/sin of two joint angles + angular velocities)
- 3 discrete actions (torque: -1, 0, +1)
- Negative rewards (-1 per step; max episode = 500 steps)
- 20x20 developmental grid (vs 10x10 for CartPole)

Strata are defined post-hoc based on the observed reward distribution,
since Acrobot reward ranges differ from CartPole.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import gymnasium as gym
import networkx as nx
import numpy as np
import torch

from MorphoNAS.genome import Genome
from MorphoNAS.grid import Grid
from MorphoNAS.neural_propagation import NeuralPropagator
from MorphoNAS_PL.plasticity_hooks import HebbianEdgeHook

logger = logging.getLogger(__name__)

_shutdown_event = None

ENV_NAME = "Acrobot-v1"
INPUT_DIM = 6
OUTPUT_DIM = 3
MIN_NEURONS = INPUT_DIM + OUTPUT_DIM  # 9


def set_shutdown_event(event: Any) -> None:
    global _shutdown_event
    _shutdown_event = event


# ── Stratum definitions ──────────────────────────────────────────────
# Acrobot rewards are negative: -500 (never solves) to ~-80 (solves fast).
# These boundaries are set conservatively; update after seeing pool distribution.

class Stratum(str, Enum):
    WEAK = "weak"
    LOW_MID = "low_mid"
    HIGH_MID = "high_mid"
    NEAR_PERFECT = "near_perfect"
    PERFECT = "perfect"


STRATUM_BOUNDS = {
    Stratum.WEAK: (-500, -450),
    Stratum.LOW_MID: (-450, -300),
    Stratum.HIGH_MID: (-300, -200),
    Stratum.NEAR_PERFECT: (-200, -100),
    Stratum.PERFECT: (-100, float("inf")),
}


def get_stratum(reward: float) -> Stratum:
    """Assign stratum based on mean baseline reward."""
    for stratum, (low, high) in STRATUM_BOUNDS.items():
        if low <= reward < high:
            return stratum
    # Rewards >= -100 count as perfect (Gymnasium solved threshold)
    return Stratum.PERFECT


def get_stratum_label(stratum: Stratum) -> str:
    labels = {
        Stratum.WEAK: "Weak [-500, -450)",
        Stratum.LOW_MID: "Low-mid [-450, -300)",
        Stratum.HIGH_MID: "High-mid [-300, -200)",
        Stratum.NEAR_PERFECT: "Near-perfect [-200, -100)",
        Stratum.PERFECT: "Perfect [>=-100]",
    }
    return labels.get(stratum, str(stratum))


# ── Eta / decay grid (22-point focused grid, same as B0.6) ──────────

ETA_VALUES = [
    -0.20, -0.10, -0.05, -0.03, -0.01,
    0.0,
    0.01, 0.03, 0.05, 0.10, 0.20,
]

DECAY_VALUES = [0.0, 0.01]

VERIFICATION_SEEDS = list(range(42, 62))  # 20 fixed seeds


@dataclass
class GridPoint:
    eta: float
    decay: float

    def to_dict(self) -> dict:
        return {"eta": self.eta, "decay": self.decay}

    @classmethod
    def from_dict(cls, d: dict) -> "GridPoint":
        return cls(eta=d["eta"], decay=d["decay"])


def generate_grid() -> list[GridPoint]:
    grid = []
    for eta in ETA_VALUES:
        for decay in DECAY_VALUES:
            grid.append(GridPoint(eta=eta, decay=decay))
    return grid


# ── Network loading and creation ─────────────────────────────────────

def load_network_from_file(filepath: str) -> tuple[Genome, dict]:
    with open(filepath, "r") as f:
        data = json.load(f)

    genome = Genome.from_dict(data["genome"])
    metadata = {
        "network_id": data.get("network_id"),
        "seed": data.get("seed"),
        "stratum": data.get("stratum"),
        "baseline_reward": data.get("baseline_reward"),
        "baseline_fitness": data.get("baseline_fitness"),
        "network_stats": data.get("network_stats", {}),
    }
    return genome, metadata


def create_propagator(
    grid: Grid,
    *,
    edge_hook=None,
    graph_diameter: Optional[int] = None,
) -> NeuralPropagator:
    G = grid.get_graph()
    return NeuralPropagator(
        G=G,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        activation_function=NeuralPropagator.tanh_activation,
        extra_thinking_time=2,
        additive_update=False,
        graph_diameter=graph_diameter,
        edge_hook=edge_hook,
    )


# ── Evaluation functions ─────────────────────────────────────────────

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
    record_step_trace: bool = False,
    env: Optional[gym.Env] = None,
) -> dict:
    """Run multiple Acrobot episodes with optional per-step plasticity."""
    if record_step_trace:
        record_plasticity = True

    close_env = False
    if env is None:
        env = gym.make(ENV_NAME, render_mode=None)
        close_env = True

    base_weights = propagator.get_weights().copy()
    rewards: list[float] = []
    lengths: list[int] = []
    mean_abs_deltas: list[float] = []
    max_abs_deltas: list[float] = []
    all_step_traces: list[list[float]] = []

    for i in range(int(num_rollouts)):
        if _shutdown_event is not None and _shutdown_event.is_set():
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
        episode_step_trace: list[float] = []

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
            if record_step_trace:
                episode_step_trace.append(step_delta)

        if hook is not None and hasattr(hook, "end_episode"):
            hook.end_episode(total_reward)

        rewards.append(float(total_reward))
        lengths.append(int(steps))

        if record_plasticity:
            mean_abs_deltas.append(step_delta_sum / max(steps, 1))
            max_abs_deltas.append(step_delta_max)
        if record_step_trace:
            all_step_traces.append(episode_step_trace)

    if close_env:
        env.close()

    result = {
        "rewards": rewards,
        "lengths": lengths,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "min_reward": float(np.min(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
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
        if record_step_trace:
            result["plasticity"]["step_traces"] = all_step_traces

    return result


def run_grid_evaluation(
    *,
    genome: Genome,
    metadata: dict,
    eta: float,
    decay: float,
    rollouts: int = 20,
    reset_plastic_each_episode: bool = True,
) -> dict:
    """Baseline + plastic evaluation for one (network, eta, decay) on Acrobot."""
    network_id = metadata.get("network_id")
    stratum = metadata.get("stratum")

    grid = Grid(genome)
    grid.run_simulation(verbose=False)
    G = grid.get_graph()

    num_nodes = int(G.number_of_nodes())
    num_edges = int(G.number_of_edges())

    if num_nodes < MIN_NEURONS:
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

    env = gym.make(ENV_NAME, render_mode=None)

    results = {
        "network_id": network_id,
        "stratum": stratum,
        "original_seed": metadata.get("seed"),
        "baseline_reward_stored": metadata.get("baseline_reward"),
        "eta": float(eta),
        "decay": float(decay),
        "rollouts": int(rollouts),
        "reset_plastic_each_episode": bool(reset_plastic_each_episode),
        "network_stats": {"num_neurons": num_nodes, "num_connections": num_edges},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        # Baseline (no plasticity)
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

        # Plastic evaluation
        if abs(eta) < 1e-12:
            plastic_results = baseline_results.copy()
            plastic_results["plasticity"] = {
                "mean_abs_delta_per_step": 0.0,
                "max_abs_delta_per_step": 0.0,
            }
        else:
            hook = HebbianEdgeHook(
                learning_rate=float(eta), weight_decay=float(decay)
            )
            plastic_propagator = create_propagator(
                grid, edge_hook=hook, graph_diameter=graph_diameter
            )
            plastic_results = run_rollouts(
                plastic_propagator,
                int(rollouts),
                seeds=VERIFICATION_SEEDS[: int(rollouts)],
                reset_plastic_each_episode=reset_plastic_each_episode,
                record_plasticity=True,
                env=env,
            )
        results["plastic"] = plastic_results

        # Summary
        baseline_reward = float(baseline_results["avg_reward"])
        plastic_reward = float(plastic_results["avg_reward"])

        results["summary"] = {
            "baseline_reward": baseline_reward,
            "plastic_reward": plastic_reward,
            "delta_reward": plastic_reward - baseline_reward,
            "improved": bool(plastic_reward > baseline_reward),
            "mean_abs_delta_per_step": float(
                plastic_results.get("plasticity", {}).get("mean_abs_delta_per_step", 0.0)
            ),
        }
    finally:
        env.close()

    return results


# ── Summary helpers ──────────────────────────────────────────────────

def compute_stratum_summary(results: list[dict], stratum: Stratum) -> dict:
    stratum_results = [r for r in results if r.get("stratum") == stratum.value]
    if not stratum_results:
        return {"n": 0}

    delta_rewards = [r.get("summary", {}).get("delta_reward", 0) for r in stratum_results]
    improved = [r.get("summary", {}).get("improved", False) for r in stratum_results]

    return {
        "n": len(stratum_results),
        "delta_reward_mean": float(np.mean(delta_rewards)),
        "delta_reward_std": float(np.std(delta_rewards)),
        "delta_reward_median": float(np.median(delta_rewards)),
        "improvement_rate": float(np.mean(improved)),
    }


def format_eta(eta: float) -> str:
    if eta == 0:
        return "eta_0"
    sign = "pos" if eta > 0 else "neg"
    return f"eta_{sign}_{abs(eta):.1e}".replace(".", "p").replace("-", "m")


def format_decay(decay: float) -> str:
    if decay == 0:
        return "decay_0"
    return f"decay_{decay:.0e}".replace(".", "p").replace("-", "m")


def format_grid_point(eta: float, decay: float) -> str:
    return f"{format_eta(eta)}_{format_decay(decay)}"
