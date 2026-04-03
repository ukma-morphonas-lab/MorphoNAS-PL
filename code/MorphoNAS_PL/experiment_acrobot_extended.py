"""
Acrobot Extended Validation Experiment Module

Determines whether finer η resolution reveals beneficial fixed plasticity
settings that the coarse 22-point grid missed. Supports both static and
non-stationary (heavy_link2_2x) conditions.

Analogous to what B0.5+ did for CartPole.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import gymnasium as gym
import networkx as nx
import numpy as np

from MorphoNAS.genome import Genome
from MorphoNAS.grid import Grid
from MorphoNAS.neural_propagation import NeuralPropagator
from MorphoNAS_PL.env_wrappers_acrobot import AcrobotMidEpisodeSwitchWrapper
from MorphoNAS_PL.experiment_acrobot import (
    ENV_NAME,
    MIN_NEURONS,
    VERIFICATION_SEEDS,
    create_propagator,
    load_network_from_file,
)
from MorphoNAS_PL.experiment_acrobot_nonstationary import VARIANTS
from MorphoNAS_PL.plasticity_hooks import HebbianEdgeHook

logger = logging.getLogger(__name__)

_shutdown_event = None


def set_shutdown_event(event: Any) -> None:
    global _shutdown_event
    _shutdown_event = event


# ── Grid definitions ─────────────────────────────────────────────────

# Full candidate grid (31 η × 8 λ = 248 points)
ETA_CANDIDATES = [
    -0.1, -0.05, -0.03, -0.02, -0.01, -0.007, -0.005, -0.003, -0.002,
    -0.001, -0.0007, -0.0005, -0.0003, -0.0001, -0.00005,
    0,
    0.00005, 0.0001, 0.0003, 0.0005, 0.0007,
    0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.1,
]

DECAY_CANDIDATES = [0, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

NS_VARIANT = "heavy_link2_2x"
NS_SWITCH_STEP = 50


def sample_pilot_grid(n_samples: int = 30, seed: int = 42) -> list[tuple[float, float]]:
    """Stratified random sample from the full candidate grid.

    Ensures coverage in the low-|η| region by stratifying into 3 bands:
    |η| < 0.001, 0.001 ≤ |η| < 0.01, |η| ≥ 0.01, plus η=0.
    """
    rng = np.random.default_rng(seed)

    # Build full grid excluding η=0
    all_pairs = [
        (eta, decay) for eta in ETA_CANDIDATES for decay in DECAY_CANDIDATES
        if abs(eta) > 1e-15
    ]

    # Stratify by |η| band
    bands = {
        "micro": [(e, d) for e, d in all_pairs if abs(e) < 0.001],
        "small": [(e, d) for e, d in all_pairs if 0.001 <= abs(e) < 0.01],
        "large": [(e, d) for e, d in all_pairs if abs(e) >= 0.01],
    }

    # Always include η=0 (1 sample), split rest across bands
    remaining = n_samples - 1
    per_band = remaining // 3
    extra = remaining - per_band * 3

    selected: list[tuple[float, float]] = [(0.0, 0.0)]

    for i, (band_name, pairs) in enumerate(bands.items()):
        n = per_band + (1 if i < extra else 0)
        n = min(n, len(pairs))
        idx = rng.choice(len(pairs), size=n, replace=False)
        selected.extend(pairs[j] for j in idx)

    return selected


def generate_dense_grid(
    eta_min: float,
    eta_max: float,
    n_eta: int,
    lambda_min: float,
    lambda_max: float,
    n_lambda: int,
) -> list[tuple[float, float]]:
    """Generate a dense uniform grid in the specified subregion."""
    etas = np.linspace(eta_min, eta_max, n_eta).tolist()
    decays = np.linspace(lambda_min, lambda_max, n_lambda).tolist()
    return [(eta, decay) for eta in etas for decay in decays]


# ── Evaluation ───────────────────────────────────────────────────────

def _run_rollouts(
    propagator: NeuralPropagator,
    num_rollouts: int,
    *,
    seeds: Optional[list[int]] = None,
    env: gym.Env,
) -> dict:
    """Run rollouts and return reward statistics."""
    base_weights = propagator.get_weights().copy()
    hook = getattr(propagator, "edge_hook", None)
    rewards: list[float] = []
    lengths: list[int] = []
    dw_means: list[float] = []

    for i in range(int(num_rollouts)):
        if _shutdown_event is not None and _shutdown_event.is_set():
            break

        propagator.reset()
        propagator.set_weights(base_weights.copy())

        seed = int(seeds[i]) if seeds is not None and i < len(seeds) else None
        if seed is not None:
            observation, info = env.reset(seed=seed)
        else:
            observation, info = env.reset()

        total_reward = 0.0
        steps = 0
        dw_sum = 0.0
        done = False

        while not done:
            obs = np.array(observation).flatten()
            propagator.propagate(obs)
            output_values = propagator.get_output()
            action = int(output_values.argmax().item())

            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            reward_f = float(reward)

            if hook is not None:
                hook.on_reward(reward_f)

            # Apply step learning
            edge_hook = getattr(propagator, "edge_hook", None)
            if edge_hook is not None and hasattr(edge_hook, "end_step"):
                deltas = edge_hook.end_step()
                propagator.apply_edge_weight_deltas(deltas)
                dw_sum += float(deltas.abs().mean().item()) if deltas.numel() > 0 else 0.0

            total_reward += reward_f
            steps += 1

        if hook is not None and hasattr(hook, "end_episode"):
            hook.end_episode(total_reward)

        rewards.append(total_reward)
        lengths.append(steps)
        dw_means.append(dw_sum / max(steps, 1))

    return {
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_abs_delta_per_step": float(np.mean(dw_means)) if dw_means else 0.0,
    }


def evaluate_network_extended(
    *,
    genome: Genome,
    metadata: dict,
    eta: float,
    decay: float,
    condition: str,
    rollouts: int = 20,
) -> dict:
    """Evaluate one (network, eta, decay, condition) combination.

    Args:
        condition: ``"static"`` or ``"ns"`` (non-stationary heavy_link2_2x).
    """
    network_id = metadata.get("network_id")
    stratum = metadata.get("stratum")

    grid = Grid(genome)
    grid.run_simulation(verbose=False)
    G = grid.get_graph()
    num_nodes = int(G.number_of_nodes())
    num_edges = int(G.number_of_edges())

    if num_nodes < MIN_NEURONS:
        return {"error": "insufficient_neurons", "num_neurons": num_nodes}

    try:
        graph_diameter = int(nx.diameter(G))
    except nx.NetworkXError:
        max_length = 0
        for source in G.nodes():
            lengths = nx.shortest_path_length(G, source)
            max_length = max(max_length, max(lengths.values()))
        graph_diameter = int(max_length)

    # Create env
    base_env = gym.make(ENV_NAME, render_mode=None)
    if condition == "ns":
        target_params = VARIANTS[NS_VARIANT]
        env = AcrobotMidEpisodeSwitchWrapper(
            base_env, switch_step=NS_SWITCH_STEP, target_params=target_params,
        )
    else:
        env = base_env

    seeds = VERIFICATION_SEEDS[:int(rollouts)]

    results = {
        "network_id": int(network_id) if network_id is not None else None,
        "stratum": stratum,
        "eta": float(eta),
        "decay": float(decay),
        "condition": condition,
        "num_neurons": num_nodes,
        "num_connections": num_edges,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        # Baseline (no plasticity)
        baseline_prop = create_propagator(
            grid, edge_hook=None, graph_diameter=graph_diameter,
        )
        baseline = _run_rollouts(baseline_prop, int(rollouts), seeds=seeds, env=env)
        results["baseline"] = baseline

        if _shutdown_event is not None and _shutdown_event.is_set():
            return results

        # Plastic
        if abs(eta) < 1e-15:
            plastic = baseline.copy()
        else:
            hook = HebbianEdgeHook(learning_rate=float(eta), weight_decay=float(decay))
            plastic_prop = create_propagator(
                grid, edge_hook=hook, graph_diameter=graph_diameter,
            )
            plastic = _run_rollouts(plastic_prop, int(rollouts), seeds=seeds, env=env)
        results["plastic"] = plastic

        results["delta_reward"] = plastic["avg_reward"] - baseline["avg_reward"]
        results["baseline_reward"] = baseline["avg_reward"]
        results["plastic_reward"] = plastic["avg_reward"]
        results["mean_abs_delta_per_step"] = plastic.get("mean_abs_delta_per_step", 0.0)
    finally:
        env.close()

    return results


def flatten_result_to_row(result: dict) -> dict:
    """Flatten to a flat dict for parquet."""
    if "error" in result:
        return {"network_id": result.get("network_id"), "error": result["error"]}

    return {
        "network_id": result.get("network_id"),
        "stratum": result.get("stratum"),
        "eta": result.get("eta"),
        "decay": result.get("decay"),
        "condition": result.get("condition"),
        "num_neurons": result.get("num_neurons"),
        "num_connections": result.get("num_connections"),
        "baseline_reward": result.get("baseline_reward"),
        "plastic_reward": result.get("plastic_reward"),
        "delta_reward": result.get("delta_reward"),
        "mean_abs_delta_per_step": result.get("mean_abs_delta_per_step", 0.0),
    }
