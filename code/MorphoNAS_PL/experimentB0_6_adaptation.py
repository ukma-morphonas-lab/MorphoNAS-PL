"""
B0.6 Adaptation Control Experiment Module

Tests whether plasticity benefit is genuine within-lifetime adaptation or
merely robustification by disabling plasticity during Phase 1 (pre-switch)
and enabling it only at the switch point (Phase 2).

The OFF→ON mechanism: set HebbianEdgeHook.learning_rate = 0 during Phase 1,
restore to actual η at the switch point. No modifications to HebbianEdgeHook
are needed — just attribute assignment on the existing object.
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
from MorphoNAS_PL.env_wrappers import MidEpisodeSwitchWrapper
from MorphoNAS_PL.experimentB0_5_natural import (
    VERIFICATION_SEEDS,
    create_propagator,
    load_network_from_file,
)
from MorphoNAS_PL.experimentB0_6_nonstationary import (
    DECAY_VALUES_FOCUSED,
    DEFAULT_SWITCH_STEP,
    ETA_VALUES_FOCUSED,
    EpisodeResult,
    PhaseResult,
    VARIANTS,
    _apply_step_learning,
    run_rollouts_phased,
)
from MorphoNAS_PL.plasticity_hooks import HebbianEdgeHook

logger = logging.getLogger(__name__)

_shutdown_event = None


def set_shutdown_event(event: Any) -> None:
    global _shutdown_event
    _shutdown_event = event


def run_rollouts_phased_offon(
    propagator: NeuralPropagator,
    num_rollouts: int,
    *,
    actual_eta: float,
    actual_decay: float,
    seeds: Optional[list[int]] = None,
    reset_plastic_each_episode: bool = True,
    env: gym.Env,
    switch_step: int = DEFAULT_SWITCH_STEP,
) -> dict:
    """Run rollouts with plasticity OFF during Phase 1 and ON during Phase 2.

    The hook's learning_rate and weight_decay are set to zero before each
    episode and restored to actual values when the environment transitions
    to Phase 2.
    """
    base_weights = propagator.get_weights().copy()
    episodes: list[EpisodeResult] = []

    hook = getattr(propagator, "edge_hook", None)

    for i in range(int(num_rollouts)):
        if _shutdown_event is not None and _shutdown_event.is_set():
            break

        propagator.reset()
        if reset_plastic_each_episode:
            propagator.set_weights(base_weights.copy())

        seed = int(seeds[i]) if seeds is not None and i < len(seeds) else None
        if seed is not None:
            observation, info = env.reset(seed=seed)
        else:
            observation, info = env.reset()

        # Suppress plasticity for Phase 1
        if hook is not None:
            hook.learning_rate = 0.0
            hook.weight_decay = 0.0

        ep = EpisodeResult()
        done = False
        plasticity_restored = False

        # Per-phase accumulators
        p1_dw_sum = 0.0
        p1_dw_sq_sum = 0.0
        p1_dw_count = 0
        p2_dw_sum = 0.0
        p2_dw_sq_sum = 0.0
        p2_dw_count = 0

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

            phase = info.get("phase", 1)

            # Restore plasticity at phase transition
            if phase == 2 and not plasticity_restored and hook is not None:
                hook.learning_rate = actual_eta
                hook.weight_decay = actual_decay
                plasticity_restored = True

            step_dw = _apply_step_learning(propagator)

            if phase == 1:
                ep.phase1.reward += reward_f
                ep.phase1.steps += 1
                p1_dw_sum += step_dw
                p1_dw_sq_sum += step_dw * step_dw
                p1_dw_count += 1
            else:
                ep.phase2.reward += reward_f
                ep.phase2.steps += 1
                p2_dw_sum += step_dw
                p2_dw_sq_sum += step_dw * step_dw
                p2_dw_count += 1

            ep.total_reward += reward_f
            ep.total_steps += 1

        # Finalize per-phase mean and std |dw|
        ep.phase1.mean_abs_dw = p1_dw_sum / max(p1_dw_count, 1)
        ep.phase2.mean_abs_dw = p2_dw_sum / max(p2_dw_count, 1)
        if p1_dw_count > 1:
            p1_var = p1_dw_sq_sum / p1_dw_count - ep.phase1.mean_abs_dw ** 2
            ep.phase1.std_abs_dw = max(0.0, p1_var) ** 0.5
        if p2_dw_count > 1:
            p2_var = p2_dw_sq_sum / p2_dw_count - ep.phase2.mean_abs_dw ** 2
            ep.phase2.std_abs_dw = max(0.0, p2_var) ** 0.5

        if hook is not None and hasattr(hook, "end_episode"):
            hook.end_episode(ep.total_reward)

        episodes.append(ep)

    # Aggregate results (same as run_rollouts_phased)
    total_rewards = [e.total_reward for e in episodes]
    p1_rewards = [e.phase1.reward for e in episodes]
    p2_rewards = [e.phase2.reward for e in episodes]
    p1_steps = [e.phase1.steps for e in episodes]
    p2_steps = [e.phase2.steps for e in episodes]
    p1_dw = [e.phase1.mean_abs_dw for e in episodes]
    p2_dw = [e.phase2.mean_abs_dw for e in episodes]
    p1_dw_stds = [e.phase1.std_abs_dw for e in episodes]
    p2_dw_stds = [e.phase2.std_abs_dw for e in episodes]

    p1_mean = float(np.mean(p1_dw)) if p1_dw else 0.0
    p2_mean = float(np.mean(p2_dw)) if p2_dw else 0.0

    return {
        "total": {
            "rewards": total_rewards,
            "avg_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
            "std_reward": float(np.std(total_rewards)) if total_rewards else 0.0,
        },
        "phase1": {
            "rewards": p1_rewards,
            "avg_reward": float(np.mean(p1_rewards)) if p1_rewards else 0.0,
            "std_reward": float(np.std(p1_rewards)) if p1_rewards else 0.0,
            "avg_steps": float(np.mean(p1_steps)) if p1_steps else 0.0,
            "mean_abs_dw": p1_mean,
            "std_abs_dw": float(np.mean(p1_dw_stds)) if p1_dw_stds else 0.0,
        },
        "phase2": {
            "rewards": p2_rewards,
            "avg_reward": float(np.mean(p2_rewards)) if p2_rewards else 0.0,
            "std_reward": float(np.std(p2_rewards)) if p2_rewards else 0.0,
            "avg_steps": float(np.mean(p2_steps)) if p2_steps else 0.0,
            "mean_abs_dw": p2_mean,
            "std_abs_dw": float(np.mean(p2_dw_stds)) if p2_dw_stds else 0.0,
        },
        "p2_p1_dw_ratio": p2_mean / p1_mean if p1_mean > 1e-15 else 0.0,
    }


def evaluate_network_adaptation(
    *,
    genome: Genome,
    metadata: dict,
    eta: float,
    decay: float,
    variant: str,
    switch_step: int = DEFAULT_SWITCH_STEP,
    rollouts: int = 20,
    plasticity_mode: str = "off_on",
) -> dict:
    """Evaluate one (network, eta, decay, variant, plasticity_mode) combination.

    Args:
        plasticity_mode: ``"off_on"`` disables plasticity during Phase 1 and
            enables it at the switch point. ``"always_on"`` uses standard
            always-active plasticity (same as B0.6).

    Returns baseline (no plasticity) and plastic results with phase-level
    breakdown, plus the ``plasticity_mode`` tag.
    """
    network_id = metadata.get("network_id")
    stratum = metadata.get("stratum")

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from: {list(VARIANTS)}")

    target_params = VARIANTS[variant]

    grid = Grid(genome)
    grid.run_simulation(verbose=False)
    G = grid.get_graph()
    num_nodes = int(G.number_of_nodes())
    num_edges = int(G.number_of_edges())

    if num_nodes < 6:
        return {"error": "insufficient_neurons", "num_neurons": num_nodes}

    try:
        graph_diameter = int(nx.diameter(G))
    except nx.NetworkXError:
        max_length = 0
        for source in G.nodes():
            lengths = nx.shortest_path_length(G, source)
            max_length = max(max_length, max(lengths.values()))
        graph_diameter = int(max_length)

    base_env = gym.make("CartPole-v1", render_mode=None)
    env = MidEpisodeSwitchWrapper(
        base_env, switch_step=switch_step, target_params=target_params
    )

    seeds = VERIFICATION_SEEDS[: int(rollouts)]

    results = {
        "network_id": int(network_id) if network_id is not None else None,
        "stratum": stratum,
        "eta": float(eta),
        "decay": float(decay),
        "variant": variant,
        "switch_step": switch_step,
        "plasticity_mode": plasticity_mode,
        "target_params": target_params,
        "rollouts": int(rollouts),
        "network_stats": {"num_neurons": num_nodes, "num_connections": num_edges},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        # Baseline: no plasticity, non-stationary env
        baseline_prop = create_propagator(
            grid, edge_hook=None, graph_diameter=graph_diameter
        )
        baseline_results = run_rollouts_phased(
            baseline_prop,
            int(rollouts),
            seeds=seeds,
            reset_plastic_each_episode=True,
            env=env,
            switch_step=switch_step,
        )
        results["baseline"] = baseline_results

        if _shutdown_event is not None and _shutdown_event.is_set():
            return results

        # Plastic evaluation
        if abs(eta) < 1e-12:
            plastic_results = baseline_results.copy()
        else:
            hook = HebbianEdgeHook(
                learning_rate=float(eta), weight_decay=float(decay)
            )
            plastic_prop = create_propagator(
                grid, edge_hook=hook, graph_diameter=graph_diameter
            )

            if plasticity_mode == "off_on":
                plastic_results = run_rollouts_phased_offon(
                    plastic_prop,
                    int(rollouts),
                    actual_eta=float(eta),
                    actual_decay=float(decay),
                    seeds=seeds,
                    reset_plastic_each_episode=True,
                    env=env,
                    switch_step=switch_step,
                )
            else:
                # always_on: standard behavior
                plastic_results = run_rollouts_phased(
                    plastic_prop,
                    int(rollouts),
                    seeds=seeds,
                    reset_plastic_each_episode=True,
                    env=env,
                    switch_step=switch_step,
                )
        results["plastic"] = plastic_results

        # Summary
        b_total = float(baseline_results["total"]["avg_reward"])
        p_total = float(plastic_results["total"]["avg_reward"])
        b_p1 = float(baseline_results["phase1"]["avg_reward"])
        p_p1 = float(plastic_results["phase1"]["avg_reward"])
        b_p2 = float(baseline_results["phase2"]["avg_reward"])
        p_p2 = float(plastic_results["phase2"]["avg_reward"])

        results["summary"] = {
            "baseline_total_reward": b_total,
            "plastic_total_reward": p_total,
            "delta_reward_total": p_total - b_total,
            "baseline_phase1_reward": b_p1,
            "plastic_phase1_reward": p_p1,
            "delta_reward_phase1": p_p1 - b_p1,
            "baseline_phase2_reward": b_p2,
            "plastic_phase2_reward": p_p2,
            "delta_reward_phase2": p_p2 - b_p2,
            "baseline_survived_switch": b_p2 > 0,
            "plastic_survived_switch": p_p2 > 0,
            "phase1_mean_abs_dw": float(
                plastic_results["phase1"].get("mean_abs_dw", 0.0)
            ),
            "phase2_mean_abs_dw": float(
                plastic_results["phase2"].get("mean_abs_dw", 0.0)
            ),
            "p1_dw_std": float(
                plastic_results["phase1"].get("std_abs_dw", 0.0)
            ),
            "p2_dw_std": float(
                plastic_results["phase2"].get("std_abs_dw", 0.0)
            ),
            "p2_p1_dw_ratio": float(
                plastic_results.get("p2_p1_dw_ratio", 0.0)
            ),
        }
    finally:
        env.close()

    return results


def flatten_result_to_row(result: dict) -> dict:
    """Flatten a single evaluation result into a flat dict for DataFrame/parquet.

    Returns the standard B0.6 schema (28 columns) plus ``plasticity_mode``.
    """
    if "error" in result:
        return {"network_id": result.get("network_id"), "error": result["error"]}

    summary = result.get("summary", {})
    baseline = result.get("baseline", {})
    plastic = result.get("plastic", {})
    stats = result.get("network_stats", {})

    return {
        "network_id": result.get("network_id"),
        "stratum": result.get("stratum"),
        "eta": result.get("eta"),
        "decay": result.get("decay"),
        "variant": result.get("variant"),
        "switch_step": result.get("switch_step"),
        "plasticity_mode": result.get("plasticity_mode"),
        "num_neurons": stats.get("num_neurons"),
        "num_connections": stats.get("num_connections"),
        # Total
        "baseline_total_reward": summary.get("baseline_total_reward"),
        "plastic_total_reward": summary.get("plastic_total_reward"),
        "delta_reward_total": summary.get("delta_reward_total"),
        # Phase 1
        "baseline_phase1_reward": summary.get("baseline_phase1_reward"),
        "plastic_phase1_reward": summary.get("plastic_phase1_reward"),
        "delta_reward_phase1": summary.get("delta_reward_phase1"),
        # Phase 2
        "baseline_phase2_reward": summary.get("baseline_phase2_reward"),
        "plastic_phase2_reward": summary.get("plastic_phase2_reward"),
        "delta_reward_phase2": summary.get("delta_reward_phase2"),
        # Diagnostics
        "baseline_survived_switch": summary.get("baseline_survived_switch"),
        "plastic_survived_switch": summary.get("plastic_survived_switch"),
        "phase1_mean_abs_dw": summary.get("phase1_mean_abs_dw", 0.0),
        "phase2_mean_abs_dw": summary.get("phase2_mean_abs_dw", 0.0),
        "p1_dw_std": summary.get("p1_dw_std", 0.0),
        "p2_dw_std": summary.get("p2_dw_std", 0.0),
        "p2_p1_dw_ratio": summary.get("p2_p1_dw_ratio", 0.0),
        # Step counts
        "baseline_phase1_steps": baseline.get("phase1", {}).get("avg_steps", 0.0),
        "baseline_phase2_steps": baseline.get("phase2", {}).get("avg_steps", 0.0),
        "plastic_phase1_steps": plastic.get("phase1", {}).get("avg_steps", 0.0),
        "plastic_phase2_steps": plastic.get("phase2", {}).get("avg_steps", 0.0),
    }
