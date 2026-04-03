"""
Acrobot Temporal Trace Experiment Module

Records per-timestep traces (|Δw|, observations, actions, phase) for
selected Acrobot networks under non-stationary conditions. Produces NPZ
files matching the B0.6 CartPole temporal profile format.
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
from MorphoNAS_PL.experiment_acrobot_nonstationary import (
    DEFAULT_SWITCH_STEP,
    VARIANTS,
)
from MorphoNAS_PL.plasticity_hooks import HebbianEdgeHook

logger = logging.getLogger(__name__)

_shutdown_event = None


def set_shutdown_event(event: Any) -> None:
    global _shutdown_event
    _shutdown_event = event


def _apply_step_learning(propagator: NeuralPropagator) -> float:
    """Apply per-step Hebbian learning and return |delta_w|."""
    hook = getattr(propagator, "edge_hook", None)
    if hook is None or not hasattr(hook, "end_step"):
        return 0.0
    deltas = hook.end_step()
    propagator.apply_edge_weight_deltas(deltas)
    return float(deltas.abs().mean().item()) if deltas.numel() > 0 else 0.0


def run_rollouts_temporal(
    propagator: NeuralPropagator,
    num_rollouts: int,
    *,
    seeds: Optional[list[int]] = None,
    reset_plastic_each_episode: bool = True,
    env: gym.Env,
    switch_step: int = DEFAULT_SWITCH_STEP,
) -> list[dict]:
    """Run rollouts recording per-timestep traces.

    Returns a list of dicts, one per episode, each containing:
        - ``dw``: float32 array of shape (T,) — mean |Δw| per step
        - ``obs``: float32 array of shape (T, 6) — observations
        - ``action``: int8 array of shape (T,) — actions taken
        - ``phase``: int8 array of shape (T,) — phase labels (1 or 2)
        - ``total_reward``: float — episode total reward
        - ``total_steps``: int — episode length
    """
    base_weights = propagator.get_weights().copy()
    hook = getattr(propagator, "edge_hook", None)
    episodes: list[dict] = []

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

        dw_trace: list[float] = []
        obs_trace: list[np.ndarray] = []
        action_trace: list[int] = []
        phase_trace: list[int] = []
        total_reward = 0.0
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

            step_dw = _apply_step_learning(propagator)
            phase = info.get("phase", 1)

            dw_trace.append(step_dw)
            obs_trace.append(obs.copy())
            action_trace.append(action)
            phase_trace.append(phase)
            total_reward += reward_f

        if hook is not None and hasattr(hook, "end_episode"):
            hook.end_episode(total_reward)

        episodes.append({
            "dw": np.array(dw_trace, dtype=np.float32),
            "obs": np.array(obs_trace, dtype=np.float32),
            "action": np.array(action_trace, dtype=np.int8),
            "phase": np.array(phase_trace, dtype=np.int8),
            "total_reward": total_reward,
            "total_steps": len(dw_trace),
        })

    return episodes


def evaluate_network_temporal(
    *,
    genome: Genome,
    metadata: dict,
    eta: float,
    decay: float,
    variant: str,
    switch_step: int = DEFAULT_SWITCH_STEP,
    rollouts: int = 20,
) -> dict:
    """Evaluate one network and return per-episode temporal traces.

    Returns a dict with:
        - ``traces``: list of per-episode dicts from ``run_rollouts_temporal``
        - ``metadata``: network metadata + summary stats
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

    base_env = gym.make(ENV_NAME, render_mode=None)
    env = AcrobotMidEpisodeSwitchWrapper(
        base_env, switch_step=switch_step, target_params=target_params
    )

    seeds = VERIFICATION_SEEDS[: int(rollouts)]

    try:
        hook = HebbianEdgeHook(learning_rate=float(eta), weight_decay=float(decay))
        prop = create_propagator(grid, edge_hook=hook, graph_diameter=graph_diameter)
        traces = run_rollouts_temporal(
            prop,
            int(rollouts),
            seeds=seeds,
            reset_plastic_each_episode=True,
            env=env,
            switch_step=switch_step,
        )
    finally:
        env.close()

    # Compute summary stats
    rewards = [t["total_reward"] for t in traces]
    p1_dws = []
    p2_dws = []
    for t in traces:
        mask1 = t["phase"] == 1
        mask2 = t["phase"] == 2
        if mask1.any():
            p1_dws.append(float(t["dw"][mask1].mean()))
        if mask2.any():
            p2_dws.append(float(t["dw"][mask2].mean()))

    p1_mean_dw = float(np.mean(p1_dws)) if p1_dws else 0.0
    p2_mean_dw = float(np.mean(p2_dws)) if p2_dws else 0.0

    return {
        "traces": traces,
        "metadata": {
            "network_id": int(network_id) if network_id is not None else None,
            "stratum": stratum,
            "variant": variant,
            "eta": float(eta),
            "decay": float(decay),
            "switch_step": switch_step,
            "n_episodes": len(traces),
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "p1_mean_dw": p1_mean_dw,
            "p2_mean_dw": p2_mean_dw,
            "p2_p1_ratio": p2_mean_dw / p1_mean_dw if p1_mean_dw > 1e-15 else 0.0,
            "num_neurons": num_nodes,
            "num_connections": num_edges,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
