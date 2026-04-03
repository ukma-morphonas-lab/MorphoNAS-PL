"""
B2 Random RNN Control Experiment Module

Generates random RNNs matching the topology statistics (neuron count, edge count)
of competent MorphoNAS networks but with random weights (not distance-dependent
developmental weights). Used as a control to test whether topology-dependence of
plasticity is specific to morphogenetically grown networks.

Key functions:
    generate_random_rnn() — create a random directed graph with specified (N, E)
    evaluate_random_rnn() — evaluate on CartPole with optional plasticity
    load_morphonas_pool_stats() — extract (N, E) from B0.5 competent pool
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import numpy as np

from MorphoNAS.neural_propagation import NeuralPropagator
from MorphoNAS_PL.plasticity_hooks import HebbianEdgeHook
from MorphoNAS_PL.experimentB0_5_natural import (
    VERIFICATION_SEEDS,
    Stratum,
    get_stratum,
    run_rollouts,
)

logger = logging.getLogger(__name__)

# Minimum neurons for CartPole (4 input + 2 output)
MIN_NEURONS = 6


def generate_random_rnn(
    num_nodes: int,
    num_edges: int,
    rng: np.random.Generator,
    weight_range: tuple[float, float] = (0.01, 1.0),
    max_retries: int = 100,
) -> Optional[nx.DiGraph]:
    """Generate a random directed graph matching specified topology stats.

    Args:
        num_nodes: Number of neurons (nodes)
        num_edges: Number of synapses (directed edges)
        rng: Random number generator
        weight_range: (min, max) for uniform weight initialization.
            Default (0.01, 1.0) matches MorphoNAS's w = max(0.01, c/(1+d)) range.
        max_retries: Max attempts to generate a weakly connected graph

    Returns:
        nx.DiGraph with 'weight' on each edge, or None if generation failed.
    """
    if num_nodes < MIN_NEURONS:
        logger.warning(f"num_nodes={num_nodes} < {MIN_NEURONS}, skipping")
        return None

    max_possible_edges = num_nodes * (num_nodes - 1)  # no self-loops
    if num_edges > max_possible_edges:
        logger.warning(
            f"num_edges={num_edges} > max possible {max_possible_edges} "
            f"for {num_nodes} nodes, clamping"
        )
        num_edges = max_possible_edges

    for attempt in range(max_retries):
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))

        # Sample random edges without replacement
        all_edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    all_edges.append((i, j))

        chosen_indices = rng.choice(len(all_edges), size=num_edges, replace=False)
        for idx in chosen_indices:
            src, dst = all_edges[idx]
            w = float(rng.uniform(*weight_range))
            G.add_edge(src, dst, weight=w)

        # Check weak connectivity
        if nx.is_weakly_connected(G):
            return G

    logger.warning(
        f"Failed to generate connected graph after {max_retries} retries "
        f"(N={num_nodes}, E={num_edges})"
    )
    return None


def evaluate_random_rnn(
    G: nx.DiGraph,
    eta: float = 0.0,
    decay: float = 0.0,
    num_rollouts: int = 20,
    seeds: Optional[list[int]] = None,
) -> dict:
    """Evaluate a random RNN on CartPole with optional plasticity.

    Args:
        G: Directed graph (from generate_random_rnn)
        eta: Learning rate (0 = no plasticity)
        decay: Weight decay
        num_rollouts: Number of evaluation episodes
        seeds: Episode seeds

    Returns:
        dict with keys: avg_reward, fitness, rewards, etc. (from run_rollouts)
    """
    if seeds is None:
        seeds = VERIFICATION_SEEDS[:num_rollouts]

    num_nodes = G.number_of_nodes()
    if num_nodes < MIN_NEURONS:
        return {"avg_reward": 0.0, "fitness": 0.0, "error": "too_small"}

    # Graph diameter
    try:
        graph_diameter = int(nx.diameter(G))
    except nx.NetworkXError:
        max_length = 0
        for source in G.nodes():
            lengths = nx.shortest_path_length(G, source)
            max_length = max(max_length, max(lengths.values()))
        graph_diameter = int(max_length) if max_length > 0 else 1

    # Plasticity hook
    edge_hook = None
    if abs(eta) > 1e-12:
        edge_hook = HebbianEdgeHook(learning_rate=float(eta), weight_decay=float(decay))

    try:
        propagator = NeuralPropagator(
            G=G,
            input_dim=4,
            output_dim=2,
            activation_function=NeuralPropagator.tanh_activation,
            extra_thinking_time=2,
            additive_update=False,
            graph_diameter=graph_diameter,
            edge_hook=edge_hook,
        )
    except Exception as e:
        logger.debug(f"Failed to create propagator: {e}")
        return {"avg_reward": 0.0, "fitness": 0.0, "error": str(e)}

    try:
        result = run_rollouts(
            propagator,
            num_rollouts,
            seeds=seeds,
            reset_plastic_each_episode=True,
            record_plasticity=abs(eta) > 1e-12,
        )
        return result
    except Exception as e:
        logger.debug(f"Rollout failed: {e}")
        return {"avg_reward": 0.0, "fitness": 0.0, "error": str(e)}


def graph_to_dict(G: nx.DiGraph) -> dict:
    """Serialize a DiGraph to a dict for JSON storage."""
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({"src": int(u), "dst": int(v), "weight": float(data.get("weight", 1.0))})
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "edges": edges,
    }


def graph_from_dict(d: dict) -> nx.DiGraph:
    """Deserialize a DiGraph from a dict."""
    G = nx.DiGraph()
    G.add_nodes_from(range(d["num_nodes"]))
    for e in d["edges"]:
        G.add_edge(e["src"], e["dst"], weight=e["weight"])
    return G


def load_morphonas_pool_stats(
    pool_dir: str,
    min_stratum: str = "low_mid",
) -> list[dict]:
    """Load topology stats from B0.5 competent network pool.

    Args:
        pool_dir: Path to B0.5 pool directory containing network JSON files
        min_stratum: Minimum stratum to include (default: low_mid, excludes Weak)

    Returns:
        List of dicts with: network_id, num_neurons, num_connections, baseline_reward, stratum
    """
    import json
    from pathlib import Path

    pool_path = Path(pool_dir)
    stratum_order = [s.value for s in Stratum]
    min_idx = stratum_order.index(min_stratum)

    stats = []

    # Try networks/ subdirectory first, then pool files
    networks_dir = pool_path / "networks"
    if networks_dir.exists():
        for f in sorted(networks_dir.glob("*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                stratum = data.get("stratum", "weak")
                if stratum_order.index(stratum) < min_idx:
                    continue
                ns = data.get("network_stats", {})
                stats.append({
                    "network_id": data.get("network_id"),
                    "num_neurons": ns.get("num_neurons", ns.get("neurons", 0)),
                    "num_connections": ns.get("num_connections", ns.get("connections", 0)),
                    "baseline_reward": data.get("baseline_reward", 0),
                    "stratum": stratum,
                })
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
                continue
    else:
        # Try parquet
        try:
            import pandas as pd

            parquet_path = pool_path / "pool_natural.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                for _, row in df.iterrows():
                    stratum = row.get("stratum", "weak")
                    if stratum_order.index(stratum) < min_idx:
                        continue
                    stats.append({
                        "network_id": row.get("network_id"),
                        "num_neurons": int(row.get("n_neurons", row.get("num_neurons", 0))),
                        "num_connections": int(row.get("n_connections", row.get("num_connections", 0))),
                        "baseline_reward": float(row.get("baseline_reward", 0)),
                        "stratum": stratum,
                    })
        except ImportError:
            logger.error("pandas required for parquet loading")

    logger.info(f"Loaded {len(stats)} competent MorphoNAS network stats from {pool_dir}")
    return stats
