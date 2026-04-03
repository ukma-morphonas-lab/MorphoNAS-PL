"""Comprehensive genome feature extraction for prediction and regression analyses.

Loads genome JSON files from a pool directory and extracts developmental
parameters, network statistics, and derived topology features into a
DataFrame suitable for machine learning and statistical analysis.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_genome_features(fpath: str) -> dict | None:
    """Extract comprehensive features from a single genome JSON file."""
    try:
        with open(fpath) as f:
            data = json.load(f)

        genome = data.get("genome", {})
        ns = data.get("network_stats", {})

        num_neurons = ns.get("neurons", 0) or ns.get("num_neurons", 0)
        num_connections = ns.get("connections", 0) or ns.get("num_connections", 0)

        max_possible = num_neurons * (num_neurons - 1) if num_neurons > 1 else 1
        connectivity_density = num_connections / max_possible if max_possible > 0 else 0.0
        mean_degree = num_connections / num_neurons if num_neurons > 0 else 0.0

        row = {
            "network_id": data["network_id"],
            "stratum": data.get("stratum", "unknown"),
            "baseline_reward": data.get("baseline_reward", np.nan),
            "baseline_fitness": data.get("baseline_fitness", np.nan),
            # Network stats
            "num_neurons": num_neurons,
            "num_connections": num_connections,
            # Derived topology
            "connectivity_density": connectivity_density,
            "mean_degree": mean_degree,
            # Developmental parameters
            "diffusion_rate": genome.get("diffusion_rate", np.nan),
            "division_threshold": genome.get("division_threshold", np.nan),
            "cell_differentiation_threshold": genome.get(
                "cell_differentiation_threshold", np.nan
            ),
            "max_axon_length": genome.get("max_axon_length", np.nan),
            "axon_growth_threshold": genome.get("axon_growth_threshold", np.nan),
            "axon_connect_threshold": genome.get("axon_connect_threshold", np.nan),
            "weight_adjustment_target": genome.get("weight_adjustment_target", np.nan),
            "weight_adjustment_rate": genome.get("weight_adjustment_rate", np.nan),
            "self_connect_fraction": genome.get(
                "self_connect_isolated_neurons_fraction", np.nan
            ),
            "num_morphogens": genome.get("num_morphogens", np.nan),
        }

        # Secretion rates (3 values each)
        for prefix, key in [
            ("psr", "progenitor_secretion_rates"),
            ("nsr", "neuron_secretion_rates"),
        ]:
            rates = genome.get(key, [])
            for i in range(3):
                row[f"{prefix}_{i}"] = float(rates[i]) if i < len(rates) else np.nan

        # Inhibition matrix (3x3)
        inhib = genome.get("inhibition_matrix", [])
        for i in range(3):
            for j in range(3):
                val = np.nan
                if i < len(inhib) and j < len(inhib[i]):
                    val = float(inhib[i][j])
                row[f"inhib_{i}{j}"] = val

        return row

    except Exception as e:
        logger.debug("Failed to load %s: %s", fpath, e)
        return None


def load_genome_features(pool_dir: str) -> pd.DataFrame:
    """Load comprehensive genome features from all network JSONs in a pool.

    Args:
        pool_dir: Path to pool directory containing a ``networks/`` subdirectory
            with individual genome JSON files.

    Returns:
        DataFrame with one row per network and ~30 feature columns.
    """
    networks_dir = os.path.join(pool_dir, "networks")
    files = sorted(
        os.path.join(networks_dir, f)
        for f in os.listdir(networks_dir)
        if f.endswith(".json")
    )
    logger.info("Loading genome features from %d network files...", len(files))

    workers = max(1, (os.cpu_count() or 4) - 2)
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_genome_features, f): f for f in files}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results.append(r)

    df = pd.DataFrame(results)
    logger.info("Loaded genome features for %d networks", len(df))
    return df


# Feature column names (excluding metadata columns)
FEATURE_COLUMNS = [
    "num_neurons",
    "num_connections",
    "connectivity_density",
    "mean_degree",
    "diffusion_rate",
    "division_threshold",
    "cell_differentiation_threshold",
    "max_axon_length",
    "axon_growth_threshold",
    "axon_connect_threshold",
    "weight_adjustment_target",
    "weight_adjustment_rate",
    "self_connect_fraction",
    "num_morphogens",
    "psr_0",
    "psr_1",
    "psr_2",
    "nsr_0",
    "nsr_1",
    "nsr_2",
    "inhib_00",
    "inhib_01",
    "inhib_02",
    "inhib_10",
    "inhib_11",
    "inhib_12",
    "inhib_20",
    "inhib_21",
    "inhib_22",
]

METADATA_COLUMNS = [
    "network_id",
    "stratum",
    "baseline_reward",
    "baseline_fitness",
]
