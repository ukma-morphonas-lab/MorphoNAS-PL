"""Shared utilities for B0.5 analysis scripts.

Provides data loading, constants, formatting helpers, and plotting utilities
used across comprehensive and supplementary analysis scripts.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MorphoNAS_PL.experimentB0_5_natural import (
    DECAY_VALUES,
    ETA_VALUES,
    STRATUM_BOUNDS,
    Stratum,
    get_stratum_label,
)

logger = logging.getLogger(__name__)

# ── Derived constants ─────────────────────────────────────────────────

STRATA_ORDER: List[str] = [s.value for s in Stratum]

STRATUM_LABELS: dict[str, str] = {s.value: get_stratum_label(s) for s in Stratum}

STRATUM_COLORS: dict[str, str] = {
    "weak": "#d62728",
    "low_mid": "#ff7f0e",
    "high_mid": "#bcbd22",
    "near_perfect": "#2ca02c",
    "perfect": "#1f77b4",
}

STRATUM_BOUNDS_LIST: List[Tuple[float, float]] = [
    (0, 200),
    (200, 350),
    (350, 450),
    (450, 475),
    (475, float("inf")),
]

# Outcome classification thresholds
IMPROVE_THRESHOLD = 0.001   # delta_fitness > +0.001 -> improved
HARM_THRESHOLD = -0.001     # delta_fitness < -0.001 -> harmed


# ── Publication matplotlib style ──────────────────────────────────────

PUBLICATION_RCPARAMS = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
}


def apply_publication_style() -> None:
    """Apply publication-quality matplotlib rcParams."""
    matplotlib.use("Agg")
    plt.rcParams.update(PUBLICATION_RCPARAMS)


# ── Formatting helpers ────────────────────────────────────────────────

def fmt_eta(e: float) -> str:
    """Format an eta value for axis labels."""
    if e == 0:
        return "0"
    return f"{e:+.0e}".replace("+", "+").replace("-0", "-")


def fmt_decay(d: float) -> str:
    """Format a decay value for axis labels."""
    if d == 0:
        return "0"
    return f"{d:.0e}"


# ── Data loading ──────────────────────────────────────────────────────

def load_sweep(sweep_dir: str) -> pd.DataFrame:
    """Load a sweep directory in either sharded or merged parquet layout."""
    sweep_path = Path(sweep_dir)
    shard_files = sorted(sweep_path.glob("eta_*.parquet"))
    if shard_files:
        files = shard_files
        layout = "sharded"
    else:
        files = sorted(sweep_path.glob("*.parquet"))
        layout = "merged"
    if not files:
        raise FileNotFoundError(f"No parquet files in {sweep_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    logger.info(
        "Loaded %s rows from %s %s file(s)",
        f"{len(df):,}",
        len(files),
        layout,
    )
    return df


def load_pool_rewards(pool_dir: str) -> np.ndarray:
    """Load baseline rewards from pool network JSONs."""
    networks_dir = os.path.join(pool_dir, "networks")
    rewards = []
    if os.path.isdir(networks_dir):
        for fname in os.listdir(networks_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(networks_dir, fname)) as f:
                        d = json.load(f)
                    r = d.get("baseline_reward")
                    if r is not None:
                        rewards.append(r)
                except Exception:
                    continue
    return np.array(rewards)


def _load_single_network(fpath: str) -> dict | None:
    """Load a single network JSON and extract topology features."""
    try:
        with open(fpath) as f:
            d = json.load(f)
        ns = d.get("network_stats", {})
        genome = d.get("genome", {})
        neurons = ns.get("neurons", 0)
        connections = ns.get("connections", 0)
        max_possible = neurons * (neurons - 1) if neurons > 1 else 1
        density = connections / max_possible if max_possible > 0 else 0.0

        return {
            "network_id": d["network_id"],
            "stratum": d["stratum"],
            "baseline_reward": d["baseline_reward"],
            "baseline_fitness": d.get("baseline_fitness", np.nan),
            "neurons": neurons,
            "connections": connections,
            "density": density,
            "max_axon_length": genome.get("max_axon_length", np.nan),
            "self_connect_fraction": genome.get(
                "self_connect_isolated_neurons_fraction", np.nan
            ),
            "diffusion_rate": genome.get("diffusion_rate", np.nan),
            "division_threshold": genome.get("division_threshold", np.nan),
            "axon_growth_threshold": genome.get("axon_growth_threshold", np.nan),
            "axon_connect_threshold": genome.get("axon_connect_threshold", np.nan),
            "weight_adjustment_rate": genome.get("weight_adjustment_rate", np.nan),
            "num_morphogens": genome.get("num_morphogens", np.nan),
        }
    except Exception as e:
        logger.debug(f"Failed to load {fpath}: {e}")
        return None


def load_pool_topology(pool_dir: str) -> pd.DataFrame:
    """Load network topology features from all pool JSONs (parallelized)."""
    networks_dir = os.path.join(pool_dir, "networks")
    files = sorted(
        os.path.join(networks_dir, f)
        for f in os.listdir(networks_dir)
        if f.endswith(".json")
    )
    logger.info(f"Loading topology from {len(files):,} network files...")

    workers = max(1, (os.cpu_count() or 4) - 2)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load_single_network, f): f for f in files}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results.append(r)

    df = pd.DataFrame(results)
    logger.info(f"Loaded topology for {len(df):,} networks")
    return df


# ── Plotting helpers ──────────────────────────────────────────────────

def save_figure(fig: plt.Figure, fig_dir: str, name: str) -> None:
    """Save a figure as both PNG and PDF, then close it."""
    fig.savefig(os.path.join(fig_dir, f"{name}.png"))
    fig.savefig(os.path.join(fig_dir, f"{name}.pdf"))
    plt.close(fig)
    logger.info(f"  Saved {name}.png/.pdf")


def get_stratum_idx(reward: float) -> int:
    """Return stratum index (0=weak, ..., 4=perfect) for a reward value."""
    for i, (lo, hi) in enumerate(STRATUM_BOUNDS_LIST):
        if lo <= reward < hi:
            return i
    return 4


def strata_present_in(df: pd.DataFrame, column: str = "stratum") -> List[str]:
    """Return STRATA_ORDER entries that appear in *df[column]*."""
    unique = set(df[column].unique())
    return [s for s in STRATA_ORDER if s in unique]
