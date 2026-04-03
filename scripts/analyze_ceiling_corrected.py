#!/usr/bin/env python3
"""
Analysis 4: Ceiling-Corrected Effect Sizes

Mean Δreward is structurally misleading due to asymmetric reward ceilings.
This script computes normalized metrics that give fair cross-stratum and
cross-task comparisons.

Normalization:
  improved:  delta / (ceiling - baseline)   — fraction of headroom captured
  harmed:    delta / (baseline - floor)     — fraction of downside lost

Figures:
  F0_normalized_effect_sizes  — Per-stratum normalized oracle gains
  F1_raw_vs_normalized        — Side-by-side raw vs normalized (B0.5+)
  F2_cross_task_normalized    — CartPole vs Acrobot direct comparison
  F3_headroom_scatter         — Headroom vs raw oracle gain
  F4_normalized_heatmap       — η×λ heatmap of mean normalized_delta

Usage:
  python scripts/analyze_ceiling_corrected.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from MorphoNAS_PL.analysis_utils import (
    STRATA_ORDER,
    STRATUM_COLORS,
    STRATUM_LABELS,
    apply_publication_style,
    load_sweep,
    save_figure,
)

logger = logging.getLogger(__name__)
apply_publication_style()

# ── Paths ────────────────────────────────────────────────────────────
B05_SWEEP_DIR = "experiments/B0.5/sweep"
B05P_SWEEP_DIR = "experiments/B0.5+/sweep"
ACROBOT_SWEEP_FILE = "experiments/acrobot/sweep_static/acrobot_static_sweep.parquet"
DEFAULT_OUTPUT_DIR = "experiments/B0.5+/analysis/ceiling_corrected"

NON_WEAK_STRATA = ["low_mid", "high_mid", "near_perfect", "perfect"]

# Task-specific constants
CARTPOLE_CEILING = 500.0
CARTPOLE_FLOOR = 8.0

# Acrobot: rewards are negative; ceiling = least negative (best), floor = most negative (worst)
ACROBOT_CEILING = -75.0
ACROBOT_FLOOR = -500.0


# ── Core computation ────────────────────────────────────────────────

def compute_normalized_delta(
    delta: np.ndarray,
    baseline: np.ndarray,
    ceiling: float,
    floor: float,
) -> np.ndarray:
    """Compute ceiling-corrected normalized delta.

    For improved (delta > 0): delta / (ceiling - baseline)
    For harmed (delta < 0): delta / (baseline - floor)
    For unchanged (delta == 0): 0.0
    """
    eps = 1e-6
    result = np.zeros_like(delta, dtype=np.float64)

    # Clip baseline to avoid division by zero
    baseline_clipped = np.clip(baseline, floor + eps, ceiling - eps)

    improved = delta > 0
    harmed = delta < 0

    headroom = ceiling - baseline_clipped
    downside = baseline_clipped - floor

    result[improved] = delta[improved] / np.maximum(headroom[improved], eps)
    result[harmed] = delta[harmed] / np.maximum(downside[harmed], eps)

    return result


def add_normalized_columns(
    df: pd.DataFrame,
    ceiling: float,
    floor: float,
    baseline_col: str,
    delta_col: str,
) -> pd.DataFrame:
    """Add 'normalized_delta' column to a sweep DataFrame."""
    baseline = df[baseline_col].values.astype(np.float64)
    delta = df[delta_col].values.astype(np.float64)
    df = df.copy()
    df["normalized_delta"] = compute_normalized_delta(delta, baseline, ceiling, floor)
    df["headroom"] = ceiling - np.clip(baseline, floor, ceiling)
    return df


def compute_oracle_normalized(
    df: pd.DataFrame,
    ceiling: float,
    floor: float,
    baseline_col: str,
) -> pd.DataFrame:
    """Per-network oracle using normalized_delta."""
    # First compute raw oracle (best delta_reward)
    idx_raw = df.groupby("network_id")["delta_reward"].idxmax()
    oracle_raw = df.loc[idx_raw, ["network_id", "stratum", "delta_reward", baseline_col]].copy()
    oracle_raw = oracle_raw.rename(columns={"delta_reward": "raw_oracle_delta"})

    # Also compute oracle on normalized scale
    idx_norm = df.groupby("network_id")["normalized_delta"].idxmax()
    oracle_norm = df.loc[idx_norm, ["network_id", "normalized_delta"]].copy()
    oracle_norm = oracle_norm.rename(columns={"normalized_delta": "norm_oracle_delta"})

    # Also add normalized version of the raw oracle
    baseline = oracle_raw[baseline_col].values.astype(np.float64)
    raw_delta = oracle_raw["raw_oracle_delta"].values.astype(np.float64)
    oracle_raw["raw_oracle_normalized"] = compute_normalized_delta(
        raw_delta, baseline, ceiling, floor,
    )
    oracle_raw["improved"] = oracle_raw["raw_oracle_delta"] > 0

    merged = oracle_raw.merge(oracle_norm, on="network_id")
    return merged


# ── Figures ──────────────────────────────────────────────────────────

def figure_normalized_effect_sizes(
    datasets: dict[str, pd.DataFrame],
    fig_dir: str,
) -> dict:
    """F0: Per-stratum normalized oracle gains for each dataset."""
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Mean normalized oracle delta per stratum
    ax = axes[0]
    width = 0.8 / n_datasets
    dataset_colors = {"B0.5": "#2196F3", "B0.5+": "#4CAF50", "Acrobot": "#FF9800"}
    results = {}

    for d_idx, (name, oracle_df) in enumerate(datasets.items()):
        strata = [s for s in NON_WEAK_STRATA if s in oracle_df["stratum"].unique()]
        means = []
        sems = []
        xs = []
        for s_idx, s in enumerate(NON_WEAK_STRATA):
            sdf = oracle_df[oracle_df["stratum"] == s]
            if len(sdf) == 0:
                continue
            vals = sdf["raw_oracle_normalized"].values
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
            xs.append(s_idx + d_idx * width)

        ax.bar(xs, means, width=width, yerr=sems, capsize=3,
               label=name, color=dataset_colors.get(name, "#999"), alpha=0.8)
        results[name] = {
            s: {"mean_norm": m, "sem": se}
            for s, m, se in zip(NON_WEAK_STRATA, means, sems)
        }

    ax.set_xticks([i + width * (n_datasets - 1) / 2 for i in range(len(NON_WEAK_STRATA))])
    ax.set_xticklabels([STRATUM_LABELS.get(s, s) for s in NON_WEAK_STRATA],
                       rotation=20, ha="right")
    ax.set_ylabel("Normalized Oracle Δreward")
    ax.set_title("(a) Ceiling-Corrected Oracle Gain")
    ax.legend()
    ax.axhline(0, color="k", ls="--", lw=1, alpha=0.3)

    # (b) Improvement rate per stratum
    ax = axes[1]
    for d_idx, (name, oracle_df) in enumerate(datasets.items()):
        rates = []
        xs = []
        for s_idx, s in enumerate(NON_WEAK_STRATA):
            sdf = oracle_df[oracle_df["stratum"] == s]
            if len(sdf) == 0:
                continue
            rates.append(float(sdf["improved"].mean()))
            xs.append(s_idx + d_idx * width)
        ax.bar(xs, rates, width=width, label=name,
               color=dataset_colors.get(name, "#999"), alpha=0.8)

    ax.set_xticks([i + width * (n_datasets - 1) / 2 for i in range(len(NON_WEAK_STRATA))])
    ax.set_xticklabels([STRATUM_LABELS.get(s, s) for s in NON_WEAK_STRATA],
                       rotation=20, ha="right")
    ax.set_ylabel("Improvement Rate")
    ax.set_title("(b) % Networks Improved (Oracle)")
    ax.legend()

    fig.suptitle("Ceiling-Corrected Effect Sizes Across Datasets")
    fig.tight_layout()
    save_figure(fig, fig_dir, "F0_normalized_effect_sizes")

    return results


def figure_raw_vs_normalized(
    oracle_df: pd.DataFrame,
    fig_dir: str,
) -> None:
    """F1: Side-by-side raw vs normalized for B0.5+."""
    strata = [s for s in NON_WEAK_STRATA if s in oracle_df["stratum"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (col, label) in enumerate([
        ("raw_oracle_delta", "Raw Oracle Δreward"),
        ("raw_oracle_normalized", "Normalized Oracle Δreward"),
    ]):
        ax = axes[idx]
        means = []
        sems = []
        colors = []
        for s in strata:
            vals = oracle_df.loc[oracle_df["stratum"] == s, col].values
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
            colors.append(STRATUM_COLORS.get(s, "#999"))

        ax.bar(range(len(strata)), means, yerr=sems, capsize=4, color=colors)
        ax.set_xticks(range(len(strata)))
        ax.set_xticklabels([STRATUM_LABELS.get(s, s) for s in strata],
                           rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"({'a' if idx == 0 else 'b'}) {label}")
        ax.axhline(0, color="k", ls="--", lw=1, alpha=0.3)

    fig.suptitle("Raw vs Ceiling-Corrected Oracle Gains (B0.5+)")
    fig.tight_layout()
    save_figure(fig, fig_dir, "F1_raw_vs_normalized")


def figure_cross_task(
    cartpole_oracle: pd.DataFrame,
    acrobot_oracle: pd.DataFrame,
    fig_dir: str,
) -> None:
    """F2: CartPole vs Acrobot direct comparison on normalized scale."""
    fig, ax = plt.subplots(figsize=(8, 5))

    width = 0.35
    for d_idx, (name, oracle_df, color) in enumerate([
        ("CartPole (B0.5+)", cartpole_oracle, "#2196F3"),
        ("Acrobot", acrobot_oracle, "#FF9800"),
    ]):
        means = []
        sems = []
        xs = []
        for s_idx, s in enumerate(NON_WEAK_STRATA):
            sdf = oracle_df[oracle_df["stratum"] == s]
            if len(sdf) == 0:
                means.append(0)
                sems.append(0)
            else:
                vals = sdf["raw_oracle_normalized"].values
                means.append(float(np.mean(vals)))
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
            xs.append(s_idx + d_idx * width)

        ax.bar(xs, means, width=width, yerr=sems, capsize=3,
               label=name, color=color, alpha=0.8)

    ax.set_xticks([i + width / 2 for i in range(len(NON_WEAK_STRATA))])
    ax.set_xticklabels([STRATUM_LABELS.get(s, s) for s in NON_WEAK_STRATA],
                       rotation=20, ha="right")
    ax.set_ylabel("Normalized Oracle Δreward")
    ax.set_title("Cross-Task Comparison: Ceiling-Corrected Oracle Gains")
    ax.legend()
    ax.axhline(0, color="k", ls="--", lw=1, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, fig_dir, "F2_cross_task_normalized")


def figure_headroom_scatter(
    oracle_df: pd.DataFrame,
    ceiling: float,
    fig_dir: str,
) -> None:
    """F3: Headroom vs raw oracle gain, color by stratum."""
    fig, ax = plt.subplots(figsize=(8, 6))

    strata = [s for s in NON_WEAK_STRATA if s in oracle_df["stratum"].unique()]
    for s in strata:
        sdf = oracle_df[oracle_df["stratum"] == s]
        headroom = ceiling - sdf["baseline_avg_reward"].values
        delta = sdf["raw_oracle_delta"].values
        ax.scatter(headroom, delta, alpha=0.3, s=10,
                   color=STRATUM_COLORS.get(s, "#999"),
                   label=STRATUM_LABELS.get(s, s))

    ax.set_xlabel("Headroom (Ceiling - Baseline)")
    ax.set_ylabel("Raw Oracle Δreward")
    ax.set_title("Ceiling Compression: Headroom vs Plasticity Gain")
    ax.legend(markerscale=3, fontsize=9)
    ax.axhline(0, color="k", ls="--", lw=1, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, fig_dir, "F3_headroom_scatter")


def figure_normalized_heatmap(
    sweep_df: pd.DataFrame,
    fig_dir: str,
) -> None:
    """F4: η×λ heatmap of mean normalized_delta per stratum."""
    strata = [s for s in NON_WEAK_STRATA if s in sweep_df["stratum"].unique()]
    n = len(strata)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    etas = sorted(sweep_df["eta"].unique())
    decays = sorted(sweep_df["decay"].unique())

    for idx, s in enumerate(strata):
        ax = axes[idx]
        sdf = sweep_df[sweep_df["stratum"] == s]

        # Pivot to heatmap
        pivot = sdf.pivot_table(
            index="decay", columns="eta", values="normalized_delta", aggfunc="mean",
        )

        im = ax.imshow(
            pivot.values, cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto",
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{e:.2f}" for e in pivot.columns], rotation=90, fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{d:.3f}" for d in pivot.index], fontsize=8)
        ax.set_xlabel("η")
        if idx == 0:
            ax.set_ylabel("λ")
        ax.set_title(STRATUM_LABELS.get(s, s))

    fig.suptitle("Normalized Δreward: η × λ Heatmap per Stratum (B0.5+)")
    fig.tight_layout()
    save_figure(fig, fig_dir, "F4_normalized_heatmap")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analysis 4: Ceiling-Corrected Effect Sizes",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    fig_dir = os.path.join(args.output_dir, "figures")
    tbl_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    # ── Load B0.5+ ───────────────────────────────────────────────────
    print("Loading B0.5+ sweep...")
    b05p = load_sweep(B05P_SWEEP_DIR)
    b05p = b05p[b05p["stratum"].isin(NON_WEAK_STRATA)]
    b05p = add_normalized_columns(
        b05p, CARTPOLE_CEILING, CARTPOLE_FLOOR,
        baseline_col="baseline_avg_reward", delta_col="delta_reward",
    )
    b05p_oracle = compute_oracle_normalized(
        b05p, CARTPOLE_CEILING, CARTPOLE_FLOOR, "baseline_avg_reward",
    )

    # ── Load B0.5 ────────────────────────────────────────────────────
    print("Loading B0.5 sweep...")
    b05 = load_sweep(B05_SWEEP_DIR)
    b05 = b05[b05["stratum"].isin(NON_WEAK_STRATA)]
    b05 = add_normalized_columns(
        b05, CARTPOLE_CEILING, CARTPOLE_FLOOR,
        baseline_col="baseline_avg_reward", delta_col="delta_reward",
    )
    b05_oracle = compute_oracle_normalized(
        b05, CARTPOLE_CEILING, CARTPOLE_FLOOR, "baseline_avg_reward",
    )

    # ── Load Acrobot ─────────────────────────────────────────────────
    acrobot_oracle = None
    try:
        print("Loading Acrobot sweep...")
        acrobot = pd.read_parquet(ACROBOT_SWEEP_FILE)
        acrobot = acrobot[acrobot["stratum"].isin(NON_WEAK_STRATA)]

        # Acrobot uses different column names — adapt
        if "baseline_avg_reward" not in acrobot.columns and "baseline_reward" in acrobot.columns:
            acrobot = acrobot.rename(columns={"baseline_reward": "baseline_avg_reward"})

        acrobot = add_normalized_columns(
            acrobot, ACROBOT_CEILING, ACROBOT_FLOOR,
            baseline_col="baseline_avg_reward", delta_col="delta_reward",
        )
        acrobot_oracle = compute_oracle_normalized(
            acrobot, ACROBOT_CEILING, ACROBOT_FLOOR, "baseline_avg_reward",
        )
    except Exception as e:
        logger.warning("Could not load Acrobot data: %s", e)

    # ── Figures ──────────────────────────────────────────────────────
    datasets = {"B0.5": b05_oracle, "B0.5+": b05p_oracle}
    if acrobot_oracle is not None:
        datasets["Acrobot"] = acrobot_oracle

    print("\n--- F0: Normalized Effect Sizes ---")
    norm_results = figure_normalized_effect_sizes(datasets, fig_dir)

    print("--- F1: Raw vs Normalized ---")
    figure_raw_vs_normalized(b05p_oracle, fig_dir)

    if acrobot_oracle is not None:
        print("--- F2: Cross-Task Comparison ---")
        figure_cross_task(b05p_oracle, acrobot_oracle, fig_dir)

    print("--- F3: Headroom Scatter ---")
    figure_headroom_scatter(b05p_oracle, CARTPOLE_CEILING, fig_dir)

    print("--- F4: Normalized Heatmap ---")
    figure_normalized_heatmap(b05p, fig_dir)

    # ── Save summary ─────────────────────────────────────────────────
    summary = {
        "cartpole": {"ceiling": CARTPOLE_CEILING, "floor": CARTPOLE_FLOOR},
        "acrobot": {"ceiling": ACROBOT_CEILING, "floor": ACROBOT_FLOOR},
        "normalized_effects": norm_results,
    }

    # Add per-dataset per-stratum stats
    for name, oracle_df in datasets.items():
        stats_dict = {}
        for s in NON_WEAK_STRATA:
            sdf = oracle_df[oracle_df["stratum"] == s]
            if len(sdf) == 0:
                continue
            stats_dict[s] = {
                "n": int(len(sdf)),
                "raw_mean": float(sdf["raw_oracle_delta"].mean()),
                "norm_mean": float(sdf["raw_oracle_normalized"].mean()),
                "improvement_rate": float(sdf["improved"].mean()),
            }
        summary[f"{name}_strata"] = stats_dict

    summary_path = os.path.join(args.output_dir, "ceiling_corrected_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print text summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CEILING-CORRECTED EFFECT SIZE SUMMARY")
    print(f"{'='*60}")

    for name, oracle_df in datasets.items():
        print(f"\n  {name}:")
        for s in NON_WEAK_STRATA:
            sdf = oracle_df[oracle_df["stratum"] == s]
            if len(sdf) == 0:
                continue
            raw = sdf["raw_oracle_delta"].mean()
            norm = sdf["raw_oracle_normalized"].mean()
            rate = sdf["improved"].mean()
            print(f"    {STRATUM_LABELS.get(s, s):20s} "
                  f"raw={raw:+7.1f}  norm={norm:+.3f}  "
                  f"improved={100*rate:.1f}% (n={len(sdf)})")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
