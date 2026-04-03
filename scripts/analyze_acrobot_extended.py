#!/usr/bin/env python3
"""
Acrobot Extended Validation Analysis

Loads pilot and/or dense sweep results, computes stats, saves figures.

Analyses:
  - Best fixed (η, λ) by mean Δreward per stratum
  - Whether any fixed setting improves the average network
  - Oracle improvement rates
  - Regret vs coarse grid
  - η response curves and heatmaps

Figures:
  F0_eta_response  — Mean Δreward vs η for each stratum (both conditions)
  F1_heatmaps      — η × λ heatmap of mean Δreward per stratum
  F2_oracle_rates  — Oracle improvement rate: coarse vs extended grid
  F3_regret        — Regret reduction from finer grid

Usage:
  python scripts/analyze_acrobot_extended.py
  python scripts/analyze_acrobot_extended.py --phase pilot
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
    STRATUM_COLORS,
    STRATUM_LABELS,
    apply_publication_style,
    save_figure,
)

logger = logging.getLogger(__name__)
apply_publication_style()

SWEEP_DIR = Path("experiments/acrobot/sweep_extended")
COARSE_STATIC = Path("experiments/acrobot/sweep_static/acrobot_static_sweep.parquet")
COARSE_NS = Path(
    "experiments/acrobot/sweep_nonstationary/sweep/acrobot_ns_heavy_link2_2x.parquet"
)
DEFAULT_OUTPUT_DIR = Path("experiments/acrobot/sweep_extended/analysis")

NON_WEAK_STRATA = ["low_mid", "high_mid", "near_perfect", "perfect"]


def load_data(phase: str, suffix: str = "") -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load static and NS parquets for the given phase."""
    static_path = SWEEP_DIR / f"{phase}_static{suffix}.parquet"
    ns_path = SWEEP_DIR / f"{phase}_ns{suffix}.parquet"

    static_df = pd.read_parquet(static_path) if static_path.exists() else None
    ns_df = pd.read_parquet(ns_path) if ns_path.exists() else None

    if static_df is not None:
        static_df = static_df[static_df["stratum"].isin(NON_WEAK_STRATA)]
    if ns_df is not None:
        ns_df = ns_df[ns_df["stratum"].isin(NON_WEAK_STRATA)]

    return static_df, ns_df


def load_coarse() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load existing coarse-grid sweeps for comparison."""
    static_df = None
    ns_df = None
    if COARSE_STATIC.exists():
        static_df = pd.read_parquet(COARSE_STATIC)
        static_df = static_df[static_df["stratum"].isin(NON_WEAK_STRATA)]
    if COARSE_NS.exists():
        ns_df = pd.read_parquet(COARSE_NS)
        ns_df = ns_df[ns_df["stratum"].isin(NON_WEAK_STRATA)]
    return static_df, ns_df


def compute_oracle(df: pd.DataFrame, delta_col: str = "delta_reward") -> pd.DataFrame:
    """Per-network oracle: best delta_reward across all (η, λ)."""
    idx = df.groupby("network_id")[delta_col].idxmax()
    oracle = df.loc[idx, ["network_id", "stratum", "eta", "decay", delta_col]].copy()
    oracle = oracle.rename(columns={delta_col: "oracle_delta", "eta": "oracle_eta", "decay": "oracle_decay"})
    oracle["improved"] = oracle["oracle_delta"] > 0
    return oracle.reset_index(drop=True)


# ── Figures ──────────────────────────────────────────────────────────

def figure_eta_response(
    static_df: pd.DataFrame | None,
    ns_df: pd.DataFrame | None,
    fig_dir: str,
) -> None:
    """F0: Mean Δreward vs η for each stratum."""
    datasets = []
    if static_df is not None:
        datasets.append(("Static", static_df))
    if ns_df is not None:
        datasets.append(("NS", ns_df))

    n = len(datasets)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for idx, (label, df) in enumerate(datasets):
        ax = axes[0, idx]
        for s in NON_WEAK_STRATA:
            sdf = df[df["stratum"] == s]
            if len(sdf) == 0:
                continue
            agg = sdf.groupby("eta")["delta_reward"].mean().reset_index().sort_values("eta")
            ax.plot(
                agg["eta"], agg["delta_reward"],
                marker="o", markersize=3, lw=1.5,
                color=STRATUM_COLORS.get(s, "#999"),
                label=STRATUM_LABELS.get(s, s),
            )
        ax.axhline(0, color="k", ls="--", lw=1, alpha=0.3)
        ax.axvline(0, color="k", ls=":", lw=0.5, alpha=0.3)
        ax.set_xlabel("η (learning rate)")
        ax.set_ylabel("Mean Δreward")
        ax.set_title(f"{label} Condition")
        ax.legend(fontsize=8)

    fig.suptitle("η Response Curves (Extended Grid)")
    fig.tight_layout()
    save_figure(fig, fig_dir, "F0_eta_response")


def figure_heatmaps(
    static_df: pd.DataFrame | None,
    ns_df: pd.DataFrame | None,
    fig_dir: str,
) -> None:
    """F1: η × λ heatmap of mean Δreward per stratum."""
    for label, df in [("static", static_df), ("ns", ns_df)]:
        if df is None:
            continue

        strata = [s for s in NON_WEAK_STRATA if s in df["stratum"].unique()]
        n = len(strata)
        if n == 0:
            continue

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
        if n == 1:
            axes = [axes]

        for idx, s in enumerate(strata):
            ax = axes[idx]
            sdf = df[df["stratum"] == s]
            pivot = sdf.pivot_table(
                index="decay", columns="eta", values="delta_reward", aggfunc="mean",
            )
            if pivot.empty:
                continue

            vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 1)
            im = ax.imshow(
                pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
            )
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{e:.4f}" for e in pivot.columns], rotation=90, fontsize=6)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{d:.4f}" for d in pivot.index], fontsize=7)
            ax.set_xlabel("η")
            if idx == 0:
                ax.set_ylabel("λ")
            ax.set_title(STRATUM_LABELS.get(s, s))

        fig.suptitle(f"Δreward Heatmap — {label.upper()} (Extended Grid)")
        fig.tight_layout()
        save_figure(fig, fig_dir, f"F1_heatmap_{label}")


def figure_oracle_rates(
    ext_static: pd.DataFrame | None,
    ext_ns: pd.DataFrame | None,
    coarse_static: pd.DataFrame | None,
    coarse_ns: pd.DataFrame | None,
    fig_dir: str,
) -> dict:
    """F2: Oracle improvement rate comparison."""
    results = {}
    pairs = [
        ("Static", ext_static, coarse_static, "delta_reward"),
        ("NS", ext_ns, coarse_ns, "delta_reward_total" if coarse_ns is not None and "delta_reward_total" in (coarse_ns.columns if coarse_ns is not None else []) else "delta_reward"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    x_pos = 0

    for label, ext_df, coarse_df, coarse_delta_col in pairs:
        for s_idx, s in enumerate(NON_WEAK_STRATA):
            coarse_rate = None
            ext_rate = None

            if coarse_df is not None:
                cs = coarse_df[coarse_df["stratum"] == s]
                if len(cs) > 0:
                    c_oracle = compute_oracle(cs, coarse_delta_col)
                    coarse_rate = float(c_oracle["improved"].mean())

            if ext_df is not None:
                es = ext_df[ext_df["stratum"] == s]
                if len(es) > 0:
                    e_oracle = compute_oracle(es)
                    ext_rate = float(e_oracle["improved"].mean())

            if coarse_rate is not None:
                ax.bar(x_pos, coarse_rate, width, color=STRATUM_COLORS.get(s, "#999"),
                       alpha=0.5, edgecolor="black", linewidth=0.5)
            if ext_rate is not None:
                ax.bar(x_pos + width, ext_rate, width, color=STRATUM_COLORS.get(s, "#999"),
                       edgecolor="black", linewidth=1.5)

            results[f"{label}_{s}"] = {
                "coarse_oracle_rate": coarse_rate,
                "extended_oracle_rate": ext_rate,
            }
            x_pos += 1

        x_pos += 0.5  # gap between conditions

    # Labels
    n_groups = len(NON_WEAK_STRATA)
    tick_positions = []
    tick_labels = []
    offset = 0
    for label, _, _, _ in pairs:
        for s in NON_WEAK_STRATA:
            tick_positions.append(offset + width / 2)
            tick_labels.append(f"{STRATUM_LABELS.get(s, s)}\n({label})")
            offset += 1
        offset += 0.5

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Oracle Improvement Rate")
    ax.set_title("Oracle Improvement: Coarse (faded) vs Extended (solid)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_figure(fig, fig_dir, "F2_oracle_rates")

    return results


def figure_regret(
    ext_df: pd.DataFrame | None,
    coarse_df: pd.DataFrame | None,
    coarse_delta_col: str,
    condition_label: str,
    fig_dir: str,
) -> dict | None:
    """F3: Per-network regret comparison."""
    if ext_df is None or coarse_df is None:
        return None

    # Merge on network_id: compare best delta from each grid
    ext_oracle = compute_oracle(ext_df)
    coarse_oracle = compute_oracle(coarse_df, coarse_delta_col)

    merged = ext_oracle[["network_id", "stratum", "oracle_delta"]].merge(
        coarse_oracle[["network_id", "oracle_delta"]],
        on="network_id", suffixes=("_ext", "_coarse"),
    )
    merged["regret_reduction"] = merged["oracle_delta_ext"] - merged["oracle_delta_coarse"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for s in NON_WEAK_STRATA:
        sdf = merged[merged["stratum"] == s]
        if len(sdf) == 0:
            continue
        ax.hist(
            sdf["regret_reduction"], bins=30, alpha=0.5,
            color=STRATUM_COLORS.get(s, "#999"),
            label=f"{STRATUM_LABELS.get(s, s)} (mean={sdf['regret_reduction'].mean():+.1f})",
        )

    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Regret Reduction (Extended − Coarse Oracle)")
    ax.set_ylabel("Count")
    ax.set_title(f"Regret Reduction from Extended Grid ({condition_label})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, fig_dir, f"F3_regret_{condition_label.lower()}")

    return {
        s: {"mean": float(merged[merged["stratum"] == s]["regret_reduction"].mean()),
            "median": float(merged[merged["stratum"] == s]["regret_reduction"].median())}
        for s in NON_WEAK_STRATA if s in merged["stratum"].unique()
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Acrobot Extended Validation Analysis",
    )
    parser.add_argument("--phase", default="pilot", choices=["pilot", "dense"])
    parser.add_argument("--suffix", default="", help="File suffix (e.g., '_pilot')")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    fig_dir = str(args.output_dir / "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Loading {args.phase} data...")
    ext_static, ext_ns = load_data(args.phase, args.suffix)

    print("Loading coarse-grid data for comparison...")
    coarse_static, coarse_ns = load_coarse()

    if ext_static is not None:
        print(f"  Extended static: {len(ext_static):,} rows, {ext_static['network_id'].nunique()} nets")
    if ext_ns is not None:
        print(f"  Extended NS: {len(ext_ns):,} rows, {ext_ns['network_id'].nunique()} nets")

    # Figures
    print("\n--- F0: η Response Curves ---")
    figure_eta_response(ext_static, ext_ns, fig_dir)

    print("--- F1: Heatmaps ---")
    figure_heatmaps(ext_static, ext_ns, fig_dir)

    print("--- F2: Oracle Rates ---")
    oracle_results = figure_oracle_rates(
        ext_static, ext_ns, coarse_static, coarse_ns, fig_dir,
    )

    print("--- F3: Regret ---")
    regret_static = figure_regret(
        ext_static, coarse_static, "delta_reward", "Static", fig_dir,
    )
    ns_delta_col = "delta_reward_total" if coarse_ns is not None and "delta_reward_total" in coarse_ns.columns else "delta_reward"
    regret_ns = figure_regret(ext_ns, coarse_ns, ns_delta_col, "NS", fig_dir)

    # Summary
    summary: dict = {
        "phase": args.phase,
        "oracle_comparison": oracle_results,
        "regret_static": regret_static,
        "regret_ns": regret_ns,
    }

    # Add best fixed settings
    for label, df in [("static", ext_static), ("ns", ext_ns)]:
        if df is None:
            continue
        agg = df.groupby(["eta", "decay"])["delta_reward"].mean().reset_index()
        best = agg.loc[agg["delta_reward"].idxmax()]
        summary[f"best_fixed_{label}"] = {
            "eta": float(best["eta"]),
            "decay": float(best["decay"]),
            "mean_delta": round(float(best["delta_reward"]), 2),
            "positive": bool(best["delta_reward"] > 0),
        }

    summary_path = args.output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print(f"\n{'='*60}")
    print("ACROBOT EXTENDED VALIDATION SUMMARY")
    print(f"{'='*60}")
    for label in ["static", "ns"]:
        bf = summary.get(f"best_fixed_{label}")
        if bf:
            sign = "+" if bf["mean_delta"] > 0 else ""
            print(f"  Best fixed ({label}): η={bf['eta']:+.5f}, λ={bf['decay']:.4f} "
                  f"→ Δ={sign}{bf['mean_delta']:.1f} "
                  f"({'POSITIVE' if bf['positive'] else 'negative'})")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
