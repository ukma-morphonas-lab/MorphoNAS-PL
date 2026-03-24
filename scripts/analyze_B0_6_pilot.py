#!/usr/bin/env python3
"""
B0.6 Pilot Analysis: Non-Stationary CartPole

Key questions:
1. Does gravity-2x actually disrupt fixed-weight networks? (Phase 2 drop)
2. At which eta values does plasticity help in Phase 2?
3. Is the Phase 2 benefit larger than the static benefit from B0.5+?
4. Is plasticity genuinely active (|dw| > 0) or is it recurrent dynamics?

Usage:
  python scripts/analyze_B0_6_pilot.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.abspath("code"))

# ── Paths ──────────────────────────────────────────────────────────────
PILOT_DIR = Path("experiments/B0.6/sweep_pilot")
B05P_SWEEP = Path("experiments/B0.5+/sweep")
OUTPUT_DIR = Path("experiments/B0.6/analysis_pilot")

STRATA_ORDER = ["weak", "low_mid", "high_mid", "near_perfect", "perfect"]
STRATA_LABELS = {
    "weak": "Weak",
    "low_mid": "Low-mid",
    "high_mid": "High-mid",
    "near_perfect": "Near-perfect",
    "perfect": "Perfect",
}


def load_pilot_data() -> pd.DataFrame:
    path = PILOT_DIR / "B0_6_gravity_2x.parquet"
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows from pilot data")
    print(f"Networks: {df['network_id'].nunique()}")
    print(f"Strata: {dict(df.groupby('stratum')['network_id'].nunique())}")
    return df


def load_static_comparison(network_ids: set[int]) -> pd.DataFrame:
    """Load B0.5+ static data for the same networks."""
    frames = []
    for pfile in sorted(B05P_SWEEP.glob("*.parquet")):
        chunk = pd.read_parquet(pfile)
        chunk = chunk[chunk["network_id"].isin(network_ids)]
        if len(chunk) > 0:
            frames.append(chunk)

    if not frames:
        print("WARNING: No B0.5+ static data found for comparison")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} static comparison rows from B0.5+")
    return df


def analysis_1_disruption(df: pd.DataFrame, output_dir: Path) -> dict:
    """Q1: Does gravity-2x actually disrupt fixed-weight networks?"""
    print("\n" + "="*60)
    print("ANALYSIS 1: Environment Disruption")
    print("="*60)

    # Use baseline rows (eta=0) for disruption analysis
    baseline = df[df["eta"].abs() < 1e-12].copy()

    results = {}
    for stratum in STRATA_ORDER:
        sdf = baseline[baseline["stratum"] == stratum]
        if len(sdf) == 0:
            continue

        p1 = sdf["baseline_phase1_reward"].values
        p2 = sdf["baseline_phase2_reward"].values
        max_possible_p1 = 200.0  # switch at step 200
        max_possible_p2 = 300.0  # steps 200-499

        p1_pct = float(np.mean(p1)) / max_possible_p1 * 100
        p2_pct = float(np.mean(p2)) / max_possible_p2 * 100
        survival_rate = float((p2 > 0).mean()) * 100

        results[stratum] = {
            "n": len(sdf),
            "phase1_mean": round(float(np.mean(p1)), 1),
            "phase1_pct": round(p1_pct, 1),
            "phase2_mean": round(float(np.mean(p2)), 1),
            "phase2_pct": round(p2_pct, 1),
            "drop_pct": round(p1_pct - p2_pct, 1),
            "survival_rate": round(survival_rate, 1),
        }

        label = STRATA_LABELS[stratum]
        print(f"  {label:15s}: Phase1={np.mean(p1):6.1f}/{max_possible_p1:.0f} ({p1_pct:.0f}%)  "
              f"Phase2={np.mean(p2):6.1f}/{max_possible_p2:.0f} ({p2_pct:.0f}%)  "
              f"Drop={p1_pct-p2_pct:+.0f}pp  Survived={survival_rate:.0f}%")

    return results


def analysis_2_per_eta(df: pd.DataFrame, output_dir: Path) -> dict:
    """Q2: At which eta values does plasticity help Phase 2?"""
    print("\n" + "="*60)
    print("ANALYSIS 2: Phase 2 Benefit by Eta")
    print("="*60)

    results = {}
    actionable_strata = ["low_mid", "high_mid", "near_perfect", "perfect"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, stratum in enumerate(actionable_strata):
        ax = axes[idx]
        sdf = df[df["stratum"] == stratum].copy()

        if len(sdf) == 0:
            continue

        # Per-eta mean Phase 2 delta
        eta_groups = sdf.groupby(["eta", "decay"]).agg(
            delta_p2_mean=("delta_reward_phase2", "mean"),
            delta_p2_std=("delta_reward_phase2", "std"),
            delta_total_mean=("delta_reward_total", "mean"),
            n=("delta_reward_phase2", "count"),
            phase2_dw=("phase2_mean_abs_dw", "mean"),
        ).reset_index()

        # Focus on decay=0.01 (the safe value from B0.5)
        d001 = eta_groups[eta_groups["decay"].between(0.009, 0.011)]

        if len(d001) > 0:
            best_row = d001.loc[d001["delta_p2_mean"].idxmax()]
            best_eta = float(best_row["eta"])
            best_delta = float(best_row["delta_p2_mean"])
        else:
            best_row = eta_groups.loc[eta_groups["delta_p2_mean"].idxmax()]
            best_eta = float(best_row["eta"])
            best_delta = float(best_row["delta_p2_mean"])

        results[stratum] = {
            "best_eta": best_eta,
            "best_delta_phase2": round(best_delta, 2),
            "best_delta_total": round(float(best_row["delta_total_mean"]), 2),
            "best_phase2_dw": float(best_row["phase2_dw"]),
        }

        label = STRATA_LABELS[stratum]
        print(f"  {label:15s}: best eta={best_eta:+.3f} → Phase2 Δ={best_delta:+.1f}, |dw|={best_row['phase2_dw']:.2e}")

        # Plot: Phase 2 delta vs eta (decay=0.01)
        if len(d001) > 0:
            d001_sorted = d001.sort_values("eta")
            ax.bar(range(len(d001_sorted)), d001_sorted["delta_p2_mean"].values,
                   color=["#2196F3" if e < 0 else "#FF5722" if e > 0 else "#9E9E9E"
                          for e in d001_sorted["eta"].values],
                   alpha=0.8)
            ax.set_xticks(range(len(d001_sorted)))
            ax.set_xticklabels([f"{e:.3f}" for e in d001_sorted["eta"].values],
                              rotation=45, ha="right", fontsize=7)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Phase 2 Δreward")
            ax.set_title(f"{label} (N={sdf['network_id'].nunique()})")

    fig.suptitle("B0.6 Pilot: Phase 2 (post-switch) plasticity benefit by eta\n(decay=0.01, gravity 9.8→20.0 at step 200)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "F1_phase2_delta_by_eta.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "F1_phase2_delta_by_eta.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: F1_phase2_delta_by_eta")

    return results


def analysis_3_adaptation_premium(df: pd.DataFrame, static_df: pd.DataFrame, output_dir: Path) -> dict:
    """Q3: Is the plasticity benefit LARGER under non-stationarity than static?"""
    print("\n" + "="*60)
    print("ANALYSIS 3: Adaptation Premium (Non-stationary vs Static)")
    print("="*60)

    if static_df.empty:
        print("  SKIP: No static data available for comparison")
        return {}

    results = {}
    pilot_networks = set(df["network_id"].unique())
    actionable_strata = ["low_mid", "high_mid", "near_perfect"]

    for stratum in actionable_strata:
        # Non-stationary: per-network best delta (across eta/decay)
        ns_df = df[(df["stratum"] == stratum)].copy()
        ns_best = ns_df.groupby("network_id")["delta_reward_total"].max()

        # Static: per-network best delta (across all B0.5+ eta/decay)
        st_df = static_df[(static_df["stratum"] == stratum) &
                          (static_df["network_id"].isin(pilot_networks))].copy()
        st_best = st_df.groupby("network_id")["delta_reward"].max()

        # Match networks
        common = sorted(set(ns_best.index) & set(st_best.index))
        if len(common) < 3:
            print(f"  {STRATA_LABELS[stratum]}: too few common networks ({len(common)})")
            continue

        ns_vals = ns_best.loc[common].values
        st_vals = st_best.loc[common].values
        premium = ns_vals - st_vals

        # Paired test
        if len(common) >= 5:
            t_stat, p_val = stats.ttest_rel(ns_vals, st_vals)
        else:
            t_stat, p_val = float("nan"), float("nan")

        results[stratum] = {
            "n_matched": len(common),
            "static_best_mean": round(float(np.mean(st_vals)), 2),
            "nonstat_best_mean": round(float(np.mean(ns_vals)), 2),
            "premium_mean": round(float(np.mean(premium)), 2),
            "premium_median": round(float(np.median(premium)), 2),
            "t_stat": round(float(t_stat), 3) if not np.isnan(t_stat) else None,
            "p_value": round(float(p_val), 4) if not np.isnan(p_val) else None,
        }

        label = STRATA_LABELS[stratum]
        print(f"  {label:15s}: static best={np.mean(st_vals):+.1f}, "
              f"nonstat best={np.mean(ns_vals):+.1f}, "
              f"premium={np.mean(premium):+.1f} "
              f"(p={'%.3f' % p_val if not np.isnan(p_val) else 'N/A'})")

    return results


def analysis_4_dw_diagnostic(df: pd.DataFrame, output_dir: Path) -> dict:
    """Q4: Is plasticity genuinely active (|dw| > 0)?"""
    print("\n" + "="*60)
    print("ANALYSIS 4: Plasticity Activity Diagnostic")
    print("="*60)

    results = {}

    # Look at rows where eta != 0
    plastic_df = df[df["eta"].abs() >= 1e-12].copy()

    for stratum in STRATA_ORDER:
        sdf = plastic_df[plastic_df["stratum"] == stratum]
        if len(sdf) == 0:
            continue

        p1_dw = sdf["phase1_mean_abs_dw"].values
        p2_dw = sdf["phase2_mean_abs_dw"].values

        # Fraction with non-zero |dw|
        p1_active = float((p1_dw > 1e-10).mean()) * 100
        p2_active = float((p2_dw > 1e-10).mean()) * 100

        results[stratum] = {
            "phase1_mean_dw": float(np.mean(p1_dw)),
            "phase2_mean_dw": float(np.mean(p2_dw)),
            "phase1_pct_active": round(p1_active, 1),
            "phase2_pct_active": round(p2_active, 1),
            "dw_ratio_p2_over_p1": round(float(np.mean(p2_dw)) / max(float(np.mean(p1_dw)), 1e-15), 2),
        }

        label = STRATA_LABELS[stratum]
        print(f"  {label:15s}: P1 |dw|={np.mean(p1_dw):.2e} ({p1_active:.0f}% active)  "
              f"P2 |dw|={np.mean(p2_dw):.2e} ({p2_active:.0f}% active)  "
              f"ratio={results[stratum]['dw_ratio_p2_over_p1']:.2f}")

    return results


def analysis_5_phase_comparison_figure(df: pd.DataFrame, output_dir: Path):
    """Summary figure: Phase 1 vs Phase 2 performance by stratum."""
    print("\n" + "="*60)
    print("ANALYSIS 5: Summary Figure")
    print("="*60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Baseline Phase 1 vs Phase 2 by stratum
    ax = axes[0]
    baseline = df[df["eta"].abs() < 1e-12]
    strata_data = []
    for s in STRATA_ORDER:
        sdf = baseline[baseline["stratum"] == s]
        if len(sdf) > 0:
            strata_data.append({
                "stratum": STRATA_LABELS[s],
                "Phase 1": float(sdf["baseline_phase1_reward"].mean()),
                "Phase 2": float(sdf["baseline_phase2_reward"].mean()),
            })

    if strata_data:
        x = range(len(strata_data))
        w = 0.35
        p1_vals = [d["Phase 1"] for d in strata_data]
        p2_vals = [d["Phase 2"] for d in strata_data]
        ax.bar([i - w/2 for i in x], p1_vals, w, label="Phase 1 (pre-switch)", color="#4CAF50", alpha=0.8)
        ax.bar([i + w/2 for i in x], p2_vals, w, label="Phase 2 (post-switch)", color="#F44336", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([d["stratum"] for d in strata_data], rotation=30, ha="right")
        ax.set_ylabel("Mean reward (no plasticity)")
        ax.set_title("A. Gravity-2x disruption\n(fixed weights)")
        ax.legend(fontsize=8)

    # Panel B: Best-eta Phase 2 delta by stratum
    ax = axes[1]
    actionable = ["low_mid", "high_mid", "near_perfect", "perfect"]
    best_deltas = []
    for s in actionable:
        sdf = df[(df["stratum"] == s) & (df["decay"].between(0.009, 0.011))]
        if len(sdf) == 0:
            continue
        by_eta = sdf.groupby("eta")["delta_reward_phase2"].mean()
        best_idx = by_eta.idxmax()
        best_deltas.append({
            "stratum": STRATA_LABELS[s],
            "best_eta": best_idx,
            "delta": float(by_eta[best_idx]),
        })

    if best_deltas:
        colors = ["#2196F3" if d["delta"] > 0 else "#FF5722" for d in best_deltas]
        bars = ax.bar(range(len(best_deltas)),
                      [d["delta"] for d in best_deltas],
                      color=colors, alpha=0.8)
        ax.set_xticks(range(len(best_deltas)))
        ax.set_xticklabels([f"{d['stratum']}\nη={d['best_eta']:+.3f}" for d in best_deltas],
                          fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Phase 2 Δreward (best η)")
        ax.set_title("B. Plasticity benefit after switch\n(best η, decay=0.01)")

        for bar, d in zip(bars, best_deltas):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f"{d['delta']:+.1f}", ha="center", va="bottom" if d["delta"] > 0 else "top",
                   fontsize=9, fontweight="bold")

    # Panel C: |dw| comparison Phase 1 vs Phase 2
    ax = axes[2]
    plastic = df[(df["eta"].abs() >= 1e-12) & (df["decay"].between(0.009, 0.011))]
    dw_data = []
    for s in STRATA_ORDER:
        sdf = plastic[plastic["stratum"] == s]
        if len(sdf) > 0 and sdf["phase1_mean_abs_dw"].mean() > 1e-15:
            dw_data.append({
                "stratum": STRATA_LABELS[s],
                "Phase 1": float(sdf["phase1_mean_abs_dw"].mean()),
                "Phase 2": float(sdf["phase2_mean_abs_dw"].mean()),
            })

    if dw_data:
        x = range(len(dw_data))
        w = 0.35
        ax.bar([i - w/2 for i in x], [d["Phase 1"] for d in dw_data], w,
               label="Phase 1", color="#4CAF50", alpha=0.8)
        ax.bar([i + w/2 for i in x], [d["Phase 2"] for d in dw_data], w,
               label="Phase 2", color="#F44336", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([d["stratum"] for d in dw_data], rotation=30, ha="right")
        ax.set_ylabel("Mean |Δw| per step")
        ax.set_title("C. Plasticity activity\n(weight change magnitude)")
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))

    fig.suptitle("B0.6 Pilot: Non-Stationary CartPole (gravity 9.8→20.0 at step 200)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "F2_pilot_summary.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "F2_pilot_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: F2_pilot_summary")


def analysis_6_per_network_scatter(df: pd.DataFrame, output_dir: Path):
    """Per-network scatter: baseline Phase 2 reward vs plasticity Phase 2 delta."""
    print("\n" + "="*60)
    print("ANALYSIS 6: Per-Network Phase 2 Response")
    print("="*60)

    # For each network, find the best eta (decay=0.01)
    plastic = df[(df["eta"].abs() >= 1e-12) & (df["decay"].between(0.009, 0.011))].copy()
    best_per_network = plastic.loc[plastic.groupby("network_id")["delta_reward_phase2"].idxmax()]

    # Also get baseline Phase 2 for each network
    baselines = df[df["eta"].abs() < 1e-12].set_index("network_id")

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"weak": "#9E9E9E", "low_mid": "#FF9800", "high_mid": "#2196F3",
              "near_perfect": "#4CAF50", "perfect": "#9C27B0"}

    for stratum in STRATA_ORDER:
        sdf = best_per_network[best_per_network["stratum"] == stratum]
        if len(sdf) == 0:
            continue
        x = sdf["baseline_phase2_reward"].values
        y = sdf["delta_reward_phase2"].values
        ax.scatter(x, y, c=colors[stratum], label=STRATA_LABELS[stratum],
                  alpha=0.7, s=60, edgecolors="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Baseline Phase 2 reward (no plasticity)", fontsize=11)
    ax.set_ylabel("Best Phase 2 Δreward (plasticity benefit)", fontsize=11)
    ax.set_title("B0.6 Pilot: Per-network plasticity benefit after gravity switch\n(best η per network, decay=0.01)", fontsize=12)
    ax.legend(fontsize=10)

    # Annotate counts
    n_helped = int((best_per_network["delta_reward_phase2"] > 0).sum())
    n_total = len(best_per_network)
    ax.text(0.02, 0.98, f"Helped: {n_helped}/{n_total} ({n_helped/n_total*100:.0f}%)",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_dir / "F3_per_network_scatter.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "F3_per_network_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: F3_per_network_scatter")
    print(f"  Networks helped: {n_helped}/{n_total} ({n_helped/n_total*100:.0f}%)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_pilot_data()
    pilot_networks = set(df["network_id"].unique())
    static_df = load_static_comparison(pilot_networks)

    # Run analyses
    r1 = analysis_1_disruption(df, OUTPUT_DIR)
    r2 = analysis_2_per_eta(df, OUTPUT_DIR)
    r3 = analysis_3_adaptation_premium(df, static_df, OUTPUT_DIR)
    r4 = analysis_4_dw_diagnostic(df, OUTPUT_DIR)
    analysis_5_phase_comparison_figure(df, OUTPUT_DIR)
    analysis_6_per_network_scatter(df, OUTPUT_DIR)

    # Save combined results
    all_results = {
        "experiment": "B0.6 Pilot",
        "variant": "gravity_2x",
        "n_networks": len(pilot_networks),
        "disruption": r1,
        "per_eta": r2,
        "adaptation_premium": r3,
        "plasticity_activity": r4,
    }

    with open(OUTPUT_DIR / "B0_6_pilot_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Figures: F1_phase2_delta_by_eta, F2_pilot_summary, F3_per_network_scatter")
    print(f"Data: B0_6_pilot_analysis.json")


if __name__ == "__main__":
    main()
