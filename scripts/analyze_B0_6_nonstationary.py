#!/usr/bin/env python3
"""
B0.6 Full Analysis: Non-Stationary CartPole

Comprehensive analysis of B0.6 non-stationary experiments.
Compares results to B0.5+ static data to quantify the "adaptation premium."

Usage:
  python scripts/analyze_B0_6_nonstationary.py --variant heavy_pole
  python scripts/analyze_B0_6_nonstationary.py --variant gravity_2x
  python scripts/analyze_B0_6_nonstationary.py --variant heavy_pole --sweep-dir experiments/B0.6/sweep_pilot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.abspath("code"))

# ── Paths ──────────────────────────────────────────────────────────────
B05P_SWEEP = Path("experiments/B0.5+/sweep")
B05P_SWEEP_EXT = Path("experiments/B0.5+/sweep_extended")

STRATA_ORDER = ["weak", "low_mid", "high_mid", "near_perfect", "perfect"]
STRATA_LABELS = {
    "weak": "Weak",
    "low_mid": "Low-mid",
    "high_mid": "High-mid",
    "near_perfect": "Near-perfect",
    "perfect": "Perfect",
}
STRATA_COLORS = {
    "weak": "#9E9E9E",
    "low_mid": "#FF9800",
    "high_mid": "#2196F3",
    "near_perfect": "#4CAF50",
    "perfect": "#9C27B0",
}


VARIANT_LABELS = {
    "gravity_2x": "gravity 9.8 → 20.0",
    "heavy_pole": "pole mass 0.1 → 1.0",
    "weak_force": "force 10.0 → 4.0",
}


def load_nonstationary_data(sweep_dir: Path, variant: str) -> pd.DataFrame:
    """Load B0.6 non-stationary data for a given variant."""
    pfile = sweep_dir / f"B0_6_{variant}.parquet"
    if not pfile.exists():
        raise FileNotFoundError(f"No data at {pfile}. Run the sweep first.")
    df = pd.read_parquet(pfile)
    print(f"B0.6 data ({variant}): {len(df):,} rows, {df['network_id'].nunique()} networks")
    return df


def load_static_data(network_ids: set[int]) -> pd.DataFrame:
    """Load B0.5+ static data for matched comparison.

    Reads only from B05P_SWEEP (the consolidated canonical parquets).
    sweep_extended is the raw v1 directory and is intentionally excluded —
    all its data is already merged into sweep/ via consolidate_B0_5plus.py.
    """
    frames = []
    if not B05P_SWEEP.exists():
        print("WARNING: No B0.5+ static data found")
        return pd.DataFrame()

    for pfile in sorted(B05P_SWEEP.glob("*.parquet")):
        chunk = pd.read_parquet(pfile)
        chunk = chunk[chunk["network_id"].isin(network_ids)]
        if len(chunk) > 0:
            frames.append(chunk)

    if not frames:
        print("WARNING: No B0.5+ static data found for matched network IDs")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["network_id", "eta", "decay"])
    print(f"B0.5+ static data: {len(df):,} rows, {df['network_id'].nunique()} networks, "
          f"{df[['eta','decay']].drop_duplicates().shape[0]} grid points")
    return df


# ── Analysis 1: Disruption Characterization ───────────────────────────

def analysis_1_disruption(df: pd.DataFrame, output_dir: Path) -> dict:
    """How much does gravity-2x disrupt each stratum?"""
    print("\n" + "="*60)
    print("ANALYSIS 1: Environment Disruption (baseline only)")
    print("="*60)

    baseline = df[df["eta"].abs() < 1e-12].copy()
    results = {}

    for stratum in STRATA_ORDER:
        sdf = baseline[baseline["stratum"] == stratum]
        if len(sdf) == 0:
            continue

        p1 = sdf["baseline_phase1_reward"].values
        p2 = sdf["baseline_phase2_reward"].values

        # Survival: did the network make it to Phase 2 at all?
        survived = p2 > 0
        # Utilization: fraction of max possible reward per phase
        # Phase 1: steps 1–199 (switch fires at _step_count=200), max 199 reward points
        # Phase 2: steps 200–500 (CartPole-v1 truncates at 500), max 301 reward points
        p1_pct = float(np.mean(p1)) / 199.0 * 100
        p2_pct = float(np.mean(p2)) / 301.0 * 100

        n_nets = int(sdf["network_id"].nunique())
        results[stratum] = {
            "n_networks": n_nets,
            "phase1_mean": round(float(np.mean(p1)), 2),
            "phase2_mean": round(float(np.mean(p2)), 2),
            "phase1_pct": round(p1_pct, 1),
            "phase2_pct": round(p2_pct, 1),
            "survival_rate": round(float(survived.mean()) * 100, 1),
            "phase2_std": round(float(np.std(p2)), 2),
        }

        label = STRATA_LABELS[stratum]
        print(f"  {label:15s} (N={n_nets:4d}): "
              f"P1={p1_pct:5.1f}%  P2={p2_pct:5.1f}%  "
              f"survived={survived.mean()*100:.0f}%")

    return results


# ── Analysis 2: Best-eta Phase 2 Benefit ──────────────────────────────

def analysis_2_best_eta(df: pd.DataFrame, output_dir: Path, variant_label: str = "") -> dict:
    """Find best eta for Phase 2 performance per stratum."""
    print("\n" + "="*60)
    print("ANALYSIS 2: Best Eta for Phase 2 Adaptation")
    print("="*60)

    results = {}
    actionable = ["low_mid", "high_mid", "near_perfect", "perfect"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, stratum in enumerate(actionable):
        ax = axes[idx]
        sdf = df[df["stratum"] == stratum].copy()
        if len(sdf) == 0:
            continue

        # Group by (eta, decay) and compute mean Phase 2 delta
        grouped = sdf.groupby(["eta", "decay"]).agg(
            delta_p2_mean=("delta_reward_phase2", "mean"),
            delta_p2_se=("delta_reward_phase2", lambda x: x.std() / np.sqrt(len(x))),
            delta_total_mean=("delta_reward_total", "mean"),
            n=("delta_reward_phase2", "count"),
            phase2_dw=("phase2_mean_abs_dw", "mean"),
        ).reset_index()

        # Best per decay
        for decay_val in [0.0, 0.01]:
            subset = grouped[grouped["decay"].between(decay_val - 0.001, decay_val + 0.001)]
            if len(subset) == 0:
                continue
            best = subset.loc[subset["delta_p2_mean"].idxmax()]
            decay_label = f"d={decay_val}"
            results.setdefault(stratum, {})[decay_label] = {
                "best_eta": float(best["eta"]),
                "delta_phase2": round(float(best["delta_p2_mean"]), 2),
                "delta_total": round(float(best["delta_total_mean"]), 2),
                "dw": float(best["phase2_dw"]),
                "n_per_eta": int(best["n"]),
            }

        # Plot decay=0.01 curve
        d001 = grouped[grouped["decay"].between(0.009, 0.011)].sort_values("eta")
        d000 = grouped[grouped["decay"].abs() < 0.001].sort_values("eta")

        if len(d001) > 0:
            ax.plot(d001["eta"], d001["delta_p2_mean"], "o-", color="#2196F3",
                    label="decay=0.01", markersize=5)
            ax.fill_between(d001["eta"],
                           d001["delta_p2_mean"] - 1.96 * d001["delta_p2_se"],
                           d001["delta_p2_mean"] + 1.96 * d001["delta_p2_se"],
                           color="#2196F3", alpha=0.15)
        if len(d000) > 0:
            ax.plot(d000["eta"], d000["delta_p2_mean"], "s--", color="#FF5722",
                    label="decay=0", markersize=4, alpha=0.7)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.3)
        ax.set_xlabel("η (learning rate)")
        ax.set_ylabel("Phase 2 Δreward")
        n_nets = sdf["network_id"].nunique()
        ax.set_title(f"{STRATA_LABELS[stratum]} (N={n_nets})")
        ax.legend(fontsize=8)

        # Print best
        if "d=0.01" in results.get(stratum, {}):
            r = results[stratum]["d=0.01"]
            print(f"  {STRATA_LABELS[stratum]:15s}: best η={r['best_eta']:+.3f} → "
                  f"Phase2 Δ={r['delta_phase2']:+.1f}, |dw|={r['dw']:.2e}")

    fig.suptitle(f"B0.6: Phase 2 Δreward vs η (after {variant_label} at step 200)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "F1_phase2_delta_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "F1_phase2_delta_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: F1_phase2_delta_curves")

    return results


# ── Analysis 3: Adaptation Premium vs Static ──────────────────────────

def analysis_3_adaptation_premium(
    ns_df: pd.DataFrame, static_df: pd.DataFrame, output_dir: Path,
    variant_label: str = "",
) -> dict:
    """Compare per-network oracle benefit: non-stationary vs static."""
    print("\n" + "="*60)
    print("ANALYSIS 3: Adaptation Premium (Non-stationary vs Static)")
    print("="*60)

    if static_df.empty:
        print("  SKIP: No static data")
        return {}

    results = {}
    actionable = ["low_mid", "high_mid", "near_perfect", "perfect"]

    for stratum in actionable:
        # Non-stationary oracle: per-network best delta_total
        ns_s = ns_df[ns_df["stratum"] == stratum].copy()
        ns_oracle = ns_s.groupby("network_id")["delta_reward_total"].max()

        # Static oracle: per-network best delta_reward
        st_s = static_df[static_df["stratum"] == stratum].copy()
        st_oracle = st_s.groupby("network_id")["delta_reward"].max()

        common = sorted(set(ns_oracle.index) & set(st_oracle.index))
        if len(common) < 5:
            print(f"  {STRATA_LABELS[stratum]}: only {len(common)} matched networks, skipping")
            continue

        ns_vals = ns_oracle.loc[common].values
        st_vals = st_oracle.loc[common].values
        premium = ns_vals - st_vals

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(ns_vals, st_vals)

        # Wilcoxon signed-rank as backup
        if len(common) >= 10:
            w_stat, w_p = stats.wilcoxon(premium)
        else:
            w_stat, w_p = float("nan"), float("nan")

        results[stratum] = {
            "n_matched": len(common),
            "static_oracle_mean": round(float(np.mean(st_vals)), 2),
            "nonstat_oracle_mean": round(float(np.mean(ns_vals)), 2),
            "premium_mean": round(float(np.mean(premium)), 2),
            "premium_median": round(float(np.median(premium)), 2),
            "premium_std": round(float(np.std(premium)), 2),
            "pct_premium_positive": round(float((premium > 0).mean()) * 100, 1),
            "ttest_t": round(float(t_stat), 3),
            "ttest_p": round(float(p_val), 4),
            "wilcoxon_p": round(float(w_p), 4) if not np.isnan(w_p) else None,
        }

        label = STRATA_LABELS[stratum]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {label:15s} (N={len(common):4d}): "
              f"static={np.mean(st_vals):+6.1f}, nonstat={np.mean(ns_vals):+6.1f}, "
              f"premium={np.mean(premium):+6.1f} [{sig}] "
              f"(t={t_stat:.2f}, p={p_val:.4f})")

    # Figure: adaptation premium distribution
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for idx, (stratum, r) in enumerate(results.items()):
        ax = axes[idx]
        ns_s = ns_df[ns_df["stratum"] == stratum]
        st_s = static_df[static_df["stratum"] == stratum]

        ns_oracle = ns_s.groupby("network_id")["delta_reward_total"].max()
        st_oracle = st_s.groupby("network_id")["delta_reward"].max()
        common = sorted(set(ns_oracle.index) & set(st_oracle.index))

        premium = ns_oracle.loc[common].values - st_oracle.loc[common].values

        ax.hist(premium, bins=30, color=STRATA_COLORS[stratum], alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.axvline(np.mean(premium), color="red", linewidth=2, label=f"mean={np.mean(premium):+.1f}")
        ax.set_xlabel("Adaptation premium (nonstat - static)")
        ax.set_ylabel("Count")
        sig = "***" if r["ttest_p"] < 0.001 else "**" if r["ttest_p"] < 0.01 else "*" if r["ttest_p"] < 0.05 else "ns"
        ax.set_title(f"{STRATA_LABELS[stratum]} (N={r['n_matched']}) [{sig}]")
        ax.legend(fontsize=9)

    fig.suptitle(f"B0.6 ({variant_label}): Adaptation premium — does plasticity help MORE under non-stationarity?",
                fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "F2_adaptation_premium.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "F2_adaptation_premium.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: F2_adaptation_premium")

    return results


# ── Analysis 4: Plasticity Activity ───────────────────────────────────

def analysis_4_plasticity_activity(df: pd.DataFrame, output_dir: Path) -> dict:
    """Verify plasticity is genuinely active (|dw| > 0)."""
    print("\n" + "="*60)
    print("ANALYSIS 4: Plasticity Activity (|Δw|)")
    print("="*60)

    plastic = df[(df["eta"].abs() >= 1e-12) & (df["decay"].between(0.009, 0.011))].copy()
    results = {}

    for stratum in STRATA_ORDER:
        sdf = plastic[plastic["stratum"] == stratum]
        if len(sdf) == 0:
            continue

        p1_dw = sdf["phase1_mean_abs_dw"].values
        p2_dw = sdf["phase2_mean_abs_dw"].values

        results[stratum] = {
            "n": len(sdf),
            "phase1_dw_mean": float(np.mean(p1_dw)),
            "phase2_dw_mean": float(np.mean(p2_dw)),
            "phase1_pct_active": round(float((p1_dw > 1e-10).mean()) * 100, 1),
            "phase2_pct_active": round(float((p2_dw > 1e-10).mean()) * 100, 1),
        }

        label = STRATA_LABELS[stratum]
        print(f"  {label:15s}: P1 |dw|={np.mean(p1_dw):.2e} ({results[stratum]['phase1_pct_active']:.0f}%)  "
              f"P2 |dw|={np.mean(p2_dw):.2e} ({results[stratum]['phase2_pct_active']:.0f}%)")

    return results


# ── Analysis 5: Per-Network Scatter ───────────────────────────────────

def analysis_5_per_network(df: pd.DataFrame, output_dir: Path, variant_label: str = "") -> dict:
    """Per-network: baseline Phase 2 vs best plasticity Phase 2 delta."""
    print("\n" + "="*60)
    print("ANALYSIS 5: Per-Network Phase 2 Response")
    print("="*60)

    # For each network, best Phase 2 delta with decay=0.01
    plastic = df[(df["eta"].abs() >= 1e-12) & (df["decay"].between(0.009, 0.011))].copy()
    best = plastic.loc[plastic.groupby("network_id")["delta_reward_phase2"].idxmax()]

    # Baseline Phase 2 — deduplicate (eta=0 has 2 rows per network, one per decay)
    baselines = (
        df[df["eta"].abs() < 1e-12]
        .drop_duplicates(subset=["network_id"])
        .set_index("network_id")[["baseline_phase2_reward", "stratum"]]
    )

    merged = best.set_index("network_id")[["delta_reward_phase2", "eta"]].join(baselines, how="inner")

    n_helped = int((merged["delta_reward_phase2"] > 0).sum())
    n_total = len(merged)

    # Multi-panel: one per actionable stratum, hexbin for dense strata
    actionable = [s for s in STRATA_ORDER if s != "weak"]
    fig, axes = plt.subplots(1, len(actionable), figsize=(4.5 * len(actionable), 5), sharey=True)
    if len(actionable) == 1:
        axes = [axes]

    for idx, stratum in enumerate(actionable):
        ax = axes[idx]
        sdf = merged[merged["stratum"] == stratum]
        if len(sdf) == 0:
            continue

        if len(sdf) > 500:
            hb = ax.hexbin(
                sdf["baseline_phase2_reward"], sdf["delta_reward_phase2"],
                gridsize=25, mincnt=1, cmap="YlOrRd", edgecolors="grey", linewidths=0.3,
            )
            fig.colorbar(hb, ax=ax, shrink=0.7, label="Count")
        else:
            ax.scatter(
                sdf["baseline_phase2_reward"], sdf["delta_reward_phase2"],
                c=STRATA_COLORS[stratum], alpha=0.5, s=20,
                edgecolors="black", linewidth=0.3,
            )

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        helped = int((sdf["delta_reward_phase2"] > 0).sum())
        pct = helped / len(sdf) * 100
        ax.set_title(f"{STRATA_LABELS[stratum]} (N={len(sdf)})\n{helped}/{len(sdf)} helped ({pct:.0f}%)",
                     fontsize=10)
        ax.set_xlabel("Baseline Phase 2 reward", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Best Phase 2 Δreward", fontsize=10)

    fig.suptitle(f"B0.6 ({variant_label}): Per-network plasticity benefit after switch (best η, decay=0.01)",
                fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "F3_per_network_scatter.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "F3_per_network_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    per_stratum = {}
    for stratum in STRATA_ORDER:
        sdf = merged[merged["stratum"] == stratum]
        if len(sdf) == 0:
            continue
        helped = (sdf["delta_reward_phase2"] > 0).sum()
        per_stratum[stratum] = {
            "n": len(sdf),
            "helped": int(helped),
            "pct_helped": round(float(helped) / len(sdf) * 100, 1),
            "mean_delta": round(float(sdf["delta_reward_phase2"].mean()), 2),
            "median_delta": round(float(sdf["delta_reward_phase2"].median()), 2),
        }
        label = STRATA_LABELS[stratum]
        print(f"  {label:15s}: {helped}/{len(sdf)} helped ({helped/len(sdf)*100:.0f}%), "
              f"mean Δ={sdf['delta_reward_phase2'].mean():+.1f}")

    print(f"\n  Overall: {n_helped}/{n_total} ({n_helped/n_total*100:.0f}%) networks helped")
    print(f"  Saved: F3_per_network_scatter")

    return {"overall_helped": n_helped, "overall_total": n_total, "per_stratum": per_stratum}


# ── Analysis 7: |Δw| vs Improvement Correlation ──────────────────────

def analysis_7_dw_vs_improvement(df: pd.DataFrame, output_dir: Path, variant_label: str = "") -> dict:
    """|Δw| in Phase 2 vs Phase 2 Δreward — does more weight change mean more benefit?"""
    print("\n" + "="*60)
    print("ANALYSIS 7: |Δw| vs Improvement Correlation")
    print("="*60)

    plastic = df[(df["eta"].abs() >= 1e-12) & (df["decay"].between(0.009, 0.011))].copy()
    # Per-network best: pick the eta that maximises Phase 2 delta
    best = plastic.loc[plastic.groupby("network_id")["delta_reward_phase2"].idxmax()]

    results = {}
    actionable = ["low_mid", "high_mid", "near_perfect", "perfect"]

    fig, axes = plt.subplots(1, len(actionable), figsize=(4.5 * len(actionable), 5))
    if len(actionable) == 1:
        axes = [axes]

    for idx, stratum in enumerate(actionable):
        ax = axes[idx]
        sdf = best[best["stratum"] == stratum]
        if len(sdf) < 5:
            continue

        dw = sdf["phase2_mean_abs_dw"].values
        delta = sdf["delta_reward_phase2"].values

        rho, p_val = stats.spearmanr(dw, delta)
        results[stratum] = {
            "n": len(sdf),
            "spearman_rho": round(float(rho), 3),
            "p_value": round(float(p_val), 4),
        }

        ax.scatter(dw, delta, c=STRATA_COLORS[stratum], alpha=0.4, s=15,
                   edgecolors="black", linewidth=0.2)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.set_title(f"{STRATA_LABELS[stratum]} (N={len(sdf)})\nρ={rho:+.2f} [{sig}]", fontsize=10)
        ax.set_xlabel("Phase 2 mean |Δw|", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Phase 2 Δreward", fontsize=10)
        ax.ticklabel_format(axis="x", style="scientific", scilimits=(-4, -4))

        label = STRATA_LABELS[stratum]
        print(f"  {label:15s}: ρ={rho:+.3f} (p={p_val:.4f}) [{sig}]")

    fig.suptitle(f"B0.6 ({variant_label}): Does more weight change mean more improvement?",
                fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "F4_dw_vs_improvement.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "F4_dw_vs_improvement.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: F4_dw_vs_improvement")

    return results


# ── Analysis 8: Phase 1 Cost ─────────────────────────────────────────

def analysis_8_phase1_cost(df: pd.DataFrame, output_dir: Path, variant_label: str = "") -> dict:
    """How much Phase 1 reward does plasticity cost? (plastic P1 - baseline P1)."""
    print("\n" + "="*60)
    print("ANALYSIS 8: Phase 1 Cost of Plasticity")
    print("="*60)

    results = {}
    actionable = ["low_mid", "high_mid", "near_perfect", "perfect"]

    fig, axes = plt.subplots(1, len(actionable), figsize=(4.5 * len(actionable), 5))
    if len(actionable) == 1:
        axes = [axes]

    for idx, stratum in enumerate(actionable):
        ax = axes[idx]
        sdf = df[(df["stratum"] == stratum) & (df["decay"].between(0.009, 0.011))].copy()
        if len(sdf) == 0:
            continue

        # Mean Phase 1 cost per eta
        grouped = sdf.groupby("eta").agg(
            p1_cost_mean=("delta_reward_phase1", "mean"),
            p1_cost_se=("delta_reward_phase1", lambda x: x.std() / np.sqrt(len(x))),
            p2_gain_mean=("delta_reward_phase2", "mean"),
            n=("delta_reward_phase1", "count"),
        ).reset_index()

        # Plot P1 cost and P2 gain on same axes
        ax.plot(grouped["eta"], grouped["p1_cost_mean"], "v-", color="#F44336",
                label="Phase 1 Δ (cost)", markersize=4)
        ax.fill_between(grouped["eta"],
                       grouped["p1_cost_mean"] - 1.96 * grouped["p1_cost_se"],
                       grouped["p1_cost_mean"] + 1.96 * grouped["p1_cost_se"],
                       color="#F44336", alpha=0.15)

        ax.plot(grouped["eta"], grouped["p2_gain_mean"], "^-", color="#4CAF50",
                label="Phase 2 Δ (gain)", markersize=4)

        # Net total
        net = grouped["p1_cost_mean"] + grouped["p2_gain_mean"]
        ax.plot(grouped["eta"], net, "o--", color="#2196F3",
                label="Net total", markersize=3, alpha=0.7)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.3)
        n_nets = sdf["network_id"].nunique()
        ax.set_title(f"{STRATA_LABELS[stratum]} (N={n_nets})", fontsize=10)
        ax.set_xlabel("η", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Δreward", fontsize=10)
        ax.legend(fontsize=6.5, loc="best")

        # Summary: at best-total eta, what's the P1 cost?
        best_row = grouped.loc[
            (grouped["p1_cost_mean"] + grouped["p2_gain_mean"]).idxmax()
        ]
        results[stratum] = {
            "best_eta": float(best_row["eta"]),
            "phase1_cost": round(float(best_row["p1_cost_mean"]), 2),
            "phase2_gain": round(float(best_row["p2_gain_mean"]), 2),
            "net_total": round(float(best_row["p1_cost_mean"] + best_row["p2_gain_mean"]), 2),
            "n_per_eta": int(best_row["n"]),
        }

        label = STRATA_LABELS[stratum]
        r = results[stratum]
        print(f"  {label:15s}: best η={r['best_eta']:+.3f} → "
              f"P1 cost={r['phase1_cost']:+.1f}, P2 gain={r['phase2_gain']:+.1f}, "
              f"net={r['net_total']:+.1f}")

    fig.suptitle(f"B0.6 ({variant_label}): Phase 1 cost vs Phase 2 gain by eta (decay=0.01)",
                fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "F5_phase1_cost.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "F5_phase1_cost.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: F5_phase1_cost")

    return results


# ── Analysis 6: Headline Summary Figure ───────────────────────────────

def analysis_6_headline(df: pd.DataFrame, static_df: pd.DataFrame, output_dir: Path, variant_label: str = ""):
    """Four-panel headline figure."""
    print("\n" + "="*60)
    print("ANALYSIS 6: Headline Figure")
    print("="*60)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: 4-bar comparison — static ± plasticity vs non-stationary ± plasticity
    ax = fig.add_subplot(gs[0, 0])
    ns_baseline = df[df["eta"].abs() < 1e-12]
    actionable_strata = ["low_mid", "high_mid", "near_perfect", "perfect"]
    strata_data = []
    for s in actionable_strata:
        ns_sdf = ns_baseline[ns_baseline["stratum"] == s]
        if len(ns_sdf) == 0:
            continue
        n_nets = ns_sdf["network_id"].nunique()

        # Static baseline from B0.5+ (eta=0, same networks)
        static_base = float("nan")
        static_plastic = float("nan")
        if not static_df.empty:
            st_s = static_df[static_df["stratum"] == s]
            st_base = st_s[st_s["eta"].abs() < 1e-12]
            if len(st_base) > 0:
                static_base = float(st_base["baseline_avg_reward"].mean())
            # Static + best plasticity (best eta per stratum, decay=0.01)
            st_plastic = st_s[
                (st_s["eta"].abs() >= 1e-12) & (st_s["decay"].between(0.009, 0.011))
            ]
            if len(st_plastic) > 0:
                best_eta_st = st_plastic.groupby("eta")["plastic_avg_reward"].mean().idxmax()
                static_plastic = float(
                    st_plastic[st_plastic["eta"] == best_eta_st]["plastic_avg_reward"].mean()
                )

        # Non-stationary baseline (no plasticity)
        ns_val = float(ns_sdf["baseline_total_reward"].mean())

        # Non-stationary + best plasticity (best eta per stratum, decay=0.01)
        ns_plastic_sdf = df[
            (df["stratum"] == s)
            & (df["eta"].abs() >= 1e-12)
            & (df["decay"].between(0.009, 0.011))
        ]
        if len(ns_plastic_sdf) > 0:
            best_eta = ns_plastic_sdf.groupby("eta")["plastic_total_reward"].mean().idxmax()
            ns_plastic_val = float(
                ns_plastic_sdf[ns_plastic_sdf["eta"] == best_eta]["plastic_total_reward"].mean()
            )
        else:
            ns_plastic_val = ns_val

        strata_data.append({
            "stratum": STRATA_LABELS[s],
            "static_base": static_base,
            "static_plastic": static_plastic,
            "ns_base": ns_val,
            "ns_plastic": ns_plastic_val,
            "n": n_nets,
        })

    x = np.arange(len(strata_data))
    w = 0.19
    ax.bar(x - 1.5*w, [d["static_base"] for d in strata_data], w,
           label="Static", color="#81C784", edgecolor="black", linewidth=0.3)
    ax.bar(x - 0.5*w, [d["static_plastic"] for d in strata_data], w,
           label="Static + plasticity", color="#2E7D32", edgecolor="black", linewidth=0.3)
    ax.bar(x + 0.5*w, [d["ns_base"] for d in strata_data], w,
           label=f"NS ({variant_label})", color="#EF9A9A", edgecolor="black", linewidth=0.3)
    ax.bar(x + 1.5*w, [d["ns_plastic"] for d in strata_data], w,
           label=f"NS + plasticity", color="#C62828", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d['stratum']}\n(N={d['n']})" for d in strata_data], fontsize=8)
    ax.set_ylabel("Mean total reward (500 steps)")
    ax.set_title("A. Static vs non-stationary, ± plasticity")
    ax.legend(fontsize=6.5, loc="upper left")

    # Panel B: Best-eta Phase 2 delta (decay=0.01)
    ax = fig.add_subplot(gs[0, 1])
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
            "n": sdf["network_id"].nunique(),
        })

    colors = [STRATA_COLORS[s] for s in actionable if STRATA_LABELS[s] in [d["stratum"] for d in best_deltas]]
    bars = ax.bar(range(len(best_deltas)),
                  [d["delta"] for d in best_deltas],
                  color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(best_deltas)))
    ax.set_xticklabels([f"{d['stratum']}\nη={d['best_eta']:+.3f}" for d in best_deltas], fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Phase 2 Δreward")
    ax.set_title("B. Plasticity benefit after switch (best η)")
    for bar, d in zip(bars, best_deltas):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y,
               f"{d['delta']:+.1f}", ha="center",
               va="bottom" if y > 0 else "top", fontsize=9, fontweight="bold")

    # Panel C: Adaptation premium histogram (High-mid)
    ax = fig.add_subplot(gs[1, 0])
    if not static_df.empty:
        for target_stratum in ["high_mid"]:
            ns_s = df[df["stratum"] == target_stratum]
            st_s = static_df[static_df["stratum"] == target_stratum]

            ns_oracle = ns_s.groupby("network_id")["delta_reward_total"].max()
            st_oracle = st_s.groupby("network_id")["delta_reward"].max()
            common = sorted(set(ns_oracle.index) & set(st_oracle.index))

            if len(common) >= 5:
                premium = ns_oracle.loc[common].values - st_oracle.loc[common].values
                ax.hist(premium, bins=40, color=STRATA_COLORS[target_stratum],
                       alpha=0.7, edgecolor="black", linewidth=0.5)
                ax.axvline(0, color="black", linewidth=1, linestyle="--")
                ax.axvline(np.mean(premium), color="red", linewidth=2,
                          label=f"mean={np.mean(premium):+.1f}")
                _, p_val = stats.ttest_rel(
                    ns_oracle.loc[common].values, st_oracle.loc[common].values
                )
                pct_pos = (premium > 0).mean() * 100
                ax.set_title(f"C. Adaptation premium — {STRATA_LABELS[target_stratum]}\n"
                           f"(p={p_val:.4f}, {pct_pos:.0f}% positive)")
                ax.legend(fontsize=9)
    ax.set_xlabel("Oracle(nonstat) − Oracle(static)")
    ax.set_ylabel("Count")

    # Panel D: |dw| Phase 1 vs Phase 2
    ax = fig.add_subplot(gs[1, 1])
    plastic = df[(df["eta"].abs() >= 1e-12) & (df["decay"].between(0.009, 0.011))]
    dw_data = []
    for s in STRATA_ORDER:
        sdf = plastic[plastic["stratum"] == s]
        if len(sdf) > 0 and sdf["phase1_mean_abs_dw"].mean() > 1e-15:
            dw_data.append({
                "stratum": STRATA_LABELS[s],
                "P1": float(sdf["phase1_mean_abs_dw"].mean()),
                "P2": float(sdf["phase2_mean_abs_dw"].mean()),
            })
    if dw_data:
        x = range(len(dw_data))
        w = 0.35
        ax.bar([i - w/2 for i in x], [d["P1"] for d in dw_data], w,
               label="Phase 1", color="#4CAF50", alpha=0.8)
        ax.bar([i + w/2 for i in x], [d["P2"] for d in dw_data], w,
               label="Phase 2", color="#F44336", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([d["stratum"] for d in dw_data], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Mean |Δw| per step")
        ax.set_title("D. Plasticity activity (weight change magnitude)")
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))

    fig.suptitle(f"B0.6: Non-Stationary CartPole — {variant_label} at Step 200",
                fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "F0_headline.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "F0_headline.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: F0_headline")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B0.6 Non-Stationary CartPole Analysis")
    parser.add_argument(
        "--variant",
        default="heavy_pole",
        help="Non-stationarity variant (default: heavy_pole)",
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("experiments/B0.6/sweep"),
        help="Directory with sweep parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: derived from sweep-dir)",
    )
    args = parser.parse_args()

    variant = args.variant
    variant_label = VARIANT_LABELS.get(variant, variant)

    if args.output_dir is None:
        if "pilot" in str(args.sweep_dir):
            args.output_dir = Path("experiments/B0.6/analysis_pilot")
        else:
            args.output_dir = Path("experiments/B0.6/analysis")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ns_df = load_nonstationary_data(args.sweep_dir, variant)
    network_ids = set(ns_df["network_id"].unique())
    static_df = load_static_data(network_ids)

    # Run all analyses
    r1 = analysis_1_disruption(ns_df, args.output_dir)
    r2 = analysis_2_best_eta(ns_df, args.output_dir, variant_label)
    r3 = analysis_3_adaptation_premium(ns_df, static_df, args.output_dir, variant_label)
    r4 = analysis_4_plasticity_activity(ns_df, args.output_dir)
    r5 = analysis_5_per_network(ns_df, args.output_dir, variant_label)
    analysis_6_headline(ns_df, static_df, args.output_dir, variant_label)
    r7 = analysis_7_dw_vs_improvement(ns_df, args.output_dir, variant_label)
    r8 = analysis_8_phase1_cost(ns_df, args.output_dir, variant_label)

    # Save combined JSON
    all_results = {
        "experiment": "B0.6",
        "variant": variant,
        "switch_step": 200,
        "n_networks": len(network_ids),
        "disruption": r1,
        "best_eta": r2,
        "adaptation_premium": r3,
        "plasticity_activity": r4,
        "per_network": r5,
        "dw_vs_improvement": r7,
        "phase1_cost": r8,
    }

    with open(args.output_dir / "B0_6_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("ALL ANALYSES COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
