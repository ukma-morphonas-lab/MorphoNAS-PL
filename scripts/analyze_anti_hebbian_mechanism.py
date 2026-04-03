#!/usr/bin/env python3
"""
Analysis 3: Anti-Hebbian Mechanism Hypothesis Testing

Tests two hypotheses about why anti-Hebbian plasticity outperforms Hebbian:

Hypothesis A: Networks with higher initial weight targets are more susceptible
to co-activation collapse and benefit more from anti-Hebbian.

Hypothesis B: Networks with larger per-step |Δw| prefer anti-Hebbian more
strongly (Hebbian overshoots).

Figures:
  F0_hypothesis_a_weight_target  — Violin plots + logistic regression
  F1_hypothesis_b_delta_w        — Violin plots + quintile dose-response
  F2_combined_mechanism          — Forest plot of effect sizes per stratum

Usage:
  python scripts/analyze_anti_hebbian_mechanism.py
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from MorphoNAS_PL.analysis_utils import (
    STRATA_ORDER,
    STRATUM_COLORS,
    STRATUM_LABELS,
    apply_publication_style,
    load_sweep,
    save_figure,
)
from MorphoNAS_PL.genome_features import load_genome_features

logger = logging.getLogger(__name__)
apply_publication_style()

# ── Paths ────────────────────────────────────────────────────────────
B05_SWEEP_DIR = "experiments/B0.5/sweep"
B05P_SWEEP_DIR = "experiments/B0.5+/sweep"
B05P_POOL_DIR = "experiments/B0.5+/pool_subsample"
DEFAULT_OUTPUT_DIR = "experiments/B0.5+/analysis/mechanism"

NON_WEAK_STRATA = ["low_mid", "high_mid", "near_perfect", "perfect"]


# ── Data preparation ────────────────────────────────────────────────

def compute_anti_hebbian_preference(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Per-network: determine if optimal η is anti-Hebbian (η < 0)."""
    idx = sweep_df.groupby("network_id")["delta_reward"].idxmax()
    oracle = sweep_df.loc[idx, ["network_id", "stratum", "eta", "delta_reward"]].copy()
    oracle = oracle.rename(columns={"eta": "best_eta", "delta_reward": "best_delta_reward"})
    oracle["anti_hebbian"] = (oracle["best_eta"] < -1e-12).astype(int)
    oracle["improved"] = (oracle["best_delta_reward"] > 0).astype(int)
    return oracle.reset_index(drop=True)


# ── Hypothesis A: weight_adjustment_target ───────────────────────────

def test_hypothesis_a(
    oracle_df: pd.DataFrame,
    genome_df: pd.DataFrame,
) -> dict:
    """Test correlation between weight_adjustment_target and anti-Hebbian preference."""
    merged = oracle_df.merge(
        genome_df[["network_id", "weight_adjustment_target"]],
        on="network_id",
    )
    merged = merged.dropna(subset=["weight_adjustment_target"])
    # Only consider networks where optimal eta is nonzero
    merged = merged[merged["best_eta"].abs() > 1e-12]

    results = {"overall": {}, "per_stratum": {}}

    # Overall
    anti = merged[merged["anti_hebbian"] == 1]["weight_adjustment_target"]
    hebb = merged[merged["anti_hebbian"] == 0]["weight_adjustment_target"]

    if len(anti) > 0 and len(hebb) > 0:
        u_stat, u_p = stats.mannwhitneyu(anti, hebb, alternative="two-sided")
        r_pb, r_p = stats.pointbiserialr(
            merged["anti_hebbian"], merged["weight_adjustment_target"],
        )
        results["overall"] = {
            "n_anti": int(len(anti)),
            "n_hebb": int(len(hebb)),
            "mean_wt_anti": float(anti.mean()),
            "mean_wt_hebb": float(hebb.mean()),
            "mannwhitney_U": float(u_stat),
            "mannwhitney_p": float(u_p),
            "pointbiserial_r": float(r_pb),
            "pointbiserial_p": float(r_p),
        }

    # Per stratum
    for s in NON_WEAK_STRATA:
        sdf = merged[merged["stratum"] == s]
        anti_s = sdf[sdf["anti_hebbian"] == 1]["weight_adjustment_target"]
        hebb_s = sdf[sdf["anti_hebbian"] == 0]["weight_adjustment_target"]
        if len(anti_s) >= 5 and len(hebb_s) >= 5:
            u, p = stats.mannwhitneyu(anti_s, hebb_s, alternative="two-sided")
            rpb, rpb_p = stats.pointbiserialr(
                sdf["anti_hebbian"], sdf["weight_adjustment_target"],
            )
            # Cohen's d
            pooled_std = np.sqrt(
                ((len(anti_s) - 1) * anti_s.std()**2 + (len(hebb_s) - 1) * hebb_s.std()**2)
                / (len(anti_s) + len(hebb_s) - 2)
            )
            d = (anti_s.mean() - hebb_s.mean()) / pooled_std if pooled_std > 0 else 0
            results["per_stratum"][s] = {
                "n_anti": int(len(anti_s)),
                "n_hebb": int(len(hebb_s)),
                "mean_wt_anti": float(anti_s.mean()),
                "mean_wt_hebb": float(hebb_s.mean()),
                "cohens_d": float(d),
                "mannwhitney_p": float(p),
                "pointbiserial_r": float(rpb),
            }

    results["merged_df"] = merged
    return results


def figure_hypothesis_a(
    results: dict,
    fig_dir: str,
) -> None:
    """F0: Violin plots + per-stratum effect sizes for Hypothesis A."""
    merged = results.pop("merged_df")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Violin plots per stratum
    ax = axes[0]
    strata = [s for s in NON_WEAK_STRATA if s in merged["stratum"].unique()]
    positions = []
    pos = 0
    for s in strata:
        sdf = merged[merged["stratum"] == s]
        anti = sdf[sdf["anti_hebbian"] == 1]["weight_adjustment_target"].values
        hebb = sdf[sdf["anti_hebbian"] == 0]["weight_adjustment_target"].values

        if len(anti) > 1:
            vp1 = ax.violinplot([anti], positions=[pos], showmedians=True, widths=0.4)
            for body in vp1["bodies"]:
                body.set_facecolor("#E53935")
                body.set_alpha(0.6)
        if len(hebb) > 1:
            vp2 = ax.violinplot([hebb], positions=[pos + 0.5], showmedians=True, widths=0.4)
            for body in vp2["bodies"]:
                body.set_facecolor("#1E88E5")
                body.set_alpha(0.6)

        positions.append(pos + 0.25)
        pos += 1.5

    ax.set_xticks(positions)
    ax.set_xticklabels([STRATUM_LABELS.get(s, s) for s in strata], rotation=20, ha="right")
    ax.set_ylabel("Weight Adjustment Target")
    ax.set_title("(a) Weight Target by η Preference")
    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        [Patch(facecolor="#E53935", alpha=0.6), Patch(facecolor="#1E88E5", alpha=0.6)],
        ["Anti-Hebbian", "Hebbian"],
        loc="upper right",
    )

    # (b) Effect sizes (Cohen's d) per stratum
    ax = axes[1]
    ds = []
    labels = []
    for s in strata:
        if s in results["per_stratum"]:
            ds.append(results["per_stratum"][s]["cohens_d"])
            labels.append(STRATUM_LABELS.get(s, s))

    if ds:
        colors = [STRATUM_COLORS.get(s, "#999") for s in strata if s in results["per_stratum"]]
        bars = ax.barh(range(len(ds)), ds, color=colors)
        ax.set_yticks(range(len(ds)))
        ax.set_yticklabels(labels)
        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("Cohen's d (Anti-Hebb vs Hebb weight target)")
        ax.set_title("(b) Effect Size per Stratum")

    fig.suptitle("Hypothesis A: Weight Target → Anti-Hebbian Preference")
    fig.tight_layout()
    save_figure(fig, fig_dir, "F0_hypothesis_a_weight_target")


# ── Hypothesis B: |Δw| per step ──────────────────────────────────────

def test_hypothesis_b(sweep_df: pd.DataFrame) -> dict:
    """Test correlation between mean |Δw| and anti-Hebbian preference.

    Uses the full B0.5 sweep (50K networks).
    """
    # Per-network: get mean |Δw| across all active (eta != 0) settings
    active = sweep_df[sweep_df["eta"].abs() > 1e-12].copy()
    per_net_dw = (
        active.groupby(["network_id", "stratum"])["mean_abs_delta_per_step"]
        .mean()
        .reset_index()
        .rename(columns={"mean_abs_delta_per_step": "mean_dw"})
    )

    # Per-network oracle
    idx = sweep_df.groupby("network_id")["delta_reward"].idxmax()
    oracle = sweep_df.loc[idx, ["network_id", "eta"]].copy()
    oracle = oracle.rename(columns={"eta": "best_eta"})
    oracle["anti_hebbian"] = (oracle["best_eta"] < -1e-12).astype(int)

    merged = per_net_dw.merge(oracle[["network_id", "best_eta", "anti_hebbian"]], on="network_id")
    merged = merged[merged["stratum"].isin(NON_WEAK_STRATA)]
    # Only consider networks with nonzero optimal eta
    merged = merged[merged["best_eta"].abs() > 1e-12]

    results = {"overall": {}, "per_stratum": {}, "dose_response": {}}

    # Overall
    anti = merged[merged["anti_hebbian"] == 1]["mean_dw"]
    hebb = merged[merged["anti_hebbian"] == 0]["mean_dw"]

    if len(anti) > 0 and len(hebb) > 0:
        u_stat, u_p = stats.mannwhitneyu(anti, hebb, alternative="two-sided")
        r_pb, r_p = stats.pointbiserialr(merged["anti_hebbian"], merged["mean_dw"])
        results["overall"] = {
            "n_anti": int(len(anti)),
            "n_hebb": int(len(hebb)),
            "mean_dw_anti": float(anti.mean()),
            "mean_dw_hebb": float(hebb.mean()),
            "mannwhitney_U": float(u_stat),
            "mannwhitney_p": float(u_p),
            "pointbiserial_r": float(r_pb),
            "pointbiserial_p": float(r_p),
        }

    # Dose-response: quintile bins
    merged["dw_quintile"] = pd.qcut(merged["mean_dw"], 5, labels=False, duplicates="drop")
    dose = (
        merged.groupby("dw_quintile")
        .agg(
            mean_dw=("mean_dw", "mean"),
            anti_hebb_rate=("anti_hebbian", "mean"),
            n=("anti_hebbian", "count"),
        )
        .reset_index()
    )
    rho, rho_p = stats.spearmanr(dose["dw_quintile"], dose["anti_hebb_rate"])
    results["dose_response"] = {
        "quintiles": dose.to_dict("records"),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
    }

    # Per stratum
    for s in NON_WEAK_STRATA:
        sdf = merged[merged["stratum"] == s]
        anti_s = sdf[sdf["anti_hebbian"] == 1]["mean_dw"]
        hebb_s = sdf[sdf["anti_hebbian"] == 0]["mean_dw"]
        if len(anti_s) >= 5 and len(hebb_s) >= 5:
            u, p = stats.mannwhitneyu(anti_s, hebb_s, alternative="two-sided")
            pooled_std = np.sqrt(
                ((len(anti_s) - 1) * anti_s.std()**2 + (len(hebb_s) - 1) * hebb_s.std()**2)
                / (len(anti_s) + len(hebb_s) - 2)
            )
            d = (anti_s.mean() - hebb_s.mean()) / pooled_std if pooled_std > 0 else 0
            results["per_stratum"][s] = {
                "n_anti": int(len(anti_s)),
                "n_hebb": int(len(hebb_s)),
                "mean_dw_anti": float(anti_s.mean()),
                "mean_dw_hebb": float(hebb_s.mean()),
                "cohens_d": float(d),
                "mannwhitney_p": float(p),
            }

    results["merged_df"] = merged
    return results


def figure_hypothesis_b(
    results: dict,
    fig_dir: str,
) -> None:
    """F1: Violin plots + quintile dose-response for Hypothesis B."""
    merged = results.pop("merged_df")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Violin plots per stratum
    ax = axes[0]
    strata = [s for s in NON_WEAK_STRATA if s in merged["stratum"].unique()]
    positions = []
    pos = 0
    for s in strata:
        sdf = merged[merged["stratum"] == s]
        anti = sdf[sdf["anti_hebbian"] == 1]["mean_dw"].values
        hebb = sdf[sdf["anti_hebbian"] == 0]["mean_dw"].values

        if len(anti) > 1:
            vp1 = ax.violinplot([anti], positions=[pos], showmedians=True, widths=0.4)
            for body in vp1["bodies"]:
                body.set_facecolor("#E53935")
                body.set_alpha(0.6)
        if len(hebb) > 1:
            vp2 = ax.violinplot([hebb], positions=[pos + 0.5], showmedians=True, widths=0.4)
            for body in vp2["bodies"]:
                body.set_facecolor("#1E88E5")
                body.set_alpha(0.6)

        positions.append(pos + 0.25)
        pos += 1.5

    ax.set_xticks(positions)
    ax.set_xticklabels([STRATUM_LABELS.get(s, s) for s in strata], rotation=20, ha="right")
    ax.set_ylabel("Mean |Δw| per Step")
    ax.set_title("(a) |Δw| by η Preference")
    from matplotlib.patches import Patch
    ax.legend(
        [Patch(facecolor="#E53935", alpha=0.6), Patch(facecolor="#1E88E5", alpha=0.6)],
        ["Anti-Hebbian", "Hebbian"],
        loc="upper right",
    )

    # (b) Dose-response
    ax = axes[1]
    dose = results["dose_response"]["quintiles"]
    quintiles = [d["dw_quintile"] for d in dose]
    rates = [d["anti_hebb_rate"] for d in dose]
    ax.bar(quintiles, rates, color="#FF9800", edgecolor="black", width=0.7)
    ax.set_xlabel("|Δw| Quintile (1=lowest, 5=highest)")
    ax.set_ylabel("Anti-Hebbian Preference Rate")
    rho = results["dose_response"]["spearman_rho"]
    rho_p = results["dose_response"]["spearman_p"]
    ax.set_title(f"(b) Dose-Response (ρ={rho:.3f}, p={rho_p:.2e})")
    ax.axhline(0.5, color="k", ls="--", lw=1, alpha=0.5)

    fig.suptitle("Hypothesis B: |Δw| per Step → Anti-Hebbian Preference")
    fig.tight_layout()
    save_figure(fig, fig_dir, "F1_hypothesis_b_delta_w")


def figure_combined(
    results_a: dict,
    results_b: dict,
    fig_dir: str,
) -> None:
    """F2: Combined forest plot of effect sizes per stratum."""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    effects = []
    colors = []
    y_pos = 0
    positions = []

    for s in NON_WEAK_STRATA:
        # Hypothesis A
        if s in results_a.get("per_stratum", {}):
            d = results_a["per_stratum"][s]["cohens_d"]
            labels.append(f"{STRATUM_LABELS.get(s, s)} — Weight Target")
            effects.append(d)
            colors.append("#E53935")
            positions.append(y_pos)
            y_pos += 1

        # Hypothesis B
        if s in results_b.get("per_stratum", {}):
            d = results_b["per_stratum"][s]["cohens_d"]
            labels.append(f"{STRATUM_LABELS.get(s, s)} — |Δw|/step")
            effects.append(d)
            colors.append("#1E88E5")
            positions.append(y_pos)
            y_pos += 1

        y_pos += 0.5  # gap between strata

    if positions:
        ax.barh(positions, effects, color=colors, height=0.7, alpha=0.8)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("Cohen's d (Anti-Hebbian vs Hebbian)")
        ax.set_title("Anti-Hebbian Mechanism: Effect Sizes per Stratum")
        ax.invert_yaxis()

    fig.tight_layout()
    save_figure(fig, fig_dir, "F2_combined_mechanism")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analysis 3: Anti-Hebbian Mechanism Hypothesis Testing",
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

    # ── Hypothesis A ─────────────────────────────────────────────────
    print("=== HYPOTHESIS A: Weight Adjustment Target ===")
    print("Loading B0.5+ sweep...")
    b05p_sweep = load_sweep(B05P_SWEEP_DIR)
    b05p_sweep = b05p_sweep[b05p_sweep["stratum"].isin(NON_WEAK_STRATA)]

    print("Loading genome features...")
    genome_df = load_genome_features(B05P_POOL_DIR)

    print("Computing anti-Hebbian preference (B0.5+)...")
    oracle_a = compute_anti_hebbian_preference(b05p_sweep)

    print("Testing Hypothesis A...")
    results_a = test_hypothesis_a(oracle_a, genome_df)
    figure_hypothesis_a(results_a, fig_dir)

    # ── Hypothesis B ─────────────────────────────────────────────────
    print("\n=== HYPOTHESIS B: |Δw| per Step ===")
    print("Loading B0.5 sweep (may take a moment)...")
    b05_sweep = load_sweep(B05_SWEEP_DIR)
    b05_sweep = b05_sweep[b05_sweep["stratum"].isin(NON_WEAK_STRATA)]

    print("Testing Hypothesis B...")
    results_b = test_hypothesis_b(b05_sweep)
    figure_hypothesis_b(results_b, fig_dir)

    # ── Combined figure ──────────────────────────────────────────────
    print("\n--- Combined Effect Sizes ---")
    figure_combined(results_a, results_b, fig_dir)

    # ── Save summary ─────────────────────────────────────────────────
    summary = {
        "hypothesis_a": {k: v for k, v in results_a.items() if k != "merged_df"},
        "hypothesis_b": {k: v for k, v in results_b.items() if k != "merged_df"},
    }
    summary_path = os.path.join(args.output_dir, "mechanism_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print text summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ANTI-HEBBIAN MECHANISM SUMMARY")
    print(f"{'='*60}")

    ov_a = results_a.get("overall", {})
    if ov_a:
        print(f"\n  Hypothesis A (weight target → anti-Hebbian):")
        print(f"    Anti-Hebb mean wt: {ov_a.get('mean_wt_anti', 'N/A'):.3f}")
        print(f"    Hebbian mean wt:   {ov_a.get('mean_wt_hebb', 'N/A'):.3f}")
        print(f"    Point-biserial r:  {ov_a.get('pointbiserial_r', 'N/A'):.3f} "
              f"(p={ov_a.get('pointbiserial_p', 'N/A'):.2e})")

    ov_b = results_b.get("overall", {})
    if ov_b:
        print(f"\n  Hypothesis B (|Δw| → anti-Hebbian):")
        print(f"    Anti-Hebb mean |Δw|: {ov_b.get('mean_dw_anti', 'N/A'):.6f}")
        print(f"    Hebbian mean |Δw|:   {ov_b.get('mean_dw_hebb', 'N/A'):.6f}")
        print(f"    Point-biserial r:    {ov_b.get('pointbiserial_r', 'N/A'):.3f} "
              f"(p={ov_b.get('pointbiserial_p', 'N/A'):.2e})")

    dr = results_b.get("dose_response", {})
    if dr:
        print(f"\n    Dose-response: ρ={dr.get('spearman_rho', 'N/A'):.3f} "
              f"(p={dr.get('spearman_p', 'N/A'):.2e})")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
