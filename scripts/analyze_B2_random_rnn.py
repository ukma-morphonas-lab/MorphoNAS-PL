#!/usr/bin/env python3
"""
B2 Random RNN Control Experiment Analysis
==========================================
Reads B2 random RNN results and compares them to MorphoNAS B0.5 results.

Analyses:
  1. Baseline competence rate — % of random RNNs that are competent (non-Weak)
     vs MorphoNAS competence rate.
  2. Oracle improvement — Per competent random RNN, find best (eta, decay) and
     compute oracle delta_reward. Compare distribution to MorphoNAS.
  3. Regret comparison — Best fixed (eta, decay) vs per-network oracle for
     random RNNs. Compare regret to MorphoNAS regret (52-100%).
  4. Anti-Hebbian dominance — For each competent random RNN, is optimal eta
     negative? Cohen's d (anti-Hebbian vs Hebbian mean delta) per stratum.
     Compare to MorphoNAS d=0.53-0.64.
  5. Topology-dependence — Is plasticity benefit concentrated in
     competent-but-imperfect strata (same pattern as MorphoNAS)?

Outputs:
  figures/  — PDF + PNG figures
  tables/   — CSV tables
  summary_stats.json — machine-readable summary

Usage:
  python scripts/analyze_B2_random_rnn.py \\
      --b2-pool experiments/B2/pool/random_rnn_pool.jsonl \\
      --b2-sweep experiments/B2/sweep/random_rnn_sweep.jsonl \\
      --b05-sweep-dir experiments/B0.5/sweep \\
      --output-dir experiments/B2/analysis
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

# Allow imports from the code/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from MorphoNAS_PL.experimentB0_5_natural import (
    DECAY_VALUES,
    ETA_VALUES,
    Stratum,
    get_stratum,
    get_stratum_label,
)
from MorphoNAS_PL.analysis_utils import (
    STRATA_ORDER,
    STRATUM_LABELS,
    STRATUM_COLORS,
    apply_publication_style,
    fmt_eta,
    fmt_decay,
    save_figure,
)

logger = logging.getLogger(__name__)

apply_publication_style()

# ── Constants ────────────────────────────────────────────────────────

NON_WEAK_STRATA = ["low_mid", "high_mid", "near_perfect", "perfect"]

CLR_B2 = "#CC6677"      # rose for random RNN
CLR_B05 = "#4477AA"     # blue for MorphoNAS
CLR_ANTI = "#4477AA"    # anti-Hebbian
CLR_HEBB = "#CC6677"    # Hebbian
CLR_ORACLE = "#EE6677"  # oracle bar
CLR_FIXED = "#4477AA"   # best-fixed bar

# CartPole ceiling
CARTPOLE_CEILING = 500.0


# ── Data loading ─────────────────────────────────────────────────────

def load_pool_jsonl(path: str) -> pd.DataFrame:
    """Load B2 pool JSONL into a DataFrame."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append({
                    "source_network_id": rec["source_network_id"],
                    "random_seed": rec["random_seed"],
                    "num_neurons": rec.get("num_neurons", 0),
                    "num_connections": rec.get("num_connections", 0),
                    "valid": rec.get("valid", False),
                    "baseline_reward": rec.get("baseline_reward", 0.0),
                    "baseline_fitness": rec.get("baseline_fitness", 0.0),
                    "stratum": rec.get("stratum", "weak"),
                })
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Skipping malformed pool line: {e}")
                continue
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df):,} pool records from {path}")
    return df


def load_sweep_jsonl(path: str) -> pd.DataFrame:
    """Load B2 sweep JSONL into a DataFrame."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append({
                    "source_network_id": rec["source_network_id"],
                    "random_seed": rec["random_seed"],
                    "num_neurons": rec.get("num_neurons", 0),
                    "num_connections": rec.get("num_connections", 0),
                    "eta": rec["eta"],
                    "decay": rec["decay"],
                    "baseline_reward": rec["baseline_reward"],
                    "plastic_reward": rec["plastic_reward"],
                    "delta_reward": rec["delta_reward"],
                    "improved": rec.get("improved", False),
                    "plastic_fitness": rec.get("plastic_fitness", 0.0),
                    "mean_abs_delta_per_step": rec.get("mean_abs_delta_per_step", 0.0),
                })
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Skipping malformed sweep line: {e}")
                continue
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df):,} sweep records from {path}")
    return df


def load_b05_sweep(sweep_dir: str) -> pd.DataFrame | None:
    """Load B0.5 MorphoNAS sweep for comparison (parquet format)."""
    from MorphoNAS_PL.analysis_utils import load_sweep
    try:
        df = load_sweep(sweep_dir)
        return df
    except FileNotFoundError:
        logger.warning(f"B0.5 sweep not found at {sweep_dir}, skipping comparison")
        return None


# ── Unique network identifier ────────────────────────────────────────

def add_network_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a unique network_id column combining source + random seed."""
    df = df.copy()
    df["network_id"] = (
        df["source_network_id"].astype(str) + "_" + df["random_seed"].astype(str)
    )
    return df


def assign_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Assign stratum based on baseline_reward if not already present."""
    df = df.copy()
    if "stratum" not in df.columns:
        df["stratum"] = df["baseline_reward"].apply(lambda r: get_stratum(r).value)
    return df


# =====================================================================
# Analysis 1 — Baseline competence rate
# =====================================================================

def analysis_1_competence(
    pool_df: pd.DataFrame,
    fig_dir: str,
    tbl_dir: str,
    b05_sweep_df: pd.DataFrame | None = None,
) -> dict:
    """Compare competence rates: random RNN pool vs MorphoNAS."""
    logger.info("Analysis 1: Baseline competence rate")

    valid_pool = pool_df[pool_df["valid"]].copy()
    total_valid = len(valid_pool)

    # Stratum distribution
    stratum_counts = {}
    for s in STRATA_ORDER:
        count = (valid_pool["stratum"] == s).sum()
        stratum_counts[s] = int(count)

    competent = valid_pool[valid_pool["stratum"] != "weak"]
    competent_rate = 100.0 * len(competent) / total_valid if total_valid > 0 else 0.0
    weak_rate = 100.0 - competent_rate

    results = {
        "total_valid": total_valid,
        "total_invalid": int((~pool_df["valid"]).sum()),
        "competent_count": len(competent),
        "competent_rate_pct": float(competent_rate),
        "weak_rate_pct": float(weak_rate),
        "stratum_counts": stratum_counts,
    }

    # MorphoNAS comparison (if available)
    b05_stratum_counts = None
    if b05_sweep_df is not None:
        col = "stratum"
        if col in b05_sweep_df.columns:
            b05_nets = b05_sweep_df.groupby("network_id")[col].first()
            b05_total = len(b05_nets)
            b05_competent = (b05_nets != "weak").sum()
            b05_competent_rate = 100.0 * b05_competent / b05_total if b05_total > 0 else 0.0
            b05_stratum_counts = {}
            for s in STRATA_ORDER:
                b05_stratum_counts[s] = int((b05_nets == s).sum())
            results["b05_total"] = int(b05_total)
            results["b05_competent_count"] = int(b05_competent)
            results["b05_competent_rate_pct"] = float(b05_competent_rate)
            results["b05_stratum_counts"] = b05_stratum_counts

    # Table
    rows = []
    for s in STRATA_ORDER:
        row = {
            "Stratum": STRATUM_LABELS[s],
            "B2 Count": stratum_counts[s],
            "B2 %": f"{100 * stratum_counts[s] / total_valid:.1f}%" if total_valid > 0 else "0%",
        }
        if b05_stratum_counts is not None:
            b05_total = results["b05_total"]
            row["B0.5 Count"] = b05_stratum_counts[s]
            row["B0.5 %"] = f"{100 * b05_stratum_counts[s] / b05_total:.1f}%" if b05_total > 0 else "0%"
        rows.append(row)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(tbl_dir, "T1_competence_rate.csv"), index=False)

    # Figure: side-by-side stratum distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(STRATA_ORDER))
    width = 0.35

    b2_pcts = [100 * stratum_counts[s] / total_valid if total_valid > 0 else 0 for s in STRATA_ORDER]
    bars_b2 = ax.bar(x - width / 2, b2_pcts, width, color=CLR_B2,
                     edgecolor="black", linewidth=0.5, label="Random RNN (B2)")

    if b05_stratum_counts is not None:
        b05_total = results["b05_total"]
        b05_pcts = [100 * b05_stratum_counts[s] / b05_total if b05_total > 0 else 0 for s in STRATA_ORDER]
        bars_b05 = ax.bar(x + width / 2, b05_pcts, width, color=CLR_B05,
                          edgecolor="black", linewidth=0.5, label="MorphoNAS (B0.5)")
        for i, (v2, v5) in enumerate(zip(b2_pcts, b05_pcts)):
            ax.text(i - width / 2, v2 + 0.5, f"{v2:.0f}%", ha="center", va="bottom", fontsize=8)
            ax.text(i + width / 2, v5 + 0.5, f"{v5:.0f}%", ha="center", va="bottom", fontsize=8)
    else:
        for i, v in enumerate(b2_pcts):
            ax.text(i - width / 2, v + 0.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    labels = [STRATUM_LABELS[s].split(" [")[0] for s in STRATA_ORDER]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("% of networks")
    ax.set_title("Stratum Distribution: Random RNN vs MorphoNAS")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fig_dir, "F1_competence_rate")

    return results


# =====================================================================
# Analysis 2 — Oracle improvement
# =====================================================================

def analysis_2_oracle_improvement(
    sweep_df: pd.DataFrame,
    fig_dir: str,
    tbl_dir: str,
    b05_sweep_df: pd.DataFrame | None = None,
) -> dict:
    """Per-network oracle delta_reward for random RNNs vs MorphoNAS."""
    logger.info("Analysis 2: Oracle improvement")

    # Assign strata to sweep records
    sweep_df = assign_strata(sweep_df)

    # Compute per-network oracle
    oracle = sweep_df.groupby("network_id").agg(
        oracle_delta_reward=("delta_reward", "max"),
        baseline_reward=("baseline_reward", "first"),
        stratum=("stratum", "first"),
    ).reset_index()

    results = {"per_stratum": {}}
    rows = []

    for s in STRATA_ORDER:
        sub = oracle[oracle["stratum"] == s]
        if sub.empty:
            continue

        n = len(sub)
        oracle_vals = sub["oracle_delta_reward"].values
        improved = (oracle_vals > 0).sum()
        improved_pct = 100.0 * improved / n

        results["per_stratum"][s] = {
            "n_networks": int(n),
            "oracle_mean": float(np.mean(oracle_vals)),
            "oracle_median": float(np.median(oracle_vals)),
            "oracle_std": float(np.std(oracle_vals, ddof=1)) if n > 1 else 0.0,
            "improved_count": int(improved),
            "improved_pct": float(improved_pct),
        }

        rows.append({
            "Stratum": STRATUM_LABELS[s],
            "N": n,
            "Oracle mean delta_r": f"{np.mean(oracle_vals):+.1f}",
            "Oracle median": f"{np.median(oracle_vals):+.1f}",
            "% improved": f"{improved_pct:.1f}%",
        })

    # B0.5 comparison
    if b05_sweep_df is not None:
        results["b05_per_stratum"] = {}
        b05_oracle = b05_sweep_df.groupby("network_id").agg(
            oracle_delta=("delta_reward", "max"),
            baseline=("baseline_avg_reward", "first") if "baseline_avg_reward" in b05_sweep_df.columns
            else ("baseline_reward", "first"),
            stratum=("stratum", "first"),
        ).reset_index()

        for s in STRATA_ORDER:
            sub = b05_oracle[b05_oracle["stratum"] == s]
            if sub.empty:
                continue
            vals = sub["oracle_delta"].values
            results["b05_per_stratum"][s] = {
                "n_networks": int(len(sub)),
                "oracle_mean": float(np.mean(vals)),
                "oracle_median": float(np.median(vals)),
                "improved_pct": float(100.0 * (vals > 0).sum() / len(sub)),
            }

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(tbl_dir, "T2_oracle_improvement.csv"), index=False)

    # Figure: oracle delta_reward distribution per stratum (violin + box)
    strata_present = [s for s in STRATA_ORDER if s in results["per_stratum"]]
    fig, ax = plt.subplots(figsize=(10, 5.5))

    violin_data = []
    positions = []
    for i, s in enumerate(strata_present):
        sub = oracle[oracle["stratum"] == s]["oracle_delta_reward"].values
        violin_data.append(sub)
        positions.append(i)

    if violin_data:
        parts = ax.violinplot(violin_data, positions=positions, showmedians=True,
                              showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(CLR_B2)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")

        # Overlay box plots
        bp = ax.boxplot(violin_data, positions=positions, widths=0.15,
                        patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(CLR_B2)
            patch.set_alpha(0.4)

    ax.axhline(0, color="black", lw=0.5, ls="--")
    labels = [STRATUM_LABELS[s].split(" [")[0] for s in strata_present]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Oracle delta_reward")
    ax.set_title("Random RNN: Per-Network Oracle Improvement by Stratum")

    # Annotate mean values
    for i, s in enumerate(strata_present):
        mean_val = results["per_stratum"][s]["oracle_mean"]
        ax.text(i, ax.get_ylim()[1] * 0.95, f"mean={mean_val:+.1f}",
                ha="center", va="top", fontsize=8, fontweight="bold")

    fig.tight_layout()
    save_figure(fig, fig_dir, "F2_oracle_improvement")

    # If B0.5 is available, make a comparison figure
    if b05_sweep_df is not None and "b05_per_stratum" in results:
        _figure_oracle_comparison(oracle, results, strata_present, fig_dir)

    return results


def _figure_oracle_comparison(
    b2_oracle: pd.DataFrame,
    results: dict,
    strata_present: list[str],
    fig_dir: str,
) -> None:
    """Side-by-side bar chart: B2 vs B0.5 oracle mean delta_reward."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(strata_present))
    width = 0.35

    b2_means = [results["per_stratum"][s]["oracle_mean"] for s in strata_present]
    b05_means = [results["b05_per_stratum"].get(s, {}).get("oracle_mean", 0.0)
                 for s in strata_present]

    ax.bar(x - width / 2, b2_means, width, color=CLR_B2,
           edgecolor="black", linewidth=0.5, label="Random RNN (B2)")
    ax.bar(x + width / 2, b05_means, width, color=CLR_B05,
           edgecolor="black", linewidth=0.5, label="MorphoNAS (B0.5)")

    ax.axhline(0, color="black", lw=0.5)
    labels = [STRATUM_LABELS[s].split(" [")[0] for s in strata_present]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Oracle mean delta_reward")
    ax.set_title("Oracle Improvement: Random RNN vs MorphoNAS")
    ax.legend()

    for i in range(len(strata_present)):
        ax.text(i - width / 2, b2_means[i] + 0.5, f"{b2_means[i]:+.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(i + width / 2, b05_means[i] + 0.5, f"{b05_means[i]:+.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    save_figure(fig, fig_dir, "F2b_oracle_comparison")


# =====================================================================
# Analysis 3 — Regret comparison
# =====================================================================

def analysis_3_regret(
    sweep_df: pd.DataFrame,
    fig_dir: str,
    tbl_dir: str,
    b05_sweep_df: pd.DataFrame | None = None,
) -> dict:
    """Regret: best fixed (eta, decay) vs per-network oracle."""
    logger.info("Analysis 3: Regret comparison")

    sweep_df = assign_strata(sweep_df)

    # Per-network oracle
    net_oracle = sweep_df.groupby("network_id")["delta_reward"].max()
    net_stratum = sweep_df.groupby("network_id")["stratum"].first()

    # Pivot: network_id x (eta, decay) -> delta_reward
    net_param = sweep_df.pivot_table(
        values="delta_reward",
        index="network_id",
        columns=["eta", "decay"],
        aggfunc="mean",
    )

    # Regret = oracle - fixed_param_delta (per network)
    regret_df = net_param.sub(net_oracle, axis=0).mul(-1)

    # Best fixed parameter overall (minimizes mean regret)
    mean_regret = regret_df.mean()
    best_fixed_idx = mean_regret.idxmin()
    best_fixed_regret = mean_regret.min()

    results = {"per_stratum": {}}
    rows = []

    for s in STRATA_ORDER:
        nets = net_stratum[net_stratum == s].index
        if len(nets) == 0:
            continue

        oracle_s = net_oracle.loc[nets]
        oracle_mean = float(oracle_s.mean())

        # Best fixed for this stratum
        regret_s = regret_df.loc[nets]
        mean_reg_s = regret_s.mean()
        best_fixed_s = mean_reg_s.idxmin()
        best_reg_s = float(mean_reg_s.min())

        # Best fixed delta (what the best fixed param achieves on average)
        best_fixed_delta = float(net_param.loc[nets, best_fixed_s].mean())

        # Regret percentage
        regret_pct = 100.0 * best_reg_s / oracle_mean if oracle_mean > 1e-8 else 0.0

        results["per_stratum"][s] = {
            "n_networks": int(len(nets)),
            "oracle_mean_delta": oracle_mean,
            "best_fixed_param": {"eta": float(best_fixed_s[0]), "decay": float(best_fixed_s[1])},
            "best_fixed_delta": best_fixed_delta,
            "regret": best_reg_s,
            "regret_pct": float(regret_pct),
        }

        rows.append({
            "Stratum": STRATUM_LABELS[s],
            "N": len(nets),
            "Oracle mean delta_r": f"{oracle_mean:+.1f}",
            f"Best fixed (eta, decay)": f"({fmt_eta(best_fixed_s[0])}, {fmt_decay(best_fixed_s[1])})",
            "Best fixed delta_r": f"{best_fixed_delta:+.1f}",
            "Regret": f"{best_reg_s:.1f}",
            "Regret %": f"{regret_pct:.1f}%",
        })

    results["overall_best_fixed"] = {
        "param": {"eta": float(best_fixed_idx[0]), "decay": float(best_fixed_idx[1])},
        "mean_regret": float(best_fixed_regret),
    }

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(tbl_dir, "T3_regret.csv"), index=False)

    # Figure: oracle vs best-fixed bar chart with regret annotation
    strata_present = [s for s in STRATA_ORDER if s in results["per_stratum"]]
    labels = [STRATUM_LABELS[s].split(" [")[0] for s in strata_present]
    x = np.arange(len(strata_present))
    width = 0.35

    oracle_vals = [results["per_stratum"][s]["oracle_mean_delta"] for s in strata_present]
    fixed_vals = [results["per_stratum"][s]["best_fixed_delta"] for s in strata_present]
    regret_pcts = [results["per_stratum"][s]["regret_pct"] for s in strata_present]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.bar(x - width / 2, fixed_vals, width, color=CLR_FIXED,
           edgecolor="black", linewidth=0.5, label="Best fixed parameter")
    ax.bar(x + width / 2, oracle_vals, width, color=CLR_ORACLE,
           edgecolor="black", linewidth=0.5, label="Per-network oracle")

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean delta_reward")
    ax.set_title("Regret: Best Fixed vs Per-Network Oracle (Random RNN)")
    ax.legend()

    for i in range(len(strata_present)):
        f_val, o_val, r_pct = fixed_vals[i], oracle_vals[i], regret_pcts[i]
        y_top = max(f_val, o_val)
        ax.text(i - width / 2, f_val + 0.5, f"{f_val:+.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(i + width / 2, o_val + 0.5, f"{o_val:+.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
        # Regret percentage annotation
        ax.text(i, y_top + 3, f"regret={r_pct:.0f}%",
                ha="center", va="bottom", fontsize=7, color="red", fontweight="bold")

    fig.tight_layout()
    save_figure(fig, fig_dir, "F3_regret")

    return results


# =====================================================================
# Analysis 4 — Anti-Hebbian dominance
# =====================================================================

def analysis_4_anti_hebbian(
    sweep_df: pd.DataFrame,
    fig_dir: str,
    tbl_dir: str,
    b05_sweep_df: pd.DataFrame | None = None,
) -> dict:
    """Anti-Hebbian vs Hebbian comparison with Cohen's d per stratum."""
    logger.info("Analysis 4: Anti-Hebbian dominance")

    sweep_df = assign_strata(sweep_df)

    # --- Part A: Per-network optimal eta sign ---
    oracle_idx = sweep_df.groupby("network_id")["delta_reward"].idxmax()
    oracle_rows = sweep_df.loc[oracle_idx, ["network_id", "stratum", "eta", "delta_reward"]].copy()
    oracle_rows = oracle_rows.rename(columns={"eta": "best_eta", "delta_reward": "oracle_delta"})
    oracle_rows["anti_hebbian"] = oracle_rows["best_eta"] < -1e-12
    oracle_rows["hebbian"] = oracle_rows["best_eta"] > 1e-12
    oracle_rows["zero_eta"] = oracle_rows["best_eta"].abs() < 1e-12

    # --- Part B: Cohen's d (anti-Hebbian vs Hebbian mean delta per stratum) ---
    df_nonzero = sweep_df[sweep_df["eta"].abs() > 1e-12].copy()
    df_nonzero["sign"] = np.where(df_nonzero["eta"] < 0, "anti-Hebbian", "Hebbian")

    results = {"per_stratum": {}, "oracle_eta_sign": {}}
    rows_cohens = []
    rows_sign = []

    for s in STRATA_ORDER:
        # Oracle sign distribution
        sub_oracle = oracle_rows[oracle_rows["stratum"] == s]
        if sub_oracle.empty:
            continue

        n = len(sub_oracle)
        n_anti = int(sub_oracle["anti_hebbian"].sum())
        n_hebb = int(sub_oracle["hebbian"].sum())
        n_zero = int(sub_oracle["zero_eta"].sum())

        results["oracle_eta_sign"][s] = {
            "n_networks": n,
            "n_anti_hebbian": n_anti,
            "n_hebbian": n_hebb,
            "n_zero": n_zero,
            "pct_anti_hebbian": float(100 * n_anti / n) if n > 0 else 0,
        }

        rows_sign.append({
            "Stratum": STRATUM_LABELS[s],
            "N": n,
            "Anti-Hebbian": f"{n_anti} ({100*n_anti/n:.0f}%)" if n > 0 else "0",
            "Hebbian": f"{n_hebb} ({100*n_hebb/n:.0f}%)" if n > 0 else "0",
            "Zero": f"{n_zero} ({100*n_zero/n:.0f}%)" if n > 0 else "0",
        })

        # Cohen's d
        sub_nz = df_nonzero[df_nonzero["stratum"] == s]
        anti_vals = sub_nz[sub_nz["sign"] == "anti-Hebbian"]["delta_reward"].values
        hebb_vals = sub_nz[sub_nz["sign"] == "Hebbian"]["delta_reward"].values

        if len(anti_vals) < 2 or len(hebb_vals) < 2:
            continue

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(anti_vals, hebb_vals, equal_var=False)

        # Mann-Whitney U
        u_stat, u_pval = stats.mannwhitneyu(anti_vals, hebb_vals, alternative="two-sided")

        # Cohen's d (pooled SD)
        n_a, n_h = len(anti_vals), len(hebb_vals)
        pooled_sd = np.sqrt(
            (np.var(anti_vals, ddof=1) * (n_a - 1) + np.var(hebb_vals, ddof=1) * (n_h - 1))
            / (n_a + n_h - 2)
        )
        cohens_d = (np.mean(anti_vals) - np.mean(hebb_vals)) / pooled_sd if pooled_sd > 0 else 0.0
        winner = "anti-Hebbian" if np.mean(anti_vals) > np.mean(hebb_vals) else "Hebbian"

        results["per_stratum"][s] = {
            "anti_mean_delta": float(np.mean(anti_vals)),
            "anti_se": float(stats.sem(anti_vals)),
            "hebb_mean_delta": float(np.mean(hebb_vals)),
            "hebb_se": float(stats.sem(hebb_vals)),
            "t_stat": float(t_stat),
            "t_pvalue": float(p_val),
            "u_stat": float(u_stat),
            "u_pvalue": float(u_pval),
            "cohens_d": float(cohens_d),
            "winner": winner,
            "n_anti_obs": n_a,
            "n_hebb_obs": n_h,
        }

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        rows_cohens.append({
            "Stratum": STRATUM_LABELS[s],
            "Anti-Hebb mean +/- SE": f"{np.mean(anti_vals):+.2f} +/- {stats.sem(anti_vals):.2f}",
            "Hebbian mean +/- SE": f"{np.mean(hebb_vals):+.2f} +/- {stats.sem(hebb_vals):.2f}",
            "t": f"{t_stat:.2f}",
            "p (Welch)": f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}",
            "U": f"{u_stat:.0f}",
            "p (MW)": f"{u_pval:.2e}" if u_pval < 0.001 else f"{u_pval:.4f}",
            "Cohen's d": f"{cohens_d:.3f}",
            "Sig": sig,
            "Winner": winner,
        })

    pd.DataFrame(rows_sign).to_csv(os.path.join(tbl_dir, "T4a_oracle_eta_sign.csv"), index=False)
    pd.DataFrame(rows_cohens).to_csv(os.path.join(tbl_dir, "T4b_cohens_d.csv"), index=False)

    # Figure: 2-panel — (a) anti- vs Hebbian mean delta bar, (b) Cohen's d horizontal bars
    strata_present = [s for s in STRATA_ORDER if s in results["per_stratum"]]
    if not strata_present:
        logger.warning("No strata with sufficient data for anti-Hebbian analysis")
        return results

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(strata_present))
    width = 0.35
    labels = [STRATUM_LABELS[s].split(" [")[0] for s in strata_present]

    # Panel (a): mean delta bars
    ax = axes[0]
    anti_means = [results["per_stratum"][s]["anti_mean_delta"] for s in strata_present]
    anti_cis = [results["per_stratum"][s]["anti_se"] * 1.96 for s in strata_present]
    hebb_means = [results["per_stratum"][s]["hebb_mean_delta"] for s in strata_present]
    hebb_cis = [results["per_stratum"][s]["hebb_se"] * 1.96 for s in strata_present]

    ax.bar(x - width / 2, anti_means, width, yerr=anti_cis, capsize=3,
           color=CLR_ANTI, alpha=0.8, label="Anti-Hebbian (eta < 0)")
    ax.bar(x + width / 2, hebb_means, width, yerr=hebb_cis, capsize=3,
           color=CLR_HEBB, alpha=0.8, label="Hebbian (eta > 0)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean delta_reward (95% CI)")
    ax.set_title("Anti-Hebbian vs Hebbian (Random RNN)")
    ax.legend(fontsize=9)

    for i, s in enumerate(strata_present):
        p = results["per_stratum"][s]["t_pvalue"]
        marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if marker:
            ymax = max(anti_means[i] + anti_cis[i], hebb_means[i] + hebb_cis[i])
            ax.text(i, ymax + 1, marker, ha="center", fontsize=12, fontweight="bold")

    ax.text(-0.08, 1.02, "(a)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")

    # Panel (b): Cohen's d horizontal bars
    ax2 = axes[1]
    ds = [results["per_stratum"][s]["cohens_d"] for s in strata_present]
    colors_d = [CLR_ANTI if d > 0 else CLR_HEBB for d in ds]

    ax2.barh(x, ds, color=colors_d, edgecolor="black", alpha=0.8)
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Cohen's d (anti-Hebbian - Hebbian)")
    ax2.set_title("Effect Size by Stratum (Random RNN)")
    ax2.axvline(0, color="black", lw=0.5)
    for val, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax2.axvline(val, color="gray", ls=":", lw=0.7, alpha=0.5)
        ax2.axvline(-val, color="gray", ls=":", lw=0.7, alpha=0.5)

    # Annotate with d values
    for i, d in enumerate(ds):
        offset = 0.02 if d >= 0 else -0.02
        ha = "left" if d >= 0 else "right"
        ax2.text(d + offset, i, f"{d:.3f}", ha=ha, va="center", fontsize=9, fontweight="bold")

    ax2.text(-0.08, 1.02, "(b)", transform=ax2.transAxes,
             fontsize=13, fontweight="bold", va="bottom")

    fig.tight_layout()
    save_figure(fig, fig_dir, "F4_anti_hebbian")

    return results


# =====================================================================
# Analysis 5 — Topology-dependence (stratum pattern)
# =====================================================================

def analysis_5_topology_dependence(
    sweep_df: pd.DataFrame,
    fig_dir: str,
    tbl_dir: str,
    b05_sweep_df: pd.DataFrame | None = None,
) -> dict:
    """Is plasticity benefit concentrated in competent-but-imperfect strata?"""
    logger.info("Analysis 5: Topology-dependence of plasticity benefit")

    sweep_df = assign_strata(sweep_df)

    results = {"per_stratum": {}}
    rows = []

    for s in STRATA_ORDER:
        sub = sweep_df[sweep_df["stratum"] == s]
        if sub.empty:
            continue

        n_networks = sub["network_id"].nunique()

        # Oracle
        oracle_per_net = sub.groupby("network_id")["delta_reward"].max()
        oracle_mean = float(oracle_per_net.mean())
        oracle_improved_pct = float(100 * (oracle_per_net > 0).mean())

        # Best fixed
        cell_means = sub.groupby(["eta", "decay"])["delta_reward"].mean()
        if cell_means.empty:
            continue
        best_param = cell_means.idxmax()
        best_sub = sub[(sub["eta"] == best_param[0]) & (sub["decay"] == best_param[1])]
        best_mean_delta = float(best_sub["delta_reward"].mean())
        best_improved_pct = float(100 * (best_sub["delta_reward"] > 0).mean())

        # Headroom: how much room to improve to ceiling
        baseline_mean = float(sub.groupby("network_id")["baseline_reward"].first().mean())
        headroom = CARTPOLE_CEILING - baseline_mean

        # Normalized improvement: oracle / headroom
        norm_improvement = oracle_mean / headroom if headroom > 1e-3 else 0.0

        results["per_stratum"][s] = {
            "n_networks": int(n_networks),
            "baseline_mean": baseline_mean,
            "headroom": float(headroom),
            "oracle_mean_delta": oracle_mean,
            "oracle_improved_pct": oracle_improved_pct,
            "best_fixed_mean_delta": best_mean_delta,
            "best_fixed_improved_pct": best_improved_pct,
            "best_fixed_param": {"eta": float(best_param[0]), "decay": float(best_param[1])},
            "normalized_improvement": float(norm_improvement),
        }

        rows.append({
            "Stratum": STRATUM_LABELS[s],
            "N": n_networks,
            "Baseline mean": f"{baseline_mean:.1f}",
            "Headroom": f"{headroom:.1f}",
            "Oracle mean delta_r": f"{oracle_mean:+.1f}",
            "Norm improvement": f"{norm_improvement:.3f}",
            "% improved (oracle)": f"{oracle_improved_pct:.1f}%",
            "Best fixed delta_r": f"{best_mean_delta:+.1f}",
            "% improved (fixed)": f"{best_improved_pct:.1f}%",
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(tbl_dir, "T5_topology_dependence.csv"), index=False)

    # Figure: 3-panel
    # (a) Oracle mean delta by stratum (bar) — shows benefit concentration
    # (b) Normalized improvement by stratum
    # (c) Comparison with B0.5 if available
    strata_present = [s for s in STRATA_ORDER if s in results["per_stratum"]]
    labels = [STRATUM_LABELS[s].split(" [")[0] for s in strata_present]
    x = np.arange(len(strata_present))

    n_panels = 3 if b05_sweep_df is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))

    # Panel (a): Oracle mean delta
    ax = axes[0]
    oracle_means = [results["per_stratum"][s]["oracle_mean_delta"] for s in strata_present]
    colors = [STRATUM_COLORS[s] for s in strata_present]
    ax.bar(x, oracle_means, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Oracle mean delta_reward")
    ax.set_title("Plasticity Benefit by Stratum")
    for i, v in enumerate(oracle_means):
        ax.text(i, v + 0.5, f"{v:+.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(-0.08, 1.02, "(a)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")

    # Panel (b): Normalized improvement
    ax = axes[1]
    norm_vals = [results["per_stratum"][s]["normalized_improvement"] for s in strata_present]
    ax.bar(x, norm_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Normalized improvement (delta / headroom)")
    ax.set_title("Headroom-Corrected Benefit")
    for i, v in enumerate(norm_vals):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(-0.08, 1.02, "(b)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")

    # Panel (c): B0.5 comparison if available
    if b05_sweep_df is not None and n_panels == 3:
        ax = axes[2]
        _plot_b05_comparison_panel(ax, results, strata_present, b05_sweep_df)
        ax.text(-0.08, 1.02, "(c)", transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom")

    fig.tight_layout()
    save_figure(fig, fig_dir, "F5_topology_dependence")

    return results


def _plot_b05_comparison_panel(
    ax: plt.Axes,
    b2_results: dict,
    strata_present: list[str],
    b05_sweep_df: pd.DataFrame,
) -> None:
    """Draw B2 vs B0.5 oracle comparison panel."""
    x = np.arange(len(strata_present))
    width = 0.35
    labels = [STRATUM_LABELS[s].split(" [")[0] for s in strata_present]

    b2_vals = [b2_results["per_stratum"][s]["oracle_mean_delta"] for s in strata_present]

    # Compute B0.5 oracle per stratum
    b05_vals = []
    for s in strata_present:
        sub = b05_sweep_df[b05_sweep_df["stratum"] == s]
        if sub.empty:
            b05_vals.append(0.0)
        else:
            oracle = sub.groupby("network_id")["delta_reward"].max()
            b05_vals.append(float(oracle.mean()))

    ax.bar(x - width / 2, b2_vals, width, color=CLR_B2,
           edgecolor="black", linewidth=0.5, label="Random RNN (B2)")
    ax.bar(x + width / 2, b05_vals, width, color=CLR_B05,
           edgecolor="black", linewidth=0.5, label="MorphoNAS (B0.5)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Oracle mean delta_reward")
    ax.set_title("B2 vs B0.5 by Stratum")
    ax.legend(fontsize=9)


# =====================================================================
# Summary statistics and JSON export
# =====================================================================

def compile_summary(
    competence: dict,
    oracle: dict,
    regret: dict,
    anti_hebbian: dict,
    topology: dict,
) -> dict:
    """Compile all analysis results into a single summary dict."""
    summary = {
        "competence": competence,
        "oracle_improvement": oracle,
        "regret": regret,
        "anti_hebbian_dominance": anti_hebbian,
        "topology_dependence": topology,
    }
    return summary


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="B2 Random RNN Control Experiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--b2-pool",
        type=str,
        default="experiments/B2/pool/random_rnn_pool.jsonl",
        help="Path to B2 pool JSONL (default: experiments/B2/pool/random_rnn_pool.jsonl)",
    )
    parser.add_argument(
        "--b2-sweep",
        type=str,
        default="experiments/B2/sweep/random_rnn_sweep.jsonl",
        help="Path to B2 sweep JSONL (default: experiments/B2/sweep/random_rnn_sweep.jsonl)",
    )
    parser.add_argument(
        "--b05-sweep-dir",
        type=str,
        default=None,
        help="Path to B0.5 sweep directory (optional, for comparison)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/B2/analysis",
        help="Output directory (default: experiments/B2/analysis)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create output directories
    output_dir = Path(args.output_dir)
    fig_dir = str(output_dir / "figures")
    tbl_dir = str(output_dir / "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    # Auto-extract from tar.gz if the JSONL is missing
    pool_path = Path(args.b2_pool)
    if not pool_path.exists():
        tgz = pool_path.with_suffix(pool_path.suffix + ".tar.gz")
        if tgz.exists():
            import tarfile
            logger.info(f"Extracting {tgz} ...")
            with tarfile.open(tgz, "r:gz") as tf:
                tf.extract(pool_path.name, path=str(pool_path.parent))

    logger.info("Loading B2 pool data...")
    pool_df = load_pool_jsonl(args.b2_pool)

    logger.info("Loading B2 sweep data...")
    sweep_df = load_sweep_jsonl(args.b2_sweep)

    # Add composite network_id and assign strata to sweep
    pool_df = add_network_id(pool_df)
    sweep_df = add_network_id(sweep_df)
    sweep_df = assign_strata(sweep_df)

    # Assign stratum from baseline_reward (use pool strata if available)
    pool_strata = pool_df.set_index("network_id")["stratum"].to_dict()
    sweep_df["stratum"] = sweep_df["network_id"].map(pool_strata).fillna(
        sweep_df["baseline_reward"].apply(lambda r: get_stratum(r).value)
    )

    logger.info(
        f"B2 data: {pool_df['network_id'].nunique():,} pool networks, "
        f"{sweep_df['network_id'].nunique():,} sweep networks, "
        f"{len(sweep_df):,} sweep observations"
    )

    # Optional B0.5 comparison data
    b05_sweep_df = None
    if args.b05_sweep_dir:
        logger.info(f"Loading B0.5 comparison data from {args.b05_sweep_dir}...")
        b05_sweep_df = load_b05_sweep(args.b05_sweep_dir)
        if b05_sweep_df is not None:
            logger.info(f"B0.5: {b05_sweep_df['network_id'].nunique():,} networks, {len(b05_sweep_df):,} rows")

    # ── Run analyses ─────────────────────────────────────────────────
    print("=" * 70)
    print("B2 RANDOM RNN CONTROL EXPERIMENT ANALYSIS")
    print("=" * 70)

    r1 = analysis_1_competence(pool_df, fig_dir, tbl_dir, b05_sweep_df)
    print(f"\n[1] Competence: {r1['competent_count']:,}/{r1['total_valid']:,} "
          f"({r1['competent_rate_pct']:.1f}%) random RNNs are competent (non-Weak)")
    if "b05_competent_rate_pct" in r1:
        print(f"    MorphoNAS comparison: {r1['b05_competent_rate_pct']:.1f}% competent")

    r2 = analysis_2_oracle_improvement(sweep_df, fig_dir, tbl_dir, b05_sweep_df)
    print("\n[2] Oracle improvement per stratum:")
    for s in STRATA_ORDER:
        if s in r2["per_stratum"]:
            d = r2["per_stratum"][s]
            print(f"    {STRATUM_LABELS[s]}: oracle mean={d['oracle_mean']:+.1f}, "
                  f"{d['improved_pct']:.0f}% improved (N={d['n_networks']})")

    r3 = analysis_3_regret(sweep_df, fig_dir, tbl_dir, b05_sweep_df)
    print("\n[3] Regret per stratum:")
    for s in STRATA_ORDER:
        if s in r3["per_stratum"]:
            d = r3["per_stratum"][s]
            print(f"    {STRATUM_LABELS[s]}: regret={d['regret']:.1f} ({d['regret_pct']:.0f}%), "
                  f"oracle={d['oracle_mean_delta']:+.1f}, "
                  f"fixed={d['best_fixed_delta']:+.1f}")

    r4 = analysis_4_anti_hebbian(sweep_df, fig_dir, tbl_dir, b05_sweep_df)
    print("\n[4] Anti-Hebbian dominance per stratum:")
    for s in STRATA_ORDER:
        if s in r4.get("per_stratum", {}):
            d = r4["per_stratum"][s]
            print(f"    {STRATUM_LABELS[s]}: Cohen's d={d['cohens_d']:.3f} "
                  f"({d['winner']}), p={d['t_pvalue']:.2e}")
        if s in r4.get("oracle_eta_sign", {}):
            d = r4["oracle_eta_sign"][s]
            print(f"      Optimal eta sign: {d['pct_anti_hebbian']:.0f}% anti-Hebbian "
                  f"(N={d['n_networks']})")

    r5 = analysis_5_topology_dependence(sweep_df, fig_dir, tbl_dir, b05_sweep_df)
    print("\n[5] Topology-dependence (stratum pattern):")
    for s in STRATA_ORDER:
        if s in r5["per_stratum"]:
            d = r5["per_stratum"][s]
            print(f"    {STRATUM_LABELS[s]}: oracle={d['oracle_mean_delta']:+.1f}, "
                  f"norm={d['normalized_improvement']:.3f}, "
                  f"headroom={d['headroom']:.0f}")

    # ── Save summary JSON ────────────────────────────────────────────
    summary = compile_summary(r1, r2, r3, r4, r5)

    summary_path = str(output_dir / "summary_stats.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary stats saved to {summary_path}")

    print(f"\n{'=' * 70}")
    print(f"Output: {output_dir}")
    print(f"  Figures: {fig_dir}/")
    print(f"  Tables:  {tbl_dir}/")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
