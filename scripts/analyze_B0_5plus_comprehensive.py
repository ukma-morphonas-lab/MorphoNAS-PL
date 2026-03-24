#!/usr/bin/env python3
"""
B0.5 Comprehensive Analysis — Publication-Ready
================================================
Produces all figures, tables, and narrative for the 50K-network
Hebbian plasticity sweep (75 grid points, ~3.75M evaluations).

Analyses:
  0. Headline impact — practical improvement magnitude in reward points
  1. Natural distribution characterization
  2. Grid heatmaps per stratum (eta x decay)
  3. Stratum-dependent optimal parameters
  4. Anti-Hebbian vs Hebbian comparison (with effect sizes & p-values)
  5. Decay importance (ANOVA + partial eta-squared)
  6. Perfect network stability (one-sample t-test)
  7. Per-network regret analysis (meta-plasticity justification)
  8. Failure mode characterization
  9. 3D surface plots (eta x decay -> delta-fitness) per stratum
 10. Outcome distribution stacked bars (improved / unchanged / harmed)
 11. Per-grid-cell outcome heatmaps (% improved and % harmed)
 12. Severity distributions (violin plots: magnitude of change by outcome)
 13. eta x decay Interaction Effects (two-way ANOVA with interaction term)
 14. Dose-Response Curves (mean delta_fitness vs |eta|, by Hebbian sign)
 15. Network Topology Correlates (structural features vs plasticity responsiveness)
 16. Stratum Transition Analysis (transition matrices under best-fixed / oracle)
 17. Dose-Response by Decay (mean delta_fitness vs decay, by |eta| bins)
 18. Oracle Eta Distribution per Stratum (violin plots of per-network best eta)
 19. Within-Stratum Gradient (baseline_reward vs oracle delta_fitness scatter)

Outputs: figures/ and tables/ subdirectories under --output-dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
import numpy as np
import pandas as pd
from scipy import stats

# Allow imports from the code/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from MorphoNAS_PL.experimentB0_5_natural import (
    DECAY_VALUES,
    ETA_VALUES,
    STRATUM_BOUNDS,
    Stratum,
    get_stratum_label,
)
from MorphoNAS_PL.analysis_utils import (
    STRATA_ORDER,
    STRATUM_LABELS,
    STRATUM_COLORS,
    STRATUM_BOUNDS_LIST,
    IMPROVE_THRESHOLD,
    HARM_THRESHOLD,
    apply_publication_style,
    fmt_eta,
    fmt_decay,
    load_sweep,
    load_pool_rewards,
    load_pool_topology,
    save_figure,
    get_stratum_idx,
    strata_present_in,
)

logger = logging.getLogger(__name__)

# Apply publication-quality plot style
apply_publication_style()


# =====================================================================
# Analysis 0 — Headline impact: practical improvement magnitude
# =====================================================================
def analysis_0_headline(df: pd.DataFrame, fig_dir: str, tbl_dir: str) -> dict:
    """How much does plasticity actually improve networks, in reward points?"""
    logger.info("Analysis 0: Headline impact — practical improvement magnitude")

    df_active = df[df["eta"] != 0].copy()

    results = {}
    rows = []

    for s in STRATA_ORDER:
        sub = df[df["stratum"] == s]
        sub_active = df_active[df_active["stratum"] == s]
        if sub.empty:
            continue

        n_networks = sub["network_id"].nunique()

        # Best fixed parameter for this stratum
        best_cell = sub.groupby(["eta", "decay"])["delta_reward"].mean()
        best_param = best_cell.idxmax()
        best_mean_dr = best_cell.max()
        best_sub = sub[(sub["eta"] == best_param[0]) & (sub["decay"] == best_param[1])]
        best_improved_pct = 100 * (best_sub["delta_reward"] > 0).mean()
        best_mean_dr_improved = best_sub.loc[best_sub["delta_reward"] > 0, "delta_reward"].mean()
        best_improved_n = (best_sub["delta_reward"] > 0).sum()

        # Per-network oracle
        oracle_per_net = sub.groupby("network_id").agg(
            oracle_dr=("delta_reward", "max"),
            baseline_reward=("baseline_avg_reward", "first"),
            stratum=("stratum", "first"),
        )
        oracle_mean_dr = oracle_per_net["oracle_dr"].mean()
        oracle_improved_pct = 100 * (oracle_per_net["oracle_dr"] > 0).mean()
        oracle_improved_n = (oracle_per_net["oracle_dr"] > 0).sum()
        oracle_mean_dr_improved = oracle_per_net.loc[
            oracle_per_net["oracle_dr"] > 0, "oracle_dr"
        ].mean()

        best_se = stats.sem(best_sub["delta_reward"].values)
        oracle_se = stats.sem(oracle_per_net["oracle_dr"].values)
        baseline_mean = oracle_per_net["baseline_reward"].mean()
        t_stat, p_val = stats.ttest_1samp(best_sub["delta_reward"].values, 0)

        # Bootstrap 95% CIs (10K resamples, percentile method)
        rng = np.random.default_rng(42)
        n_boot = 10_000
        best_vals = best_sub["delta_reward"].values
        oracle_vals_arr = oracle_per_net["oracle_dr"].values
        best_boot = np.array([
            rng.choice(best_vals, size=len(best_vals), replace=True).mean()
            for _ in range(n_boot)
        ])
        oracle_boot = np.array([
            rng.choice(oracle_vals_arr, size=len(oracle_vals_arr), replace=True).mean()
            for _ in range(n_boot)
        ])
        best_ci_lo, best_ci_hi = np.percentile(best_boot, [2.5, 97.5])
        oracle_ci_lo, oracle_ci_hi = np.percentile(oracle_boot, [2.5, 97.5])

        # Strata-crossing
        new_reward = best_sub["baseline_avg_reward"] + best_sub["delta_reward"]
        old_strata = best_sub["baseline_avg_reward"].apply(get_stratum_idx)
        new_strata = new_reward.apply(get_stratum_idx)
        promoted = (new_strata > old_strata).sum()
        demoted = (new_strata < old_strata).sum()

        r = {
            "n_networks": int(n_networks),
            "baseline_mean_reward": float(baseline_mean),
            "best_param": {"eta": float(best_param[0]), "decay": float(best_param[1])},
            "best_param_mean_delta_reward": float(best_mean_dr),
            "best_param_improved_pct": float(best_improved_pct),
            "best_param_improved_n": int(best_improved_n),
            "best_param_mean_delta_when_improved": float(best_mean_dr_improved)
                if np.isfinite(best_mean_dr_improved) else None,
            "best_param_t": float(t_stat),
            "best_param_p": float(p_val),
            "best_param_promoted": int(promoted),
            "best_param_demoted": int(demoted),
            "oracle_mean_delta_reward": float(oracle_mean_dr),
            "oracle_improved_pct": float(oracle_improved_pct),
            "oracle_improved_n": int(oracle_improved_n),
            "oracle_mean_delta_when_improved": float(oracle_mean_dr_improved)
                if np.isfinite(oracle_mean_dr_improved) else None,
            "best_param_se": float(best_se),
            "oracle_se": float(oracle_se),
            "best_param_ci_lo": float(best_ci_lo),
            "best_param_ci_hi": float(best_ci_hi),
            "oracle_ci_lo": float(oracle_ci_lo),
            "oracle_ci_hi": float(oracle_ci_hi),
        }
        results[s] = r

        # Significance marker
        if s == "perfect":
            sig_marker = "\u2014"
        elif p_val < 0.001:
            sig_marker = "***"
        elif p_val < 0.01:
            sig_marker = "**"
        elif p_val < 0.05:
            sig_marker = "*"
        else:
            sig_marker = "ns"

        rows.append({
            "Stratum": STRATUM_LABELS[s],
            "N networks": n_networks,
            "Baseline reward": f"{baseline_mean:.1f}",
            "Best (eta, decay)": f"({fmt_eta(best_param[0])}, {fmt_decay(best_param[1])})",
            "Mean delta-reward (best param)": f"{best_mean_dr:+.1f}",
            "% improved (best param)": f"{best_improved_pct:.1f}%",
            "Mean delta when improved": f"+{best_mean_dr_improved:.1f}"
                if np.isfinite(best_mean_dr_improved) else "—",
            "p (vs 0)": f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}",
            "Sig": sig_marker,
            "Best param 95% CI": f"[{best_ci_lo:+.1f}, {best_ci_hi:+.1f}]",
            "Oracle mean delta-reward": f"{oracle_mean_dr:+.1f}",
            "Oracle 95% CI": f"[{oracle_ci_lo:+.1f}, {oracle_ci_hi:+.1f}]",
            "% improved (oracle)": f"{oracle_improved_pct:.1f}%",
            "Oracle delta when improved": f"+{oracle_mean_dr_improved:.1f}"
                if np.isfinite(oracle_mean_dr_improved) else "—",
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(tbl_dir, "T0_headline_impact.csv"), index=False)

    # Colorblind-safe palette for best-fixed vs oracle
    CLR_FIXED = "#4477AA"   # blue
    CLR_ORACLE = "#EE6677"  # red-pink

    def _sig_label(p: float) -> str:
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    # Figure: 2-panel headline (a) Mean Δr with 95% CIs, (b) % improved
    strata_present = [s for s in STRATA_ORDER if s in results]
    labels = [STRATUM_LABELS[s].split(" [")[0] for s in strata_present]
    x = np.arange(len(strata_present))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel (a): Mean Δr ──
    best_drs = [results[s]["best_param_mean_delta_reward"] for s in strata_present]
    oracle_drs = [results[s]["oracle_mean_delta_reward"] for s in strata_present]
    best_ci_los = [results[s]["best_param_ci_lo"] for s in strata_present]
    best_ci_his = [results[s]["best_param_ci_hi"] for s in strata_present]
    oracle_ci_los = [results[s]["oracle_ci_lo"] for s in strata_present]
    oracle_ci_his = [results[s]["oracle_ci_hi"] for s in strata_present]

    # Error bars (asymmetric)
    best_errs = [
        [max(0, b - lo) for b, lo in zip(best_drs, best_ci_los)],
        [max(0, hi - b) for b, hi in zip(best_drs, best_ci_his)],
    ]
    oracle_errs = [
        [max(0, o - lo) for o, lo in zip(oracle_drs, oracle_ci_los)],
        [max(0, hi - o) for o, hi in zip(oracle_drs, oracle_ci_his)],
    ]

    ax1.bar(x - width / 2, best_drs, width, color=CLR_FIXED, edgecolor="black",
            linewidth=0.5, label="Best fixed parameter", yerr=best_errs,
            capsize=3, error_kw={"linewidth": 1})
    ax1.bar(x + width / 2, oracle_drs, width, color=CLR_ORACLE, edgecolor="black",
            linewidth=0.5, label="Per-network oracle", yerr=oracle_errs,
            capsize=3, error_kw={"linewidth": 1})

    ax1.axhline(0, color="black", lw=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Mean $\\Delta r$ (reward points)")
    ax1.legend(fontsize=9, loc="upper left")

    # Annotate bars with value + significance
    for i, s in enumerate(strata_present):
        b = best_drs[i]
        o = oracle_drs[i]
        sig = _sig_label(results[s]["best_param_p"])
        # Best-fixed: value + significance
        label_b = f"{b:+.1f}" if b != 0 else "0.0"
        if sig != "***" and s != "perfect":
            label_b += f" ({sig})"
        y_b = best_ci_his[i] + 1.5
        ax1.text(i - width / 2, y_b, label_b,
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
        # Oracle
        y_o = oracle_ci_his[i] + 1.5
        ax1.text(i + width / 2, y_o, f"{o:+.1f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.text(-0.08, 1.02, "(a)", transform=ax1.transAxes,
             fontsize=13, fontweight="bold", va="bottom")

    # ── Panel (b): % of networks improved ──
    best_pcts = [results[s]["best_param_improved_pct"] for s in strata_present]
    oracle_pcts = [results[s]["oracle_improved_pct"] for s in strata_present]

    ax2.bar(x - width / 2, best_pcts, width, color=CLR_FIXED, edgecolor="black",
            linewidth=0.5, label="Best fixed parameter")
    ax2.bar(x + width / 2, oracle_pcts, width, color=CLR_ORACLE, edgecolor="black",
            linewidth=0.5, label="Per-network oracle")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Networks improved (%)")
    ax2.set_ylim(0, 109)
    ax2.legend(fontsize=9, loc="upper left")

    for i in range(len(strata_present)):
        b, o = best_pcts[i], oracle_pcts[i]
        ax2.text(i - width / 2, b + 1.5, f"{b:.0f}%",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax2.text(i + width / 2, o + 1.5, f"{o:.0f}%",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.text(-0.08, 1.02, "(b)", transform=ax2.transAxes,
             fontsize=13, fontweight="bold", va="bottom")

    fig.tight_layout()
    save_figure(fig, fig_dir, "F0_headline_impact")

    logger.info("  Headline summary:")
    for s in strata_present:
        r = results[s]
        logger.info(f"    {STRATUM_LABELS[s]}: best-param delta-reward = {r['best_param_mean_delta_reward']:+.1f} pts "
                    f"({r['best_param_improved_pct']:.0f}% improved), "
                    f"oracle delta-reward = {r['oracle_mean_delta_reward']:+.1f} pts "
                    f"({r['oracle_improved_pct']:.0f}% improved)")

    return results


# =====================================================================
# Analysis 1 — Natural distribution characterization
# =====================================================================
def analysis_1_distribution(df: pd.DataFrame, pool_rewards: np.ndarray,
                            fig_dir: str, tbl_dir: str) -> dict:
    """Characterize the natural distribution of network performance."""
    logger.info("Analysis 1: Natural distribution characterization")

    ctrl = df[(df["eta"] == 0) & (df["decay"] == 0)].copy()
    n_unique = ctrl["network_id"].nunique()

    stratum_counts = ctrl["stratum"].value_counts()
    total = stratum_counts.sum()

    rows = []
    for s in STRATA_ORDER:
        count = stratum_counts.get(s, 0)
        pct = 100 * count / total if total > 0 else 0
        low, high = STRATUM_BOUNDS[Stratum(s)]
        rows.append({
            "Stratum": STRATUM_LABELS[s],
            "Range": f"[{low}, {high})" if high < float('inf') else f"[{low}, inf)",
            "Count": int(count),
            "Percent": f"{pct:.1f}%",
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(tbl_dir, "T1_distribution.csv"), index=False)

    baseline_rewards = ctrl["baseline_avg_reward"].values
    summary = {
        "n_networks": int(n_unique),
        "mean_reward": float(np.mean(baseline_rewards)),
        "median_reward": float(np.median(baseline_rewards)),
        "std_reward": float(np.std(baseline_rewards)),
        "min_reward": float(np.min(baseline_rewards)),
        "max_reward": float(np.max(baseline_rewards)),
    }

    from matplotlib.patches import Patch

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"width_ratios": [2.2, 1]})

    ax = axes[0]
    bounds = list(STRATUM_BOUNDS.items())

    # Color histogram bars by stratum; log y-scale reveals structure across all strata
    counts_hist, bin_edges = np.histogram(baseline_rewards, bins=80, range=(0, 500))
    for lo_b, hi_b, c in zip(bin_edges[:-1], bin_edges[1:], counts_hist):
        if c == 0:
            continue
        mid = (lo_b + hi_b) / 2
        bar_color = STRATUM_COLORS["weak"]  # default
        for stratum, (slo, shi) in bounds:
            if slo <= mid < min(shi, 500):
                bar_color = STRATUM_COLORS[stratum.value]
                break
        ax.bar(lo_b, c, width=hi_b - lo_b, color=bar_color,
               alpha=0.85, edgecolor="white", linewidth=0.3)

    # Stratum boundary lines
    for stratum, (lo, hi) in bounds[1:]:
        ax.axvline(lo, color="gray", ls="--", lw=0.7, alpha=0.6)

    ax.set_yscale("log")
    ax.set_xlim(0, 500)
    ax.set_xlabel("Baseline Reward (20-episode mean)", fontsize=11)
    ax.set_ylabel("Count (log scale)", fontsize=11)
    ax.tick_params(labelsize=10)

    # Legend covers all 5 strata
    legend_handles = [
        Patch(facecolor=STRATUM_COLORS[s], label=STRATUM_LABELS[s].split(" [")[0])
        for s in STRATA_ORDER
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8.5, framealpha=0.85, edgecolor="#cccccc")

    ax2 = axes[1]
    bar_labels = [STRATUM_LABELS[s].split(" [")[0] for s in STRATA_ORDER]
    counts = [stratum_counts.get(s, 0) for s in STRATA_ORDER]
    bar_colors = [STRATUM_COLORS[s] for s in STRATA_ORDER]
    bars = ax2.barh(bar_labels, counts, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, c in zip(bars, counts):
        pct = 100 * c / total
        label = f"{c:,} ({pct:.1f}%)"
        bar_mid_log = 10 ** ((np.log10(max(c, 11)) + np.log10(10)) / 2)
        ax2.text(bar_mid_log, bar.get_y() + bar.get_height() / 2,
                 label, va="center", ha="center", fontsize=8.5,
                 color="white", fontweight="bold")
    ax2.set_xlabel("Count (log scale)", fontsize=11)
    ax2.set_title("Networks per Stratum", fontsize=11)
    ax2.set_xscale("log")
    ax2.set_xlim(10, max(counts) * 3)
    ax2.tick_params(labelsize=10)
    ax2.invert_yaxis()

    fig.tight_layout()
    save_figure(fig, fig_dir, "F1_natural_distribution")

    logger.info(f"  {n_unique} unique networks, mean reward = {summary['mean_reward']:.1f}")
    return summary


# =====================================================================
# Analysis 2 — Grid heatmaps per stratum
# =====================================================================
def analysis_2_heatmaps(df: pd.DataFrame, fig_dir: str, tbl_dir: str) -> dict:
    """Produce eta x decay heatmaps of mean delta-fitness for each stratum + overall."""
    logger.info("Analysis 2: Grid heatmaps per stratum")

    eta_vals = sorted(df["eta"].unique())
    decay_vals = sorted(df["decay"].unique())

    results = {}

    strata_to_plot = ["all"] + STRATA_ORDER
    pivot_cache = {}
    for s in strata_to_plot:
        sub = df if s == "all" else df[df["stratum"] == s]
        piv = sub.pivot_table(
            values="delta_fitness", index="eta", columns="decay",
            aggfunc="mean"
        ).reindex(index=eta_vals, columns=decay_vals)
        pivot_cache[s] = piv
        results[s] = {
            "global_mean": float(piv.values[np.isfinite(piv.values)].mean()),
            "best_cell": None,
        }
        if piv.notna().any().any():
            best_idx = np.unravel_index(np.nanargmax(piv.values), piv.shape)
            results[s]["best_cell"] = {
                "eta": float(eta_vals[best_idx[0]]),
                "decay": float(decay_vals[best_idx[1]]),
                "value": float(piv.values[best_idx]),
            }

    class _AsymPowerNorm(mcolors.Normalize):
        """Asymmetric norm with power-law compression on the negative side.

        Maps data to [0, 1] for the colormap:
        - Negative values [vmin, 0] → [0, 0.5] with gamma compression
          (gamma < 1 stretches values near zero, compresses deep negatives)
        - Positive values [0, vmax] → [0.5, 1.0] linearly

        This gives proper gradation across the full negative range while
        reserving enough colormap space for the green (positive) region.
        """

        def __init__(self, vmin, vmax, gamma=0.4):
            super().__init__(vmin=vmin, vmax=vmax)
            self.gamma = gamma

        def __call__(self, value, clip=None):
            val = np.ma.asarray(value, dtype=float)
            result = np.ma.empty_like(val)
            # Negative side: power-law compression into [0, 0.5]
            neg_mask = val < 0
            if self.vmin < 0:
                neg_frac = np.ma.where(neg_mask, val / self.vmin, 0.0)  # 0..1
                neg_frac = np.ma.clip(neg_frac, 0, 1)
                result[neg_mask] = 0.5 * (1 - neg_frac[neg_mask] ** self.gamma)
            # Positive side: linear into [0.5, 1.0]
            pos_mask = val >= 0
            if self.vmax > 0:
                pos_frac = np.ma.where(pos_mask, val / self.vmax, 0.0)
                pos_frac = np.ma.clip(pos_frac, 0, 1)
                result[pos_mask] = 0.5 + 0.5 * pos_frac[pos_mask]
            else:
                result[pos_mask] = 0.5
            return result

    def _heatmap_norm(vals):
        """Create asymmetric norm for heatmaps.

        Full negative range is shown with power-law compression (gamma=0.4),
        giving gradation from light red (near zero) to deep red (extreme harm).
        Positive values map linearly to yellow→green, ensuring the best cells
        appear clearly green.
        """
        vmin_raw = float(np.nanmin(vals))
        vmax_raw = float(np.nanmax(vals))
        if vmax_raw <= 0:
            vmax_raw = -vmin_raw * 0.1 or 0.001
        return _AsymPowerNorm(vmin=vmin_raw, vmax=vmax_raw, gamma=0.4)

    # 2x3 panel of heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    for idx, s in enumerate(strata_to_plot):
        ax = axes_flat[idx]
        piv = pivot_cache[s]
        vals = piv.values

        norm = _heatmap_norm(vals)
        im = ax.imshow(vals, cmap="RdYlGn", norm=norm,
                       aspect="auto", origin="upper")

        ax.set_xticks(range(len(decay_vals)))
        ax.set_xticklabels([fmt_decay(d) for d in decay_vals], fontsize=8)
        ax.set_yticks(range(len(eta_vals)))
        ax.set_yticklabels([fmt_eta(e) for e in eta_vals], fontsize=8)

        if s == "all":
            n_net = df["network_id"].nunique()
            title = f"All Networks (N={n_net:,})"
        else:
            n_net = df[df["stratum"] == s]["network_id"].nunique()
            title = f"{STRATUM_LABELS[s]} (N={n_net:,})"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Decay")
        ax.set_ylabel(u"\u03B7 (learning rate)")

        for i in range(len(eta_vals)):
            for j in range(len(decay_vals)):
                v = vals[i, j]
                if np.isfinite(v):
                    cval = float(norm(v))
                    txt_col = "white" if cval < 0.15 or cval > 0.85 else "black"
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=6.5, color=txt_col)
        # No colorbar — cell annotations provide exact values; asymmetric norm
        # would make the bar visually misleading (50% green for tiny positive range).

    axes_flat[5].axis("off")

    fig.suptitle(u"B0.5 Hebbian Plasticity Sweep: Mean \u0394fitness (\u03B7 \u00D7 decay)",
                 fontsize=15, y=1.01)
    save_figure(fig, fig_dir, "F2_heatmaps_all_strata")

    # F2b: 3 key strata (using TwoSlopeNorm for publication quality)
    def _heatmap_norm_twoslope(vals):
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        if vmin >= 0:
            return mcolors.TwoSlopeNorm(vmin=0, vcenter=0, vmax=max(vmax, 0.001))
        if vmax <= 0:
            return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=abs(vmin) * 0.01)
        return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    key_strata = ["high_mid", "near_perfect", "perfect"]
    fig_key, axes_key = plt.subplots(1, 3, figsize=(22, 7))
    for idx, s in enumerate(key_strata):
        ax_k = axes_key[idx]
        piv = pivot_cache[s]
        vals = piv.values
        norm = _heatmap_norm_twoslope(vals)
        im = ax_k.imshow(vals, cmap="RdYlGn", norm=norm,
                         aspect="auto", origin="upper")
        ax_k.set_xticks(range(len(decay_vals)))
        ax_k.set_xticklabels([fmt_decay(d) for d in decay_vals], fontsize=12)
        ax_k.set_yticks(range(len(eta_vals)))
        ax_k.set_yticklabels([fmt_eta(e) for e in eta_vals], fontsize=10)
        ax_k.set_xlabel("Decay", fontsize=12)
        ax_k.set_ylabel(u"\u03B7 (learning rate)", fontsize=12)
        n_net = df[df["stratum"] == s]["network_id"].nunique()
        label = STRATUM_LABELS[s]
        ax_k.set_title(f"{label} (N={n_net:,})", fontsize=13)
        for i in range(len(eta_vals)):
            for j in range(len(decay_vals)):
                v = vals[i, j]
                if np.isfinite(v):
                    cval = float(norm(v))
                    txt_col = "white" if cval < 0.15 or cval > 0.85 else "black"
                    ax_k.text(j, i, f"{v:.3f}", ha="center", va="center",
                              fontsize=9, color=txt_col, fontweight="bold")

    # No suptitle — LaTeX caption handles it
    fig_key.tight_layout()
    save_figure(fig_key, fig_dir, "F2b_heatmaps_key_strata")

    # Individual high-res heatmaps
    for s in strata_to_plot:
        piv = pivot_cache[s]
        vals = piv.values
        norm = _heatmap_norm(vals)

        fig2, ax2 = plt.subplots(figsize=(10, 7))
        im = ax2.imshow(vals, cmap="RdYlGn", norm=norm,
                        aspect="auto", origin="upper")
        ax2.set_xticks(range(len(decay_vals)))
        ax2.set_xticklabels([fmt_decay(d) for d in decay_vals])
        ax2.set_yticks(range(len(eta_vals)))
        ax2.set_yticklabels([fmt_eta(e) for e in eta_vals])
        ax2.set_xlabel("Decay")
        ax2.set_ylabel(u"\u03B7 (learning rate)")

        if s == "all":
            ax2.set_title(u"B0.5 Grid: Mean \u0394fitness \u2014 All Networks")
        else:
            ax2.set_title(f"B0.5 Grid: Mean \u0394fitness \u2014 {STRATUM_LABELS[s]}")

        for i in range(len(eta_vals)):
            for j in range(len(decay_vals)):
                v = vals[i, j]
                if np.isfinite(v):
                    cval = float(norm(v))
                    txt_col = "white" if cval < 0.15 or cval > 0.85 else "black"
                    ax2.text(j, i, f"{v:.4f}", ha="center", va="center",
                             fontsize=7, color=txt_col)
        # No colorbar — see comment above.
        save_figure(fig2, fig_dir, f"F2_heatmap_{s}")

    return results


# =====================================================================
# Analysis 3 — Stratum-dependent optimal parameters
# =====================================================================
def analysis_3_optimal_params(df: pd.DataFrame, fig_dir: str, tbl_dir: str) -> dict:
    """Find best (eta, decay) per stratum; test for universality."""
    logger.info("Analysis 3: Stratum-dependent optimal parameters")

    eta_vals = sorted(df["eta"].unique())
    decay_vals = sorted(df["decay"].unique())

    rows = []
    best_params = {}
    for s in STRATA_ORDER:
        sub = df[df["stratum"] == s]
        if sub.empty:
            continue
        n_networks = sub["network_id"].nunique()
        piv = sub.pivot_table(values="delta_fitness", index="eta",
                              columns="decay", aggfunc="mean")
        piv = piv.reindex(index=eta_vals, columns=decay_vals)

        best_idx = np.unravel_index(np.nanargmax(piv.values), piv.shape)
        worst_idx = np.unravel_index(np.nanargmin(piv.values), piv.shape)

        best_eta = eta_vals[best_idx[0]]
        best_decay = decay_vals[best_idx[1]]
        best_val = piv.values[best_idx]
        worst_val = piv.values[worst_idx]

        if 0 in eta_vals:
            ctrl_row = sub[sub["eta"] == 0]
            ctrl_mean = ctrl_row["delta_fitness"].mean()
        else:
            ctrl_mean = np.nan

        best_params[s] = {"eta": best_eta, "decay": best_decay}
        rows.append({
            "Stratum": STRATUM_LABELS[s],
            "N_networks": n_networks,
            u"Best \u03B7": fmt_eta(best_eta),
            "Best decay": fmt_decay(best_decay),
            u"Best \u0394fitness": f"{best_val:.4f}",
            u"Worst \u0394fitness": f"{worst_val:.4f}",
            "Range": f"{best_val - worst_val:.4f}",
            u"Control (\u03B7=0)": f"{ctrl_mean:.4f}" if np.isfinite(ctrl_mean) else "—",
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(tbl_dir, "T3_optimal_params.csv"), index=False)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    strata_present = [s for s in STRATA_ORDER if s in best_params]
    x = np.arange(len(strata_present))
    colors = [STRATUM_COLORS[s] for s in strata_present]

    best_etas = [best_params[s]["eta"] for s in strata_present]
    best_decays = [best_params[s]["decay"] for s in strata_present]
    best_deltas = [float(tbl[tbl["Stratum"] == STRATUM_LABELS[s]][u"Best \u0394fitness"].values[0])
                   for s in strata_present]
    labels = [STRATUM_LABELS[s].split(" [")[0] for s in strata_present]

    axes[0].bar(x, best_etas, color=colors, edgecolor="black", alpha=0.8)
    axes[0].axhline(0, color="black", ls="-", lw=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel(u"Best \u03B7")
    axes[0].set_title(u"Optimal \u03B7 by Stratum")

    axes[1].bar(x, best_decays, color=colors, edgecolor="black", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("Best Decay")
    axes[1].set_title("Optimal Decay by Stratum")
    for i, d in enumerate(best_decays):
        axes[1].text(i, d + max(best_decays)*0.02, fmt_decay(d),
                     ha="center", fontsize=8)

    axes[2].bar(x, best_deltas, color=colors, edgecolor="black", alpha=0.8)
    axes[2].axhline(0, color="black", ls="-", lw=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    axes[2].set_ylabel(u"Best Mean \u0394fitness")
    axes[2].set_title("Best Achievable Improvement by Stratum")
    for i, v in enumerate(best_deltas):
        axes[2].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)

    save_figure(fig, fig_dir, "F3_optimal_params")

    return {"best_params": best_params, "table": rows}


# =====================================================================
# Analysis 4 — Anti-Hebbian vs Hebbian comparison
# =====================================================================
def analysis_4_hebbian_comparison(df: pd.DataFrame, fig_dir: str, tbl_dir: str) -> dict:
    """Statistical comparison of positive vs negative eta."""
    logger.info("Analysis 4: Anti-Hebbian vs Hebbian comparison")

    df_nonzero = df[df["eta"] != 0].copy()
    df_nonzero["sign"] = np.where(df_nonzero["eta"] < 0, "anti-Hebbian", "Hebbian")

    rows = []
    results = {}
    for s in STRATA_ORDER:
        sub = df_nonzero[df_nonzero["stratum"] == s]
        if sub.empty:
            continue

        anti = sub[sub["sign"] == "anti-Hebbian"]["delta_fitness"].values
        hebb = sub[sub["sign"] == "Hebbian"]["delta_fitness"].values

        if len(anti) == 0 or len(hebb) == 0:
            continue

        t_stat, p_val = stats.ttest_ind(anti, hebb, equal_var=False)

        pooled_sd = np.sqrt((np.var(anti, ddof=1) * (len(anti)-1) +
                             np.var(hebb, ddof=1) * (len(hebb)-1)) /
                            (len(anti) + len(hebb) - 2))
        cohens_d = (np.mean(anti) - np.mean(hebb)) / pooled_sd if pooled_sd > 0 else 0
        winner = "anti-Hebbian" if np.mean(anti) > np.mean(hebb) else "Hebbian"

        results[s] = {
            "anti_mean": float(np.mean(anti)),
            "anti_se": float(stats.sem(anti)),
            "hebb_mean": float(np.mean(hebb)),
            "hebb_se": float(stats.sem(hebb)),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(cohens_d),
            "winner": winner,
            "n_anti": len(anti),
            "n_hebb": len(hebb),
        }

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        rows.append({
            "Stratum": STRATUM_LABELS[s],
            "Anti-Hebb mean+/-SE": f"{np.mean(anti):.4f}+/-{stats.sem(anti):.4f}",
            "Hebbian mean+/-SE": f"{np.mean(hebb):.4f}+/-{stats.sem(hebb):.4f}",
            "t": f"{t_stat:.2f}",
            "p": f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}",
            "Cohen's d": f"{cohens_d:.3f}",
            "Sig": sig,
            "Winner": winner,
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(tbl_dir, "T4_hebbian_comparison.csv"), index=False)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    strata_present = [s for s in STRATA_ORDER if s in results]
    x = np.arange(len(strata_present))
    width = 0.35
    labels = [STRATUM_LABELS[s].split(" [")[0] for s in strata_present]

    anti_means = [results[s]["anti_mean"] for s in strata_present]
    anti_ses = [results[s]["anti_se"] * 1.96 for s in strata_present]
    hebb_means = [results[s]["hebb_mean"] for s in strata_present]
    hebb_ses = [results[s]["hebb_se"] * 1.96 for s in strata_present]

    ax = axes[0]
    ax.bar(x - width/2, anti_means, width, yerr=anti_ses, capsize=3,
           color="#4477AA", alpha=0.8, label=u"Anti-Hebbian (\u03B7 < 0)")
    ax.bar(x + width/2, hebb_means, width, yerr=hebb_ses, capsize=3,
           color="#CC6677", alpha=0.8, label=u"Hebbian (\u03B7 > 0)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(u"Mean \u0394fitness (95% CI)")
    ax.set_title(u"Anti-Hebbian vs Hebbian by Stratum")
    ax.legend()

    for i, (am, hm) in enumerate(zip(anti_means, hebb_means)):
        offset_a = 0.002 if am >= 0 else -0.002
        offset_h = 0.002 if hm >= 0 else -0.002
        ax.text(i - width/2, am + anti_ses[i] + offset_a, f"{am:.4f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.text(i + width/2, hm + hebb_ses[i] + offset_h if hm >= 0 else hm - hebb_ses[i] + offset_h,
                f"{hm:.4f}", ha="center",
                va="bottom" if hm >= 0 else "top", fontsize=7, fontweight="bold")

    for i, s in enumerate(strata_present):
        p = results[s]["p_value"]
        marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if marker:
            ymax = max(anti_means[i] + anti_ses[i],
                       hebb_means[i] + hebb_ses[i])
            ax.text(i, ymax + 0.006, marker, ha="center", fontsize=12, fontweight="bold")

    ax.text(0.02, 0.02, u"Welch\u2019s t-test (unequal variance); ***p<0.001",
            transform=ax.transAxes, fontsize=7, fontstyle="italic", color="gray")

    ax2 = axes[1]
    ds = [results[s]["cohens_d"] for s in strata_present]
    colors_d = ["#4477AA" if d > 0 else "#CC6677" for d in ds]
    ax2.barh(x, ds, color=colors_d, edgecolor="black", alpha=0.8)
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Cohen's d (anti-Hebbian - Hebbian)")
    ax2.set_title("Effect Size by Stratum")
    ax2.axvline(0, color="black", lw=0.5)
    for val, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax2.axvline(val, color="gray", ls=":", lw=0.7, alpha=0.5)
        ax2.axvline(-val, color="gray", ls=":", lw=0.7, alpha=0.5)

    save_figure(fig, fig_dir, "F4_hebbian_comparison")

    return results


# =====================================================================
# Analysis 5 — Decay importance
# =====================================================================
def analysis_5_decay_importance(df: pd.DataFrame, fig_dir: str, tbl_dir: str) -> dict:
    """Assess the effect of decay relative to eta."""
    logger.info("Analysis 5: Decay importance")

    df_active = df[df["eta"] != 0].copy()
    decay_vals = sorted(df_active["decay"].unique())

    results = {}
    rows = []
    for s in STRATA_ORDER:
        sub = df_active[df_active["stratum"] == s]
        if sub.empty:
            continue

        by_decay = sub.groupby("decay")["delta_fitness"].agg(["mean", "std", "count"])

        groups = [sub[sub["decay"] == d]["delta_fitness"].values for d in decay_vals]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            F_stat, p_val = stats.f_oneway(*groups)
            grand_mean = sub["delta_fitness"].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
            ss_total = sum(np.sum((g - grand_mean)**2) for g in groups)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0
        else:
            F_stat, p_val, eta_sq = np.nan, np.nan, np.nan

        results[s] = {
            "F": float(F_stat) if np.isfinite(F_stat) else None,
            "p": float(p_val) if np.isfinite(p_val) else None,
            "eta_sq": float(eta_sq) if np.isfinite(eta_sq) else None,
            "by_decay": {str(d): float(by_decay.loc[d, "mean"]) for d in decay_vals if d in by_decay.index},
        }

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        rows.append({
            "Stratum": STRATUM_LABELS[s],
            "F": f"{F_stat:.2f}" if np.isfinite(F_stat) else "—",
            "p": f"{p_val:.2e}" if np.isfinite(p_val) and p_val < 0.001 else f"{p_val:.4f}" if np.isfinite(p_val) else "—",
            u"\u03B7\u00B2 (decay)": f"{eta_sq:.4f}" if np.isfinite(eta_sq) else "—",
            "Sig": sig,
        })

    # Two-way: eta effect ANOVA for comparison
    for s in STRATA_ORDER:
        sub = df_active[df_active["stratum"] == s]
        if sub.empty:
            continue
        eta_vals = sorted(sub["eta"].unique())
        groups_eta = [sub[sub["eta"] == e]["delta_fitness"].values for e in eta_vals]
        groups_eta = [g for g in groups_eta if len(g) > 0]
        if len(groups_eta) >= 2:
            F_eta, p_eta = stats.f_oneway(*groups_eta)
            grand_mean = sub["delta_fitness"].mean()
            ss_between_eta = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups_eta)
            ss_total = sum(np.sum((g - grand_mean)**2) for g in groups_eta)
            eta_sq_eta = ss_between_eta / ss_total if ss_total > 0 else 0

            for row in rows:
                if row["Stratum"] == STRATUM_LABELS[s]:
                    row[u"F (\u03B7 effect)"] = f"{F_eta:.2f}"
                    row[u"\u03B7\u00B2 (eta effect)"] = f"{eta_sq_eta:.4f}"
                    results[s]["F_eta"] = float(F_eta)
                    results[s]["eta_sq_eta"] = float(eta_sq_eta)

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(tbl_dir, "T5_decay_importance.csv"), index=False)

    # Figure
    key_strata_f5 = [s for s in ["high_mid", "near_perfect", "perfect"] if s in results]
    fig, axes = plt.subplots(1, len(key_strata_f5),
                             figsize=(5 * len(key_strata_f5), 5),
                             sharey=True)
    if not hasattr(axes, '__len__'):
        axes = [axes]

    for idx, s in enumerate(key_strata_f5):
        ax = axes[idx]
        sub = df_active[df_active["stratum"] == s]
        means = sub.groupby("decay")["delta_fitness"].mean()
        sems = sub.groupby("decay")["delta_fitness"].sem() * 1.96

        ax.errorbar(range(len(decay_vals)), [means.get(d, np.nan) for d in decay_vals],
                    yerr=[sems.get(d, 0) for d in decay_vals],
                    marker="o", capsize=3, color=STRATUM_COLORS[s], lw=2)
        ax.set_xticks(range(len(decay_vals)))
        ax.set_xticklabels([fmt_decay(d) for d in decay_vals], rotation=30)
        ax.set_xlabel("Decay")
        if idx == 0:
            ax.set_ylabel(u"Mean \u0394fitness (95% CI)")
        ax.set_title(STRATUM_LABELS[s].split(" [")[0])
        ax.axhline(0, color="black", ls="-", lw=0.5)

    plt.suptitle(u"Effect of Decay on \u0394fitness by Stratum (averaged over \u03B7)", fontsize=13)
    save_figure(fig, fig_dir, "F5_decay_importance")

    return results


# =====================================================================
# Analysis 6 — Perfect network stability
# =====================================================================
def analysis_6_perfect_stability(df: pd.DataFrame, fig_dir: str, tbl_dir: str) -> dict:
    """Test whether plasticity harms perfect networks."""
    logger.info("Analysis 6: Perfect network stability")

    perf = df[df["stratum"] == "perfect"].copy()
    if perf.empty:
        logger.warning("  No perfect networks found")
        return {}

    n_networks = perf["network_id"].nunique()
    n_evals = len(perf)

    deltas = perf["delta_fitness"].values
    t_stat, p_val = stats.ttest_1samp(deltas, 0)
    cohens_d = np.mean(deltas) / np.std(deltas, ddof=1) if np.std(deltas) > 0 else 0

    harmed = (deltas < -0.01).sum()
    helped = (deltas > 0.01).sum()
    neutral = len(deltas) - harmed - helped

    eta_vals = sorted(perf["eta"].unique())
    decay_vals = sorted(perf["decay"].unique())

    rows = []
    for e in eta_vals:
        for d in decay_vals:
            cell = perf[(perf["eta"] == e) & (perf["decay"] == d)]
            if cell.empty:
                continue
            cd = cell["delta_fitness"].values
            t, p = stats.ttest_1samp(cd, 0) if len(cd) > 1 else (0, 1)
            rows.append({
                u"\u03B7": fmt_eta(e),
                "decay": fmt_decay(d),
                u"mean \u0394fit": f"{np.mean(cd):.4f}",
                "SE": f"{stats.sem(cd):.4f}" if len(cd) > 1 else "—",
                "t": f"{t:.2f}",
                "p": f"{p:.4f}",
                "harm %": f"{100*(cd < -0.01).mean():.1f}%",
            })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(tbl_dir, "T6_perfect_stability.csv"), index=False)

    result = {
        "n_networks": int(n_networks),
        "n_evaluations": int(n_evals),
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas)),
        "se_delta": float(stats.sem(deltas)),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "harmed_pct": float(100 * harmed / len(deltas)),
        "helped_pct": float(100 * helped / len(deltas)),
        "neutral_pct": float(100 * neutral / len(deltas)),
    }

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(deltas, bins=100, color=STRATUM_COLORS["perfect"], alpha=0.7,
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", ls="--", lw=1.5, label="No change")
    ax.axvline(np.mean(deltas), color="black", ls="-", lw=1.5,
               label=f"Mean = {np.mean(deltas):.4f}")
    ax.set_xlabel(u"\u0394fitness")
    ax.set_ylabel("Count")
    ax.set_title(f"Perfect Networks: Plasticity Effect\n"
                 f"(N={n_networks:,} networks, {n_evals:,} evaluations)")
    ax.legend()
    data_min, data_max = np.min(deltas), np.max(deltas)
    padding = (data_max - data_min) * 0.05
    ax.set_xlim(data_min - padding, data_max + padding)

    ax2 = axes[1]
    eta_groups = []
    eta_labels_plot = []
    for e in eta_vals:
        vals = perf[perf["eta"] == e]["delta_fitness"].values
        if len(vals) > 0:
            eta_groups.append(vals)
            eta_labels_plot.append(fmt_eta(e))

    bp = ax2.boxplot(eta_groups, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(STRATUM_COLORS["perfect"])
        patch.set_alpha(0.5)
    ax2.set_xticklabels(eta_labels_plot, rotation=45, ha="right", fontsize=8)
    ax2.axhline(0, color="red", ls="--", lw=1)
    ax2.set_xlabel(u"\u03B7")
    ax2.set_ylabel(u"\u0394fitness")
    ax2.set_title(u"Perfect Network Stability by \u03B7")

    save_figure(fig, fig_dir, "F6_perfect_stability")

    return result


# =====================================================================
# Analysis 7 — Per-network regret
# =====================================================================
def analysis_7_regret(df: pd.DataFrame, fig_dir: str, tbl_dir: str) -> dict:
    """Compute regret: gap between per-network oracle vs fixed param."""
    logger.info("Analysis 7: Per-network regret analysis")

    net_best = df.groupby("network_id")["delta_fitness"].max().rename("oracle_delta")
    net_stratum = df.groupby("network_id")["stratum"].first()

    net_param = df.pivot_table(values="delta_fitness", index="network_id",
                               columns=["eta", "decay"], aggfunc="mean")

    regret_df = net_param.sub(net_best, axis=0).mul(-1)

    mean_regret = regret_df.mean()
    best_fixed_idx = mean_regret.idxmin()
    best_fixed_regret = mean_regret.min()
    oracle_mean = net_best.mean()

    rows = []
    stratum_regret = {}
    for s in STRATA_ORDER:
        nets_in_stratum = net_stratum[net_stratum == s].index
        if len(nets_in_stratum) == 0:
            continue
        oracle_s = net_best.loc[nets_in_stratum]
        regret_s = regret_df.loc[nets_in_stratum]
        mean_reg_s = regret_s.mean()
        best_fixed_s = mean_reg_s.idxmin()
        best_reg_s = mean_reg_s.min()

        stratum_regret[s] = {
            "n_networks": len(nets_in_stratum),
            "oracle_mean_delta": float(oracle_s.mean()),
            "best_fixed_param": str(best_fixed_s),
            "best_fixed_regret": float(best_reg_s),
            "median_oracle_delta": float(oracle_s.median()),
        }
        rows.append({
            "Stratum": STRATUM_LABELS[s],
            "N": len(nets_in_stratum),
            u"Oracle mean \u0394fit": f"{oracle_s.mean():.4f}",
            u"Best fixed (\u03B7, decay)": f"({best_fixed_s[0]:.0e}, {best_fixed_s[1]:.0e})",
            u"Best fixed \u0394fit": f"{(net_param.loc[nets_in_stratum, best_fixed_s].mean()):.4f}",
            "Regret": f"{best_reg_s:.4f}",
            "Regret %": f"{100 * best_reg_s / max(oracle_s.mean(), 1e-8):.1f}%" if oracle_s.mean() > 0 else "—",
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(tbl_dir, "T7_regret.csv"), index=False)

    result = {
        "n_networks": int(len(net_best)),
        "oracle_mean_delta": float(oracle_mean),
        "best_fixed_param": str(best_fixed_idx),
        "best_fixed_regret": float(best_fixed_regret),
        "stratum_regret": stratum_regret,
    }

    # Figure: stacked bar chart — oracle as full height, best-fixed overlaid
    # Recompute values in delta_reward scale (0-500) for readable bar heights
    CLR_FIXED = "#4477AA"   # blue
    CLR_ORACLE = "#EE6677"  # red-pink

    strata_present = [s for s in STRATA_ORDER if s in stratum_regret]
    labels = [STRATUM_LABELS[s].split(" [")[0] for s in strata_present]
    x = np.arange(len(strata_present))
    width = 0.55

    oracle_vals = []
    fixed_vals = []
    regret_pcts = []
    for s in strata_present:
        sub = df[df["stratum"] == s]
        oracle_per_net = sub.groupby("network_id")["delta_reward"].max()
        oracle_mean = oracle_per_net.mean()
        oracle_vals.append(oracle_mean)

        cell_means = sub.groupby(["eta", "decay"])["delta_reward"].mean()
        best_param = cell_means.idxmax()
        best_sub = sub[(sub["eta"] == best_param[0]) & (sub["decay"] == best_param[1])]
        fixed_mean = best_sub.groupby("network_id")["delta_reward"].mean().mean()
        fixed_vals.append(fixed_mean)

        if oracle_mean > 0:
            regret_pcts.append(100 * (1 - fixed_mean / oracle_mean))
        else:
            regret_pcts.append(0)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Oracle as full bar (light pink, background)
    ax.bar(x, oracle_vals, width, color=CLR_ORACLE, edgecolor="black",
           linewidth=0.5, alpha=0.35, label="Regret (unreached by best-fixed)")
    # Best-fixed overlaid on top (solid blue, foreground)
    ax.bar(x, fixed_vals, width, color=CLR_FIXED, edgecolor="black",
           linewidth=0.5, label="Best fixed parameter")

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean $\\Delta r$ (reward points)")
    ax.legend(fontsize=10, loc="upper left")

    for i in range(len(strata_present)):
        f_val = fixed_vals[i]
        o_val = oracle_vals[i]
        reg = regret_pcts[i]

        # Oracle value at top
        ax.text(i, max(o_val, 0) + 2.5, f"{o_val:+.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=CLR_ORACLE)

        # Fixed value inside bar
        if f_val > 3:
            ax.text(i, f_val / 2, f"{f_val:+.1f}",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white")

        # Regret % in the gap
        if o_val > 1 and reg > 10:
            mid_gap = (f_val + o_val) / 2
            ax.text(i, mid_gap, f"{reg:.0f}%",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="#333333")

    # Add headroom so top label doesn't clip
    y_max = max(oracle_vals) if oracle_vals else 10
    ax.set_ylim(top=y_max * 1.15)

    fig.tight_layout()
    save_figure(fig, fig_dir, "F7_regret")

    return result


# =====================================================================
# Analysis 8 — Failure mode characterization
# =====================================================================
def analysis_8_failure_modes(df: pd.DataFrame, fig_dir: str, tbl_dir: str) -> dict:
    """Characterize when and why plasticity hurts."""
    logger.info("Analysis 8: Failure mode characterization")

    df_active = df[df["eta"] != 0].copy()
    df_active["abs_eta"] = df_active["eta"].abs()
    df_active["sign"] = np.where(df_active["eta"] < 0, "Anti-Hebbian", "Hebbian")

    total_harm = (df_active["delta_fitness"] < -0.01).mean()

    # ── Per-stratum statistics (CSV output unchanged) ────────────────
    rows = []
    stratum_results = {}
    for s in STRATA_ORDER:
        sub = df_active[df_active["stratum"] == s]
        if sub.empty:
            continue

        harm_rate = (sub["delta_fitness"] < -0.01).mean()
        severe_harm = (sub["delta_fitness"] < -0.05).mean()
        mean_delta_when_harmed = sub[sub["delta_fitness"] < -0.01]["delta_fitness"].mean()

        harmed_mask = sub["delta_fitness"] < -0.01
        r_abs_eta, p_abs_eta = stats.pearsonr(sub["abs_eta"], harmed_mask.astype(float))

        worst_by_param = sub.groupby(["eta", "decay"])["delta_fitness"].mean()
        worst_param = worst_by_param.idxmin()
        worst_val = worst_by_param.min()

        stratum_results[s] = {
            "harm_rate": float(harm_rate),
            "severe_harm_rate": float(severe_harm),
            "mean_delta_when_harmed": float(mean_delta_when_harmed) if np.isfinite(mean_delta_when_harmed) else None,
            "worst_param": {"eta": worst_param[0], "decay": worst_param[1]},
            "worst_mean_delta": float(worst_val),
            "abs_eta_harm_corr": float(r_abs_eta),
        }

        rows.append({
            "Stratum": STRATUM_LABELS[s],
            u"Harm rate (\u0394<-0.01)": f"{100*harm_rate:.1f}%",
            u"Severe harm (\u0394<-0.05)": f"{100*severe_harm:.1f}%",
            u"Mean \u0394 when harmed": f"{mean_delta_when_harmed:.4f}" if np.isfinite(mean_delta_when_harmed) else "\u2014",
            u"Worst (\u03B7, decay)": f"({worst_param[0]:.0e}, {worst_param[1]:.0e})",
            u"Worst mean \u0394": f"{worst_val:.4f}",
            u"|\u03B7|\u2194harm r": f"{r_abs_eta:.3f}",
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(tbl_dir, "T8_failure_modes.csv"), index=False)

    # ── Figure: 2 panels ────────────────────────────────────────────
    display_strata = [s for s in STRATA_ORDER if s != "weak" and s in stratum_results]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5),
                                      gridspec_kw={"width_ratios": [1, 1.5]})

    # ── Panel (a): Per-tier harm rates (grouped bar chart) ──────────
    x_pos = np.arange(len(display_strata))
    bar_w = 0.35

    harm_vals = [stratum_results[s]["harm_rate"] * 100 for s in display_strata]
    severe_vals = [stratum_results[s]["severe_harm_rate"] * 100 for s in display_strata]
    bar_colors = [STRATUM_COLORS[s] for s in display_strata]

    bars_harm = ax_a.bar(x_pos - bar_w / 2, harm_vals, bar_w,
                         color=bar_colors, edgecolor="black", linewidth=0.5,
                         label=u"Any harm (\u0394 < \u22120.01)")
    bars_severe = ax_a.bar(x_pos + bar_w / 2, severe_vals, bar_w,
                           color=bar_colors, edgecolor="black", linewidth=0.5,
                           alpha=0.5, hatch="//",
                           label=u"Severe (\u0394 < \u22120.05)")

    for bar in bars_harm:
        h = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width() / 2, h + 1,
                  f"{h:.0f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars_severe:
        h = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width() / 2, h + 1,
                  f"{h:.0f}%", ha="center", va="bottom", fontsize=8)

    ax_a.axhline(50, color="grey", ls="--", lw=0.8, alpha=0.5)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels([STRATUM_LABELS[s].split(" [")[0] for s in display_strata],
                         fontsize=9, rotation=20, ha="right")
    ax_a.set_ylabel("Configs causing harm (%)")
    ax_a.text(0.02, 0.97, "(a)", transform=ax_a.transAxes,
              fontsize=11, fontweight="bold", va="top")
    ax_a.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    ax_a.set_ylim(0, max(harm_vals) * 1.25)

    # ── Panel (b): Anti-Hebbian harm vs |η| per decay (High-mid) ───
    # plus Hebbian reference band
    ref_tier = "high_mid"
    z = 1.96

    # Highlight two key decay values; gray out the rest
    decay_vals = sorted(df_active["decay"].unique())
    key_decays = {0.01: ("#2166ac", "d=0.01 (recommended)"),
                  0.0:  ("#b2182b", "d=0 (no decay)")}

    # Anti-Hebbian lines per decay value
    sub_anti = df_active[
        (df_active["sign"] == "Anti-Hebbian") & (df_active["stratum"] == ref_tier)
    ]
    for d_val in decay_vals:
        sub_d = sub_anti[sub_anti["decay"] == d_val]
        if sub_d.empty:
            continue
        harm_by_eta = sub_d.groupby("abs_eta").apply(
            lambda g: pd.Series({
                "harm_rate": (g["delta_fitness"] < -0.01).mean(),
                "n": len(g),
            }),
            include_groups=False,
        )
        hr = harm_by_eta["harm_rate"].values
        n_per = harm_by_eta["n"].values
        ci_lo = np.maximum(0, hr - z * np.sqrt(hr * (1 - hr) / n_per))
        ci_hi = np.minimum(1, hr + z * np.sqrt(hr * (1 - hr) / n_per))
        eta_idx = harm_by_eta.index.values

        if d_val in key_decays:
            color, label = key_decays[d_val]
            lw, ls, alpha, zorder = 2.5, "-", 0.15, 5
        else:
            color, label = "#999999", None
            lw, ls, alpha, zorder = 0.8, "--", 0.0, 1

        ax_b.plot(eta_idx, hr * 100, marker="o" if d_val in key_decays else None,
                  markersize=3, label=label,
                  color=color, lw=lw, ls=ls, zorder=zorder)
        if alpha > 0:
            ax_b.fill_between(eta_idx, ci_lo * 100, ci_hi * 100,
                              alpha=alpha, color=color)

    # Add a single "other decay values" entry to the legend
    ax_b.plot([], [], color="#999999", ls="--", lw=0.8,
              label="other decay values")

    ax_b.axhline(50, color="grey", ls="--", lw=0.8, alpha=0.5)
    ax_b.set_xlabel(u"|\u03B7|")
    ax_b.set_ylabel("Harm rate (%)")
    ax_b.text(0.02, 0.97, "(b)", transform=ax_b.transAxes,
              fontsize=11, fontweight="bold", va="top")
    ax_b.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.06, 1.0),
                ncol=1)
    ax_b.set_ylim(0, 105)

    fig.tight_layout()
    save_figure(fig, fig_dir, "F8_failure_modes")

    return {
        "overall_harm_rate": float(total_harm),
        "stratum_results": stratum_results,
    }


# =====================================================================
# Analysis 9 — 3D Surface Plots
# =====================================================================
def analysis_9_3d_surfaces(df: pd.DataFrame, fig_dir: str) -> None:
    """Generate 3D surface plots showing the eta x decay landscape per stratum."""
    logger.info("Analysis 9: 3D surface plots per stratum")

    eta_vals = sorted(df["eta"].unique())
    decay_vals = sorted(df["decay"].unique())

    decay_indices = np.arange(len(decay_vals))
    eta_indices = np.arange(len(eta_vals))
    X, Y = np.meshgrid(decay_indices, eta_indices)

    strata_to_plot = STRATA_ORDER

    # Panel figure: all strata in 2x3
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(
        u"B0.5 Plasticity Landscape: Mean \u0394fitness (\u03B7 \u00D7 decay)",
        fontsize=15, y=0.98,
    )

    for idx, s in enumerate(strata_to_plot):
        sub = df[df["stratum"] == s]
        n_net = sub["network_id"].nunique()
        piv = sub.pivot_table(
            values="delta_fitness", index="eta", columns="decay", aggfunc="mean",
        ).reindex(index=eta_vals, columns=decay_vals)
        Z = piv.values

        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")

        vabs = max(abs(np.nanmin(Z)), abs(np.nanmax(Z)), 1e-6)
        norm = plt.Normalize(-vabs, vabs)

        ax.plot_surface(
            X, Y, Z,
            cmap="RdYlGn", norm=norm,
            alpha=0.85, edgecolor="gray", linewidth=0.3,
        )
        ax.plot_surface(X, Y, np.zeros_like(Z), alpha=0.15, color="gray")

        ax.set_xticks(decay_indices)
        ax.set_xticklabels([fmt_decay(d) for d in decay_vals], fontsize=7, rotation=15)
        ax.set_yticks(eta_indices[::2])
        ax.set_yticklabels([fmt_eta(eta_vals[i]) for i in range(0, len(eta_vals), 2)], fontsize=7)

        ax.set_xlabel("Decay", fontsize=9, labelpad=8)
        ax.set_ylabel(u"\u03B7", fontsize=9, labelpad=8)
        ax.set_zlabel(u"\u0394fitness", fontsize=9, labelpad=5)

        label = STRATUM_LABELS[s].split(" [")[0]
        ax.set_title(f"{label} (N={n_net:,})", fontsize=11, pad=5)
        ax.view_init(elev=25, azim=-55)
        ax.tick_params(axis="z", labelsize=7)

    ax_empty = fig.add_subplot(2, 3, 6)
    ax_empty.axis("off")
    ax_empty.text(
        0.5, 0.5,
        u"3D surfaces show the \u03B7 \u00D7 decay\n"
        "landscape for each performance\n"
        u"stratum. The gray plane marks\n"
        u"\u0394fitness = 0 (no change).\n\n"
        "Green = improvement\n"
        "Red = degradation",
        ha="center", va="center", fontsize=12,
        transform=ax_empty.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, fig_dir, "F9_3d_surfaces_all_strata")

    # Individual high-res 3D surfaces
    for s in strata_to_plot:
        sub = df[df["stratum"] == s]
        n_net = sub["network_id"].nunique()
        piv = sub.pivot_table(
            values="delta_fitness", index="eta", columns="decay", aggfunc="mean",
        ).reindex(index=eta_vals, columns=decay_vals)
        Z = piv.values
        vabs = max(abs(np.nanmin(Z)), abs(np.nanmax(Z)), 1e-6)
        norm = plt.Normalize(-vabs, vabs)

        fig2 = plt.figure(figsize=(12, 8))
        ax2 = fig2.add_subplot(111, projection="3d")

        ax2.plot_surface(
            X, Y, Z,
            cmap="RdYlGn", norm=norm,
            alpha=0.85, edgecolor="gray", linewidth=0.3,
        )
        ax2.plot_surface(X, Y, np.zeros_like(Z), alpha=0.12, color="gray")

        ax2.set_xticks(decay_indices)
        ax2.set_xticklabels([fmt_decay(d) for d in decay_vals], fontsize=9)
        ax2.set_yticks(eta_indices[::2])
        ax2.set_yticklabels([fmt_eta(eta_vals[i]) for i in range(0, len(eta_vals), 2)], fontsize=9)
        ax2.set_xlabel("Decay", fontsize=11, labelpad=10)
        ax2.set_ylabel(u"\u03B7", fontsize=11, labelpad=10)
        ax2.set_zlabel(u"Mean \u0394fitness", fontsize=11, labelpad=8)

        label = STRATUM_LABELS[s]
        ax2.set_title(f"Plasticity Landscape \u2014 {label} (N={n_net:,})", fontsize=13)
        ax2.view_init(elev=25, azim=-55)

        fig2.colorbar(
            cm.ScalarMappable(norm=norm, cmap="RdYlGn"),
            ax=ax2, shrink=0.6, label=u"Mean \u0394fitness",
        )

        save_figure(fig2, fig_dir, f"F9_3d_surface_{s}")

    logger.info("Analysis 9: Done — panel + individual 3D surfaces saved")


# =====================================================================
# Analysis 10 — Outcome Distribution Stacked Bars
# =====================================================================
def analysis_10_outcome_distributions(df: pd.DataFrame, fig_dir: str, tbl_dir: str) -> None:
    """Stacked bars: % improved / unchanged / harmed per stratum."""
    logger.info("Analysis 10: Outcome distribution stacked bars")

    fig = plt.figure(figsize=(18, 7))
    gs = GridSpec(1, 3, wspace=0.3)

    rows = []

    for panel_idx, (mode_label, mode) in enumerate([
        ("Best Fixed Parameter", "best_fixed"),
        (u"Active Plasticity Only (\u03B7\u22600)", "active_only"),
        ("Per-Network Oracle", "oracle"),
    ]):
        ax = fig.add_subplot(gs[0, panel_idx])

        strata_labels = []
        pct_improved = []
        pct_unchanged = []
        pct_harmed = []

        for s in STRATA_ORDER:
            sub = df[df["stratum"] == s]
            n_net = sub["network_id"].nunique()
            if sub.empty:
                continue

            if mode == "best_fixed":
                best_cell = sub.groupby(["eta", "decay"])["delta_fitness"].mean()
                best_param = best_cell.idxmax()
                data = sub[
                    (sub["eta"] == best_param[0]) & (sub["decay"] == best_param[1])
                ]["delta_fitness"]
            elif mode == "all_params":
                data = sub["delta_fitness"]
            elif mode == "oracle":
                data = sub.groupby("network_id")["delta_fitness"].max()
            elif mode == "active_only":
                data = sub[sub["eta"] != 0]["delta_fitness"]
            else:
                data = sub["delta_fitness"]

            n = len(data)
            improved = (data > IMPROVE_THRESHOLD).sum() / n * 100
            harmed = (data < HARM_THRESHOLD).sum() / n * 100
            unchanged = 100 - improved - harmed

            label = STRATUM_LABELS[s].split(" [")[0]
            strata_labels.append(f"{label}\n(N={n_net:,})")
            pct_improved.append(improved)
            pct_unchanged.append(unchanged)
            pct_harmed.append(harmed)

            rows.append({
                "Mode": mode_label,
                "Stratum": STRATUM_LABELS[s],
                "N_networks": n_net,
                "N_evaluations": n,
                "% Improved": f"{improved:.1f}",
                "% Unchanged": f"{unchanged:.1f}",
                "% Harmed": f"{harmed:.1f}",
            })

        x = np.arange(len(strata_labels))
        width = 0.65

        ax.bar(x, pct_improved, width, label="Improved", color="#2ca02c", alpha=0.85)
        ax.bar(x, pct_unchanged, width, bottom=pct_improved,
               label="Unchanged", color="#aaaaaa", alpha=0.7)
        ax.bar(x, pct_harmed, width,
               bottom=[i + u for i, u in zip(pct_improved, pct_unchanged)],
               label="Harmed", color="#d62728", alpha=0.85)

        for i in range(len(strata_labels)):
            imp = pct_improved[i]
            unc = pct_unchanged[i]
            har = pct_harmed[i]
            if imp > 5:
                ax.text(i, imp / 2, f"{imp:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
            if unc > 8:
                ax.text(i, imp + unc / 2, f"{unc:.0f}%", ha="center", va="center",
                        fontsize=9, color="black")
            if har > 5:
                ax.text(i, imp + unc + har / 2, f"{har:.0f}%", ha="center",
                        va="center", fontsize=9, fontweight="bold", color="white")

        ax.set_xticks(x)
        ax.set_xticklabels(strata_labels, fontsize=9)
        ax.set_ylabel("% of evaluations")
        ax.set_title(mode_label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        u"B0.5 Outcome Distribution: Improved vs Unchanged vs Harmed",
        fontsize=15, y=1.03,
    )
    save_figure(fig, fig_dir, "F10_outcome_distributions")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(tbl_dir, "T10_outcome_distributions.csv"), index=False)

    logger.info("Analysis 10: Done")


# =====================================================================
# Analysis 11 — Per-Grid-Cell Outcome Heatmaps
# =====================================================================
def analysis_11_outcome_heatmaps(df: pd.DataFrame, fig_dir: str) -> None:
    """For each stratum, show eta x decay heatmaps of % improved and % harmed."""
    logger.info("Analysis 11: Per-grid-cell outcome heatmaps")

    eta_vals = sorted(df["eta"].unique())
    decay_vals = sorted(df["decay"].unique())

    interesting_strata = ["low_mid", "high_mid", "near_perfect", "perfect"]

    fig, axes = plt.subplots(len(interesting_strata), 2, figsize=(14, 4 * len(interesting_strata)))

    for row_idx, s in enumerate(interesting_strata):
        sub = df[df["stratum"] == s]
        n_net = sub["network_id"].nunique()

        improved_piv = sub.assign(
            improved_flag=(sub["delta_fitness"] > IMPROVE_THRESHOLD).astype(float)
        ).pivot_table(
            values="improved_flag", index="eta", columns="decay", aggfunc="mean",
        ).reindex(index=eta_vals, columns=decay_vals) * 100

        harmed_piv = sub.assign(
            harmed_flag=(sub["delta_fitness"] < HARM_THRESHOLD).astype(float)
        ).pivot_table(
            values="harmed_flag", index="eta", columns="decay", aggfunc="mean",
        ).reindex(index=eta_vals, columns=decay_vals) * 100

        label = STRATUM_LABELS[s].split(" [")[0]

        ax_imp = axes[row_idx, 0]
        im1 = ax_imp.imshow(improved_piv.values, cmap="Greens", aspect="auto",
                            origin="upper", vmin=0, vmax=100)
        ax_imp.set_title(f"{label} \u2014 % Improved (N={n_net:,})", fontsize=11)
        ax_imp.set_xticks(range(len(decay_vals)))
        ax_imp.set_xticklabels([fmt_decay(d) for d in decay_vals], fontsize=8)
        ax_imp.set_yticks(range(len(eta_vals)))
        ax_imp.set_yticklabels([fmt_eta(e) for e in eta_vals], fontsize=8)
        ax_imp.set_xlabel("Decay")
        ax_imp.set_ylabel(u"\u03B7")
        for i in range(len(eta_vals)):
            for j in range(len(decay_vals)):
                v = improved_piv.values[i, j]
                if np.isfinite(v):
                    txt_col = "white" if v > 60 else "black"
                    ax_imp.text(j, i, f"{v:.0f}%", ha="center", va="center",
                                fontsize=6.5, color=txt_col)
        plt.colorbar(im1, ax=ax_imp, shrink=0.8, label="% Improved")

        ax_hrm = axes[row_idx, 1]
        im2 = ax_hrm.imshow(harmed_piv.values, cmap="Reds", aspect="auto",
                            origin="upper", vmin=0, vmax=100)
        ax_hrm.set_title(f"{label} \u2014 % Harmed (N={n_net:,})", fontsize=11)
        ax_hrm.set_xticks(range(len(decay_vals)))
        ax_hrm.set_xticklabels([fmt_decay(d) for d in decay_vals], fontsize=8)
        ax_hrm.set_yticks(range(len(eta_vals)))
        ax_hrm.set_yticklabels([fmt_eta(e) for e in eta_vals], fontsize=8)
        ax_hrm.set_xlabel("Decay")
        ax_hrm.set_ylabel(u"\u03B7")
        for i in range(len(eta_vals)):
            for j in range(len(decay_vals)):
                v = harmed_piv.values[i, j]
                if np.isfinite(v):
                    txt_col = "white" if v > 60 else "black"
                    ax_hrm.text(j, i, f"{v:.0f}%", ha="center", va="center",
                                fontsize=6.5, color=txt_col)
        plt.colorbar(im2, ax=ax_hrm, shrink=0.8, label="% Harmed")

    fig.suptitle(
        u"B0.5 Outcome Rates by Grid Cell (\u03B7 \u00D7 decay)",
        fontsize=14, y=1.01,
    )
    save_figure(fig, fig_dir, "F11_outcome_heatmaps")

    logger.info("Analysis 11: Done")


# =====================================================================
# Analysis 12 — Severity Distribution Violin Plots
# =====================================================================
def analysis_12_severity_distributions(df: pd.DataFrame, fig_dir: str) -> None:
    """Violin plots: distribution of delta-fitness per stratum, improved vs harmed."""
    logger.info("Analysis 12: Severity distribution violin plots")

    interesting_strata = [s for s in STRATA_ORDER if s != "weak"]

    fig, axes = plt.subplots(1, len(interesting_strata), figsize=(5 * len(interesting_strata), 6))
    if len(interesting_strata) == 1:
        axes = [axes]

    for idx, s in enumerate(interesting_strata):
        ax = axes[idx]
        sub = df[(df["stratum"] == s) & (df["eta"] != 0)]
        n_net = sub["network_id"].nunique()

        improved = sub[sub["delta_fitness"] > IMPROVE_THRESHOLD]["delta_fitness"].values
        harmed = sub[sub["delta_fitness"] < HARM_THRESHOLD]["delta_fitness"].values

        if len(improved) > 10:
            p99 = np.percentile(improved, 99)
            improved_clipped = np.clip(improved, None, p99)
        else:
            improved_clipped = improved
        if len(harmed) > 10:
            p1 = np.percentile(harmed, 1)
            harmed_clipped = np.clip(harmed, p1, None)
        else:
            harmed_clipped = harmed

        data = []
        labels_list = []
        colors_list = []

        if len(improved_clipped) > 10:
            data.append(improved_clipped)
            labels_list.append(f"Improved\n({len(improved):,})")
            colors_list.append("#2ca02c")
        if len(harmed_clipped) > 10:
            data.append(harmed_clipped)
            labels_list.append(f"Harmed\n({len(harmed):,})")
            colors_list.append("#d62728")

        if data:
            parts = ax.violinplot(data, positions=range(len(data)),
                                  showmeans=True, showmedians=True)
            for i, body in enumerate(parts["bodies"]):
                body.set_facecolor(colors_list[i])
                body.set_alpha(0.7)
            for pname in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
                if pname in parts:
                    parts[pname].set_color("black")

            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(labels_list, fontsize=9)

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        label = STRATUM_LABELS[s].split(" [")[0]
        ax.set_title(f"{label} (N={n_net:,})", fontsize=12)
        ax.set_ylabel(u"\u0394fitness")

    fig.suptitle(
        u"B0.5 Severity: How Much Change When Improved/Harmed? (active \u03B7 only)",
        fontsize=14, y=1.02,
    )
    save_figure(fig, fig_dir, "F12_severity_distributions")

    logger.info("Analysis 12: Done")


# =====================================================================
# Analysis 13 — eta x decay Interaction Effects
# =====================================================================
def analysis_13_interaction_effects(df: pd.DataFrame, fig_dir: str, tbl_dir: str) -> None:
    """Two-way ANOVA with interaction term (eta x decay) per stratum."""
    logger.info("Analysis 13: eta x decay Interaction Effects")

    df_active = df[df["eta"] != 0].copy()

    anova_rows = []

    strata_present = strata_present_in(df_active)
    n_strata = len(strata_present)
    ncols = min(3, n_strata)
    nrows = (n_strata + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for idx, s in enumerate(strata_present):
        sub = df_active[df_active["stratum"] == s].copy()
        ax = axes[idx // ncols][idx % ncols]

        eta_vals = sorted(sub["eta"].unique())
        decay_vals = sorted(sub["decay"].unique())

        cell_means = sub.groupby(["eta", "decay"])["delta_fitness"].mean()
        cell_counts = sub.groupby(["eta", "decay"])["delta_fitness"].count()
        grand_mean = sub["delta_fitness"].mean()
        n_total = len(sub)

        eta_means = sub.groupby("eta")["delta_fitness"].mean()
        decay_means = sub.groupby("decay")["delta_fitness"].mean()

        ss_eta = 0.0
        for e in eta_vals:
            n_e = (sub["eta"] == e).sum()
            ss_eta += n_e * (eta_means[e] - grand_mean) ** 2

        ss_decay = 0.0
        for d in decay_vals:
            n_d = (sub["decay"] == d).sum()
            ss_decay += n_d * (decay_means[d] - grand_mean) ** 2

        ss_cells = 0.0
        for (e, d), m in cell_means.items():
            n_ed = cell_counts[(e, d)]
            ss_cells += n_ed * (m - grand_mean) ** 2

        ss_interact = ss_cells - ss_eta - ss_decay

        ss_error = 0.0
        for (e, d), grp in sub.groupby(["eta", "decay"]):
            cell_m = cell_means[(e, d)]
            ss_error += ((grp["delta_fitness"] - cell_m) ** 2).sum()

        a = len(eta_vals)
        b = len(decay_vals)
        df_eta = a - 1
        df_decay = b - 1
        df_interact = (a - 1) * (b - 1)
        df_error = n_total - a * b

        ms_eta = ss_eta / df_eta if df_eta > 0 else 0
        ms_decay = ss_decay / df_decay if df_decay > 0 else 0
        ms_interact = ss_interact / df_interact if df_interact > 0 else 0
        ms_error = ss_error / df_error if df_error > 0 else 1e-12

        f_eta = ms_eta / ms_error if ms_error > 0 else 0
        f_decay = ms_decay / ms_error if ms_error > 0 else 0
        f_interact = ms_interact / ms_error if ms_error > 0 else 0

        p_eta = 1.0 - stats.f.cdf(f_eta, df_eta, df_error) if df_error > 0 else 1.0
        p_decay = 1.0 - stats.f.cdf(f_decay, df_decay, df_error) if df_error > 0 else 1.0
        p_interact = (
            1.0 - stats.f.cdf(f_interact, df_interact, df_error)
            if df_error > 0
            else 1.0
        )

        partial_eta2_eta = ss_eta / (ss_eta + ss_error) if (ss_eta + ss_error) > 0 else 0
        partial_eta2_decay = ss_decay / (ss_decay + ss_error) if (ss_decay + ss_error) > 0 else 0
        partial_eta2_interact = (
            ss_interact / (ss_interact + ss_error)
            if (ss_interact + ss_error) > 0
            else 0
        )

        for source, ss, dof, ms, f_val, p_val, pe2 in [
            ("eta", ss_eta, df_eta, ms_eta, f_eta, p_eta, partial_eta2_eta),
            ("decay", ss_decay, df_decay, ms_decay, f_decay, p_decay, partial_eta2_decay),
            ("eta x decay", ss_interact, df_interact, ms_interact, f_interact, p_interact, partial_eta2_interact),
            ("Residual", ss_error, df_error, ms_error, np.nan, np.nan, np.nan),
        ]:
            anova_rows.append({
                "Stratum": STRATUM_LABELS[s],
                "Source": source,
                "SS": ss,
                "df": dof,
                "MS": ms,
                "F": f_val,
                "p": p_val,
                "partial_eta2": pe2,
            })

        # Interaction plot
        cmap = plt.cm.viridis
        for j, d in enumerate(decay_vals):
            color = cmap(j / max(len(decay_vals) - 1, 1))
            means_line = []
            for e in eta_vals:
                key = (e, d)
                if key in cell_means.index:
                    means_line.append(cell_means[key])
                else:
                    means_line.append(np.nan)
            ax.plot(
                range(len(eta_vals)),
                means_line,
                marker="o",
                markersize=3,
                linewidth=1.2,
                color=color,
                label=f"decay={fmt_decay(d)}",
            )

        # Show every other eta tick for readability
        ax.set_xticks(range(0, len(eta_vals), 2))
        ax.set_xticklabels([fmt_eta(eta_vals[i]) for i in range(0, len(eta_vals), 2)],
                           rotation=45, ha="right", fontsize=6)
        ax.set_ylabel(r"Mean $\Delta$fitness")
        ax.set_title(
            f"{STRATUM_LABELS[s].split(' [')[0]}\n"
            f"Interaction F={f_interact:.1f}, p={p_interact:.2e}, "
            r"$\eta^2_p$" + f"={partial_eta2_interact:.4f}",
            fontsize=10,
        )
        ax.axhline(0, color="grey", lw=0.5, ls="--")

    for idx in range(n_strata, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared legend outside the subplots
    handles, labels_leg = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", ncol=len(decay_vals),
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        r"$\eta \times$ decay Interaction Effects on $\Delta$fitness",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, fig_dir, "F13_interaction_effects")

    tbl = pd.DataFrame(anova_rows)
    tbl.to_csv(os.path.join(tbl_dir, "T13_interaction_anova.csv"), index=False)
    logger.info("  Saved T13_interaction_anova.csv")

    for s in strata_present:
        interact_row = [
            r
            for r in anova_rows
            if r["Stratum"] == STRATUM_LABELS[s] and r["Source"] == "eta x decay"
        ]
        if interact_row:
            r = interact_row[0]
            sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "ns"
            logger.info(
                f"  {STRATUM_LABELS[s]:30s} interaction: "
                f"F({r['df']:.0f})={r['F']:.2f}, p={r['p']:.2e}, "
                f"partial eta2={r['partial_eta2']:.5f} [{sig}]"
            )


# =====================================================================
# Analysis 14 — Dose-Response Curves
# =====================================================================
def analysis_14_dose_response(df: pd.DataFrame, fig_dir: str) -> None:
    """Line plot of mean delta_fitness vs |eta|, split by sign, per stratum."""
    logger.info("Analysis 14: Dose-Response Curves")

    df_active = df[df["eta"] != 0].copy()
    df_active["abs_eta"] = df_active["eta"].abs()
    df_active["sign"] = np.where(df_active["eta"] < 0, "Anti-Hebbian", "Hebbian")

    strata_present = strata_present_in(df_active)
    n_strata = len(strata_present)
    ncols = min(3, n_strata)
    nrows = (n_strata + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for idx, s in enumerate(strata_present):
        sub = df_active[df_active["stratum"] == s]
        ax = axes[idx // ncols][idx % ncols]

        for sign_label, color, marker in [
            ("Anti-Hebbian", "#1f77b4", "s"),
            ("Hebbian", "#d62728", "o"),
        ]:
            grp = sub[sub["sign"] == sign_label]
            if grp.empty:
                continue
            agg = grp.groupby("abs_eta")["delta_fitness"].agg(["mean", "std", "count"])
            agg["se"] = agg["std"] / np.sqrt(agg["count"])

            ax.errorbar(
                agg.index,
                agg["mean"],
                yerr=1.96 * agg["se"],
                marker=marker,
                markersize=5,
                linewidth=1.5,
                capsize=3,
                label=sign_label,
                color=color,
            )

        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_xlabel(r"|$\eta$|")
        ax.set_ylabel(r"Mean $\Delta$fitness")
        ax.set_title(STRATUM_LABELS[s].split(" [")[0], fontsize=11)
        ax.set_xscale("log")
        ax.legend(fontsize=9)

    for idx in range(n_strata, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        r"Dose-Response: Mean $\Delta$fitness vs |$\eta$| by Sign",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, fig_dir, "F14_dose_response")


# =====================================================================
# Analysis 15 — Network Topology Correlates
# =====================================================================
def analysis_15_topology_correlates(
    df: pd.DataFrame, topo_df: pd.DataFrame, fig_dir: str, tbl_dir: str
) -> None:
    """Correlate structural features with plasticity responsiveness."""
    logger.info("Analysis 15: Network Topology Correlates")

    df_active = df[df["eta"] != 0].copy()

    oracle = df_active.groupby("network_id").agg(
        oracle_delta_fitness=("delta_fitness", "max"),
        oracle_delta_reward=("delta_reward", "max"),
    )

    merged = topo_df.merge(oracle, on="network_id", how="inner")
    logger.info(f"  Merged topology+oracle: {len(merged):,} networks")

    topo_features = [
        ("neurons", "Neuron count"),
        ("connections", "Connection count"),
        ("density", "Network density"),
        ("max_axon_length", "Max axon length"),
        ("self_connect_fraction", "Self-connect fraction"),
        ("diffusion_rate", "Diffusion rate"),
        ("division_threshold", "Division threshold"),
        ("axon_growth_threshold", "Axon growth threshold"),
        ("axon_connect_threshold", "Axon connect threshold"),
        ("weight_adjustment_rate", "Weight adj. rate"),
    ]

    focus_strata = ["low_mid", "high_mid", "near_perfect"]
    strata_present = [s for s in focus_strata if s in merged["stratum"].unique()]

    # Correlation table
    corr_rows = []
    for s in strata_present:
        sub = merged[merged["stratum"] == s]
        for feat_col, feat_label in topo_features:
            if feat_col not in sub.columns:
                continue
            valid = sub[[feat_col, "oracle_delta_fitness"]].dropna()
            if len(valid) < 10:
                continue
            r_pearson, p_pearson = stats.pearsonr(valid[feat_col], valid["oracle_delta_fitness"])
            r_spearman, p_spearman = stats.spearmanr(valid[feat_col], valid["oracle_delta_fitness"])
            corr_rows.append({
                "Stratum": STRATUM_LABELS[s],
                "Feature": feat_label,
                "N": len(valid),
                "Pearson_r": r_pearson,
                "Pearson_p": p_pearson,
                "Spearman_rho": r_spearman,
                "Spearman_p": p_spearman,
            })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(tbl_dir, "T15_topology_correlations.csv"), index=False)
    logger.info("  Saved T15_topology_correlations.csv")

    # Pick top features by Spearman rho
    top_features_set = set()
    if not corr_df.empty:
        top_by_rho = corr_df.reindex(
            corr_df["Spearman_rho"].abs().sort_values(ascending=False).index
        )
        seen = set()
        for _, row in top_by_rho.iterrows():
            if row["Feature"] not in seen:
                seen.add(row["Feature"])
                top_features_set.add(row["Feature"])
            if len(top_features_set) >= 4:
                break

    if len(top_features_set) < 4:
        for _, label in topo_features[:4]:
            top_features_set.add(label)
    top_features = [
        (col, label)
        for col, label in topo_features
        if label in top_features_set
    ][:4]

    # Scatter plots
    fig, axes = plt.subplots(
        len(strata_present),
        len(top_features),
        figsize=(5 * len(top_features), 4 * len(strata_present)),
        squeeze=False,
    )

    for row_idx, s in enumerate(strata_present):
        sub = merged[merged["stratum"] == s]
        color = STRATUM_COLORS[s]
        for col_idx, (feat_col, feat_label) in enumerate(top_features):
            ax = axes[row_idx][col_idx]
            valid = sub[[feat_col, "oracle_delta_fitness"]].dropna()
            if len(valid) < 10:
                ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
                continue

            x_data = valid[feat_col].values
            y_data = valid["oracle_delta_fitness"].values

            # Use hexbin for large-N strata to show density instead of blobs
            if len(x_data) > 500:
                hb = ax.hexbin(x_data, y_data, gridsize=25, cmap="YlOrRd",
                               mincnt=1, alpha=0.85)
                plt.colorbar(hb, ax=ax, shrink=0.6, label="Count")
            else:
                ax.scatter(x_data, y_data, alpha=0.25, s=12, color=color,
                           edgecolors="none")

            slope, intercept, r_val, p_val, se = stats.linregress(x_data, y_data)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="black", lw=1.5, ls="--")

            rho, p_sp = stats.spearmanr(x_data, y_data)

            ax.set_xlabel(feat_label, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(r"Oracle $\Delta$fitness", fontsize=9)
            ax.set_title(f"r={r_val:.3f}, rho={rho:.3f}", fontsize=9)

            if col_idx == 0:
                ax.annotate(
                    STRATUM_LABELS[s].split(" [")[0],
                    xy=(0.02, 0.96),
                    xycoords="axes fraction",
                    fontsize=10,
                    fontweight="bold",
                    va="top",
                    color=color,
                )

    fig.suptitle(
        "Network Topology vs Plasticity Responsiveness (Oracle " r"$\Delta$fitness)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, fig_dir, "F15_topology_correlates")


# =====================================================================
# Analysis 16 — Stratum Transition Analysis
# =====================================================================
def analysis_16_stratum_transitions(df: pd.DataFrame, fig_dir: str, tbl_dir: str) -> None:
    """Transition matrix: fraction of networks moving between strata."""
    logger.info("Analysis 16: Stratum Transition Analysis")

    strata_names = ["Weak", "Low-mid", "High-mid", "Near-perfect", "Perfect"]

    def compute_transitions(delta_rewards, baseline_rewards):
        mat = np.zeros((5, 5), dtype=int)
        new_rewards = baseline_rewards + delta_rewards
        for br, nr in zip(baseline_rewards, new_rewards):
            old_idx = get_stratum_idx(br)
            new_idx = get_stratum_idx(nr)
            mat[old_idx, new_idx] += 1
        return mat

    df_active = df[df["eta"] != 0].copy()

    # Best param per stratum
    best_params = {}
    for s in STRATA_ORDER:
        sub = df_active[df_active["stratum"] == s]
        if sub.empty:
            continue
        cell_means = sub.groupby(["eta", "decay"])["delta_reward"].mean()
        best_params[s] = cell_means.idxmax()

    best_fixed_rows = []
    for s, (best_eta, best_decay) in best_params.items():
        sub = df_active[
            (df_active["stratum"] == s)
            & (df_active["eta"] == best_eta)
            & (df_active["decay"] == best_decay)
        ]
        best_fixed_rows.append(sub[["network_id", "baseline_avg_reward", "delta_reward"]])
    df_best = pd.concat(best_fixed_rows, ignore_index=True) if best_fixed_rows else pd.DataFrame()

    oracle_per_net = df_active.loc[
        df_active.groupby("network_id")["delta_reward"].idxmax()
    ][["network_id", "baseline_avg_reward", "delta_reward"]].copy()

    mat_best = compute_transitions(
        df_best["delta_reward"].values, df_best["baseline_avg_reward"].values
    ) if not df_best.empty else np.zeros((5, 5), dtype=int)

    mat_oracle = compute_transitions(
        oracle_per_net["delta_reward"].values,
        oracle_per_net["baseline_avg_reward"].values,
    )

    def normalize_rows(mat):
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return mat / row_sums

    frac_best = normalize_rows(mat_best)
    frac_oracle = normalize_rows(mat_oracle)

    # Save tables
    for label, mat, frac in [
        ("best_fixed", mat_best, frac_best),
        ("oracle", mat_oracle, frac_oracle),
    ]:
        tbl_rows = []
        for i, sname in enumerate(strata_names):
            row = {"From": sname}
            for j, tname in enumerate(strata_names):
                row[f"To_{tname}_count"] = int(mat[i, j])
                row[f"To_{tname}_frac"] = float(frac[i, j])
            tbl_rows.append(row)
        tbl = pd.DataFrame(tbl_rows)
        tbl.to_csv(
            os.path.join(tbl_dir, f"T16_stratum_transitions_{label}.csv"),
            index=False,
        )
    logger.info("  Saved T16_stratum_transitions_best_fixed.csv / _oracle.csv")

    # ── Figure: paired transition heatmaps (best-fixed vs oracle) ──
    show_tiers = ["Low-mid", "High-mid", "Near-perfect", "Perfect"]
    show_idx = [strata_names.index(t) for t in show_tiers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, frac, title in [
        (ax1, frac_best, "Best-fixed policy"),
        (ax2, frac_oracle, "Per-network oracle"),
    ]:
        # Extract submatrix for show_tiers
        sub = frac[np.ix_(show_idx, show_idx)] * 100

        im = ax.imshow(sub, cmap="YlGn", vmin=0, vmax=100, aspect="auto")

        # Annotate cells
        for i in range(len(show_tiers)):
            for j in range(len(show_tiers)):
                val = sub[i, j]
                if val == 0:
                    continue
                color = "white" if val > 60 else "black"
                if val < 1:
                    label = f"{val:.1f}%"
                    fontsize = 7
                elif val > 99 and val < 100:
                    label = f"{val:.1f}%"
                    fontsize = 10
                else:
                    label = f"{val:.0f}%"
                    fontsize = 10
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=fontsize,
                        fontweight="bold" if i != j and val > 10 else "normal",
                        color=color)

        ax.set_xticks(range(len(show_tiers)))
        ax.set_xticklabels(show_tiers, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(len(show_tiers)))
        if ax == ax1:
            ax.set_yticklabels(show_tiers, fontsize=9)
            ax.set_ylabel("Original tier", fontsize=11)
        ax.set_xlabel("Tier after plasticity", fontsize=11)
        panel_label = "(a)" if ax == ax1 else "(b)"
        ax.text(-0.02, 1.05, panel_label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom", ha="right")

    fig.tight_layout()
    save_figure(fig, fig_dir, "F16_stratum_transitions")


# =====================================================================
# Analysis 17 — Dose-Response by Decay
# =====================================================================
def analysis_17_decay_dose_response(df: pd.DataFrame, fig_dir: str) -> None:
    """Line plots of mean delta_fitness vs decay, split by |eta| magnitude bins, per stratum."""
    logger.info("Analysis 17: Dose-Response by Decay")

    df_active = df[df["eta"] != 0].copy()
    df_active["abs_eta"] = df_active["eta"].abs()

    eta_abs_vals = sorted(df_active["abs_eta"].unique())
    tercile_edges = np.quantile(eta_abs_vals, [0, 1 / 3, 2 / 3, 1.0])
    bin_labels = [
        f"|eta| low [{tercile_edges[0]:.0e}, {tercile_edges[1]:.0e}]",
        f"|eta| mid ({tercile_edges[1]:.0e}, {tercile_edges[2]:.0e}]",
        f"|eta| high ({tercile_edges[2]:.0e}, {tercile_edges[3]:.0e}]",
    ]
    df_active["eta_bin"] = pd.cut(
        df_active["abs_eta"],
        bins=[tercile_edges[0] - 1e-12, tercile_edges[1], tercile_edges[2], tercile_edges[3] + 1e-12],
        labels=bin_labels,
    )

    strata_present = strata_present_in(df_active)
    n_strata = len(strata_present)
    ncols = min(3, n_strata)
    nrows = (n_strata + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    bin_colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    for idx, s in enumerate(strata_present):
        sub = df_active[df_active["stratum"] == s]
        ax = axes[idx // ncols][idx % ncols]

        for bin_idx, bl in enumerate(bin_labels):
            grp = sub[sub["eta_bin"] == bl]
            if grp.empty:
                continue
            agg = grp.groupby("decay")["delta_fitness"].agg(["mean", "std", "count"])
            agg["se"] = agg["std"] / np.sqrt(agg["count"])

            x_vals = agg.index.values
            x_display = np.where(x_vals == 0, 5e-6, x_vals)

            ax.errorbar(
                x_display,
                agg["mean"],
                yerr=1.96 * agg["se"],
                marker="o",
                markersize=5,
                linewidth=1.5,
                capsize=3,
                label=bl,
                color=bin_colors[bin_idx],
            )

        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_xlabel("Decay")
        ax.set_ylabel(r"Mean $\Delta$fitness")
        ax.set_title(STRATUM_LABELS[s].split(" [")[0], fontsize=11)
        ax.set_xscale("log")
        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    for idx in range(n_strata, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        r"Dose-Response: Mean $\Delta$fitness vs Decay by |$\eta$| Magnitude",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, fig_dir, "F17_decay_dose_response")


# =====================================================================
# Analysis 18 — Oracle Eta Distribution per Stratum
# =====================================================================
def analysis_18_oracle_eta_distribution(df: pd.DataFrame, fig_dir: str, tbl_dir: str) -> None:
    """Box + strip plots of per-network oracle eta (including eta=0) per stratum."""
    logger.info("Analysis 18: Oracle eta distribution per stratum")

    # Oracle across ALL (eta, decay) including eta=0
    oracle_idx = df.groupby("network_id")["delta_fitness"].idxmax()
    oracle_rows = df.loc[oracle_idx, ["network_id", "eta", "decay",
                                       "delta_fitness", "stratum"]].copy()

    # Competent tiers only (skip Weak)
    competent = ["low_mid", "high_mid", "near_perfect", "perfect"]
    strata_present = [s for s in competent if s in oracle_rows["stratum"].unique()]

    # Table
    tbl_rows = []
    for s in strata_present:
        sub = oracle_rows[oracle_rows["stratum"] == s]
        etas = sub["eta"].values
        tbl_rows.append({
            "Stratum": STRATUM_LABELS[s],
            "N": len(sub),
            "Oracle eta mean": f"{np.mean(etas):.4f}",
            "Oracle eta median": f"{np.median(etas):.4f}",
            "Oracle eta std": f"{np.std(etas):.4f}",
            "Oracle eta IQR": f"[{np.percentile(etas, 25):.4f}, {np.percentile(etas, 75):.4f}]",
            "Frac anti-Hebbian": f"{(etas < 0).mean():.2f}",
            "Frac eta=0 (no plasticity best)": f"{(etas == 0).mean():.2f}",
        })
    tbl = pd.DataFrame(tbl_rows)
    tbl.to_csv(os.path.join(tbl_dir, "T18_oracle_eta_distribution.csv"), index=False)

    # Figure: horizontal box + jittered strip, strata on y-axis
    fig, ax = plt.subplots(figsize=(7, 4))

    for i, s in enumerate(strata_present):
        sub = oracle_rows[oracle_rows["stratum"] == s]
        etas = sub["eta"].values
        color = STRATUM_COLORS[s]
        y_pos = len(strata_present) - 1 - i  # top-to-bottom: first stratum at top

        # Jittered strip (behind box)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(etas))
        ax.scatter(etas, y_pos + jitter, s=6, alpha=0.25, color=color,
                   edgecolors="none", zorder=2)

        # Box plot
        bp = ax.boxplot(etas, positions=[y_pos], vert=False, widths=0.4,
                        patch_artist=True, zorder=3,
                        flierprops=dict(marker=".", markersize=2, alpha=0.3),
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(color="grey"),
                        capprops=dict(color="grey"))
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][0].set_edgecolor("black")
        bp["boxes"][0].set_linewidth(0.8)

    # Gate boundary
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.5, zorder=1)

    # y-axis labels
    y_positions = list(range(len(strata_present) - 1, -1, -1))
    y_labels = [STRATUM_LABELS[s].split(" [")[0] for s in strata_present]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Oracle \u03B7", fontsize=10)

    # Re-position gate-off annotations now that xlim is set
    # (redraw after axis limits are finalized)
    # Remove previous text annotations and redo
    for txt in list(ax.texts):
        txt.remove()
    x_left = ax.get_xlim()[0]
    for i, s in enumerate(strata_present):
        sub = oracle_rows[oracle_rows["stratum"] == s]
        etas = sub["eta"].values
        frac_zero = (etas == 0).mean()
        y_pos = len(strata_present) - 1 - i
        ax.text(x_left + 0.003, y_pos + 0.28,
                f"{frac_zero:.0%} at \u03B7=0",
                fontsize=7, color="0.3", ha="left", va="bottom", zorder=4)

    ax.set_ylim(-0.6, len(strata_present) - 0.4)
    fig.tight_layout()

    save_figure(fig, fig_dir, "F18_oracle_eta_distribution")
    logger.info("Analysis 18: Done")


# =====================================================================
# Analysis 19 — Within-Stratum Gradient
# =====================================================================
def analysis_19_within_stratum_gradient(df: pd.DataFrame, fig_dir: str) -> None:
    """Scatter: baseline_reward vs oracle delta_fitness, colored by stratum."""
    logger.info("Analysis 19: Within-stratum gradient")

    df_active = df[df["eta"] != 0].copy()

    # Per-network oracle
    oracle_per_net = df_active.groupby("network_id").agg(
        oracle_delta_fitness=("delta_fitness", "max"),
        baseline_reward=("baseline_avg_reward", "first"),
        stratum=("stratum", "first"),
    )

    # Restrict to non-Weak (baseline >= 200)
    plot_df = oracle_per_net[oracle_per_net["baseline_reward"] >= 200].copy()
    logger.info(f"  Non-weak networks for gradient plot: {len(plot_df):,}")

    fig, ax = plt.subplots(figsize=(12, 6))

    for s in STRATA_ORDER:
        if s == "weak":
            continue
        sub = plot_df[plot_df["stratum"] == s]
        if sub.empty:
            continue
        ax.scatter(sub["baseline_reward"], sub["oracle_delta_fitness"],
                   alpha=0.3, s=15, color=STRATUM_COLORS[s], edgecolors="none",
                   label=STRATUM_LABELS[s].split(" [")[0])

    # LOWESS smoothing line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sorted_df = plot_df.sort_values("baseline_reward")
        smoothed = lowess(sorted_df["oracle_delta_fitness"].values,
                          sorted_df["baseline_reward"].values,
                          frac=0.3, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", lw=2.5,
                ls="-", label="LOWESS trend", zorder=10)
    except ImportError:
        # Fallback: binned means
        plot_df["reward_bin"] = pd.cut(plot_df["baseline_reward"], bins=20)
        binned = plot_df.groupby("reward_bin", observed=True)["oracle_delta_fitness"].agg(
            ["mean", "count"]
        )
        bin_centers = [interval.mid for interval in binned.index]
        ax.plot(bin_centers, binned["mean"], color="black", lw=2.5,
                marker="s", markersize=4, label="Binned mean", zorder=10)

    # Mark stratum boundaries
    for boundary in [200, 350, 450, 475]:
        ax.axvline(boundary, color="gray", ls=":", lw=0.8, alpha=0.5)

    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("Baseline Reward")
    ax.set_ylabel(u"Oracle \u0394fitness")
    ax.set_title(u"Within-Stratum Gradient: Baseline Reward vs Oracle \u0394fitness\n"
                 "(non-Weak networks, baseline \u2265 200)")
    ax.legend(fontsize=9, loc="upper right")

    save_figure(fig, fig_dir, "F19_within_stratum_gradient")
    logger.info("Analysis 19: Done")


# =====================================================================
# Summary report generation
# =====================================================================
def generate_summary_json(all_results: dict, output_dir: str) -> None:
    """Write comprehensive summary JSON."""
    path = os.path.join(output_dir, "B0_5_comprehensive_summary.json")

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return obj

    with open(path, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    logger.info(f"Summary saved to {path}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="B0.5 Comprehensive Analysis")
    parser.add_argument("--sweep-dir", default="experiments/B0.5+/sweep")
    parser.add_argument("--pool-dir", default="experiments/B0.5+/pool_subsample")
    parser.add_argument("--output-dir", default="experiments/B0.5+/analysis")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s | %(levelname)s | %(message)s")

    fig_dir = os.path.join(args.output_dir, "figures")
    tbl_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("B0.5 COMPREHENSIVE ANALYSIS — PUBLICATION READY")
    logger.info("=" * 70)

    # Load data
    df = load_sweep(args.sweep_dir)
    pool_rewards = load_pool_rewards(args.pool_dir)

    all_results = {}

    # ── Analyses 0-12 (core) ──
    all_results["headline_impact"] = analysis_0_headline(df, fig_dir, tbl_dir)
    all_results["distribution"] = analysis_1_distribution(df, pool_rewards, fig_dir, tbl_dir)
    all_results["heatmaps"] = analysis_2_heatmaps(df, fig_dir, tbl_dir)
    all_results["optimal_params"] = analysis_3_optimal_params(df, fig_dir, tbl_dir)
    all_results["hebbian_comparison"] = analysis_4_hebbian_comparison(df, fig_dir, tbl_dir)
    all_results["decay_importance"] = analysis_5_decay_importance(df, fig_dir, tbl_dir)
    all_results["perfect_stability"] = analysis_6_perfect_stability(df, fig_dir, tbl_dir)
    all_results["regret"] = analysis_7_regret(df, fig_dir, tbl_dir)
    all_results["failure_modes"] = analysis_8_failure_modes(df, fig_dir, tbl_dir)
    analysis_9_3d_surfaces(df, fig_dir)
    analysis_10_outcome_distributions(df, fig_dir, tbl_dir)
    analysis_11_outcome_heatmaps(df, fig_dir)
    analysis_12_severity_distributions(df, fig_dir)

    # ── Analyses 13-17 (supplementary) ──
    analysis_13_interaction_effects(df, fig_dir, tbl_dir)
    analysis_14_dose_response(df, fig_dir)

    # Analysis 15 needs topology data from pool JSONs
    topo_df = load_pool_topology(args.pool_dir)
    analysis_15_topology_correlates(df, topo_df, fig_dir, tbl_dir)

    analysis_16_stratum_transitions(df, fig_dir, tbl_dir)
    analysis_17_decay_dose_response(df, fig_dir)

    # ── Analyses 18-19 (polish additions) ──
    analysis_18_oracle_eta_distribution(df, fig_dir, tbl_dir)
    analysis_19_within_stratum_gradient(df, fig_dir)

    # Save comprehensive summary
    generate_summary_json(all_results, args.output_dir)

    logger.info("=" * 70)
    logger.info("ALL ANALYSES COMPLETE")
    logger.info(f"Figures: {fig_dir}")
    logger.info(f"Tables:  {tbl_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
