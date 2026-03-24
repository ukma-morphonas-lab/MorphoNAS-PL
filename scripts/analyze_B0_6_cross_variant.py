#!/usr/bin/env python3
"""
B0.6 Cross-Variant Comparison: heavy_pole vs gravity_2x

Produces:
  F_pct_helped: % of networks helped at per-network oracle, both variants
  F_adaptation_premium: adaptation premium per stratum, both variants
  F1: Temporal |dw| deep-dive — spike analysis around step 200
  F2: Within-Phase-2 survival curves (plastic vs baseline)
  F3: Per-stratum adaptation premium scatter (heavy_pole vs gravity_2x)

Usage:
  python scripts/analyze_B0_6_cross_variant.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────
DEFAULT_SWEEP_DIR = Path("experiments/B0.6/sweep")
DEFAULT_ANALYSIS_HP = Path("experiments/B0.6/analysis/B0_6_analysis.json")
DEFAULT_ANALYSIS_GX = Path("experiments/B0.6/analysis_gravity_2x/B0_6_analysis.json")
DEFAULT_TEMPORAL_DIR = Path("experiments/B0.6/temporal_profile")
DEFAULT_OUTPUT_DIR = Path("experiments/B0.6/analysis_cross_variant")
DEFAULT_B05P_SWEEP_DIR = Path("experiments/B0.5+/sweep")

SWEEP_DIR = DEFAULT_SWEEP_DIR
ANALYSIS_HP = DEFAULT_ANALYSIS_HP
ANALYSIS_GX = DEFAULT_ANALYSIS_GX
TEMPORAL_DIR = DEFAULT_TEMPORAL_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
B05P_SWEEP_DIR = DEFAULT_B05P_SWEEP_DIR

STRATA = ["low_mid", "high_mid", "near_perfect", "perfect"]
STRATA_LABELS = {
    "low_mid": "Low-mid",
    "high_mid": "High-mid",
    "near_perfect": "Near-perfect",
    "perfect": "Perfect",
}
STRATA_COLORS = {
    "low_mid": "#FF9800",
    "high_mid": "#2196F3",
    "near_perfect": "#4CAF50",
    "perfect": "#9C27B0",
}
VARIANT_COLORS = {"gravity_2x": "#42A5F5", "heavy_pole": "#FF7043"}
VARIANT_LABELS = {"gravity_2x": "Gravity 2x", "heavy_pole": "Heavy Pole 10x"}

SWITCH_STEP = 200


def load_analyses() -> tuple[dict, dict]:
    with open(ANALYSIS_HP) as f:
        hp = json.load(f)
    with open(ANALYSIS_GX) as f:
        gx = json.load(f)
    return hp, gx


def load_temporal_metadata() -> list[dict]:
    path = TEMPORAL_DIR / "metadata.json"
    with open(path) as f:
        meta = json.load(f)
    return meta["networks"]


def load_temporal_traces(variant: str) -> dict[str, np.ndarray]:
    path = TEMPORAL_DIR / f"traces_{variant}.npz"
    return dict(np.load(path, allow_pickle=False))


# ── F_pct_helped: % networks helped ───────────────────────────────────

def _compute_oracle_cis(parquet_path: Path, strata: list[str]) -> dict[str, tuple[float, float]]:
    """Compute 95% CI of per-network oracle delta_reward_phase2 from sweep parquet."""
    df = pd.read_parquet(parquet_path)
    # Per-network oracle: best delta_reward_phase2 across all (eta, decay) pairs
    oracle = df.groupby(["network_id", "stratum"])["delta_reward_phase2"].max().reset_index()
    cis = {}
    for s in strata:
        vals = oracle.loc[oracle["stratum"] == s, "delta_reward_phase2"].values
        if len(vals) < 2:
            cis[s] = (0.0, 0.0)
            continue
        sem = vals.std(ddof=1) / np.sqrt(len(vals))
        mean = vals.mean()
        cis[s] = (mean - 1.96 * sem, mean + 1.96 * sem)
    return cis


def figure_pct_helped(hp: dict, gx: dict):
    """Two-panel figure matching B0.5+ headline style: (a) mean oracle Δr, (b) % helped."""
    CLR_GX = "#4477AA"   # Tol blue (gravity)
    CLR_HP = "#EE6677"   # Tol pink (heavy-pole)

    x = np.arange(len(STRATA))
    width = 0.35
    labels = [STRATA_LABELS[s] for s in STRATA]

    gx_pn = {s: gx["per_network"]["per_stratum"].get(s, {}) for s in STRATA}
    hp_pn = {s: hp["per_network"]["per_stratum"].get(s, {}) for s in STRATA}

    gx_mean = [gx_pn[s].get("mean_delta", 0) for s in STRATA]
    hp_mean = [hp_pn[s].get("mean_delta", 0) for s in STRATA]
    gx_helped = [gx_pn[s].get("pct_helped", 0) for s in STRATA]
    hp_helped = [hp_pn[s].get("pct_helped", 0) for s in STRATA]

    # Compute 95% CIs from parquet data
    gx_cis = _compute_oracle_cis(SWEEP_DIR / "B0_6_gravity_2x.parquet", STRATA)
    hp_cis = _compute_oracle_cis(SWEEP_DIR / "B0_6_heavy_pole.parquet", STRATA)

    gx_errs = [
        [max(0, gx_mean[i] - gx_cis[s][0]) for i, s in enumerate(STRATA)],
        [max(0, gx_cis[s][1] - gx_mean[i]) for i, s in enumerate(STRATA)],
    ]
    hp_errs = [
        [max(0, hp_mean[i] - hp_cis[s][0]) for i, s in enumerate(STRATA)],
        [max(0, hp_cis[s][1] - hp_mean[i]) for i, s in enumerate(STRATA)],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel (a): Mean oracle Δr ──
    ax1.bar(x - width / 2, gx_mean, width, color=CLR_GX, edgecolor="black",
            linewidth=0.5, label="Gravity 2x", yerr=gx_errs,
            capsize=3, error_kw={"linewidth": 1})
    ax1.bar(x + width / 2, hp_mean, width, color=CLR_HP, edgecolor="black",
            linewidth=0.5, label="Heavy pole 10x", yerr=hp_errs,
            capsize=3, error_kw={"linewidth": 1})

    ax1.axhline(0, color="black", lw=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Mean oracle $\\Delta r$ (reward points)")
    ax1.legend(fontsize=9, loc="upper left")

    for i, s in enumerate(STRATA):
        g, h = gx_mean[i], hp_mean[i]
        y_g = gx_cis[s][1] + 1.5
        y_h = hp_cis[s][1] + 1.5
        ax1.text(i - width / 2, y_g, f"+{g:.0f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax1.text(i + width / 2, y_h, f"+{h:.0f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.text(-0.08, 1.02, "(a)", transform=ax1.transAxes,
             fontsize=13, fontweight="bold", va="bottom")

    # ── Panel (b): % networks helped ──
    ax2.bar(x - width / 2, gx_helped, width, color=CLR_GX, edgecolor="black",
            linewidth=0.5, label="Gravity 2x")
    ax2.bar(x + width / 2, hp_helped, width, color=CLR_HP, edgecolor="black",
            linewidth=0.5, label="Heavy pole 10x")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Networks helped (%)")
    ax2.set_ylim(0, 109)
    ax2.legend(fontsize=9, loc="upper left")

    for i, (g, h) in enumerate(zip(gx_helped, hp_helped)):
        ax2.text(i - width / 2, g + 1.5, f"{g:.0f}%",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax2.text(i + width / 2, h + 1.5, f"{h:.0f}%",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.text(-0.08, 1.02, "(b)", transform=ax2.transAxes,
             fontsize=13, fontweight="bold", va="bottom")

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "F_pct_helped.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "F_pct_helped.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved F_pct_helped")


# ── F_adaptation_premium: adaptation premium per stratum ─────────────

def figure_adaptation_premium(hp: dict, gx: dict):
    """Adaptation premium per stratum, both variants."""
    x = np.arange(len(STRATA))
    w = 0.35

    gx_mean = [gx["adaptation_premium"].get(s, {}).get("premium_mean", 0) for s in STRATA]
    hp_mean = [hp["adaptation_premium"].get(s, {}).get("premium_mean", 0) for s in STRATA]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - w / 2, gx_mean, w, color="#4477AA", edgecolor="black",
           linewidth=0.5, label="Gravity 2x")
    ax.bar(x + w / 2, hp_mean, w, color="#EE6677", edgecolor="black",
           linewidth=0.5, label="Heavy pole 10x")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([STRATA_LABELS[s] for s in STRATA])
    ax.set_ylabel("Adaptation premium (reward points)")
    ax.legend(fontsize=9, loc="upper left")

    # Combined value + significance annotations
    for i, s in enumerate(STRATA):
        for offset, analysis, val in [(-w / 2, gx, gx_mean[i]), (w / 2, hp, hp_mean[i])]:
            p_val = analysis["adaptation_premium"].get(s, {}).get("wilcoxon_p")
            sig = ""
            if p_val is not None:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            label = f"{val:+.1f} {sig}" if abs(val) > 1 else sig
            if not label:
                continue
            y_off = 1.0 if val >= 0 else -1.0
            va = "bottom" if val >= 0 else "top"
            ax.text(i + offset, val + y_off, label,
                    ha="center", va=va, fontsize=8, fontweight="bold")

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "F_adaptation_premium.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "F_adaptation_premium.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved F_adaptation_premium")


# ── F1: Temporal |dw| deep-dive ───────────────────────────────────────

def figure_1_temporal_deep_dive(bin_size: int = 10):
    """
    3-row figure:
      Row 1: |dw| per step for both variants, focused on High-mid
      Row 2: |dw| difference (post-switch minus pre-switch mean), all strata
      Row 3: Step-200 spike magnitude distribution per network
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    for vi, variant in enumerate(["gravity_2x", "heavy_pole"]):
        traces_npz = load_temporal_traces(variant)

        # Collect per-stratum binned |dw|
        for stratum in STRATA:
            all_dw_traces = []
            all_phases = []
            prefix = f"{stratum}_"
            # Gather all episode traces for this stratum
            dw_keys = sorted(k for k in traces_npz if k.startswith(prefix) and k.endswith("_dw"))
            for dk in dw_keys:
                phase_key = dk.replace("_dw", "_phase")
                all_dw_traces.append(traces_npz[dk])
                if phase_key in traces_npz:
                    all_phases.append(traces_npz[phase_key])

            if not all_dw_traces:
                continue

            # Bin traces
            max_steps = max(len(t) for t in all_dw_traces)
            n_bins = (max_steps + bin_size - 1) // bin_size
            bin_centers = (np.arange(n_bins) * bin_size) + bin_size / 2
            binned = np.full((len(all_dw_traces), n_bins), np.nan)
            for i, t in enumerate(all_dw_traces):
                for b in range(n_bins):
                    s, e = b * bin_size, min((b + 1) * bin_size, len(t))
                    if s < len(t):
                        binned[i, b] = t[s:e].mean()

            means = np.nanmean(binned, axis=0)
            counts = np.sum(~np.isnan(binned), axis=0)
            sems = np.nanstd(binned, axis=0) / np.sqrt(np.maximum(counts, 1))

            color = STRATA_COLORS[stratum]

            # Row 1: full |dw| time series
            ax = axes[0, vi]
            ax.plot(bin_centers, means, color=color, label=STRATA_LABELS[stratum], linewidth=1.5)
            ax.fill_between(bin_centers, means - sems, means + sems, color=color, alpha=0.15)

            # Row 2: |dw| normalised to pre-switch mean (fold-change)
            ax2 = axes[1, vi]
            pre_switch_bins = bin_centers < SWITCH_STEP
            pre_mean = np.nanmean(means[pre_switch_bins]) if pre_switch_bins.any() else 1e-15
            if pre_mean > 1e-15:
                fold_change = means / pre_mean
                ax2.plot(bin_centers, fold_change, color=color,
                         label=STRATA_LABELS[stratum], linewidth=1.5)

            # Row 3: per-network spike magnitude
            ax3 = axes[2, vi]
            # For each network-episode: compute (mean |dw| in steps 200-300) / (mean |dw| in steps 100-200)
            spike_ratios = []
            for t in all_dw_traces:
                if len(t) > 300:
                    pre = t[100:200].mean() if len(t) > 200 else 1e-15
                    post = t[200:300].mean()
                    if pre > 1e-15:
                        spike_ratios.append(post / pre)

            if spike_ratios and stratum != "perfect":
                ax3.hist(spike_ratios, bins=30, color=color, alpha=0.5,
                         edgecolor="black", linewidth=0.3,
                         label=f"{STRATA_LABELS[stratum]} (N={len(spike_ratios)})")

        # Format Row 1
        axes[0, vi].axvline(SWITCH_STEP, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
        axes[0, vi].set_ylabel("Mean |Δw| per step")
        axes[0, vi].set_title(f"{VARIANT_LABELS[variant]}: |Δw| temporal profile")
        axes[0, vi].legend(fontsize=8)
        axes[0, vi].grid(True, alpha=0.2)

        # Format Row 2
        axes[1, vi].axvline(SWITCH_STEP, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
        axes[1, vi].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        axes[1, vi].set_ylabel("Fold-change vs pre-switch mean")
        axes[1, vi].set_title(f"{VARIANT_LABELS[variant]}: |Δw| fold-change (normalised)")
        axes[1, vi].legend(fontsize=8)
        axes[1, vi].grid(True, alpha=0.2)
        axes[1, vi].set_xlabel(f"Step ({bin_size}-step bins)")

        # Format Row 3
        axes[2, vi].axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="No change")
        axes[2, vi].set_xlabel("Spike ratio: mean |Δw| steps [200,300) / [100,200)")
        axes[2, vi].set_ylabel("Count (episodes)")
        axes[2, vi].set_title(f"{VARIANT_LABELS[variant]}: Step-200 spike distribution")
        axes[2, vi].legend(fontsize=8)

    fig.suptitle("B0.6: How Does Plasticity Respond to the Physics Switch?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "F1_temporal_deep_dive.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "F1_temporal_deep_dive.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved F1: temporal deep-dive")


# ── F2: Within-Phase-2 survival ───────────────────────────────────────

def figure_2_survival_curves():
    """
    Compare episode lengths: plastic vs baseline from the sweep data.
    How much longer do plastic networks survive after the switch?
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for vi, variant in enumerate(["gravity_2x", "heavy_pole"]):
        ax = axes[vi]
        df = pd.read_parquet(SWEEP_DIR / f"B0_6_{variant}.parquet")

        for stratum in ["low_mid", "high_mid", "near_perfect"]:
            sdf = df[df["stratum"] == stratum]

            # Baseline: eta=0
            base = sdf[sdf["eta"].abs() < 1e-12]
            base_p2 = base["baseline_phase2_steps"].values

            # Best plastic: best eta at decay=0.01
            plastic = sdf[(sdf["eta"].abs() >= 1e-12) & (sdf["decay"].between(0.009, 0.011))]
            if len(plastic) == 0:
                continue
            best_eta = plastic.groupby("eta")["delta_reward_phase2"].mean().idxmax()
            best = plastic[plastic["eta"] == best_eta]
            plastic_p2 = best["plastic_phase2_steps"].values

            color = STRATA_COLORS[stratum]

            # Plot as survival distributions (fraction surviving > X steps in Phase 2)
            for data, ls, alpha, label_suffix in [
                (base_p2, "--", 0.5, "baseline"),
                (plastic_p2, "-", 1.0, f"plastic (η={best_eta:+.2f})"),
            ]:
                sorted_vals = np.sort(data)
                # Fraction surviving more than each value
                survival = 1.0 - np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                ax.plot(sorted_vals, survival, color=color, linestyle=ls,
                        alpha=alpha, linewidth=1.5,
                        label=f"{STRATA_LABELS[stratum]} {label_suffix}")

        ax.set_xlabel("Phase 2 steps survived")
        ax.set_ylabel("Fraction of networks")
        ax.set_title(f"{VARIANT_LABELS[variant]}: Phase 2 survival (plastic vs baseline)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)
        ax.axvline(0, color="gray", linewidth=0.3)

    fig.suptitle("B0.6: Does Plasticity Extend Survival After the Physics Switch?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "F2_survival_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "F2_survival_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved F2: survival curves")


# ── F3: Per-network premium scatter ───────────────────────────────────

def figure_3_premium_scatter(hp: dict, gx: dict):
    """Scatter: per-network adaptation premium (gravity_2x vs heavy_pole)."""
    hp_df = pd.read_parquet(SWEEP_DIR / "B0_6_heavy_pole.parquet")
    gx_df = pd.read_parquet(SWEEP_DIR / "B0_6_gravity_2x.parquet")

    # Load B0.5+ static oracle
    static_frames = []
    if B05P_SWEEP_DIR.exists():
        for pf in sorted(B05P_SWEEP_DIR.glob("*.parquet")):
            static_frames.append(pd.read_parquet(pf))
    if not static_frames:
        print("  SKIP F3: no B0.5+ data")
        return
    static_df = pd.concat(static_frames, ignore_index=True)
    static_df = static_df.drop_duplicates(subset=["network_id", "eta", "decay"])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for idx, stratum in enumerate(STRATA):
        ax = axes[idx]

        # Per-network oracles
        hp_s = hp_df[hp_df["stratum"] == stratum]
        gx_s = gx_df[gx_df["stratum"] == stratum]
        st_s = static_df[static_df["stratum"] == stratum]

        hp_oracle = hp_s.groupby("network_id")["delta_reward_total"].max()
        gx_oracle = gx_s.groupby("network_id")["delta_reward_total"].max()
        st_oracle = st_s.groupby("network_id")["delta_reward"].max()

        common = sorted(set(hp_oracle.index) & set(gx_oracle.index) & set(st_oracle.index))
        if len(common) < 5:
            continue

        hp_prem = hp_oracle.loc[common].values - st_oracle.loc[common].values
        gx_prem = gx_oracle.loc[common].values - st_oracle.loc[common].values

        ax.scatter(gx_prem, hp_prem, c=STRATA_COLORS[stratum], alpha=0.4, s=15,
                   edgecolors="black", linewidth=0.2)

        # Diagonal
        lim_min = min(gx_prem.min(), hp_prem.min()) - 5
        lim_max = max(gx_prem.max(), hp_prem.max()) + 5
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.3, linewidth=0.5)
        ax.axhline(0, color="gray", linewidth=0.3)
        ax.axvline(0, color="gray", linewidth=0.3)

        # Quadrant counts
        q1 = ((gx_prem > 0) & (hp_prem > 0)).sum()  # both positive
        q4 = ((gx_prem > 0) & (hp_prem <= 0)).sum()  # only gx positive

        rho, p_val = stats.spearmanr(gx_prem, hp_prem)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        ax.set_xlabel("Gravity 2× premium (reward)")
        if idx == 0:
            ax.set_ylabel("Heavy pole 10× premium (reward)")
        ax.set_title(f"{STRATA_LABELS[stratum]} (N={len(common)})", fontsize=10)

        # Annotation: correlation and quadrant info
        ax.text(0.05, 0.95, f"ρ = {rho:+.2f} [{sig}]\nboth +: {q1}  gx only: {q4}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "F3_premium_scatter.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "F3_premium_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved F3: premium scatter")


# ── F4: Temporal |dw| response aligned at switch ─────────────────────

def figure_4_switch_aligned(bin_size: int = 5):
    """
    Zoom into steps [150, 350] — centered on the switch.
    Shows how quickly |dw| responds after the physics change.
    """
    CLR_GX = "#4477AA"   # Tol blue
    CLR_HP = "#EE6677"   # Tol pink
    variant_colors = {"gravity_2x": CLR_GX, "heavy_pole": CLR_HP}

    # 3 responsive strata in a row; Perfect omitted (near-zero |dw|)
    strata_show = ["low_mid", "high_mid", "near_perfect"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Inset window around the switch
    inset_start, inset_end = 185, 225

    for si, stratum in enumerate(strata_show):
        ax = axes[si]

        # Store per-variant curves for inset reuse
        variant_curves = {}

        for variant in ["gravity_2x", "heavy_pole"]:
            traces_npz = load_temporal_traces(variant)
            prefix = f"{stratum}_"
            dw_keys = sorted(k for k in traces_npz if k.startswith(prefix) and k.endswith("_dw"))

            if not dw_keys:
                continue

            # Collect all traces, align to steps [150, 350]
            window_start, window_end = 150, 350
            n_steps = window_end - window_start
            n_bins = n_steps // bin_size
            aligned = np.full((len(dw_keys), n_bins), np.nan)

            for i, dk in enumerate(dw_keys):
                t = traces_npz[dk]
                for b in range(n_bins):
                    s = window_start + b * bin_size
                    e = min(s + bin_size, len(t))
                    if s < len(t):
                        aligned[i, b] = t[s:e].mean()

            bin_centers = window_start + (np.arange(n_bins) * bin_size) + bin_size / 2
            means = np.nanmean(aligned, axis=0)
            counts = np.sum(~np.isnan(aligned), axis=0)
            sems = np.nanstd(aligned, axis=0) / np.sqrt(np.maximum(counts, 1))

            color = variant_colors[variant]
            ax.plot(bin_centers, means, color=color, label=VARIANT_LABELS[variant], linewidth=2)
            ax.fill_between(bin_centers, means - sems, means + sems, color=color, alpha=0.15)
            variant_curves[variant] = (bin_centers, means, sems, color)

        ax.axvline(SWITCH_STEP, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
                   label="Physics switch")
        ax.set_xlabel("Step")
        if si == 0:
            ax.set_ylabel("Mean |Δw|")
        ax.set_title(f"{STRATA_LABELS[stratum]}", fontsize=11)

        # Inset zoom: upper-left
        if variant_curves:
            inset = ax.inset_axes([0.02, 0.55, 0.38, 0.38])
            for variant, (bc, mn, se, clr) in variant_curves.items():
                mask = (bc >= inset_start) & (bc <= inset_end)
                inset.plot(bc[mask], mn[mask], color=clr, linewidth=1.2)
                inset.fill_between(bc[mask], (mn - se)[mask], (mn + se)[mask],
                                   color=clr, alpha=0.12)
            inset.axvline(SWITCH_STEP, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
            inset.set_xlim(inset_start, inset_end)
            inset.tick_params(labelsize=5, length=2, pad=1)
            inset.set_title("steps 185–225", fontsize=6, pad=2)
            inset.ticklabel_format(style="sci", axis="y", scilimits=(-5, -5))
            inset.yaxis.get_offset_text().set_fontsize(5)

    # Single shared legend outside the panels, upper-right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8,
               bbox_to_anchor=(0.99, 0.99), framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 0.88, 1.0])
    fig.savefig(OUTPUT_DIR / "F4_switch_aligned.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "F4_switch_aligned.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved F4: switch-aligned zoom")


# ── Summary stats ─────────────────────────────────────────────────────

def print_comparison_table(hp: dict, gx: dict, temporal_meta: list[dict]):
    print("\n" + "=" * 90)
    print("CROSS-VARIANT COMPARISON SUMMARY")
    print("=" * 90)

    header = (f"{'Metric':<30} {'':>10} "
              + " ".join(f"{STRATA_LABELS[s]:>14}" for s in STRATA))
    print(header)
    print("-" * 90)

    def row(label, variant_label, vals):
        nums = " ".join(f"{v:>14}" for v in vals)
        print(f"{label:<30} {variant_label:>10} {nums}")

    # Disruption
    row("P2 utilisation (%)", "grav-2x",
        [f"{gx['disruption'].get(s, {}).get('phase2_pct', 0):.1f}" for s in STRATA])
    row("", "heavy-p",
        [f"{hp['disruption'].get(s, {}).get('phase2_pct', 0):.1f}" for s in STRATA])

    # Adaptation premium
    row("Adapt. premium (mean)", "grav-2x",
        [f"{gx['adaptation_premium'].get(s, {}).get('premium_mean', 0):+.1f}" for s in STRATA])
    row("", "heavy-p",
        [f"{hp['adaptation_premium'].get(s, {}).get('premium_mean', 0):+.1f}" for s in STRATA])
    row("Adapt. premium (median)", "grav-2x",
        [f"{gx['adaptation_premium'].get(s, {}).get('premium_median', 0):+.1f}" for s in STRATA])
    row("", "heavy-p",
        [f"{hp['adaptation_premium'].get(s, {}).get('premium_median', 0):+.1f}" for s in STRATA])
    row("Wilcoxon p", "grav-2x",
        [f"{gx['adaptation_premium'].get(s, {}).get('wilcoxon_p', 'N/A')}" for s in STRATA])
    row("", "heavy-p",
        [f"{hp['adaptation_premium'].get(s, {}).get('wilcoxon_p', 'N/A')}" for s in STRATA])

    # % helped
    row("% helped (Phase 2)", "grav-2x",
        [f"{gx['per_network']['per_stratum'].get(s, {}).get('pct_helped', 0):.0f}%" for s in STRATA])
    row("", "heavy-p",
        [f"{hp['per_network']['per_stratum'].get(s, {}).get('pct_helped', 0):.0f}%" for s in STRATA])

    # P2/P1 ratio from temporal
    row("P2/P1 |dw| ratio", "grav-2x",
        [f"{np.mean([m['p2_p1_ratio'] for m in temporal_meta if m['variant']=='gravity_2x' and m['stratum']==s]):.2f}" for s in STRATA])
    row("", "heavy-p",
        [f"{np.mean([m['p2_p1_ratio'] for m in temporal_meta if m['variant']=='heavy_pole' and m['stratum']==s]):.2f}" for s in STRATA])

    # |dw| vs improvement
    row("Spearman ρ (|dw|→Δ)", "grav-2x",
        [f"{gx['dw_vs_improvement'].get(s, {}).get('spearman_rho', 0):+.3f}" for s in STRATA])
    row("", "heavy-p",
        [f"{hp['dw_vs_improvement'].get(s, {}).get('spearman_rho', 0):+.3f}" for s in STRATA])

    print("=" * 90)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B0.6 cross-variant comparison")
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--analysis-hp", type=Path, default=DEFAULT_ANALYSIS_HP)
    parser.add_argument("--analysis-gx", type=Path, default=DEFAULT_ANALYSIS_GX)
    parser.add_argument("--temporal-dir", type=Path, default=DEFAULT_TEMPORAL_DIR)
    parser.add_argument("--b05plus-sweep-dir", type=Path, default=DEFAULT_B05P_SWEEP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    global SWEEP_DIR, ANALYSIS_HP, ANALYSIS_GX, TEMPORAL_DIR, OUTPUT_DIR, B05P_SWEEP_DIR
    SWEEP_DIR = args.sweep_dir
    ANALYSIS_HP = args.analysis_hp
    ANALYSIS_GX = args.analysis_gx
    TEMPORAL_DIR = args.temporal_dir
    OUTPUT_DIR = args.output_dir
    B05P_SWEEP_DIR = args.b05plus_sweep_dir

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading analysis JSONs...")
    hp, gx = load_analyses()
    temporal_meta = load_temporal_metadata()

    print_comparison_table(hp, gx, temporal_meta)

    print("\nGenerating figures...")
    figure_pct_helped(hp, gx)
    figure_adaptation_premium(hp, gx)
    figure_1_temporal_deep_dive()
    figure_2_survival_curves()
    figure_3_premium_scatter(hp, gx)
    figure_4_switch_aligned()

    # Save comparison JSON
    comparison = {
        "variants": ["gravity_2x", "heavy_pole"],
        "strata": STRATA,
    }
    for metric in ["disruption", "adaptation_premium", "per_network", "dw_vs_improvement"]:
        comparison[metric] = {
            "gravity_2x": gx.get(metric, {}),
            "heavy_pole": hp.get(metric, {}),
        }
    with open(OUTPUT_DIR / "cross_variant_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
