#!/usr/bin/env python3
"""
Generate camera-ready figures for the plasticity characterisation paper.

Produces three figures at ACM sigconf two-column width (3.33 inches):
  1. F2_heatmap_high_mid.pdf — η × λ heatmap for High-mid stratum
  2. F4_hebbian_comparison.pdf — Anti-Hebbian vs Hebbian bars + Cohen's d
  3. F_adaptation_premium.pdf — Adaptation premium per stratum

All figures use TrueType fonts (no Type-3), 8pt minimum text, no titles.

Usage:
  uv run python scripts/generate_paper_figures.py
  uv run python scripts/generate_paper_figures.py --verify  # check fonts with pdffonts
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42        # TrueType
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.titlesize"] = 9
matplotlib.rcParams["axes.labelsize"] = 8
matplotlib.rcParams["xtick.labelsize"] = 7
matplotlib.rcParams["ytick.labelsize"] = 7
matplotlib.rcParams["legend.fontsize"] = 7
matplotlib.rcParams["lines.linewidth"] = 1.5
matplotlib.rcParams["axes.linewidth"] = 0.75
matplotlib.rcParams["font.family"] = "sans-serif"

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from MorphoNAS_PL.experimentB0_5_natural import STRATUM_BOUNDS

COLUMN_WIDTH = 3.33  # inches — ACM sigconf column width

STRATA_ORDER = ["weak", "low_mid", "high_mid", "near_perfect", "perfect"]
STRATUM_SHORT = {
    "weak": "Weak",
    "low_mid": "Low-mid",
    "high_mid": "High-mid",
    "near_perfect": "Near-perf.",
    "perfect": "Perfect",
}


def fmt_eta(v: float) -> str:
    if v == 0:
        return "0"
    return f"{v:+.3g}"


def fmt_decay(v: float) -> str:
    if v == 0:
        return "0"
    if v < 0.01:
        return f"{v:.0e}"
    return f"{v:.2g}"


# ── Figure 1: Heatmap High-mid ──────────────────────────────────────

def figure_heatmap_high_mid(df: pd.DataFrame, out_path: str) -> None:
    """η × λ heatmap of mean Δfitness for High-mid stratum."""
    sub = df[df["stratum"] == "high_mid"]
    eta_vals = sorted(df["eta"].unique())
    decay_vals = sorted(df["decay"].unique())

    piv = sub.pivot_table(
        values="delta_fitness", index="eta", columns="decay", aggfunc="mean"
    ).reindex(index=eta_vals, columns=decay_vals)
    vals = piv.values

    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    if vmin >= 0:
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0, vmax=max(vmax, 0.001))
    elif vmax <= 0:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=abs(vmin) * 0.01)
    else:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.0))
    im = ax.imshow(vals, cmap="RdYlGn", norm=norm, aspect="auto", origin="upper")

    ax.set_xticks(range(len(decay_vals)))
    ax.set_xticklabels([fmt_decay(d) for d in decay_vals], fontsize=6)
    ax.set_yticks(range(len(eta_vals)))
    ax.set_yticklabels([fmt_eta(e) for e in eta_vals], fontsize=5.5)
    ax.set_xlabel(u"\u03BB (decay)")
    ax.set_ylabel(u"\u03B7 (learning rate)")

    # Cell annotations
    for i in range(len(eta_vals)):
        for j in range(len(decay_vals)):
            v = vals[i, j]
            if np.isfinite(v):
                cval = float(norm(v))
                txt_col = "white" if cval < 0.15 or cval > 0.85 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=4.5, color=txt_col)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Figure 2: Hebbian comparison ────────────────────────────────────

def _compute_hebbian_results(df: pd.DataFrame) -> tuple[dict, list[str]]:
    """Shared computation for both hebbian figures."""
    df_nz = df[df["eta"] != 0].copy()
    df_nz["sign"] = np.where(df_nz["eta"] < 0, "anti", "hebb")

    results = {}
    for s in STRATA_ORDER:
        sub = df_nz[df_nz["stratum"] == s]
        if sub.empty:
            continue
        anti = sub[sub["sign"] == "anti"]["delta_fitness"].values
        hebb = sub[sub["sign"] == "hebb"]["delta_fitness"].values
        if len(anti) == 0 or len(hebb) == 0:
            continue

        pooled_sd = np.sqrt(
            (np.var(anti, ddof=1) * (len(anti) - 1) + np.var(hebb, ddof=1) * (len(hebb) - 1))
            / (len(anti) + len(hebb) - 2)
        )
        d = (np.mean(anti) - np.mean(hebb)) / pooled_sd if pooled_sd > 0 else 0
        _, p = stats.ttest_ind(anti, hebb, equal_var=False)

        results[s] = {
            "anti_mean": float(np.mean(anti)),
            "anti_ci": float(stats.sem(anti) * 1.96),
            "hebb_mean": float(np.mean(hebb)),
            "hebb_ci": float(stats.sem(hebb) * 1.96),
            "d": float(d),
            "p": float(p),
        }

    strata_present = [s for s in STRATA_ORDER if s in results]
    return results, strata_present


def figure_hebbian_delta(df: pd.DataFrame, out_path: str) -> None:
    """Mean Δfitness bars: anti-Hebbian vs Hebbian by stratum with 95% CI."""
    results, strata_present = _compute_hebbian_results(df)
    x = np.arange(len(strata_present))
    labels = [STRATUM_SHORT[s] for s in strata_present]
    w = 0.35

    anti_m = [results[s]["anti_mean"] for s in strata_present]
    anti_ci = [results[s]["anti_ci"] for s in strata_present]
    hebb_m = [results[s]["hebb_mean"] for s in strata_present]
    hebb_ci = [results[s]["hebb_ci"] for s in strata_present]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.2))

    ax.bar(x - w / 2, anti_m, w, yerr=anti_ci, capsize=3,
           color="#88BBDD", alpha=0.85, label=u"Anti-Hebbian (\u03B7 < 0)",
           linewidth=0.5, edgecolor="black")
    ax.bar(x + w / 2, hebb_m, w, yerr=hebb_ci, capsize=3,
           color="#CC6677", alpha=0.85, label=u"Hebbian (\u03B7 > 0)",
           linewidth=0.5, edgecolor="black")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(u"Mean \u0394r (95% CI)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def figure_cohens_d(df: pd.DataFrame, out_path: str) -> None:
    """Cohen's d horizontal bars by stratum (anti-Hebbian advantage)."""
    results, strata_present = _compute_hebbian_results(df)
    x = np.arange(len(strata_present))
    labels = [STRATUM_SHORT[s] for s in strata_present]

    ds = [results[s]["d"] for s in strata_present]
    colors_d = ["#88BBDD" if d > 0 else "#CC6677" for d in ds]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.0))

    ax.barh(x, ds, color=colors_d, edgecolor="black", alpha=0.85, height=0.6, linewidth=0.5)
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d (positive = anti-Hebbian advantage)")
    ax.axvline(0, color="black", lw=0.5)
    for val in [0.2, 0.5, 0.8]:
        ax.axvline(val, color="gray", ls=":", lw=0.5, alpha=0.4)
        ax.axvline(-val, color="gray", ls=":", lw=0.5, alpha=0.4)

    d_max = max(abs(d) for d in ds)
    ax.set_xlim(-d_max - 0.25, d_max + 0.45)

    for i, d in enumerate(ds):
        ax.text(d + 0.03 if d >= 0 else d - 0.03, i, f"{d:.2f}",
                ha="left" if d >= 0 else "right", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Figure 4: Acrobot Cohen's d ─────────────────────────────────────

def figure_acrobot_cohens_d(out_path: str, repo_root: Path = None) -> None:
    """Cohen's d (anti-Hebbian advantage) by stratum for Acrobot, all-data-points method."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    df = pd.read_parquet(
        repo_root / "experiments" / "acrobot" / "sweep_static" / "acrobot_static_sweep.parquet"
    )

    # Acrobot strata (no Weak in the sweep)
    strata = ["low_mid", "high_mid", "near_perfect", "perfect"]
    labels = [STRATUM_SHORT[s] for s in strata]
    x = np.arange(len(strata))

    ds = []
    for s in strata:
        sdf = df[df["stratum"] == s]
        anti = sdf[sdf["eta"] < 0]["delta_reward"].values
        hebb = sdf[sdf["eta"] > 0]["delta_reward"].values
        pooled_sd = np.sqrt(
            (np.var(anti, ddof=1) * (len(anti) - 1) + np.var(hebb, ddof=1) * (len(hebb) - 1))
            / (len(anti) + len(hebb) - 2)
        )
        d = (np.mean(anti) - np.mean(hebb)) / pooled_sd if pooled_sd > 0 else 0
        ds.append(float(d))

    colors_d = ["#88BBDD" if d > 0 else "#CC6677" for d in ds]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.0))

    ax.barh(x, ds, color=colors_d, edgecolor="black", alpha=0.85, height=0.6, linewidth=0.5)
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d (positive = anti-Hebbian advantage)")
    ax.axvline(0, color="black", lw=0.5)
    for val in [0.2, 0.5, 0.8]:
        ax.axvline(val, color="gray", ls=":", lw=0.5, alpha=0.4)
        ax.axvline(-val, color="gray", ls=":", lw=0.5, alpha=0.4)

    # Match CartPole Cohen's d figure x-axis for visual comparison
    ax.set_xlim(-0.75, 1.0)

    for i, d in enumerate(ds):
        ax.text(d + 0.02 if d >= 0 else d - 0.02, i, f"{d:.2f}",
                ha="left" if d >= 0 else "right", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Figure 3: Adaptation premium ────────────────────────────────────

def figure_adaptation_premium(out_path: str, repo_root: Path = None) -> None:
    """Adaptation premium bars by stratum for gravity-2x and heavy-pole."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    base = repo_root / "experiments" / "B0.6"
    gx_path = base / "analysis_gravity_2x" / "B0_6_analysis.json"
    hp_path = base / "analysis" / "B0_6_analysis.json"

    import json
    with open(gx_path) as f:
        gx = json.load(f)
    with open(hp_path) as f:
        hp = json.load(f)

    strata = ["low_mid", "high_mid", "near_perfect", "perfect"]
    labels = [STRATUM_SHORT[s] for s in strata]
    x = np.arange(len(strata))
    w = 0.35

    gx_mean = [gx["adaptation_premium"].get(s, {}).get("premium_mean", 0) for s in strata]
    hp_mean = [hp["adaptation_premium"].get(s, {}).get("premium_mean", 0) for s in strata]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))

    ax.bar(x - w / 2, gx_mean, w, color="#88BBDD", edgecolor="black",
           linewidth=0.5, label=u"Gravity 2\u00D7")
    ax.bar(x + w / 2, hp_mean, w, color="#EE6677", edgecolor="black",
           linewidth=0.5, label=u"Heavy pole 10\u00D7")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Adaptation premium (reward)")
    ax.set_ylim(-33, 23)
    ax.legend(fontsize=6, loc="upper left")

    # Value + significance annotations
    for i, s in enumerate(strata):
        for offset, analysis, val in [(-w / 2, gx, gx_mean[i]), (w / 2, hp, hp_mean[i])]:
            p_val = analysis["adaptation_premium"].get(s, {}).get("wilcoxon_p")
            sig = ""
            if p_val is not None:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            label = f"{val:+.1f}" if abs(val) > 1 else ""
            if sig:
                label = f"{label} {sig}".strip()
            if not label:
                continue
            # Place label outside the bar; use white text if it must overlap
            if val >= 0:
                y_pos = val + 0.8
                va = "bottom"
                color = "black"
            else:
                # For negative bars, place below the bar tip
                y_pos = val - 0.8
                va = "top"
                color = "black"
            ax.text(i + offset, y_pos, label,
                    ha="center", va=va, fontsize=5.5, fontweight="bold",
                    color=color)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Figure 3a: OFF→ON vs always-on ────────────────────────────────────

def figure_off_on(out_path: str, repo_root: Path = None) -> None:
    """Oracle Phase-2 Δ: always-on vs OFF→ON, by stratum, both variants."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    strata = ["low_mid", "high_mid", "near_perfect", "perfect"]
    labels = [STRATUM_SHORT[s] for s in strata]

    fig, (ax_hp, ax_gx) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 2.2), sharey=True)

    for ax, variant, title in [
        (ax_hp, "heavy_pole", "Heavy pole 10\u00D7"),
        (ax_gx, "gravity_2x", "Gravity 2\u00D7"),
    ]:
        # Always-on data
        b06 = pd.read_parquet(
            repo_root / "experiments" / "B0.6" / "sweep" / f"B0_6_{variant}.parquet")
        b06_sw = b06[(b06["eta"] != 0) | (b06["decay"] != 0)]

        # OFF→ON data
        adapt = pd.read_parquet(
            repo_root / "experiments" / "B0.6_adaptation" / "sweep" / f"B0_6_adaptation_{variant}.parquet")
        adapt_sw = adapt[adapt["eta"] != 0]

        x = np.arange(len(strata))
        w = 0.35

        ao_vals, oo_vals = [], []
        for s in strata:
            # Always-on oracle
            sub = b06_sw[b06_sw["stratum"] == s]
            best = sub.loc[sub.groupby("network_id")["delta_reward_phase2"].idxmax()]
            ao_vals.append(best["delta_reward_phase2"].mean())
            # OFF→ON oracle
            sub2 = adapt_sw[adapt_sw["stratum"] == s]
            best2 = sub2.loc[sub2.groupby("network_id")["delta_reward_phase2"].idxmax()]
            oo_vals.append(best2["delta_reward_phase2"].mean())

        ax.bar(x - w / 2, ao_vals, w, color="#88BBDD", edgecolor="black",
               linewidth=0.5, label="Always-on")
        ax.bar(x + w / 2, oo_vals, w, color="#EE6677", edgecolor="black",
               linewidth=0.5, label=u"OFF\u2192ON")

        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6, rotation=20, ha="right")
        ax.set_title(title, fontsize=8)

    ax_hp.set_ylabel("Oracle Phase-2 \u0394 reward")
    ax_gx.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Figure 3b: Dose-response ──────────────────────────────────────────

def figure_dose_response(out_path: str, repo_root: Path = None) -> None:
    """Oracle Phase-2 Δ reward vs post-switch duration (plasticity OFF→ON)."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    df = pd.read_parquet(
        repo_root / "experiments" / "B0.6_dose_response" / "sweep" / "B0_6_dose_heavy_pole.parquet"
    )
    sweep = df[df["eta"] != 0].copy()

    strata = ["low_mid", "high_mid", "near_perfect", "perfect"]
    stratum_colors = {
        "low_mid": "#4477AA", "high_mid": "#EE6677",
        "near_perfect": "#228833", "perfect": "#CCBB44",
    }

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))

    switch_times = sorted(sweep["switch_step"].unique())

    for stratum in strata:
        post_steps = []
        oracle_deltas = []
        pct_improved = []
        for sw in switch_times:
            sub = sweep[(sweep["switch_step"] == sw) & (sweep["stratum"] == stratum)]
            best = sub.loc[sub.groupby("network_id")["delta_reward_phase2"].idxmax()]
            post_steps.append(500 - sw)
            oracle_deltas.append(best["delta_reward_phase2"].mean())
            pct_improved.append((best["delta_reward_phase2"] > 0).mean())

        ax.plot(post_steps, oracle_deltas, "o-",
                color=stratum_colors[stratum], linewidth=1.5, markersize=4,
                label=STRATUM_SHORT[stratum])

    ax.set_xlabel("Post-switch steps available")
    ax.set_ylabel("Oracle Phase-2 Δ reward")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=6.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Figure 5: Acrobot temporal traces ─────────────────────────────────

def _load_dw_traces(npz_path: str, strata: list[str]) -> dict[str, np.ndarray]:
    """Load and pad dw traces from NPZ, return per-stratum (n_traces, max_len) arrays."""
    import re
    from collections import defaultdict

    npz = np.load(npz_path, allow_pickle=True)
    pattern = re.compile(r"^(\w+)_(\d+)_(\d+)_dw$")
    by_stratum: dict[str, list[np.ndarray]] = defaultdict(list)

    for key in npz.keys():
        m = pattern.match(key)
        if m and m.group(1) in strata:
            by_stratum[m.group(1)].append(npz[key])

    max_len = 500
    result = {}
    for stratum in strata:
        traces = by_stratum.get(stratum, [])
        if not traces:
            continue
        padded = np.full((len(traces), max_len), np.nan)
        for i, t in enumerate(traces):
            n = min(len(t), max_len)
            padded[i, :n] = t[:n]
        result[stratum] = padded
    return result


def _smooth_and_cutoff(padded: np.ndarray, min_frac: float = 0.3) -> tuple[np.ndarray, int]:
    """Return smoothed mean trace and cutoff index."""
    mean_trace = np.nanmean(padded, axis=0)
    kernel = np.ones(5) / 5
    smoothed = np.convolve(mean_trace, kernel, mode="same")
    valid = np.sum(~np.isnan(padded), axis=0)
    cutoff = np.searchsorted(valid < padded.shape[0] * min_frac, True)
    return smoothed, min(cutoff, padded.shape[1])


def _load_episode_lengths(npz_path: str, strata: list[str]) -> dict[str, np.ndarray]:
    """Load episode lengths from NPZ dw traces."""
    import re
    from collections import defaultdict

    npz = np.load(npz_path, allow_pickle=True)
    pattern = re.compile(r"^(\w+)_(\d+)_(\d+)_dw$")
    by_stratum: dict[str, list[int]] = defaultdict(list)

    for key in npz.keys():
        m = pattern.match(key)
        if m and m.group(1) in strata:
            by_stratum[m.group(1)].append(len(npz[key]))

    return {s: np.array(ls) for s, ls in by_stratum.items()}


def _load_obs_traces(npz_path: str, strata: list[str]) -> dict[str, np.ndarray]:
    """Load obs traces from NPZ, compute angular velocity magnitude, return padded arrays."""
    import re
    from collections import defaultdict

    npz = np.load(npz_path, allow_pickle=True)
    pattern = re.compile(r"^(\w+)_(\d+)_(\d+)_obs$")
    by_stratum: dict[str, list[np.ndarray]] = defaultdict(list)

    for key in npz.keys():
        m = pattern.match(key)
        if m and m.group(1) in strata:
            obs = npz[key]  # shape (T, 6)
            vel_mag = np.sqrt(obs[:, 4] ** 2 + obs[:, 5] ** 2)
            by_stratum[m.group(1)].append(vel_mag)

    max_len = 500
    result = {}
    for stratum in strata:
        traces = by_stratum.get(stratum, [])
        if not traces:
            continue
        padded = np.full((len(traces), max_len), np.nan)
        for i, t in enumerate(traces):
            n = min(len(t), max_len)
            padded[i, :n] = t[:n]
        result[stratum] = padded
    return result


def figure_acrobot_temporal_dw(out_path: str, repo_root: Path = None) -> None:
    """Two-panel: angular velocity without vs with plasticity under non-stationarity.

    Left: no plasticity — perturbation destabilises, velocity keeps climbing.
    Right: with plasticity — perturbation hits but velocity partially stabilises.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    base = repo_root / "experiments" / "acrobot" / "temporal_profile"

    import json
    with open(base / "metadata_acrobot.json") as f:
        _meta = json.load(f)
        switch_step = _meta.get("switch_step") or _meta["networks"][0].get("switch_step", 50)

    strata = ["low_mid", "high_mid", "near_perfect", "perfect"]
    stratum_colors = {
        "low_mid": "#4477AA", "high_mid": "#EE6677",
        "near_perfect": "#228833", "perfect": "#CCBB44",
    }

    nopl_vel = _load_obs_traces(str(base / "traces_ns_no_plasticity.npz"), strata)
    pl_vel = _load_obs_traces(str(base / "traces_heavy_link2_2x.npz"), strata)

    fig, (ax_nopl, ax_pl) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 2.2), sharey=True)
    steps = np.arange(500)

    for stratum in strata:
        color = stratum_colors[stratum]
        label = STRATUM_SHORT[stratum]

        for ax, traces in [(ax_nopl, nopl_vel), (ax_pl, pl_vel)]:
            if stratum in traces:
                smoothed, cutoff = _smooth_and_cutoff(traces[stratum])
                ax.plot(steps[:cutoff], smoothed[:cutoff],
                        color=color, linewidth=1.2,
                        label=label if ax is ax_pl else None)

    for ax in (ax_nopl, ax_pl):
        ax.axvline(switch_step, color="black", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("Timestep")

    ax_nopl.set_ylabel("Angular velocity (rad/s)")
    ax_nopl.set_title("No plasticity", fontsize=8)
    ax_pl.set_title("With plasticity", fontsize=8)

    # Legend below the charts, full width
    handles = [plt.Line2D([0], [0], color=stratum_colors[s], lw=1.5,
               label=STRATUM_SHORT[s]) for s in strata]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def figure_acrobot_survival(out_path: str, repo_root: Path = None) -> None:
    """Episode survival under non-stationarity: plasticity vs no plasticity."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    base = repo_root / "experiments" / "acrobot" / "temporal_profile"

    import json
    with open(base / "metadata_acrobot.json") as f:
        _meta = json.load(f)
        switch_step = _meta.get("switch_step") or _meta["networks"][0].get("switch_step", 50)

    strata = ["low_mid", "high_mid", "near_perfect", "perfect"]
    stratum_colors = {
        "low_mid": "#4477AA", "high_mid": "#EE6677",
        "near_perfect": "#228833", "perfect": "#CCBB44",
    }

    pl_lengths = _load_episode_lengths(
        str(base / "traces_heavy_link2_2x.npz"), strata)
    nopl_lengths = _load_episode_lengths(
        str(base / "traces_ns_no_plasticity.npz"), strata)

    steps = np.arange(500)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.2))

    for stratum in strata:
        color = stratum_colors[stratum]
        for lengths, ls, alpha in [(pl_lengths, "-", 1.0), (nopl_lengths, "--", 0.7)]:
            if stratum in lengths:
                survival = np.array([np.mean(lengths[stratum] > t) for t in steps])
                ax.plot(steps, survival, color=color, ls=ls,
                        linewidth=1.2, alpha=alpha)

    ax.axvline(switch_step, color="black", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Fraction unsolved")
    ax.set_ylim(-0.05, 1.05)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=stratum_colors[s], lw=1.2,
                       label=STRATUM_SHORT[s]) for s in strata]
    handles.append(Line2D([0], [0], color="gray", ls="-", lw=1.2, label="Plasticity"))
    handles.append(Line2D([0], [0], color="gray", ls="--", lw=1.2, label="No plasticity"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6,
               frameon=False, bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


# Keep the combined version as a convenience wrapper
def figure_acrobot_temporal(out_path: str, repo_root: Path = None) -> None:
    """Generate both Acrobot temporal figures."""
    from pathlib import Path as P
    p = P(out_path)
    figure_acrobot_temporal_dw(out_path, repo_root=repo_root)
    surv_path = str(p.parent / p.name.replace("temporal_dw", "survival"))
    figure_acrobot_survival(surv_path, repo_root=repo_root)


# ── Main ─────────────────────────────────────────────────────────────

# ── Figure 6: Co-evolution η trajectory ────────────────────────────

def figure_eta_evolution(out_path: str, repo_root: Path = None) -> None:
    """Evolved η over generations for CartPole and Acrobot co-evolution (Condition C).

    2×2 panel: rows = benchmark, columns = best individual / population mean.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    import json

    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH * 2, COLUMN_WIDTH * 1.25), sharex=True)

    color = "#2ca02c"

    configs = [
        ("CartPole", repo_root / "experiments" / "B1" / "condition_C"),
        ("Acrobot", repo_root / "experiments" / "B1_acrobot" / "condition_C"),
    ]

    for row, (label, input_dir) in enumerate(configs):
        all_best_eta = []
        all_pop_eta = []
        for run_dir in sorted(input_dir.glob("run_*")):
            with open(run_dir / "generations.jsonl") as f:
                gens = [json.loads(line) for line in f]
            all_best_eta.append([g["best_eta"] for g in gens])
            all_pop_eta.append([g["eta_mean"] for g in gens])

        max_gen = min(len(x) for x in all_best_eta)
        best_arr = np.array([x[:max_gen] for x in all_best_eta])
        pop_arr = np.array([x[:max_gen] for x in all_pop_eta])
        gen_idx = np.arange(max_gen)

        for col, (data, subtitle) in enumerate([
            (best_arr, f"{label}: best individual \u03b7"),
            (pop_arr, f"{label}: population mean \u03b7"),
        ]):
            ax = axes[row, col]
            median = np.median(data, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)

            ax.fill_between(gen_idx, q25, q75, alpha=0.2, color=color)
            ax.plot(gen_idx, median, color=color, linewidth=1.2)
            ax.axhline(y=0, color="#cc0000", linestyle="--", alpha=0.5, linewidth=0.7)
            ax.set_title(subtitle)
            ax.set_ylabel("\u03b7")
            if row == 1:
                ax.set_xlabel("Generation")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate camera-ready paper figures")
    parser.add_argument("--verify", action="store_true", help="Check for Type-3 fonts with pdffonts")
    args = parser.parse_args()

    # Resolve repo root from script location
    repo_root = Path(__file__).resolve().parent.parent

    # Load B0.5+ data
    sweep_path = repo_root / "experiments" / "B0.5+" / "sweep" / "B0_5plus_sweep.parquet"
    if not sweep_path.exists():
        print(f"ERROR: {sweep_path} not found.")
        sys.exit(1)

    df = pd.read_parquet(sweep_path)
    print(f"Loaded B0.5+ sweep: {len(df):,} rows, {df.network_id.nunique()} networks")

    # Output paths
    fig_dir = repo_root / "experiments" / "B0.5+" / "analysis" / "figures"
    fig1_path = str(fig_dir / "F2_heatmap_high_mid.pdf")
    fig2a_path = str(fig_dir / "F4a_hebbian_delta.pdf")
    fig2b_path = str(fig_dir / "F4b_cohens_d.pdf")
    fig3_path = str(repo_root / "experiments" / "B0.6" / "analysis_cross_variant" / "F_adaptation_premium.pdf")

    os.makedirs(str(fig_dir), exist_ok=True)
    os.makedirs(os.path.dirname(fig3_path), exist_ok=True)

    fig4_path = str(repo_root / "experiments" / "acrobot" / "analysis_static" / "acrobot_cohens_d.pdf")
    os.makedirs(os.path.dirname(fig4_path), exist_ok=True)

    fig5_dir = repo_root / "experiments" / "acrobot" / "temporal_profile"
    fig5_path = str(fig5_dir / "acrobot_temporal_dw.pdf")

    print("\nGenerating figures at ACM column width (3.33 inches)...")
    figure_heatmap_high_mid(df, fig1_path)
    figure_hebbian_delta(df, fig2a_path)
    figure_cohens_d(df, fig2b_path)
    figure_adaptation_premium(fig3_path, repo_root=repo_root)
    figure_acrobot_cohens_d(fig4_path, repo_root=repo_root)
    figure_acrobot_temporal(fig5_path, repo_root=repo_root)

    fig6_path = str(repo_root / "experiments" / "B1" / "analysis" / "figures" / "eta_evolution.pdf")
    os.makedirs(os.path.dirname(fig6_path), exist_ok=True)
    figure_eta_evolution(fig6_path, repo_root=repo_root)

    if args.verify:
        print("\nFont verification:")
        for path in [fig1_path, fig2a_path, fig2b_path, fig3_path, fig4_path, fig5_path, fig6_path]:
            print(f"\n  {path}:")
            result = subprocess.run(["pdffonts", path], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                type3 = [l for l in lines if "Type 3" in l]
                if type3:
                    print(f"    WARNING: Type-3 fonts found!")
                    for l in type3:
                        print(f"    {l}")
                else:
                    print(f"    OK — no Type-3 fonts")
            else:
                print(f"    pdffonts not available (install poppler)")

    print("\nDone.")


if __name__ == "__main__":
    main()
