#!/usr/bin/env python3
"""
B0.5+ Gate Simulation Analysis
===============================
Using existing B0.5+ sweep data (357,750 evaluations), test whether simple
threshold-based rules (gates) can capture most of the per-network oracle benefit.

Question: Do we need per-network continuous eta (B1), or does a simple
"plasticity on/off based on baseline reward" rule suffice?

Rules tested:
  - Best-fixed: single (eta, decay) for all networks in a stratum
  - Binary gate: if baseline < threshold → apply best_eta, else → eta=0
  - K-bin rules: split baseline range into K bins, assign best eta per bin
  - Oracle: per-network best (eta, decay) — theoretical ceiling

Output: tables + figures in the B0.5+ analysis directory.
"""

from __future__ import annotations

import argparse
import json
import logging
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
    STRATUM_LABELS,
    STRATUM_COLORS,
    STRATUM_BOUNDS_LIST,
    apply_publication_style,
    load_sweep,
    save_figure,
    strata_present_in,
)

logger = logging.getLogger(__name__)

apply_publication_style()

DEFAULT_SWEEP_DIR = "experiments/B0.5+/sweep"
DEFAULT_OUTPUT_DIR = "experiments/B0.5+/analysis"


def compute_per_network_oracle(df: pd.DataFrame) -> pd.DataFrame:
    """For each network, find (eta, decay) that maximizes delta_reward."""
    idx = df.groupby("network_id")["delta_reward"].idxmax()
    oracle = df.loc[idx, [
        "network_id", "stratum", "baseline_avg_reward",
        "eta", "decay", "delta_reward",
    ]].copy()
    oracle.rename(columns={
        "eta": "oracle_eta",
        "decay": "oracle_decay",
        "delta_reward": "oracle_dr",
    }, inplace=True)
    return oracle.reset_index(drop=True)


def compute_best_fixed(df: pd.DataFrame) -> dict:
    """For each stratum, find single best (eta, decay) and its mean delta_reward."""
    results = {}
    for s in strata_present_in(df):
        sub = df[df["stratum"] == s]
        cell_means = sub.groupby(["eta", "decay"])["delta_reward"].mean()
        best_param = cell_means.idxmax()
        best_eta, best_decay = best_param
        results[s] = {
            "eta": best_eta,
            "decay": best_decay,
            "mean_dr": cell_means.max(),
        }
    return results


def apply_gate_rule(
    oracle_df: pd.DataFrame,
    df: pd.DataFrame,
    threshold: float,
    eta_on: float,
    decay: float = 0.01,
) -> pd.DataFrame:
    """Binary gate: if baseline < threshold → (eta_on, decay), else → (0, 0)."""
    result = oracle_df[["network_id", "stratum", "baseline_avg_reward", "oracle_dr"]].copy()

    # For networks below threshold: look up their delta_reward at (eta_on, decay)
    on_mask = result["baseline_avg_reward"] < threshold
    off_mask = ~on_mask

    # Get delta_reward at the specified (eta, decay) for each network
    lookup = df[(df["eta"] == eta_on) & (df["decay"] == decay)].set_index("network_id")["delta_reward"]
    result["gate_dr"] = 0.0
    result.loc[on_mask, "gate_dr"] = result.loc[on_mask, "network_id"].map(lookup).fillna(0.0)

    # For networks above threshold: eta=0 → delta_reward at (0, 0)
    lookup_off = df[(df["eta"] == 0) & (df["decay"] == 0)].set_index("network_id")["delta_reward"]
    result.loc[off_mask, "gate_dr"] = result.loc[off_mask, "network_id"].map(lookup_off).fillna(0.0)

    return result


def apply_kbin_rule(
    oracle_df: pd.DataFrame,
    df: pd.DataFrame,
    bin_edges: list[float],
    bin_etas: list[float],
    decay: float = 0.01,
) -> pd.DataFrame:
    """K-bin rule: assign different eta per baseline reward bin."""
    result = oracle_df[["network_id", "stratum", "baseline_avg_reward", "oracle_dr"]].copy()
    result["gate_dr"] = 0.0

    baselines = result["baseline_avg_reward"].values
    bin_indices = np.digitize(baselines, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_etas) - 1)

    # Pre-compute lookups for each eta
    eta_lookups = {}
    for eta in set(bin_etas):
        sub = df[(df["eta"] == eta) & (df["decay"] == decay)]
        if sub.empty and eta == 0:
            sub = df[(df["eta"] == 0) & (df["decay"] == 0)]
        eta_lookups[eta] = sub.set_index("network_id")["delta_reward"]

    for i, eta in enumerate(bin_etas):
        mask = bin_indices == i
        if not mask.any():
            continue
        nids = result.loc[mask, "network_id"]
        result.loc[mask, "gate_dr"] = nids.map(eta_lookups[eta]).fillna(0.0).values

    return result


def find_best_binary_gate(
    oracle_df: pd.DataFrame,
    df: pd.DataFrame,
    stratum: str,
) -> dict:
    """Search over thresholds and etas to find best binary gate for a stratum."""
    sub_oracle = oracle_df[oracle_df["stratum"] == stratum]
    if sub_oracle.empty:
        return {}

    baselines = sub_oracle["baseline_avg_reward"].values
    lo, hi = baselines.min(), baselines.max()

    # Available etas from the sweep (negative + zero)
    available_etas = sorted(df["eta"].unique())
    negative_etas = [e for e in available_etas if e < 0]

    best_result = {"capture_pct": 0}
    oracle_total = sub_oracle["oracle_dr"].clip(lower=0).sum()
    if oracle_total <= 0:
        return {"capture_pct": 0, "threshold": 0, "eta_on": 0}

    thresholds = np.linspace(lo, hi, 30)

    for thr in thresholds:
        for eta_on in negative_etas:
            gated = apply_gate_rule(oracle_df, df, thr, eta_on, decay=0.01)
            gated_sub = gated[gated["stratum"] == stratum]
            gate_total = gated_sub["gate_dr"].clip(lower=0).sum()
            capture = 100 * gate_total / oracle_total
            if capture > best_result["capture_pct"]:
                best_result = {
                    "threshold": float(thr),
                    "eta_on": float(eta_on),
                    "capture_pct": float(capture),
                    "mean_dr": float(gated_sub["gate_dr"].mean()),
                }

    return best_result


def find_best_kbin_rule(
    oracle_df: pd.DataFrame,
    df: pd.DataFrame,
    stratum: str,
    k: int,
) -> dict:
    """Find best K-bin rule for a stratum by optimizing bin boundaries and etas."""
    sub_oracle = oracle_df[oracle_df["stratum"] == stratum]
    if sub_oracle.empty:
        return {}

    baselines = sub_oracle["baseline_avg_reward"].values
    oracle_total = sub_oracle["oracle_dr"].clip(lower=0).sum()
    if oracle_total <= 0:
        return {"capture_pct": 0}

    # Use quantile-based bin edges
    quantiles = np.linspace(0, 100, k + 1)
    bin_edges = [np.percentile(baselines, q) for q in quantiles]
    bin_edges[0] = 0
    bin_edges[-1] = 600

    # For each bin, find best eta (brute force over available etas)
    available_etas = sorted(df["eta"].unique())
    candidate_etas = [e for e in available_etas if e <= 0]  # include 0

    best_bin_etas = []
    for i in range(k):
        lo_edge, hi_edge = bin_edges[i], bin_edges[i + 1]
        bin_mask = (baselines >= lo_edge) & (baselines < hi_edge)
        bin_nids = sub_oracle.loc[bin_mask, "network_id"].values

        best_eta = 0.0
        best_mean = -np.inf
        for eta in candidate_etas:
            sub = df[(df["eta"] == eta) & (df["decay"] == 0.01)]
            if sub.empty and eta == 0:
                sub = df[(df["eta"] == 0) & (df["decay"] == 0)]
            if sub.empty:
                continue
            lookup = sub.set_index("network_id")["delta_reward"]
            mean_dr = lookup.reindex(bin_nids).mean()
            if not np.isnan(mean_dr) and mean_dr > best_mean:
                best_mean = mean_dr
                best_eta = eta
        best_bin_etas.append(best_eta)

    gated = apply_kbin_rule(oracle_df, df, bin_edges, best_bin_etas, decay=0.01)
    gated_sub = gated[gated["stratum"] == stratum]
    gate_total = gated_sub["gate_dr"].clip(lower=0).sum()
    capture = 100 * gate_total / oracle_total

    return {
        "k": k,
        "bin_edges": [float(e) for e in bin_edges],
        "bin_etas": [float(e) for e in best_bin_etas],
        "capture_pct": float(capture),
        "mean_dr": float(gated_sub["gate_dr"].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="B0.5+ Gate Simulation Analysis")
    parser.add_argument("--sweep-dir", default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    fig_dir = Path(args.output_dir) / "figures"
    tbl_dir = Path(args.output_dir) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading B0.5+ sweep data...")
    df = load_sweep(args.sweep_dir)
    logger.info(f"  {len(df):,} rows, {df['network_id'].nunique():,} networks, "
                f"{len(df.groupby(['eta', 'decay'])):,} grid points")

    # ── Compute oracle and best-fixed ────────────────────────────────
    oracle_df = compute_per_network_oracle(df)
    best_fixed = compute_best_fixed(df)

    strata = strata_present_in(df)

    # ── Analysis 1: Best binary gate per stratum ─────────────────────
    logger.info("\n=== Binary Gate Search ===")
    gate_results = {}
    for s in strata:
        gate_results[s] = find_best_binary_gate(oracle_df, df, s)
        g = gate_results[s]
        if g:
            bf = best_fixed.get(s, {})
            oracle_mean = oracle_df[oracle_df["stratum"] == s]["oracle_dr"].mean()
            logger.info(
                f"  {STRATUM_LABELS.get(s, s):15s} | "
                f"best-fixed={bf.get('mean_dr', 0):+.1f} | "
                f"gate={g.get('mean_dr', 0):+.1f} (thr={g.get('threshold', 0):.0f}, "
                f"eta={g.get('eta_on', 0):.3f}) | "
                f"oracle={oracle_mean:+.1f} | "
                f"gate captures {g.get('capture_pct', 0):.1f}% of oracle"
            )

    # ── Analysis 2: K-bin rules (2, 3, 5 bins) ──────────────────────
    logger.info("\n=== K-Bin Rules ===")
    kbin_results = {}
    for s in strata:
        kbin_results[s] = {}
        oracle_mean = oracle_df[oracle_df["stratum"] == s]["oracle_dr"].mean()
        bf_mean = best_fixed.get(s, {}).get("mean_dr", 0)
        oracle_total = oracle_df[oracle_df["stratum"] == s]["oracle_dr"].clip(lower=0).sum()
        bf_capture = 0.0
        if oracle_total > 0:
            # Compute best-fixed capture
            bf_eta = best_fixed[s]["eta"]
            bf_decay = best_fixed[s]["decay"]
            bf_sub = df[(df["eta"] == bf_eta) & (df["decay"] == bf_decay) & (df["stratum"] == s)]
            bf_total = bf_sub["delta_reward"].clip(lower=0).sum()
            bf_capture = 100 * bf_total / oracle_total

        logger.info(f"\n  {STRATUM_LABELS.get(s, s):15s}:")
        logger.info(f"    best-fixed: mean={bf_mean:+.1f}, captures {bf_capture:.1f}% of oracle")

        for k in [2, 3, 5, 10]:
            kbin_results[s][k] = find_best_kbin_rule(oracle_df, df, s, k)
            r = kbin_results[s][k]
            logger.info(
                f"    {k:2d}-bin rule:  mean={r.get('mean_dr', 0):+.1f}, "
                f"captures {r.get('capture_pct', 0):.1f}% of oracle | "
                f"etas={r.get('bin_etas', [])}"
            )

    # ── Analysis 3: Gate-off fraction per stratum ────────────────────
    logger.info("\n=== Networks Best Served by eta=0 (No Plasticity) ===")
    for s in strata:
        sub = oracle_df[oracle_df["stratum"] == s]
        # Check: how many networks have oracle_eta == 0?
        # Also check: how many have oracle_dr <= 0 (any eta hurts)?
        pct_eta_zero = 100 * (sub["oracle_eta"] == 0).mean()
        pct_no_improve = 100 * (sub["oracle_dr"] <= 0).mean()
        logger.info(
            f"  {STRATUM_LABELS.get(s, s):15s}: "
            f"{pct_eta_zero:.1f}% best at eta=0, "
            f"{pct_no_improve:.1f}% never improve (oracle_dr <= 0)"
        )

    # ── Summary table ────────────────────────────────────────────────
    rows = []
    for s in strata:
        bf = best_fixed.get(s, {})
        g = gate_results.get(s, {})
        oracle_mean = oracle_df[oracle_df["stratum"] == s]["oracle_dr"].mean()
        n = oracle_df[oracle_df["stratum"] == s].shape[0]

        oracle_total = oracle_df[oracle_df["stratum"] == s]["oracle_dr"].clip(lower=0).sum()
        bf_capture = 0.0
        if oracle_total > 0:
            bf_eta = bf.get("eta", 0)
            bf_decay = bf.get("decay", 0)
            bf_sub = df[(df["eta"] == bf_eta) & (df["decay"] == bf_decay) & (df["stratum"] == s)]
            bf_total = bf_sub["delta_reward"].clip(lower=0).sum()
            bf_capture = 100 * bf_total / oracle_total

        row = {
            "Stratum": STRATUM_LABELS.get(s, s),
            "N": n,
            "Best-fixed mean Δr": bf.get("mean_dr", 0),
            "Best-fixed capture%": bf_capture,
            "Binary gate mean Δr": g.get("mean_dr", 0),
            "Binary gate capture%": g.get("capture_pct", 0),
            "Gate threshold": g.get("threshold", 0),
            "Gate eta": g.get("eta_on", 0),
        }
        for k in [2, 3, 5, 10]:
            kr = kbin_results.get(s, {}).get(k, {})
            row[f"{k}-bin capture%"] = kr.get("capture_pct", 0)

        row["Oracle mean Δr"] = oracle_mean
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = tbl_dir / "T_gate_simulation.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.2f")
    logger.info(f"\nSaved summary table to {summary_path}")

    # ── Figure: Capture % comparison ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(strata))
    width = 0.15

    bf_captures = []
    gate_captures = []
    k2_captures = []
    k3_captures = []
    k5_captures = []
    k10_captures = []

    for s in strata:
        bf = best_fixed.get(s, {})
        oracle_total = oracle_df[oracle_df["stratum"] == s]["oracle_dr"].clip(lower=0).sum()
        bf_capture = 0.0
        if oracle_total > 0:
            bf_eta = bf.get("eta", 0)
            bf_decay = bf.get("decay", 0)
            bf_sub = df[(df["eta"] == bf_eta) & (df["decay"] == bf_decay) & (df["stratum"] == s)]
            bf_total = bf_sub["delta_reward"].clip(lower=0).sum()
            bf_capture = 100 * bf_total / oracle_total
        bf_captures.append(bf_capture)
        gate_captures.append(gate_results.get(s, {}).get("capture_pct", 0))
        k2_captures.append(kbin_results.get(s, {}).get(2, {}).get("capture_pct", 0))
        k3_captures.append(kbin_results.get(s, {}).get(3, {}).get("capture_pct", 0))
        k5_captures.append(kbin_results.get(s, {}).get(5, {}).get("capture_pct", 0))
        k10_captures.append(kbin_results.get(s, {}).get(10, {}).get("capture_pct", 0))

    bar_kw = dict(edgecolor="black", linewidth=0.4)
    bars_bf = ax.bar(x - 2.5 * width, bf_captures, width, label="Best-fixed (1 param)", color="#4477AA", **bar_kw)
    bars_gate = ax.bar(x - 1.5 * width, gate_captures, width, label="Binary gate", color="#EE6677", **bar_kw)
    bars_k2 = ax.bar(x - 0.5 * width, k2_captures, width, label="2-bin rule", color="#228833", **bar_kw)
    bars_k3 = ax.bar(x + 0.5 * width, k3_captures, width, label="3-bin rule", color="#CCBB44", **bar_kw)
    bars_k5 = ax.bar(x + 1.5 * width, k5_captures, width, label="5-bin rule", color="#66CCEE", **bar_kw)
    bars_k10 = ax.bar(x + 2.5 * width, k10_captures, width, label="10-bin rule", color="#AA3377", **bar_kw)

    ax.set_ylabel("% of oracle benefit captured")
    ax.set_xticks(x)
    ax.set_xticklabels([STRATUM_LABELS.get(s, s) for s in strata], rotation=15, ha="right")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.set_ylim(0, 100)
    ax.axhline(100, color="#999999", linestyle="--", linewidth=0.8, alpha=0.5)

    # Add value labels on bars
    for bars in [bars_bf, bars_gate, bars_k2, bars_k3, bars_k5, bars_k10]:
        for bar in bars:
            h = bar.get_height()
            if h > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    save_figure(fig, str(fig_dir), "F_gate_simulation")

    # ── Figure: Oracle eta distribution per stratum (violin) ─────────
    fig, axes = plt.subplots(1, len(strata), figsize=(3 * len(strata), 5), sharey=True)
    if len(strata) == 1:
        axes = [axes]

    for i, s in enumerate(strata):
        ax = axes[i]
        sub = oracle_df[oracle_df["stratum"] == s]
        etas = sub["oracle_eta"].values

        # Histogram
        eta_vals = sorted(df["eta"].unique())
        counts = [(etas == e).sum() for e in eta_vals]
        total = len(etas)
        pcts = [100 * c / total for c in counts]

        colors = ["#2ca02c" if e == 0 else ("#1f77b4" if e < 0 else "#d62728") for e in eta_vals]
        ax.barh(range(len(eta_vals)), pcts, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(eta_vals)))
        if i == 0:
            ax.set_yticklabels([f"{e:.3f}" if abs(e) < 0.1 else f"{e:.2f}" for e in eta_vals], fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("% of networks")
        ax.set_title(f"{STRATUM_LABELS.get(s, s)}\n(N={total})", fontsize=10)
        ax.axvline(100 / len(eta_vals), color="gray", linestyle=":", alpha=0.5)

    fig.suptitle("Per-Network Oracle Eta Distribution by Stratum", fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, str(fig_dir), "F_oracle_eta_distribution")

    # ── Print key insight ────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("KEY INSIGHT")
    logger.info("=" * 70)
    for s in strata:
        k10 = kbin_results.get(s, {}).get(10, {}).get("capture_pct", 0)
        oracle_mean = oracle_df[oracle_df["stratum"] == s]["oracle_dr"].mean()
        n = oracle_df[oracle_df["stratum"] == s].shape[0]
        if oracle_mean > 1:
            logger.info(
                f"  {STRATUM_LABELS.get(s, s):15s}: Even 10-bin rule captures "
                f"{k10:.0f}% of oracle. "
                f"{'Per-network eta needed' if k10 < 75 else 'Binning may suffice'} "
                f"(oracle mean Δr={oracle_mean:+.1f}, N={n})"
            )

    # ── Save full results as JSON ────────────────────────────────────
    full_results = {
        "best_fixed": {s: {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                           for k, v in d.items()} for s, d in best_fixed.items()},
        "gate_results": {s: {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                             for k, v in d.items()} for s, d in gate_results.items()},
        "kbin_results": {},
    }
    for s in strata:
        full_results["kbin_results"][s] = {}
        for k in [2, 3, 5, 10]:
            r = kbin_results.get(s, {}).get(k, {})
            full_results["kbin_results"][s][k] = {
                kk: ([float(x) for x in vv] if isinstance(vv, list) else
                     float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv)
                for kk, vv in r.items()
            }

    json_path = tbl_dir / "gate_simulation_results.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"\nSaved full results to {json_path}")


if __name__ == "__main__":
    main()
