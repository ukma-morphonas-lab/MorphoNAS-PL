#!/usr/bin/env python3
"""
Analysis 2: Deeper Topology Regression

Quantifies the density reversal signal (Low-mid: denser → benefit;
Near-perfect: sparser → benefit) with partial correlations, interaction
terms, and full regression models controlling for baseline reward.

Figures:
  F0_partial_correlations  — Heatmap of partial correlations (features × strata)
  F1_density_reversal      — Scatter: density vs delta_reward per stratum
  F2_interaction_effects   — Forest plot of interaction coefficients
  F3_feature_regression    — Per-stratum regression coefficients

Usage:
  python scripts/analyze_topology_regression.py
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
from MorphoNAS_PL.genome_features import (
    FEATURE_COLUMNS,
    load_genome_features,
)

logger = logging.getLogger(__name__)
apply_publication_style()

# ── Paths ────────────────────────────────────────────────────────────
B05P_SWEEP_DIR = "experiments/B0.5+/sweep"
B05P_POOL_DIR = "experiments/B0.5+/pool_subsample"
DEFAULT_OUTPUT_DIR = "experiments/B0.5+/analysis/topology"

NON_WEAK_STRATA = ["low_mid", "high_mid", "near_perfect", "perfect"]

# Key topology features for focused analysis
TOPOLOGY_FEATURES = [
    "connectivity_density",
    "mean_degree",
    "num_neurons",
    "num_connections",
    "diffusion_rate",
    "division_threshold",
    "cell_differentiation_threshold",
    "max_axon_length",
    "axon_growth_threshold",
    "axon_connect_threshold",
    "weight_adjustment_target",
    "weight_adjustment_rate",
]


# ── Statistics ───────────────────────────────────────────────────────

def partial_correlation(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
) -> tuple[float, float]:
    """Compute partial correlation of x and y controlling for z.

    Returns (r_partial, p_value).
    """
    # Residualize x on z
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 10:
        return np.nan, np.nan

    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    resid_x = x - (slope_xz * z + intercept_xz)

    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    resid_y = y - (slope_yz * z + intercept_yz)

    r, p = stats.pearsonr(resid_x, resid_y)
    return float(r), float(p)


def compute_oracle_delta(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-network oracle delta_reward."""
    idx = sweep_df.groupby("network_id")["delta_reward"].idxmax()
    oracle = sweep_df.loc[idx, [
        "network_id", "stratum", "delta_reward", "eta", "decay",
        "baseline_avg_reward",
    ]].copy()
    oracle = oracle.rename(columns={
        "delta_reward": "best_delta_reward",
        "eta": "best_eta",
    })
    return oracle.reset_index(drop=True)


# ── Figures ──────────────────────────────────────────────────────────

def figure_partial_correlations(
    merged: pd.DataFrame,
    features: list[str],
    fig_dir: str,
) -> dict:
    """F0: Heatmap of partial correlations controlling for baseline reward."""
    strata = [s for s in NON_WEAK_STRATA if s in merged["stratum"].unique()]
    matrix = np.full((len(features), len(strata)), np.nan)
    pvals = np.full_like(matrix, np.nan)

    for j, s in enumerate(strata):
        sdf = merged[merged["stratum"] == s]
        baseline = sdf["baseline_reward"].values
        y = sdf["best_delta_reward"].values
        for i, feat in enumerate(features):
            x = sdf[feat].values
            r, p = partial_correlation(x, y, baseline)
            matrix[i, j] = r
            pvals[i, j] = p

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels([STRATUM_LABELS.get(s, s) for s in strata], rotation=30, ha="right")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)

    # Annotate significant cells
    for i in range(len(features)):
        for j in range(len(strata)):
            r = matrix[i, j]
            p = pvals[i, j]
            if np.isnan(r):
                continue
            stars = ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            color = "white" if abs(r) > 0.15 else "black"
            ax.text(j, i, f"{r:.2f}{stars}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Partial r (controlling for baseline)")
    ax.set_title("Partial Correlations: Features vs Oracle Δreward")
    fig.tight_layout()
    save_figure(fig, fig_dir, "F0_partial_correlations")

    return {
        "correlations": {
            feat: {s: float(matrix[i, j]) for j, s in enumerate(strata)}
            for i, feat in enumerate(features)
        },
    }


def figure_density_reversal(
    merged: pd.DataFrame,
    fig_dir: str,
) -> dict:
    """F1: Scatter of connectivity_density vs delta_reward per stratum."""
    strata = [s for s in NON_WEAK_STRATA if s in merged["stratum"].unique()]
    n_strata = len(strata)
    fig, axes = plt.subplots(1, n_strata, figsize=(4 * n_strata, 5), sharey=True)
    if n_strata == 1:
        axes = [axes]

    results = {}
    for idx, s in enumerate(strata):
        ax = axes[idx]
        sdf = merged[merged["stratum"] == s]
        x = sdf["connectivity_density"].values
        y = sdf["best_delta_reward"].values

        ax.scatter(x, y, alpha=0.3, s=15, color=STRATUM_COLORS.get(s, "#999"))

        # Regression line
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 5:
            slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
            x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, "k-", lw=2, alpha=0.7)
            ax.set_title(f"{STRATUM_LABELS.get(s, s)}\nr={r:.3f}, p={p:.2e}")
            results[s] = {
                "r": float(r), "p": float(p),
                "slope": float(slope), "se": float(se),
                "n": int(mask.sum()),
            }

            # Partial correlation (controlling for baseline)
            baseline = sdf["baseline_reward"].values
            r_partial, p_partial = partial_correlation(x, y, baseline)
            results[s]["partial_r"] = float(r_partial)
            results[s]["partial_p"] = float(p_partial)
        else:
            ax.set_title(STRATUM_LABELS.get(s, s))

        ax.set_xlabel("Connectivity Density")
        if idx == 0:
            ax.set_ylabel("Oracle Δreward")

    fig.suptitle("Density Reversal: Connectivity vs Plasticity Benefit", y=1.02)
    fig.tight_layout()
    save_figure(fig, fig_dir, "F1_density_reversal")

    return results


def figure_interaction_effects(
    merged: pd.DataFrame,
    fig_dir: str,
) -> dict:
    """F2: Forest plot of interaction coefficients."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    strata = [s for s in NON_WEAK_STRATA if s in merged["stratum"].unique()]

    # Build interaction model: delta ~ density * stratum_dummies + baseline
    df = merged.copy()
    df = df.dropna(subset=["connectivity_density", "best_delta_reward", "baseline_reward"])

    scaler = StandardScaler()
    df["density_scaled"] = scaler.fit_transform(
        df[["connectivity_density"]],
    )[:, 0]
    df["baseline_scaled"] = scaler.fit_transform(
        df[["baseline_reward"]],
    )[:, 0]

    # Reference stratum: low_mid
    interaction_results = {}
    for s in strata:
        if s == "low_mid":
            continue
        df[f"is_{s}"] = (df["stratum"] == s).astype(float)
        df[f"density_x_{s}"] = df["density_scaled"] * df[f"is_{s}"]

    feature_names = (
        ["density_scaled", "baseline_scaled"]
        + [f"is_{s}" for s in strata if s != "low_mid"]
        + [f"density_x_{s}" for s in strata if s != "low_mid"]
    )
    X = df[feature_names].values
    y = df["best_delta_reward"].values

    model = LinearRegression()
    model.fit(X, y)

    coefs = dict(zip(feature_names, model.coef_))

    # Bootstrap CIs for interaction terms
    interaction_terms = [f"density_x_{s}" for s in strata if s != "low_mid"]
    n_boot = 1000
    rng = np.random.default_rng(42)
    boot_coefs = {t: [] for t in interaction_terms}

    for _ in range(n_boot):
        idx = rng.choice(len(X), size=len(X), replace=True)
        m = LinearRegression()
        m.fit(X[idx], y[idx])
        for j, name in enumerate(feature_names):
            if name in boot_coefs:
                boot_coefs[name].append(m.coef_[j])

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    means = []
    ci_lows = []
    ci_highs = []

    for term in interaction_terms:
        s = term.replace("density_x_", "")
        labels.append(f"Density × {STRATUM_LABELS.get(s, s)}")
        m = coefs[term]
        boots = np.array(boot_coefs[term])
        ci_low = np.percentile(boots, 2.5)
        ci_high = np.percentile(boots, 97.5)
        means.append(m)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
        interaction_results[s] = {
            "coef": float(m),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "significant": bool(ci_low > 0 or ci_high < 0),
        }

    y_pos = range(len(labels))
    ax.errorbar(
        means, y_pos,
        xerr=[np.array(means) - np.array(ci_lows),
              np.array(ci_highs) - np.array(means)],
        fmt="o", color="#2196F3", capsize=5, markersize=8,
    )
    ax.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Interaction Coefficient (vs Low-mid reference)")
    ax.set_title("Density × Stratum Interaction Effects on Δreward")
    fig.tight_layout()
    save_figure(fig, fig_dir, "F2_interaction_effects")

    return {
        "model_r2": float(model.score(X, y)),
        "density_main": float(coefs["density_scaled"]),
        "baseline_main": float(coefs["baseline_scaled"]),
        "interactions": interaction_results,
    }


def figure_feature_regression(
    merged: pd.DataFrame,
    features: list[str],
    fig_dir: str,
) -> dict:
    """F3: Per-stratum regression coefficients for top features."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    strata = [s for s in NON_WEAK_STRATA if s in merged["stratum"].unique()]
    results = {}

    coef_matrix = np.full((len(features), len(strata)), np.nan)

    for j, s in enumerate(strata):
        sdf = merged[merged["stratum"] == s].dropna(subset=features + ["best_delta_reward"])
        if len(sdf) < 20:
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(sdf[features].values)
        y = sdf["best_delta_reward"].values

        model = LinearRegression()
        model.fit(X, y)
        coef_matrix[:, j] = model.coef_

        results[s] = {
            "r2": float(model.score(X, y)),
            "coefficients": {f: float(c) for f, c in zip(features, model.coef_)},
        }

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(coef_matrix, cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels(
        [STRATUM_LABELS.get(s, s) for s in strata], rotation=30, ha="right",
    )
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)

    for i in range(len(features)):
        for j in range(len(strata)):
            val = coef_matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 2.5 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Standardized Coefficient")
    ax.set_title("Per-Stratum Regression: Genome Features → Oracle Δreward")
    fig.tight_layout()
    save_figure(fig, fig_dir, "F3_feature_regression")

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analysis 2: Deeper Topology Regression",
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

    # ── Load data ────────────────────────────────────────────────────
    print("Loading B0.5+ sweep data...")
    sweep_df = load_sweep(B05P_SWEEP_DIR)
    sweep_df = sweep_df[sweep_df["stratum"].isin(NON_WEAK_STRATA)]

    print("Loading genome features...")
    genome_df = load_genome_features(B05P_POOL_DIR)

    print("Computing per-network oracle...")
    oracle_df = compute_oracle_delta(sweep_df)

    # Merge
    merged = oracle_df.merge(genome_df, on="network_id", suffixes=("", "_genome"))
    if "stratum_genome" in merged.columns:
        merged = merged.drop(columns=["stratum_genome"])
    merged = merged[merged["stratum"].isin(NON_WEAK_STRATA)]
    print(f"  Merged: {len(merged)} networks")

    # ── Figures ──────────────────────────────────────────────────────
    print("\n--- F0: Partial Correlations ---")
    pcorr_results = figure_partial_correlations(merged, TOPOLOGY_FEATURES, fig_dir)

    print("--- F1: Density Reversal ---")
    density_results = figure_density_reversal(merged, fig_dir)

    print("--- F2: Interaction Effects ---")
    interaction_results = figure_interaction_effects(merged, fig_dir)

    print("--- F3: Feature Regression ---")
    regression_results = figure_feature_regression(merged, TOPOLOGY_FEATURES, fig_dir)

    # ── Save summary ─────────────────────────────────────────────────
    summary = {
        "n_networks": len(merged),
        "partial_correlations": pcorr_results,
        "density_reversal": density_results,
        "interaction_model": interaction_results,
        "per_stratum_regression": regression_results,
    }
    summary_path = os.path.join(args.output_dir, "topology_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print text summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TOPOLOGY REGRESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Networks: {len(merged)}")
    print(f"  Interaction model R²: {interaction_results['model_r2']:.3f}")
    print(f"  Density main effect: {interaction_results['density_main']:.3f}")
    print(f"\n  Density × Stratum interactions (vs Low-mid):")
    for s, r in interaction_results.get("interactions", {}).items():
        sig = "*" if r["significant"] else ""
        print(f"    {s}: {r['coef']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] {sig}")

    print(f"\n  Density reversal (raw Pearson r):")
    for s, r in density_results.items():
        print(f"    {s}: r={r['r']:.3f} (partial r={r.get('partial_r', 'N/A')})")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
