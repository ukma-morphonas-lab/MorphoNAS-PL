#!/usr/bin/env python3
"""
B1 Co-Evolution Analysis
========================

Reads B1 co-evolution experiment results and produces publication-quality
figures and statistical summaries.

Outputs:
  1. Convergence curves — best & mean fitness vs generation (median + IQR)
  2. Final fitness box plots — with Mann-Whitney U tests (Bonferroni-corrected)
  3. Evolved eta distribution — Condition C histogram + sign test
  4. Eta trajectory — median eta over generations in Condition C (IQR band)
  5. Stratum distribution — bar chart of final strata per condition
  6. Structural comparison — num_neurons/num_connections box plots per condition
  7. Summary stats — JSON

Usage:
    python scripts/analyze_B1_coevolution.py \\
        --input-dir experiments/B1 \\
        --output-dir experiments/B1/analysis \\
        --conditions A,B,C
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Allow imports from the code/ directory
sys.path.append(os.path.abspath("code"))

from MorphoNAS_PL.experimentB0_5_natural import (
    Stratum,
    get_stratum_label,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Publication plot style
# ---------------------------------------------------------------------------

PUBLICATION_RCPARAMS = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
}
plt.rcParams.update(PUBLICATION_RCPARAMS)

CONDITION_COLORS = {
    "A": "#1f77b4",  # blue  — no plasticity
    "B": "#ff7f0e",  # orange — fixed plasticity
    "C": "#2ca02c",  # green — co-evolved plasticity
}

CONDITION_LABELS = {
    "A": "A: No plasticity",
    "B": "B: Fixed plasticity",
    "C": "C: Co-evolved plasticity",
}

STRATUM_COLORS = {
    "weak": "#d62728",
    "low_mid": "#ff7f0e",
    "high_mid": "#bcbd22",
    "near_perfect": "#2ca02c",
    "perfect": "#1f77b4",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_generations(input_dir: Path, condition: str) -> list[list[dict]]:
    """Load generations.jsonl from all runs for a condition.

    Returns a list of runs, each run being a list of generation dicts.
    """
    cond_dir = input_dir / f"condition_{condition}"
    if not cond_dir.exists():
        logger.warning(f"Condition directory not found: {cond_dir}")
        return []

    runs = []
    for run_dir in sorted(cond_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        jsonl_path = run_dir / "generations.jsonl"
        if not jsonl_path.exists():
            logger.warning(f"Missing generations.jsonl: {jsonl_path}")
            continue
        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            runs.append(records)

    logger.info(f"Condition {condition}: loaded {len(runs)} runs")
    return runs


def load_final_bests(input_dir: Path, condition: str) -> list[dict]:
    """Load final_best.json from all runs for a condition."""
    cond_dir = input_dir / f"condition_{condition}"
    if not cond_dir.exists():
        return []

    results = []
    for run_dir in sorted(cond_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        fb_path = run_dir / "final_best.json"
        if not fb_path.exists():
            logger.warning(f"Missing final_best.json: {fb_path}")
            continue
        with open(fb_path) as f:
            data = json.load(f)
        data["_run_dir"] = str(run_dir)
        results.append(data)

    logger.info(f"Condition {condition}: loaded {len(results)} final_best files")
    return results


def _runs_to_array(runs: list[list[dict]], field: str) -> np.ndarray:
    """Convert per-run generation records to a 2D array (runs x generations).

    Pads shorter runs with NaN so all runs have the same number of generations.
    """
    if not runs:
        return np.empty((0, 0))
    max_gen = max(len(r) for r in runs)
    arr = np.full((len(runs), max_gen), np.nan)
    for i, run in enumerate(runs):
        for j, rec in enumerate(run):
            arr[i, j] = rec[field]
    return arr


# ---------------------------------------------------------------------------
# 1. Convergence curves
# ---------------------------------------------------------------------------


def plot_convergence(
    all_runs: dict[str, list[list[dict]]],
    conditions: list[str],
    fig_dir: Path,
) -> None:
    """Best & mean fitness vs generation: median + IQR shading, per condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for metric, ax, title in [
        ("best_fitness", axes[0], "Best Fitness"),
        ("avg_fitness", axes[1], "Mean Fitness"),
    ]:
        for cond in conditions:
            runs = all_runs.get(cond, [])
            if not runs:
                continue
            arr = _runs_to_array(runs, metric)
            if arr.size == 0:
                continue

            gens = np.arange(arr.shape[1])
            median = np.nanmedian(arr, axis=0)
            q25 = np.nanpercentile(arr, 25, axis=0)
            q75 = np.nanpercentile(arr, 75, axis=0)

            color = CONDITION_COLORS[cond]
            label = CONDITION_LABELS[cond]
            ax.plot(gens, median, color=color, label=label, linewidth=1.5)
            ax.fill_between(gens, q25, q75, alpha=0.2, color=color)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("B1 Co-Evolution: Convergence Curves (Median ± IQR)", y=1.02)
    fig.tight_layout()

    out_path = fig_dir / "convergence_curves.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 2. Final fitness comparison
# ---------------------------------------------------------------------------


def analyze_final_fitness(
    all_final: dict[str, list[dict]],
    conditions: list[str],
    fig_dir: Path,
) -> dict:
    """Box plots of best fitness at final generation + Mann-Whitney U tests."""
    # Collect final fitness per condition
    fitness_by_cond: dict[str, np.ndarray] = {}
    for cond in conditions:
        finals = all_final.get(cond, [])
        if finals:
            fitness_by_cond[cond] = np.array([f["fitness"] for f in finals])

    if not fitness_by_cond:
        logger.warning("No final fitness data available")
        return {}

    # Box plot
    fig, ax = plt.subplots(figsize=(6, 5))
    data_list = []
    labels = []
    colors = []
    for cond in conditions:
        if cond in fitness_by_cond:
            data_list.append(fitness_by_cond[cond])
            labels.append(CONDITION_LABELS[cond])
            colors.append(CONDITION_COLORS[cond])

    bp = ax.boxplot(
        data_list,
        labels=labels,
        patch_artist=True,
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Best Fitness")
    ax.set_title("Final Best Fitness by Condition")
    ax.grid(True, axis="y", alpha=0.3)

    out_path = fig_dir / "final_fitness_boxplot.pdf"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")

    # Mann-Whitney U tests with Bonferroni correction
    cond_list = [c for c in conditions if c in fitness_by_cond]
    n_comparisons = len(cond_list) * (len(cond_list) - 1) // 2
    mw_results = {}

    for i in range(len(cond_list)):
        for j in range(i + 1, len(cond_list)):
            c1, c2 = cond_list[i], cond_list[j]
            x, y = fitness_by_cond[c1], fitness_by_cond[c2]
            u_stat, p_val = stats.mannwhitneyu(x, y, alternative="two-sided")
            p_corrected = min(p_val * n_comparisons, 1.0)

            # Effect size: rank-biserial correlation r = 1 - 2U/(n1*n2)
            n1, n2 = len(x), len(y)
            r_effect = 1.0 - (2.0 * u_stat) / (n1 * n2)

            key = f"{c1}_vs_{c2}"
            mw_results[key] = {
                "U": float(u_stat),
                "p_raw": float(p_val),
                "p_bonferroni": float(p_corrected),
                "effect_size_r": float(r_effect),
                "n1": n1,
                "n2": n2,
                "median_1": float(np.median(x)),
                "median_2": float(np.median(y)),
            }

            sig = "***" if p_corrected < 0.001 else (
                "**" if p_corrected < 0.01 else (
                    "*" if p_corrected < 0.05 else "ns"
                )
            )
            print(
                f"  {c1} vs {c2}: U={u_stat:.0f}, "
                f"p={p_val:.4e} (Bonferroni: {p_corrected:.4e}) {sig}, "
                f"r={r_effect:+.3f}, "
                f"median {c1}={np.median(x):.4f} vs {c2}={np.median(y):.4f}"
            )

    return mw_results


# ---------------------------------------------------------------------------
# 3. Evolved eta distribution (Condition C)
# ---------------------------------------------------------------------------


def analyze_eta_distribution(
    all_final: dict[str, list[dict]],
    fig_dir: Path,
) -> dict:
    """Histogram of best eta values from Condition C + sign test."""
    finals_c = all_final.get("C", [])
    if not finals_c:
        logger.warning("No Condition C data for eta distribution")
        return {}

    etas = np.array([f["eta"] for f in finals_c])

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(
        etas,
        bins=30,
        color=CONDITION_COLORS["C"],
        edgecolor="black",
        alpha=0.7,
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="eta = 0")
    ax.axvline(
        np.median(etas),
        color="darkgreen",
        linestyle="-.",
        linewidth=1.5,
        label=f"Median = {np.median(etas):.4f}",
    )
    ax.set_xlabel("Evolved eta (learning rate)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Evolved eta (Condition C, Final Generation)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = fig_dir / "eta_distribution_condC.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")

    # Sign test for anti-Hebbian dominance (eta < 0)
    n_negative = int(np.sum(etas < 0))
    n_positive = int(np.sum(etas > 0))
    n_zero = int(np.sum(etas == 0))
    n_total = n_negative + n_positive  # exclude exact zeros

    if n_total > 0:
        # Two-sided binomial test: H0 is P(eta < 0) = 0.5
        sign_result = stats.binomtest(n_negative, n_total, 0.5)
        sign_p = float(sign_result.pvalue)
    else:
        sign_p = 1.0

    sign_results = {
        "n_runs": len(etas),
        "n_negative": n_negative,
        "n_positive": n_positive,
        "n_zero": n_zero,
        "sign_test_p": sign_p,
        "anti_hebbian_fraction": n_negative / max(n_total, 1),
        "median_eta": float(np.median(etas)),
        "mean_eta": float(np.mean(etas)),
        "std_eta": float(np.std(etas)),
    }

    sig = "significant" if sign_p < 0.05 else "not significant"
    print(
        f"  Sign test for anti-Hebbian dominance: "
        f"{n_negative}/{n_total} negative ({100 * n_negative / max(n_total, 1):.1f}%), "
        f"p = {sign_p:.4e} ({sig})"
    )
    print(
        f"  Eta stats: median = {np.median(etas):.4f}, "
        f"mean = {np.mean(etas):.4f}, std = {np.std(etas):.4f}"
    )

    return sign_results


# ---------------------------------------------------------------------------
# 4. Eta trajectory (Condition C)
# ---------------------------------------------------------------------------


def plot_eta_trajectory(
    all_runs: dict[str, list[list[dict]]],
    fig_dir: Path,
) -> None:
    """Median eta over generations in Condition C with IQR band."""
    runs_c = all_runs.get("C", [])
    if not runs_c:
        logger.warning("No Condition C runs for eta trajectory")
        return

    # Best eta of each generation
    arr_best_eta = _runs_to_array(runs_c, "best_eta")
    arr_eta_mean = _runs_to_array(runs_c, "eta_mean")

    if arr_best_eta.size == 0:
        return

    gens = np.arange(arr_best_eta.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for arr, ax, title in [
        (arr_best_eta, axes[0], "Best Individual's eta"),
        (arr_eta_mean, axes[1], "Population Mean eta"),
    ]:
        median = np.nanmedian(arr, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)

        color = CONDITION_COLORS["C"]
        ax.plot(gens, median, color=color, linewidth=1.5, label="Median")
        ax.fill_between(gens, q25, q75, alpha=0.2, color=color, label="IQR")
        ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="eta = 0")
        ax.set_xlabel("Generation")
        ax.set_ylabel("eta")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle("B1 Condition C: Eta Evolution Over Generations (Median ± IQR)", y=1.02)
    fig.tight_layout()

    out_path = fig_dir / "eta_trajectory_condC.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 5. Stratum distribution
# ---------------------------------------------------------------------------


def plot_stratum_distribution(
    all_final: dict[str, list[dict]],
    conditions: list[str],
    fig_dir: Path,
) -> dict:
    """Bar chart of strata counts per condition."""
    strata_order = [s.value for s in Stratum]
    strata_labels = [get_stratum_label(s) for s in Stratum]

    counts_by_cond: dict[str, dict[str, int]] = {}
    for cond in conditions:
        finals = all_final.get(cond, [])
        counter = Counter(f.get("stratum", "unknown") for f in finals)
        counts_by_cond[cond] = {s: counter.get(s, 0) for s in strata_order}

    if not counts_by_cond:
        return {}

    n_strata = len(strata_order)
    n_conds = len(conditions)
    x = np.arange(n_strata)
    bar_width = 0.8 / max(n_conds, 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, cond in enumerate(conditions):
        if cond not in counts_by_cond:
            continue
        vals = [counts_by_cond[cond][s] for s in strata_order]
        offset = (i - (n_conds - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            vals,
            bar_width * 0.9,
            label=CONDITION_LABELS[cond],
            color=CONDITION_COLORS[cond],
            edgecolor="black",
            alpha=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(strata_labels, rotation=30, ha="right")
    ax.set_ylabel("Number of Runs")
    ax.set_title("Stratum Distribution of Final Best Individuals")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = fig_dir / "stratum_distribution.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")

    return counts_by_cond


# ---------------------------------------------------------------------------
# 6. Structural comparison (grow genome, extract topology)
# ---------------------------------------------------------------------------


def _extract_structure(genome_dict: dict) -> dict | None:
    """Grow a genome and extract structural features.

    Returns dict with num_neurons, num_connections or None on failure.
    """
    try:
        from MorphoNAS.genome import Genome
        from MorphoNAS.grid import Grid

        genome = Genome.from_dict(genome_dict)
        grid = Grid(genome)
        grid.run_simulation(verbose=False)
        G = grid.get_graph()
        return {
            "num_neurons": G.number_of_nodes(),
            "num_connections": G.number_of_edges(),
        }
    except Exception as e:
        logger.warning(f"Failed to grow genome: {e}")
        return None


def plot_structural_comparison(
    all_final: dict[str, list[dict]],
    conditions: list[str],
    fig_dir: Path,
) -> dict:
    """Box plots of num_neurons and num_connections per condition."""
    struct_data: dict[str, list[dict]] = {}

    for cond in conditions:
        finals = all_final.get(cond, [])
        cond_structures = []
        for fb in finals:
            genome_dict = fb.get("genome")
            if genome_dict is None:
                continue
            s = _extract_structure(genome_dict)
            if s is not None:
                cond_structures.append(s)
        struct_data[cond] = cond_structures
        logger.info(
            f"Condition {cond}: extracted structure from "
            f"{len(cond_structures)}/{len(finals)} genomes"
        )

    if not any(struct_data.values()):
        logger.warning("No structural data extracted")
        return {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for metric, ax, ylabel in [
        ("num_neurons", axes[0], "Number of Neurons"),
        ("num_connections", axes[1], "Number of Connections"),
    ]:
        data_list = []
        labels = []
        colors = []
        for cond in conditions:
            values = [s[metric] for s in struct_data.get(cond, []) if metric in s]
            if values:
                data_list.append(values)
                labels.append(CONDITION_LABELS[cond])
                colors.append(CONDITION_COLORS[cond])

        if not data_list:
            continue

        bp = ax.boxplot(
            data_list,
            labels=labels,
            patch_artist=True,
            widths=0.5,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Structural Comparison of Evolved Networks", y=1.02)
    fig.tight_layout()

    out_path = fig_dir / "structural_comparison.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")

    # Compile summary stats
    struct_summary = {}
    for cond in conditions:
        entries = struct_data.get(cond, [])
        if not entries:
            continue
        neurons = [e["num_neurons"] for e in entries]
        connections = [e["num_connections"] for e in entries]
        struct_summary[cond] = {
            "n": len(entries),
            "neurons_median": float(np.median(neurons)),
            "neurons_mean": float(np.mean(neurons)),
            "neurons_std": float(np.std(neurons)),
            "connections_median": float(np.median(connections)),
            "connections_mean": float(np.mean(connections)),
            "connections_std": float(np.std(connections)),
        }

    return struct_summary


# ---------------------------------------------------------------------------
# 7. Summary stats
# ---------------------------------------------------------------------------


def compile_summary(
    all_runs: dict[str, list[list[dict]]],
    all_final: dict[str, list[dict]],
    conditions: list[str],
    mw_results: dict,
    sign_results: dict,
    stratum_counts: dict,
    struct_summary: dict,
) -> dict:
    """Compile all statistics into a single summary dict."""
    summary: dict = {"conditions": {}}

    for cond in conditions:
        runs = all_runs.get(cond, [])
        finals = all_final.get(cond, [])

        if not finals:
            continue

        fitnesses = [f["fitness"] for f in finals]
        rewards = [f["reward"] for f in finals]

        cond_summary = {
            "n_runs": len(runs),
            "n_final_bests": len(finals),
            "fitness": {
                "mean": float(np.mean(fitnesses)),
                "median": float(np.median(fitnesses)),
                "std": float(np.std(fitnesses)),
                "min": float(np.min(fitnesses)),
                "max": float(np.max(fitnesses)),
                "q25": float(np.percentile(fitnesses, 25)),
                "q75": float(np.percentile(fitnesses, 75)),
            },
            "reward": {
                "mean": float(np.mean(rewards)),
                "median": float(np.median(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
            },
        }

        # Eta/decay stats for conditions B and C
        if cond in ("B", "C"):
            etas = [f["eta"] for f in finals]
            decays = [f["decay"] for f in finals]
            cond_summary["eta"] = {
                "mean": float(np.mean(etas)),
                "median": float(np.median(etas)),
                "std": float(np.std(etas)),
            }
            cond_summary["decay"] = {
                "mean": float(np.mean(decays)),
                "median": float(np.median(decays)),
                "std": float(np.std(decays)),
            }

        # Convergence info: generation at which median best_fitness first
        # reaches 90% of its final value
        if runs:
            arr = _runs_to_array(runs, "best_fitness")
            if arr.size > 0:
                median_curve = np.nanmedian(arr, axis=0)
                final_val = median_curve[-1]
                threshold = 0.9 * final_val
                reached = np.where(median_curve >= threshold)[0]
                conv_gen = int(reached[0]) if len(reached) > 0 else -1
                cond_summary["convergence_gen_90pct"] = conv_gen

        if cond in stratum_counts:
            cond_summary["stratum_counts"] = stratum_counts[cond]

        if cond in struct_summary:
            cond_summary["structure"] = struct_summary[cond]

        summary["conditions"][cond] = cond_summary

    if mw_results:
        summary["mann_whitney_tests"] = mw_results

    if sign_results:
        summary["sign_test_anti_hebbian"] = sign_results

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B1 Co-Evolution Analysis: figures + statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="experiments/B1",
        help="Base input directory (default: experiments/B1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/B1/analysis",
        help="Output directory for figures and stats (default: experiments/B1/analysis)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="A,B,C",
        help="Comma-separated conditions to include (default: A,B,C)",
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
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    conditions = [c.strip() for c in args.conditions.split(",")]

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Create output directories
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("B1 Co-Evolution Analysis")
    print("=" * 60)
    print(f"Input:      {input_dir}")
    print(f"Output:     {output_dir}")
    print(f"Conditions: {conditions}")
    print()

    all_runs: dict[str, list[list[dict]]] = {}
    all_final: dict[str, list[dict]] = {}

    for cond in conditions:
        all_runs[cond] = load_generations(input_dir, cond)
        all_final[cond] = load_final_bests(input_dir, cond)

    total_runs = sum(len(v) for v in all_runs.values())
    total_finals = sum(len(v) for v in all_final.values())
    print(f"Loaded: {total_runs} run trajectories, {total_finals} final results")
    print()

    if total_runs == 0 and total_finals == 0:
        logger.error("No data found. Check --input-dir path and directory structure.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Convergence curves
    # ------------------------------------------------------------------
    print("-" * 60)
    print("1. Convergence Curves")
    print("-" * 60)
    if total_runs > 0:
        plot_convergence(all_runs, conditions, fig_dir)
        print("  Done.")
    else:
        print("  Skipped (no generation data).")
    print()

    # ------------------------------------------------------------------
    # 2. Final fitness comparison
    # ------------------------------------------------------------------
    print("-" * 60)
    print("2. Final Fitness Comparison (Mann-Whitney U)")
    print("-" * 60)
    mw_results = {}
    if total_finals > 0:
        mw_results = analyze_final_fitness(all_final, conditions, fig_dir)
    else:
        print("  Skipped (no final_best data).")
    print()

    # ------------------------------------------------------------------
    # 3. Evolved eta distribution (Condition C)
    # ------------------------------------------------------------------
    print("-" * 60)
    print("3. Evolved Eta Distribution (Condition C)")
    print("-" * 60)
    sign_results = {}
    if "C" in conditions:
        sign_results = analyze_eta_distribution(all_final, fig_dir)
    else:
        print("  Skipped (Condition C not included).")
    print()

    # ------------------------------------------------------------------
    # 4. Eta trajectory (Condition C)
    # ------------------------------------------------------------------
    print("-" * 60)
    print("4. Eta Trajectory (Condition C)")
    print("-" * 60)
    if "C" in conditions and all_runs.get("C"):
        plot_eta_trajectory(all_runs, fig_dir)
        print("  Done.")
    else:
        print("  Skipped (no Condition C generation data).")
    print()

    # ------------------------------------------------------------------
    # 5. Stratum distribution
    # ------------------------------------------------------------------
    print("-" * 60)
    print("5. Stratum Distribution")
    print("-" * 60)
    stratum_counts = {}
    if total_finals > 0:
        stratum_counts = plot_stratum_distribution(all_final, conditions, fig_dir)
        for cond in conditions:
            if cond in stratum_counts:
                print(f"  Condition {cond}: {stratum_counts[cond]}")
    else:
        print("  Skipped (no final_best data).")
    print()

    # ------------------------------------------------------------------
    # 6. Structural comparison
    # ------------------------------------------------------------------
    print("-" * 60)
    print("6. Structural Comparison")
    print("-" * 60)
    struct_summary = {}
    if total_finals > 0:
        struct_summary = plot_structural_comparison(all_final, conditions, fig_dir)
        for cond, ss in struct_summary.items():
            print(
                f"  Condition {cond}: "
                f"neurons={ss['neurons_median']:.0f} (median), "
                f"connections={ss['connections_median']:.0f} (median), "
                f"n={ss['n']}"
            )
    else:
        print("  Skipped (no final_best data).")
    print()

    # ------------------------------------------------------------------
    # 7. Summary stats JSON
    # ------------------------------------------------------------------
    print("-" * 60)
    print("7. Summary Stats")
    print("-" * 60)
    summary = compile_summary(
        all_runs,
        all_final,
        conditions,
        mw_results,
        sign_results,
        stratum_counts,
        struct_summary,
    )

    summary_path = output_dir / "summary_stats.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")
    print()

    # Print key results
    print("=" * 60)
    print("Key Results")
    print("=" * 60)
    for cond in conditions:
        cs = summary.get("conditions", {}).get(cond, {})
        if not cs:
            continue
        fit = cs.get("fitness", {})
        print(
            f"  Condition {cond}: "
            f"median fitness = {fit.get('median', 0):.4f} "
            f"(IQR: {fit.get('q25', 0):.4f}–{fit.get('q75', 0):.4f}), "
            f"n = {cs.get('n_final_bests', 0)}"
        )
    print()
    print("Analysis complete.")


if __name__ == "__main__":
    main()
