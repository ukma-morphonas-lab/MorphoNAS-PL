#!/usr/bin/env python3
"""
Acrobot Static Plasticity Sweep Analysis
=========================================
Analyzes the 22-point (11 eta x 2 decay) sweep on non-Weak Acrobot networks.

Analyses:
  1. Stratum distribution and pool characterization
  2. Anti-Hebbian vs Hebbian improvement rates and mean delta rewards
  3. Cohen's d (anti-Hebbian advantage) per stratum
  4. Per-network oracle analysis (% improved, oracle delta, best eta distribution)
  5. Regret: oracle vs best-fixed-eta
  6. Per-stratum summary table (paper-ready)

Usage:
  python scripts/analyze_acrobot_static_sweep.py
  python scripts/analyze_acrobot_static_sweep.py --output-dir experiments/acrobot/analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from MorphoNAS_PL.experiment_acrobot import (
    DECAY_VALUES,
    ETA_VALUES,
    STRATUM_BOUNDS,
    Stratum,
    get_stratum_label,
)

STRATA_ORDER = ["low_mid", "high_mid", "near_perfect", "perfect"]


def load_data(sweep_path: str) -> pd.DataFrame:
    df = pd.read_parquet(sweep_path)
    print(f"Loaded {len(df):,} rows, {df.network_id.nunique()} networks")
    return df


def load_pool_stats(pool_dir: str) -> dict:
    """Load pool metadata for full distribution context."""
    meta_path = os.path.join(pool_dir, "pool_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


# ── Analysis 1: Pool characterization ────────────────────────────────

def analyze_pool(df: pd.DataFrame, pool_dir: str) -> dict:
    print("\n" + "=" * 60)
    print("1. POOL CHARACTERIZATION")
    print("=" * 60)

    pool_meta = load_pool_stats(pool_dir)
    stratum_counts = pool_meta.get("stratum_counts", {})

    total = sum(stratum_counts.values()) if stratum_counts else 0
    weak_n = stratum_counts.get("weak", 0)
    non_weak_n = total - weak_n

    print(f"Total valid networks: {total}")
    print(f"Weak (non-functional): {weak_n} ({weak_n/total*100:.1f}%)" if total > 0 else "")
    print(f"Non-weak (in sweep):   {non_weak_n} ({non_weak_n/total*100:.1f}%)" if total > 0 else "")

    print(f"\nNon-weak stratum breakdown:")
    for s in STRATA_ORDER:
        n = stratum_counts.get(s, 0)
        pct = n / non_weak_n * 100 if non_weak_n > 0 else 0
        print(f"  {s:>13s}: {n:>4d} ({pct:5.1f}%)")

    return {"total": total, "weak": weak_n, "non_weak": non_weak_n, "strata": stratum_counts}


# ── Analysis 2: Anti-Hebbian vs Hebbian ──────────────────────────────

def analyze_anti_vs_hebbian(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("2. ANTI-HEBBIAN vs HEBBIAN")
    print("=" * 60)

    anti = df[df["eta"] < 0]
    hebb = df[df["eta"] > 0]
    ctrl = df[df["eta"].abs() < 1e-12]

    print(f"\nOverall improvement rates:")
    print(f"  Anti-Hebbian (eta<0): {anti.improved.mean()*100:.1f}% "
          f"({int(anti.improved.sum())}/{len(anti)})")
    print(f"  Hebbian (eta>0):      {hebb.improved.mean()*100:.1f}% "
          f"({int(hebb.improved.sum())}/{len(hebb)})")
    print(f"  Control (eta=0):      {ctrl.improved.mean()*100:.1f}%")

    print(f"\nMean delta reward:")
    print(f"  Anti-Hebbian: {anti.delta_reward.mean():+.2f}")
    print(f"  Hebbian:      {hebb.delta_reward.mean():+.2f}")

    print(f"\nPer-stratum breakdown:")
    print(f"{'Stratum':>13s}  {'N nets':>6s}  {'Anti %imp':>9s}  {'Hebb %imp':>9s}  "
          f"{'Anti Δrew':>9s}  {'Hebb Δrew':>9s}")
    print("-" * 70)

    results = {}
    for s in STRATA_ORDER:
        sdf = df[df.stratum == s]
        if len(sdf) == 0:
            continue
        n_nets = sdf.network_id.nunique()
        s_anti = sdf[sdf.eta < 0]
        s_hebb = sdf[sdf.eta > 0]
        a_imp = s_anti.improved.mean() * 100
        h_imp = s_hebb.improved.mean() * 100
        a_delta = s_anti.delta_reward.mean()
        h_delta = s_hebb.delta_reward.mean()
        print(f"{s:>13s}  {n_nets:>6d}  {a_imp:>8.1f}%  {h_imp:>8.1f}%  "
              f"{a_delta:>+9.1f}  {h_delta:>+9.1f}")
        results[s] = {
            "n_nets": n_nets,
            "anti_imp_rate": a_imp,
            "hebb_imp_rate": h_imp,
            "anti_mean_delta": a_delta,
            "hebb_mean_delta": h_delta,
        }

    return results


# ── Analysis 3: Cohen's d ────────────────────────────────────────────

def analyze_cohens_d(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("3. COHEN'S d (ANTI-HEBBIAN ADVANTAGE)")
    print("=" * 60)
    print("Positive d = anti-Hebbian produces higher (less negative) delta reward")

    results = {}
    for s in STRATA_ORDER:
        sdf = df[df.stratum == s]
        a = sdf[sdf.eta < 0].groupby("network_id").delta_reward.mean()
        h = sdf[sdf.eta > 0].groupby("network_id").delta_reward.mean()
        if len(a) > 1 and len(h) > 1:
            pooled_std = np.sqrt((a.std() ** 2 + h.std() ** 2) / 2)
            d = (a.mean() - h.mean()) / pooled_std if pooled_std > 0 else 0
            # Welch's t-test
            t_stat, p_val = stats.ttest_ind(a, h, equal_var=False)
            print(f"  {s:>13s}: d = {d:+.3f}, p = {p_val:.4f} "
                  f"(anti mean={a.mean():+.1f}, hebb mean={h.mean():+.1f})")
            results[s] = {"cohens_d": d, "p_value": p_val}

    return results


# ── Analysis 4: Per-network oracle ───────────────────────────────────

def analyze_oracle(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("4. PER-NETWORK ORACLE ANALYSIS")
    print("=" * 60)

    n_nets = df.network_id.nunique()
    oracle = df.groupby("network_id").apply(lambda g: g.loc[g.delta_reward.idxmax()])
    oracle_improved = oracle[oracle.delta_reward > 0]
    oracle_rate = len(oracle_improved) / n_nets * 100

    print(f"Networks with >= 1 beneficial eta: {len(oracle_improved)}/{n_nets} ({oracle_rate:.1f}%)")
    print(f"Oracle mean delta (all): {oracle.delta_reward.mean():+.2f}")
    print(f"Oracle mean delta (improved only): {oracle_improved.delta_reward.mean():+.2f}")

    # Best eta distribution
    print(f"\nBest eta distribution (oracle-improved networks):")
    best_etas = oracle_improved.eta.value_counts().sort_index()
    for eta, count in best_etas.items():
        sign = "anti" if eta < 0 else "hebb" if eta > 0 else "ctrl"
        bar = "#" * count
        print(f"  eta={eta:+.2f} ({sign:>4s}): {count:>3d} {bar}")

    anti_oracle = len(oracle_improved[oracle_improved.eta < 0])
    hebb_oracle = len(oracle_improved[oracle_improved.eta > 0])
    print(f"\nOracle prefers anti-Hebbian: {anti_oracle} nets ({anti_oracle/len(oracle_improved)*100:.1f}%)")
    print(f"Oracle prefers Hebbian:      {hebb_oracle} nets ({hebb_oracle/len(oracle_improved)*100:.1f}%)")

    # Per-stratum oracle
    print(f"\nPer-stratum oracle improvement rates:")
    per_stratum = {}
    for s in STRATA_ORDER:
        s_oracle = oracle[oracle.stratum == s]
        s_improved = s_oracle[s_oracle.delta_reward > 0]
        s_total = len(s_oracle)
        if s_total == 0:
            continue
        rate = len(s_improved) / s_total * 100
        mean_d = s_oracle.delta_reward.mean()
        print(f"  {s:>13s}: {len(s_improved)}/{s_total} ({rate:.1f}%), oracle Δ={mean_d:+.2f}")
        per_stratum[s] = {"rate": rate, "mean_delta": mean_d, "n_improved": len(s_improved)}

    return {
        "oracle_rate": oracle_rate,
        "oracle_mean_delta": oracle.delta_reward.mean(),
        "anti_preferred": anti_oracle,
        "hebb_preferred": hebb_oracle,
        "per_stratum": per_stratum,
    }


# ── Analysis 5: Regret ───────────────────────────────────────────────

def analyze_regret(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("5. REGRET ANALYSIS (ORACLE vs BEST-FIXED)")
    print("=" * 60)

    # Oracle: best eta per network
    oracle = df.groupby("network_id").apply(lambda g: g.loc[g.delta_reward.idxmax()])
    oracle_mean = oracle.delta_reward.mean()

    # Best-fixed: single (eta, decay) that maximizes mean improvement across all networks
    grid_means = df.groupby(["eta", "decay"]).delta_reward.mean()
    best_fixed_idx = grid_means.idxmax()
    best_fixed_delta = grid_means.max()

    regret = 1.0 - (best_fixed_delta / oracle_mean) if oracle_mean > 0 else float("nan")

    print(f"Oracle mean delta:       {oracle_mean:+.2f}")
    print(f"Best fixed (eta={best_fixed_idx[0]:+.2f}, decay={best_fixed_idx[1]}): {best_fixed_delta:+.2f}")
    print(f"Regret: {regret:.1%} of oracle improvement lost under fixed parameters")

    # Top 5 fixed grid points
    print(f"\nTop 5 fixed (eta, decay) by mean delta reward:")
    top5 = grid_means.nlargest(5)
    for (eta, decay), delta in top5.items():
        print(f"  eta={eta:+.2f}, decay={decay}: Δ={delta:+.2f}")

    return {
        "oracle_mean": oracle_mean,
        "best_fixed_eta": best_fixed_idx[0],
        "best_fixed_decay": best_fixed_idx[1],
        "best_fixed_delta": best_fixed_delta,
        "regret": regret,
    }


# ── Analysis 6: Paper-ready summary table ────────────────────────────

def print_summary_table(df: pd.DataFrame, pool_stats: dict) -> dict:
    print("\n" + "=" * 60)
    print("6. PAPER-READY SUMMARY TABLE")
    print("=" * 60)

    header = (
        f"{'Stratum':>13s} | {'N':>4s} | {'% Improved':>10s} | "
        f"{'Best eta':>8s} | {'Cohen d':>8s} | {'Oracle Δ':>8s}"
    )
    print(header)
    print("-" * len(header))

    table_rows = []
    for s in STRATA_ORDER:
        sdf = df[df.stratum == s]
        if len(sdf) == 0:
            continue
        n_nets = sdf.network_id.nunique()

        # Oracle
        s_oracle = sdf.groupby("network_id").apply(lambda g: g.loc[g.delta_reward.idxmax()])
        s_improved = s_oracle[s_oracle.delta_reward > 0]
        imp_rate = len(s_improved) / n_nets * 100

        # Best eta (mode of oracle)
        if len(s_improved) > 0:
            best_eta = s_improved.eta.mode().iloc[0]
        else:
            best_eta = 0.0

        # Cohen's d
        a = sdf[sdf.eta < 0].groupby("network_id").delta_reward.mean()
        h = sdf[sdf.eta > 0].groupby("network_id").delta_reward.mean()
        if len(a) > 1 and len(h) > 1:
            pooled_std = np.sqrt((a.std() ** 2 + h.std() ** 2) / 2)
            d = (a.mean() - h.mean()) / pooled_std if pooled_std > 0 else 0
        else:
            d = 0

        oracle_delta = s_oracle.delta_reward.mean()

        row = {
            "stratum": s,
            "n": n_nets,
            "pct_improved": imp_rate,
            "best_eta": best_eta,
            "cohens_d": d,
            "oracle_delta": oracle_delta,
        }
        table_rows.append(row)

        print(f"{s:>13s} | {n_nets:>4d} | {imp_rate:>9.1f}% | "
              f"{best_eta:>+8.2f} | {d:>+8.3f} | {oracle_delta:>+8.2f}")

    return {"rows": table_rows}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Acrobot Static Sweep Analysis")
    parser.add_argument(
        "--sweep-path", type=str,
        default="experiments/acrobot/sweep_static/acrobot_static_sweep.parquet",
    )
    parser.add_argument(
        "--pool-dir", type=str,
        default="experiments/acrobot/pool",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/acrobot/analysis_static",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.sweep_path)

    pool_stats = analyze_pool(df, args.pool_dir)
    ah_results = analyze_anti_vs_hebbian(df)
    cohens = analyze_cohens_d(df)
    oracle = analyze_oracle(df)
    regret = analyze_regret(df)
    table = print_summary_table(df, pool_stats)

    # Save all results as JSON
    all_results = {
        "pool": pool_stats,
        "anti_vs_hebbian": ah_results,
        "cohens_d": cohens,
        "oracle": {k: v for k, v in oracle.items() if k != "per_stratum"},
        "oracle_per_stratum": oracle.get("per_stratum", {}),
        "regret": regret,
        "summary_table": table,
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import json

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            r = convert(obj)
            if r is not obj:
                return r
            return super().default(obj)

    out_path = os.path.join(args.output_dir, "analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
