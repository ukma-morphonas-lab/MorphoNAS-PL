#!/usr/bin/env python3
"""
Acrobot Non-Stationary Plasticity Sweep Analysis
=================================================
Analyzes the non-stationary sweep (2x LINK_MASS_2 at step 50).

Analyses:
  1. Phase 2 (post-switch) benefit rates by stratum
  2. Anti-Hebbian vs Hebbian under non-stationarity
  3. Plasticity engagement (P2/P1 |Δw| ratio)
  4. Oracle improvement rates
  5. Adaptation premium (NS benefit vs static benefit)
  6. Paper-ready summary

Usage:
  python scripts/analyze_acrobot_ns_sweep.py
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

STRATA_ORDER = ["low_mid", "high_mid", "near_perfect", "perfect"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows, {df.network_id.nunique()} networks")
    return df


def load_static_data(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


# ── Analysis 1: Phase 2 benefit rates ────────────────────────────────

def analyze_phase2_benefit(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("1. PHASE 2 (POST-SWITCH) BENEFIT RATES")
    print("=" * 60)

    anti = df[df["eta"] < 0]
    hebb = df[df["eta"] > 0]

    print(f"\nOverall Phase 2 improvement rates:")
    print(f"  Anti-Hebbian: {(anti.delta_reward_phase2 > 0).mean()*100:.1f}%")
    print(f"  Hebbian:      {(hebb.delta_reward_phase2 > 0).mean()*100:.1f}%")

    print(f"\nPer-stratum:")
    print(f"{'Stratum':>13s}  {'N':>4s}  {'Anti P2%':>8s}  {'Hebb P2%':>8s}  "
          f"{'Anti P2 Δ':>9s}  {'Hebb P2 Δ':>9s}")
    print("-" * 65)

    results = {}
    for s in STRATA_ORDER:
        sdf = df[df.stratum == s]
        if len(sdf) == 0:
            continue
        n = sdf.network_id.nunique()
        sa = sdf[sdf.eta < 0]
        sh = sdf[sdf.eta > 0]
        a_rate = (sa.delta_reward_phase2 > 0).mean() * 100
        h_rate = (sh.delta_reward_phase2 > 0).mean() * 100
        a_delta = sa.delta_reward_phase2.mean()
        h_delta = sh.delta_reward_phase2.mean()
        print(f"{s:>13s}  {n:>4d}  {a_rate:>7.1f}%  {h_rate:>7.1f}%  "
              f"{a_delta:>+9.2f}  {h_delta:>+9.2f}")
        results[s] = {
            "n": n, "anti_p2_rate": a_rate, "hebb_p2_rate": h_rate,
            "anti_p2_delta": a_delta, "hebb_p2_delta": h_delta,
        }

    return results


# ── Analysis 2: Plasticity engagement ────────────────────────────────

def analyze_plasticity_engagement(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("2. PLASTICITY ENGAGEMENT (|Δw| BY PHASE)")
    print("=" * 60)

    plastic = df[df["eta"].abs() >= 1e-12]

    print(f"\nOverall (plastic conditions only):")
    p1_dw = plastic.phase1_mean_abs_dw.mean()
    p2_dw = plastic.phase2_mean_abs_dw.mean()
    ratio_of_means = p2_dw / p1_dw if p1_dw > 1e-15 else 0.0
    median_ratio = plastic.p2_p1_dw_ratio.median()
    mean_ratio = plastic.p2_p1_dw_ratio.mean()
    print(f"  Mean |Δw| Phase 1:    {p1_dw:.6f}")
    print(f"  Mean |Δw| Phase 2:    {p2_dw:.6f}")
    print(f"  P2/P1 ratio of means: {ratio_of_means:.1f}x")
    print(f"  P2/P1 median ratio:   {median_ratio:.1f}x")
    print(f"  P2/P1 mean ratio:     {mean_ratio:.1f}x  (inflated by near-zero P1 outliers)")

    print(f"\nPer-stratum:")
    results = {}
    for s in STRATA_ORDER:
        sdf = plastic[plastic.stratum == s]
        if len(sdf) == 0:
            continue
        p1 = sdf.phase1_mean_abs_dw.mean()
        p2 = sdf.phase2_mean_abs_dw.mean()
        rom = p2 / p1 if p1 > 1e-15 else 0.0
        med = sdf.p2_p1_dw_ratio.median()
        print(f"  {s:>13s}: P1={p1:.6f}, P2={p2:.6f}, ratio_of_means={rom:.1f}x, median={med:.1f}x")
        results[s] = {"p1_dw": p1, "p2_dw": p2, "ratio_of_means": rom, "median_ratio": med}

    return {
        "overall": {
            "p1_dw": p1_dw, "p2_dw": p2_dw,
            "ratio_of_means": ratio_of_means,
            "median_ratio": median_ratio,
            "mean_ratio_WARNING_inflated": mean_ratio,
        },
        "per_stratum": results,
    }


# ── Analysis 3: Oracle ───────────────────────────────────────────────

def analyze_oracle(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("3. ORACLE ANALYSIS (TOTAL REWARD)")
    print("=" * 60)

    n_nets = df.network_id.nunique()
    oracle = df.groupby("network_id").apply(lambda g: g.loc[g.delta_reward_total.idxmax()])
    oracle_improved = oracle[oracle.delta_reward_total > 0]
    rate = len(oracle_improved) / n_nets * 100

    print(f"Networks improved: {len(oracle_improved)}/{n_nets} ({rate:.1f}%)")
    print(f"Oracle mean delta (all): {oracle.delta_reward_total.mean():+.2f}")
    print(f"Oracle mean delta (improved): {oracle_improved.delta_reward_total.mean():+.2f}")

    # Best eta distribution
    anti_oracle = len(oracle_improved[oracle_improved.eta < 0])
    hebb_oracle = len(oracle_improved[oracle_improved.eta > 0])
    print(f"\nOracle prefers anti-Hebbian: {anti_oracle}")
    print(f"Oracle prefers Hebbian:      {hebb_oracle}")

    # Per-stratum
    print(f"\nPer-stratum oracle rates:")
    per_stratum = {}
    for s in STRATA_ORDER:
        s_oracle = oracle[oracle.stratum == s]
        s_imp = s_oracle[s_oracle.delta_reward_total > 0]
        n = len(s_oracle)
        if n == 0:
            continue
        r = len(s_imp) / n * 100
        d = s_oracle.delta_reward_total.mean()
        print(f"  {s:>13s}: {len(s_imp)}/{n} ({r:.1f}%), oracle Δ={d:+.2f}")
        per_stratum[s] = {"rate": r, "mean_delta": d}

    return {
        "rate": rate,
        "anti_preferred": anti_oracle,
        "hebb_preferred": hebb_oracle,
        "per_stratum": per_stratum,
    }


# ── Analysis 4: Adaptation premium ──────────────────────────────────

def analyze_adaptation_premium(df: pd.DataFrame, static_df: pd.DataFrame | None) -> dict:
    print("\n" + "=" * 60)
    print("4. ADAPTATION PREMIUM (NS vs STATIC)")
    print("=" * 60)

    if static_df is None:
        print("  Static sweep data not found. Skipping.")
        return {}

    # Per-network oracle rates
    ns_oracle = df.groupby("network_id").apply(lambda g: g.loc[g.delta_reward_total.idxmax()])
    ns_rate = (ns_oracle.delta_reward_total > 0).mean() * 100

    static_oracle = static_df.groupby("network_id").apply(
        lambda g: g.loc[g.delta_reward.idxmax()]
    )
    static_rate = (static_oracle.delta_reward > 0).mean() * 100

    premium = ns_rate - static_rate

    print(f"Static oracle improvement rate:  {static_rate:.1f}%")
    print(f"NS oracle improvement rate:      {ns_rate:.1f}%")
    print(f"Adaptation premium:              {premium:+.1f}pp")

    print(f"\nPer-stratum:")
    results = {}
    for s in STRATA_ORDER:
        ns_s = ns_oracle[ns_oracle.stratum == s]
        st_s = static_oracle[static_oracle.stratum == s]
        if len(ns_s) == 0 or len(st_s) == 0:
            continue
        ns_r = (ns_s.delta_reward_total > 0).mean() * 100
        st_r = (st_s.delta_reward > 0).mean() * 100
        prem = ns_r - st_r
        print(f"  {s:>13s}: static={st_r:.1f}%, NS={ns_r:.1f}%, premium={prem:+.1f}pp")
        results[s] = {"static_rate": st_r, "ns_rate": ns_r, "premium": prem}

    return {"static_rate": static_rate, "ns_rate": ns_rate, "premium": premium, "per_stratum": results}


# ── Analysis 5: Paper-ready summary ──────────────────────────────────

def print_paper_summary(df: pd.DataFrame, static_df: pd.DataFrame | None) -> dict:
    print("\n" + "=" * 60)
    print("5. PAPER-READY SUMMARY TABLE")
    print("=" * 60)

    header = (
        f"{'Stratum':>13s} | {'N':>4s} | {'NS % Helped':>11s} | "
        f"{'Static % Helped':>15s} | {'Premium':>7s}"
    )
    print(header)
    print("-" * len(header))

    ns_oracle = df.groupby("network_id").apply(lambda g: g.loc[g.delta_reward_total.idxmax()])
    static_oracle = None
    if static_df is not None:
        static_oracle = static_df.groupby("network_id").apply(
            lambda g: g.loc[g.delta_reward.idxmax()]
        )

    rows = []
    for s in STRATA_ORDER:
        ns_s = ns_oracle[ns_oracle.stratum == s]
        n = len(ns_s)
        if n == 0:
            continue
        ns_rate = (ns_s.delta_reward_total > 0).mean() * 100

        if static_oracle is not None:
            st_s = static_oracle[static_oracle.stratum == s]
            st_rate = (st_s.delta_reward > 0).mean() * 100
        else:
            st_rate = float("nan")

        premium = ns_rate - st_rate
        print(f"{s:>13s} | {n:>4d} | {ns_rate:>10.1f}% | {st_rate:>14.1f}% | {premium:>+6.1f}pp")
        rows.append({
            "stratum": s, "n": n, "ns_rate": ns_rate,
            "static_rate": st_rate, "premium": premium,
        })

    return {"rows": rows}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Acrobot NS Sweep Analysis")
    parser.add_argument(
        "--ns-path", type=str,
        default="experiments/acrobot/sweep_nonstationary/sweep/acrobot_ns_heavy_link2_2x.parquet",
    )
    parser.add_argument(
        "--static-path", type=str,
        default="experiments/acrobot/sweep_static/acrobot_static_sweep.parquet",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/acrobot/analysis_nonstationary",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.ns_path)
    static_df = load_static_data(args.static_path)

    p2_results = analyze_phase2_benefit(df)
    engagement = analyze_plasticity_engagement(df)
    oracle = analyze_oracle(df)
    premium = analyze_adaptation_premium(df, static_df)
    table = print_paper_summary(df, static_df)

    # Save
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    all_results = {
        "phase2_benefit": p2_results,
        "plasticity_engagement": engagement,
        "oracle": oracle,
        "adaptation_premium": premium,
        "summary_table": table,
    }

    out_path = os.path.join(args.output_dir, "analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
