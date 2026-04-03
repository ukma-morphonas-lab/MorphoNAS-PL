#!/usr/bin/env python3
"""
Acrobot Extended Validation Experiment Runner

Two-phase experiment to find beneficial fixed plasticity settings at finer
η resolution than the coarse 22-point grid.

Phase 1 (pilot): Monte Carlo sample of 30 grid points from the 248-point
candidate grid, both static and NS conditions.

Phase 2 (dense): Targeted dense sweep in a promising subregion identified
by Phase 1.

Usage:
  # Quick test (10 networks per stratum, pilot grid)
  python scripts/run_acrobot_extended.py --phase pilot --pilot

  # Full pilot (all 513 non-Weak networks)
  python scripts/run_acrobot_extended.py --phase pilot

  # Dense sweep in a subregion
  python scripts/run_acrobot_extended.py --phase dense \\
    --eta-min -0.005 --eta-max 0.005 --n-eta 20 \\
    --lambda-min 0 --lambda-max 0.02 --n-lambda 8
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("code"))

from MorphoNAS_PL.experiment_acrobot_extended import (
    evaluate_network_extended,
    flatten_result_to_row,
    generate_dense_grid,
    sample_pilot_grid,
    set_shutdown_event,
)
from MorphoNAS_PL.experiment_acrobot import load_network_from_file

logger = logging.getLogger(__name__)

POOL_ROOT = Path("experiments/acrobot/pool")
POOL_NETWORKS = POOL_ROOT / "networks"
OUTPUT_ROOT = Path("experiments/acrobot/sweep_extended")

NON_WEAK_STRATA = ["low_mid", "high_mid", "near_perfect", "perfect"]


def _load_network_ids(
    pilot: bool,
    pilot_per_stratum: int = 10,
    seed: int = 42,
    max_networks: int | None = None,
    network_offset: int = 0,
) -> list[int]:
    """Load non-Weak network IDs from the Acrobot pool."""
    by_stratum: dict[str, list[int]] = {}
    for f in os.listdir(POOL_NETWORKS):
        if not f.endswith(".json"):
            continue
        fpath = POOL_NETWORKS / f
        with open(fpath) as fh:
            data = json.load(fh)
        stratum = data.get("stratum", "weak")
        if stratum in NON_WEAK_STRATA:
            by_stratum.setdefault(stratum, []).append(data["network_id"])

    all_non_weak = sorted(nid for ids in by_stratum.values() for nid in ids)

    if network_offset > 0:
        all_non_weak = all_non_weak[network_offset:]

    if max_networks is not None:
        all_non_weak = all_non_weak[:max_networks]

    if not pilot:
        return all_non_weak

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for stratum in sorted(by_stratum):
        ids = sorted(by_stratum[stratum])
        n = min(pilot_per_stratum, len(ids))
        chosen = rng.choice(ids, size=n, replace=False).tolist()
        selected.extend(chosen)
        logger.info("  Pilot: %s -> %d networks", stratum, n)

    return sorted(int(x) for x in selected)


def _worker_init(shutdown_event):
    set_shutdown_event(shutdown_event)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from MorphoNAS_PL.parallel_utils import configure_worker_threads
    configure_worker_threads()


def _evaluate_single(args: tuple) -> dict | None:
    network_path, eta, decay, condition, rollouts = args
    try:
        genome, metadata = load_network_from_file(str(network_path))
        result = evaluate_network_extended(
            genome=genome, metadata=metadata,
            eta=eta, decay=decay, condition=condition,
            rollouts=rollouts,
        )
        return result
    except Exception as e:
        logger.error("Error evaluating %s: %s", network_path, e)
        return {"error": str(e), "network_path": str(network_path)}


CHECKPOINT_INTERVAL = 500


def _save_checkpoint(rows: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    pd.DataFrame(rows).to_parquet(tmp, compression="zstd", index=False)
    tmp.rename(path)


def run_sweep(
    network_ids: list[int],
    grid_points: list[tuple[float, float]],
    condition: str,
    *,
    rollouts: int = 20,
    n_workers: int | None = None,
    parquet_path: Path,
    resume: bool = True,
) -> pd.DataFrame:
    """Run sweep for one condition (static or ns)."""
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 2)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict] = []
    completed_keys: set[tuple] = set()
    if resume and parquet_path.exists():
        existing_df = pd.read_parquet(parquet_path)
        existing_rows = existing_df.to_dict("records")
        for _, row in existing_df.iterrows():
            completed_keys.add((int(row["network_id"]), float(row["eta"]), float(row["decay"])))
        print(f"  Resuming: {len(existing_rows):,} existing results, "
              f"{len(completed_keys):,} keys done")

    work_items: list[tuple] = []
    total_possible = 0
    for nid in network_ids:
        fpath = POOL_NETWORKS / f"network_{nid:05d}.json"
        if not fpath.exists():
            continue
        for eta, decay in grid_points:
            total_possible += 1
            if (nid, eta, decay) not in completed_keys:
                work_items.append((fpath, eta, decay, condition, rollouts))

    total = len(work_items)
    skipped = total_possible - total
    print(f"\n  {condition.upper()} sweep:")
    print(f"    Networks: {len(network_ids)}, Grid points: {len(grid_points)}")
    print(f"    Evaluations: {total:,} ({skipped:,} skipped)")
    print(f"    Workers: {n_workers}")

    if total == 0:
        print("    All evaluations already completed.")
        return pd.DataFrame(existing_rows) if existing_rows else pd.DataFrame()

    shutdown_event = mp.Event()
    rows: list[dict] = list(existing_rows)
    new_count = 0
    last_checkpoint = 0
    t0 = time.time()

    try:
        with mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(shutdown_event,),
        ) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_evaluate_single, work_items, chunksize=4)
            ):
                if result is None:
                    continue
                if "error" not in result or result.get("error") is None:
                    rows.append(flatten_result_to_row(result))
                    new_count += 1

                if (i + 1) % 50 == 0 or (i + 1) == total:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta_remaining = (total - i - 1) / rate if rate > 0 else 0
                    print(
                        f"    [{i+1:,}/{total:,}] "
                        f"{elapsed:.0f}s elapsed, {rate:.1f} eval/s, "
                        f"~{eta_remaining:.0f}s remaining"
                    )

                if new_count - last_checkpoint >= CHECKPOINT_INTERVAL:
                    _save_checkpoint(rows, parquet_path)
                    last_checkpoint = new_count
                    print(f"    ** Checkpoint: {len(rows):,} rows **")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")
        shutdown_event.set()

    elapsed = time.time() - t0
    print(f"  Completed {new_count:,} new evaluations in {elapsed:.1f}s")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, compression="zstd", index=False)
    print(f"  Saved: {parquet_path} ({len(df):,} rows)")
    return df


def compute_pilot_summary(
    static_df: pd.DataFrame | None,
    ns_df: pd.DataFrame | None,
    grid_points: list[tuple[float, float]],
) -> dict:
    """Compute pilot summary: mean Δ reward at each sampled (η, λ)."""
    summary: dict = {"grid_points_sampled": len(grid_points), "conditions": {}}

    for label, df in [("static", static_df), ("ns", ns_df)]:
        if df is None or len(df) == 0:
            continue

        # Per grid point: mean delta_reward across all networks
        agg = (
            df.groupby(["eta", "decay"])["delta_reward"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg = agg.sort_values("mean", ascending=False)

        positive = agg[agg["mean"] > 0]

        # Best by stratum
        best_by_stratum = {}
        for s in NON_WEAK_STRATA:
            sdf = df[df["stratum"] == s]
            if len(sdf) == 0:
                continue
            sagg = (
                sdf.groupby(["eta", "decay"])["delta_reward"]
                .mean()
                .reset_index()
                .sort_values("delta_reward", ascending=False)
            )
            best = sagg.iloc[0]
            best_by_stratum[s] = {
                "eta": float(best["eta"]),
                "decay": float(best["decay"]),
                "mean_delta": round(float(best["delta_reward"]), 2),
            }

        summary["conditions"][label] = {
            "n_evaluations": len(df),
            "n_networks": int(df["network_id"].nunique()),
            "n_positive_grid_points": len(positive),
            "n_total_grid_points": len(agg),
            "top_5": [
                {
                    "eta": float(r["eta"]),
                    "decay": float(r["decay"]),
                    "mean_delta": round(float(r["mean"]), 2),
                    "n": int(r["count"]),
                }
                for _, r in agg.head(5).iterrows()
            ],
            "best_by_stratum": best_by_stratum,
        }

        # Recommend Phase 2 boundaries
        if len(positive) > 0:
            pos_etas = positive["eta"].values
            pos_decays = positive["decay"].values
            summary["conditions"][label]["recommended_phase2"] = {
                "eta_min": round(float(pos_etas.min()), 6),
                "eta_max": round(float(pos_etas.max()), 6),
                "lambda_min": round(float(pos_decays.min()), 6),
                "lambda_max": round(float(pos_decays.max()), 6),
            }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Acrobot Extended Validation Experiment",
    )
    parser.add_argument(
        "--phase", required=True, choices=["pilot", "dense"],
    )
    parser.add_argument("--pilot", action="store_true",
                        help="Quick test with 10 networks per stratum")
    parser.add_argument("--pilot-per-stratum", type=int, default=10)
    parser.add_argument("--pilot-samples", type=int, default=30,
                        help="Number of grid points to sample for pilot")
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--max-networks", type=int, default=None)
    parser.add_argument("--network-offset", type=int, default=0,
                        help="Skip first N networks (for parallel splits)")
    parser.add_argument("--output-tag", default="",
                        help="Tag appended to output filenames (e.g., '_part2')")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--condition", default="both", choices=["static", "ns", "both"],
        help="Run only one condition or both",
    )
    # Dense phase args
    parser.add_argument("--eta-min", type=float, default=-0.005)
    parser.add_argument("--eta-max", type=float, default=0.005)
    parser.add_argument("--n-eta", type=int, default=20)
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=0.02)
    parser.add_argument("--n-lambda", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    network_ids = _load_network_ids(
        pilot=args.pilot,
        pilot_per_stratum=args.pilot_per_stratum,
        seed=args.seed,
        max_networks=args.max_networks,
        network_offset=args.network_offset,
    )
    print(f"Loaded {len(network_ids)} networks")

    # Generate grid
    if args.phase == "pilot":
        grid_points = sample_pilot_grid(n_samples=args.pilot_samples, seed=args.seed)
        prefix = "pilot"
    else:
        grid_points = generate_dense_grid(
            args.eta_min, args.eta_max, args.n_eta,
            args.lambda_min, args.lambda_max, args.n_lambda,
        )
        prefix = "dense"

    print(f"Grid: {len(grid_points)} (η, λ) points")

    suffix = ("_pilot" if args.pilot else "") + args.output_tag
    output_dir = OUTPUT_ROOT

    print(f"\n{'='*60}")
    print(f"ACROBOT EXTENDED VALIDATION — {args.phase.upper()}")
    print(f"  Networks: {len(network_ids)}")
    print(f"  Grid points: {len(grid_points)}")
    conditions = ["static", "ns"] if args.condition == "both" else [args.condition]
    n_cond = len(conditions)
    print(f"  Conditions: {', '.join(conditions)}")
    print(f"  Total evaluations: ~{n_cond * len(network_ids) * len(grid_points):,}")
    print(f"{'='*60}")

    static_df = None
    ns_df = None

    if "static" in conditions:
        static_path = output_dir / f"{prefix}_static{suffix}.parquet"
        static_df = run_sweep(
            network_ids, grid_points, "static",
            rollouts=args.rollouts, n_workers=args.n_workers,
            parquet_path=static_path,
        )

    if "ns" in conditions:
        ns_path = output_dir / f"{prefix}_ns{suffix}.parquet"
        ns_df = run_sweep(
            network_ids, grid_points, "ns",
            rollouts=args.rollouts, n_workers=args.n_workers,
            parquet_path=ns_path,
        )

    # Summary
    summary = compute_pilot_summary(static_df, ns_df, grid_points)
    summary["phase"] = args.phase
    summary["n_networks"] = len(network_ids)
    summary["grid_points"] = [{"eta": e, "decay": d} for e, d in grid_points]

    summary_path = output_dir / f"{prefix}_summary{suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cond, cdata in summary.get("conditions", {}).items():
        print(f"\n  {cond.upper()}:")
        print(f"    Positive grid points: {cdata['n_positive_grid_points']}/{cdata['n_total_grid_points']}")
        print(f"    Top 5 (η, λ) by mean Δreward:")
        for t in cdata["top_5"]:
            print(f"      η={t['eta']:+.5f}, λ={t['decay']:.4f}: "
                  f"Δ={t['mean_delta']:+.1f} (n={t['n']})")
        if "best_by_stratum" in cdata:
            print(f"    Best by stratum:")
            for s, b in cdata["best_by_stratum"].items():
                print(f"      {s}: η={b['eta']:+.5f}, λ={b['decay']:.4f}, "
                      f"Δ={b['mean_delta']:+.1f}")
        if "recommended_phase2" in cdata:
            r = cdata["recommended_phase2"]
            print(f"    Recommended Phase 2: η∈[{r['eta_min']}, {r['eta_max']}], "
                  f"λ∈[{r['lambda_min']}, {r['lambda_max']}]")

    print(f"\nAll done! Results in: {output_dir}")


if __name__ == "__main__":
    main()
