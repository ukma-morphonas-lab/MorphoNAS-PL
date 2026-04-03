#!/usr/bin/env python3
"""
B0.6 Non-Stationary CartPole Experiment

Tests whether Hebbian plasticity enables genuine online adaptation by switching
CartPole physics mid-episode. Uses the B0.5+ subsample (2,862 networks).

Usage:
  # Pilot (50 networks, gravity_2x)
  .venv/bin/python scripts/run_B0_6_nonstationary.py --pilot

  # Full run, gravity_2x variant
  .venv/bin/python scripts/run_B0_6_nonstationary.py --variant gravity_2x

  # Full run, all variants
  .venv/bin/python scripts/run_B0_6_nonstationary.py --variant all

  # Custom
  .venv/bin/python scripts/run_B0_6_nonstationary.py --variant heavy_pole --n-workers 6
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

from MorphoNAS_PL.experimentB0_6_nonstationary import (
    DECAY_VALUES_FOCUSED,
    DEFAULT_SWITCH_STEP,
    ETA_VALUES_FOCUSED,
    VARIANTS,
    evaluate_network_nonstationary,
    flatten_result_to_row,
    set_shutdown_event,
)
from MorphoNAS_PL.experimentB0_5_natural import load_network_from_file

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
B05P_ROOT = Path("experiments/B0.5+")
B05P_POOL = B05P_ROOT / "pool_subsample" / "networks"
B06_ROOT = Path("experiments/B0.6")


def _load_subsample_ids(
    pilot: bool,
    pilot_per_stratum: int = 10,
    seed: int = 42,
    include_weak: bool = False,
) -> list[int]:
    """Load network IDs, excluding Weak by default.

    Weak networks average ~10 steps (5% P1 utilisation) and never reach the
    mid-episode switch at step 200, so they contribute no Phase 2 signal.
    Excluding them saves ~22,000 evaluations in a full sweep.
    """
    id_file = B05P_ROOT / "network_ids.txt"
    all_ids = [int(line.strip()) for line in open(id_file) if line.strip()]

    # Read strata for filtering (and pilot sampling)
    by_stratum: dict[str, list[int]] = {}
    for nid in all_ids:
        fpath = B05P_POOL / f"network_{nid:05d}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        stratum = data.get("stratum", "weak")
        by_stratum.setdefault(stratum, []).append(nid)

    if not include_weak:
        n_weak = len(by_stratum.pop("weak", []))
        if n_weak > 0:
            logger.info(f"  Excluded {n_weak} Weak networks (never reach switch step)")

    if not pilot:
        return sorted(nid for ids in by_stratum.values() for nid in ids)

    # Stratified pilot sample
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for stratum, ids in sorted(by_stratum.items()):
        n = min(pilot_per_stratum, len(ids))
        chosen = rng.choice(ids, size=n, replace=False).tolist()
        selected.extend(chosen)
        logger.info(f"  Pilot: {stratum} → {n} networks")

    return sorted(int(x) for x in selected)


def _worker_init(shutdown_event):
    """Initialize each worker process."""
    set_shutdown_event(shutdown_event)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from MorphoNAS_PL.parallel_utils import configure_worker_threads
    configure_worker_threads()


def _evaluate_single(args: tuple) -> dict | None:
    """Worker function: evaluate one (network, eta, decay, variant) combo."""
    network_path, eta, decay, variant, switch_step, rollouts = args

    try:
        genome, metadata = load_network_from_file(str(network_path))
        result = evaluate_network_nonstationary(
            genome=genome,
            metadata=metadata,
            eta=eta,
            decay=decay,
            variant=variant,
            switch_step=switch_step,
            rollouts=rollouts,
        )
        return result
    except Exception as e:
        logger.error(f"Error evaluating {network_path}: {e}")
        return {"error": str(e), "network_path": str(network_path)}


def run_sweep(
    network_ids: list[int],
    variant: str,
    *,
    switch_step: int = DEFAULT_SWITCH_STEP,
    rollouts: int = 20,
    n_workers: int | None = None,
    output_dir: Path,
) -> pd.DataFrame:
    """Run the full sweep for one variant and save results."""
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 2)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build work items
    work_items: list[tuple] = []
    for nid in network_ids:
        fpath = B05P_POOL / f"network_{nid:05d}.json"
        if not fpath.exists():
            logger.warning(f"Network file not found: {fpath}")
            continue
        for eta in ETA_VALUES_FOCUSED:
            for decay in DECAY_VALUES_FOCUSED:
                work_items.append((fpath, eta, decay, variant, switch_step, rollouts))

    total = len(work_items)
    print(f"\n{'='*60}")
    print(f"B0.6 Sweep: {variant}")
    print(f"  Networks: {len(network_ids)}")
    print(f"  Grid: {len(ETA_VALUES_FOCUSED)} eta × {len(DECAY_VALUES_FOCUSED)} decay = {len(ETA_VALUES_FOCUSED)*len(DECAY_VALUES_FOCUSED)} points")
    print(f"  Total evaluations: {total:,}")
    print(f"  Workers: {n_workers}")
    print(f"  Switch step: {switch_step}")
    print(f"  Target params: {VARIANTS[variant]}")
    print(f"{'='*60}\n")

    shutdown_event = mp.Event()
    rows: list[dict] = []
    t0 = time.time()

    try:
        with mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(shutdown_event,),
        ) as pool:
            for i, result in enumerate(pool.imap_unordered(_evaluate_single, work_items, chunksize=4)):
                if result is None:
                    continue
                if "error" not in result or result.get("error") is None:
                    rows.append(flatten_result_to_row(result))

                if (i + 1) % 100 == 0 or (i + 1) == total:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta_remaining = (total - i - 1) / rate if rate > 0 else 0
                    print(
                        f"  [{i+1:,}/{total:,}] "
                        f"{elapsed:.0f}s elapsed, "
                        f"{rate:.1f} eval/s, "
                        f"~{eta_remaining:.0f}s remaining"
                    )
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving partial results...")
        shutdown_event.set()

    elapsed = time.time() - t0
    print(f"\nCompleted {len(rows):,} evaluations in {elapsed:.1f}s")

    if not rows:
        print("No results to save.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Save parquet
    parquet_path = output_dir / f"B0_6_{variant}.parquet"
    df.to_parquet(parquet_path, compression="zstd", index=False)
    print(f"Saved: {parquet_path} ({len(df):,} rows)")

    # Save summary JSON
    summary = _compute_summary(df, variant, elapsed, switch_step)
    summary_path = output_dir / f"B0_6_{variant}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    return df


def _compute_summary(df: pd.DataFrame, variant: str, elapsed: float, switch_step: int) -> dict:
    """Compute aggregate summary statistics."""
    summary: dict = {
        "variant": variant,
        "switch_step": switch_step,
        "target_params": VARIANTS[variant],
        "n_networks": int(df["network_id"].nunique()),
        "n_evaluations": len(df),
        "elapsed_seconds": round(elapsed, 1),
        "strata": {},
    }

    for stratum in df["stratum"].unique():
        sdf = df[df["stratum"] == stratum]
        # Baseline only rows (eta=0)
        baseline_rows = sdf[sdf["eta"].abs() < 1e-12]
        # Plastic rows (eta != 0)
        plastic_rows = sdf[sdf["eta"].abs() >= 1e-12]

        stratum_summary: dict[str, Any] = {
            "n_networks": int(sdf["network_id"].nunique()),
            "n_evaluations": len(sdf),
        }

        if len(baseline_rows) > 0:
            stratum_summary["baseline"] = {
                "avg_total_reward": round(float(baseline_rows["baseline_total_reward"].mean()), 2),
                "avg_phase1_reward": round(float(baseline_rows["baseline_phase1_reward"].mean()), 2),
                "avg_phase2_reward": round(float(baseline_rows["baseline_phase2_reward"].mean()), 2),
                "pct_survived_switch": round(float(baseline_rows["baseline_survived_switch"].mean()) * 100, 1),
            }

        if len(plastic_rows) > 0:
            stratum_summary["plastic"] = {
                "avg_total_reward": round(float(plastic_rows["plastic_total_reward"].mean()), 2),
                "avg_phase1_reward": round(float(plastic_rows["plastic_phase1_reward"].mean()), 2),
                "avg_phase2_reward": round(float(plastic_rows["plastic_phase2_reward"].mean()), 2),
                "avg_delta_total": round(float(plastic_rows["delta_reward_total"].mean()), 2),
                "avg_delta_phase2": round(float(plastic_rows["delta_reward_phase2"].mean()), 2),
                "pct_survived_switch": round(float(plastic_rows["plastic_survived_switch"].mean()) * 100, 1),
                "avg_phase2_dw": round(float(plastic_rows["phase2_mean_abs_dw"].mean()), 6),
            }

        summary["strata"][stratum] = stratum_summary

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="B0.6: Non-Stationary CartPole Plasticity Experiment"
    )
    parser.add_argument(
        "--variant",
        default="gravity_2x",
        choices=list(VARIANTS.keys()) + ["all"],
        help="Which non-stationarity variant to test (default: gravity_2x)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot with 50 networks (10 per stratum)",
    )
    parser.add_argument(
        "--pilot-per-stratum",
        type=int,
        default=10,
        help="Networks per stratum in pilot mode (default: 10)",
    )
    parser.add_argument(
        "--switch-step",
        type=int,
        default=DEFAULT_SWITCH_STEP,
        help=f"Step at which physics switch (default: {DEFAULT_SWITCH_STEP})",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=20,
        help="Episodes per evaluation (default: 20)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count - 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pilot sampling (default: 42)",
    )
    parser.add_argument(
        "--include-weak",
        action="store_true",
        help="Include Weak-stratum networks (excluded by default — they never reach the switch step)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    # Load networks
    network_ids = _load_subsample_ids(
        pilot=args.pilot,
        pilot_per_stratum=args.pilot_per_stratum,
        seed=args.seed,
        include_weak=args.include_weak,
    )
    print(f"Loaded {len(network_ids)} networks")

    # Determine output dir
    suffix = "_pilot" if args.pilot else ""
    output_dir = B06_ROOT / f"sweep{suffix}"

    # Determine variants to run
    variants = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    for variant in variants:
        run_sweep(
            network_ids,
            variant,
            switch_step=args.switch_step,
            rollouts=args.rollouts,
            n_workers=args.n_workers,
            output_dir=output_dir,
        )

    print(f"\nAll done! Results in: {output_dir}")


if __name__ == "__main__":
    main()
