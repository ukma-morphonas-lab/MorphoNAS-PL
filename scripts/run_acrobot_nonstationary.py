#!/usr/bin/env python3
"""
Acrobot Non-Stationary Plasticity Experiment

Tests whether Hebbian plasticity enables genuine online adaptation by switching
Acrobot physics mid-episode. Uses the Acrobot pool (non-Weak networks).

Usage:
  # Pilot (50 networks, heavy_link2)
  python run_acrobot_nonstationary.py --pilot

  # Full run, heavy_link2 variant
  python run_acrobot_nonstationary.py --variant heavy_link2

  # Calibrate perturbation magnitude on 50 networks
  python run_acrobot_nonstationary.py --calibrate

  # Full run, all variants
  python run_acrobot_nonstationary.py --variant all
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

from MorphoNAS_PL.experiment_acrobot_nonstationary import (
    DECAY_VALUES_FOCUSED,
    DEFAULT_SWITCH_STEP,
    ETA_VALUES_FOCUSED,
    VARIANTS,
    evaluate_network_nonstationary,
    flatten_result_to_row,
    set_shutdown_event,
)
from MorphoNAS_PL.experiment_acrobot import load_network_from_file

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────
POOL_ROOT = Path("experiments/acrobot/pool")
POOL_NETWORKS = POOL_ROOT / "networks"
NS_ROOT = Path("experiments/acrobot/sweep_nonstationary")


def _load_network_ids(
    pilot: bool,
    pilot_per_stratum: int = 10,
    seed: int = 42,
    include_weak: bool = False,
    max_networks: int | None = None,
) -> list[int]:
    """Load network IDs from the Acrobot pool, excluding Weak by default."""
    id_file = POOL_ROOT / "network_ids.txt"
    if not id_file.exists():
        # Fall back to scanning the networks directory
        all_ids = sorted([
            int(f.replace("network_", "").replace(".json", ""))
            for f in os.listdir(POOL_NETWORKS)
            if f.endswith(".json")
        ])
    else:
        all_ids = [int(line.strip()) for line in open(id_file) if line.strip()]

    # Group by stratum
    by_stratum: dict[str, list[int]] = {}
    for nid in all_ids:
        fpath = POOL_NETWORKS / f"network_{nid:05d}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        stratum = data.get("stratum", "weak")
        by_stratum.setdefault(stratum, []).append(nid)

    if not include_weak:
        n_weak = len(by_stratum.pop("weak", []))
        if n_weak > 0:
            logger.info(f"  Excluded {n_weak} Weak networks")

    all_non_weak = sorted(nid for ids in by_stratum.values() for nid in ids)

    if max_networks is not None:
        all_non_weak = all_non_weak[:max_networks]

    if not pilot:
        return all_non_weak

    # Stratified pilot sample
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for stratum, ids in sorted(by_stratum.items()):
        n = min(pilot_per_stratum, len(ids))
        chosen = rng.choice(ids, size=n, replace=False).tolist()
        selected.extend(chosen)
        logger.info(f"  Pilot: {stratum} -> {n} networks")

    return sorted(int(x) for x in selected)


def _worker_init(shutdown_event):
    set_shutdown_event(shutdown_event)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from MorphoNAS_PL.parallel_utils import configure_worker_threads
    configure_worker_threads()


def _evaluate_single(args: tuple) -> dict | None:
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


CHECKPOINT_INTERVAL = 500  # save partial results every N evaluations


def _save_checkpoint(rows: list[dict], path: Path) -> None:
    """Save current results to parquet (atomic via temp file)."""
    tmp = path.with_suffix(".parquet.tmp")
    pd.DataFrame(rows).to_parquet(tmp, compression="zstd", index=False)
    tmp.rename(path)


def run_sweep(
    network_ids: list[int],
    variant: str,
    *,
    switch_step: int = DEFAULT_SWITCH_STEP,
    rollouts: int = 20,
    n_workers: int | None = None,
    output_dir: Path,
    resume: bool = True,
) -> pd.DataFrame:
    """Run the full non-stationary sweep for one variant.

    Supports resume: saves checkpoints every CHECKPOINT_INTERVAL evaluations.
    On restart, loads existing checkpoint and skips already-completed
    (network_id, eta, decay) combinations.
    """
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"acrobot_ns_{variant}.parquet"

    # Resume: load existing partial results
    existing_rows: list[dict] = []
    completed_keys: set[tuple] = set()
    if resume and parquet_path.exists():
        existing_df = pd.read_parquet(parquet_path)
        existing_rows = existing_df.to_dict("records")
        for _, row in existing_df.iterrows():
            completed_keys.add((int(row["network_id"]), float(row["eta"]), float(row["decay"])))
        print(f"  Resuming: {len(existing_rows):,} existing results loaded, "
              f"{len(completed_keys):,} unique (net, eta, decay) done")

    # Build work items, skipping completed
    work_items: list[tuple] = []
    total_possible = 0
    for nid in network_ids:
        fpath = POOL_NETWORKS / f"network_{nid:05d}.json"
        if not fpath.exists():
            logger.warning(f"Network file not found: {fpath}")
            continue
        for eta in ETA_VALUES_FOCUSED:
            for decay in DECAY_VALUES_FOCUSED:
                total_possible += 1
                if (nid, eta, decay) not in completed_keys:
                    work_items.append((fpath, eta, decay, variant, switch_step, rollouts))

    total = len(work_items)
    skipped = total_possible - total
    print(f"\n{'='*60}")
    print(f"Acrobot Non-Stationary Sweep: {variant}")
    print(f"  Networks: {len(network_ids)}")
    print(f"  Grid: {len(ETA_VALUES_FOCUSED)} eta x {len(DECAY_VALUES_FOCUSED)} decay "
          f"= {len(ETA_VALUES_FOCUSED)*len(DECAY_VALUES_FOCUSED)} points")
    print(f"  Total evaluations: {total:,} ({skipped:,} skipped from checkpoint)")
    print(f"  Workers: {n_workers}")
    print(f"  Switch step: {switch_step}")
    print(f"  Target params: {VARIANTS[variant]}")
    print(f"  Checkpoint every: {CHECKPOINT_INTERVAL} evals")
    print(f"{'='*60}\n")

    if total == 0:
        print("All evaluations already completed.")
        return pd.DataFrame(existing_rows) if existing_rows else pd.DataFrame()

    shutdown_event = mp.Event()
    rows: list[dict] = list(existing_rows)  # start with checkpoint data
    new_count = 0
    last_checkpoint = 0
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
                    new_count += 1

                if (i + 1) % 50 == 0 or (i + 1) == total:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta_remaining = (total - i - 1) / rate if rate > 0 else 0
                    print(
                        f"  [{i+1:,}/{total:,}] "
                        f"{elapsed:.0f}s elapsed, "
                        f"{rate:.1f} eval/s, "
                        f"~{eta_remaining:.0f}s remaining"
                    )

                # Incremental checkpoint
                if new_count - last_checkpoint >= CHECKPOINT_INTERVAL:
                    _save_checkpoint(rows, parquet_path)
                    last_checkpoint = new_count
                    print(f"  ** Checkpoint saved: {len(rows):,} total rows **")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")
        shutdown_event.set()

    elapsed = time.time() - t0
    print(f"\nCompleted {new_count:,} new evaluations in {elapsed:.1f}s "
          f"({len(rows):,} total rows)")

    if not rows:
        print("No results to save.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Final save
    df.to_parquet(parquet_path, compression="zstd", index=False)
    print(f"Saved: {parquet_path} ({len(df):,} rows)")

    summary = _compute_summary(df, variant, elapsed, switch_step)
    summary_path = output_dir / f"acrobot_ns_{variant}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    return df


def _compute_summary(df: pd.DataFrame, variant: str, elapsed: float, switch_step: int) -> dict:
    from typing import Any
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
        baseline_rows = sdf[sdf["eta"].abs() < 1e-12]
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
            }

        if len(plastic_rows) > 0:
            stratum_summary["plastic"] = {
                "avg_total_reward": round(float(plastic_rows["plastic_total_reward"].mean()), 2),
                "avg_delta_total": round(float(plastic_rows["delta_reward_total"].mean()), 2),
                "avg_delta_phase2": round(float(plastic_rows["delta_reward_phase2"].mean()), 2),
                "avg_phase2_dw": round(float(plastic_rows["phase2_mean_abs_dw"].mean()), 6),
            }

        summary["strata"][stratum] = stratum_summary

    return summary


def run_calibration(
    network_ids: list[int],
    *,
    n_workers: int | None = None,
    output_dir: Path,
) -> None:
    """Test perturbation magnitudes x switch steps on a small sample.

    Tests 3 link mass multipliers at 3 switch steps to find a combination
    that causes measurable but not catastrophic disruption. Competent
    Acrobot networks solve in ~65-150 steps, so late switch steps (100+)
    may miss the episode entirely for fast networks.
    """
    calibration_variants = ["heavy_link2_2x", "heavy_link2_3x", "heavy_link2_5x"]
    switch_steps = [30, 50, 75]

    print(f"\n{'='*60}")
    print("ACROBOT PERTURBATION CALIBRATION")
    print(f"  Networks: {len(network_ids)}")
    print(f"  Variants: {calibration_variants}")
    print(f"  Switch steps: {switch_steps}")
    print(f"{'='*60}\n")

    results_summary = []

    for switch_step in switch_steps:
        cal_dir = output_dir / "calibration" / f"step{switch_step}"
        for variant_name in calibration_variants:
            label = f"{variant_name}_step{switch_step}"

            # Resume: skip if parquet already exists
            existing = cal_dir / f"acrobot_ns_{variant_name}.parquet"
            if existing.exists():
                print(f"\n--- {label} --- SKIPPED (parquet exists)")
                continue

            print(f"\n--- {label} ---")
            df = run_sweep(
                network_ids,
                variant_name,
                switch_step=switch_step,
                rollouts=20,
                n_workers=n_workers,
                output_dir=cal_dir,
            )
            if len(df) > 0 and "baseline_total_reward" in df.columns:
                baseline_only = df[df["eta"].abs() < 1e-12]
                if len(baseline_only) > 0:
                    mean_total = baseline_only["baseline_total_reward"].mean()
                    pct_survived = (baseline_only["baseline_phase2_steps"] > 0).mean() * 100
                    mean_p2_reward = baseline_only["baseline_phase2_reward"].mean()
                    print(f"  Switch step {switch_step}, {variant_name}:")
                    print(f"    Mean total reward: {mean_total:.1f}")
                    print(f"    % reached Phase 2: {pct_survived:.1f}%")
                    print(f"    Mean Phase 2 reward: {mean_p2_reward:.1f}")
                    results_summary.append({
                        "variant": variant_name,
                        "params": VARIANTS[variant_name],
                        "switch_step": switch_step,
                        "mean_total_reward": round(float(mean_total), 2),
                        "pct_reached_phase2": round(float(pct_survived), 1),
                        "mean_phase2_reward": round(float(mean_p2_reward), 2),
                    })

    # Save calibration summary
    import json
    cal_base = output_dir / "calibration"
    cal_base.mkdir(parents=True, exist_ok=True)
    cal_path = cal_base / "calibration_summary.json"
    with open(cal_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':>20s} | {'Step':>4s} | {'Total R':>8s} | {'% P2':>5s} | {'P2 R':>8s}")
    print("-" * 60)
    for r in results_summary:
        print(f"{r['variant']:>20s} | {r['switch_step']:>4d} | "
              f"{r['mean_total_reward']:>+8.1f} | {r['pct_reached_phase2']:>5.1f} | "
              f"{r['mean_phase2_reward']:>+8.1f}")
    print(f"\nResults saved to {cal_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Acrobot Non-Stationary Plasticity Experiment"
    )
    parser.add_argument(
        "--variant", default="heavy_link2",
        choices=list(VARIANTS.keys()) + ["all"],
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-per-stratum", type=int, default=10)
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration on 50 networks to find right perturbation magnitude")
    parser.add_argument("--switch-step", type=int, default=DEFAULT_SWITCH_STEP)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--max-networks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-weak", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    if args.calibrate:
        # Calibration mode: small sample, multiple perturbation magnitudes
        network_ids = _load_network_ids(
            pilot=True,
            pilot_per_stratum=10,
            seed=args.seed,
            include_weak=args.include_weak,
        )
        run_calibration(network_ids, n_workers=args.n_workers, output_dir=NS_ROOT)
        return

    network_ids = _load_network_ids(
        pilot=args.pilot,
        pilot_per_stratum=args.pilot_per_stratum,
        seed=args.seed,
        include_weak=args.include_weak,
        max_networks=args.max_networks,
    )
    print(f"Loaded {len(network_ids)} networks")

    suffix = "_pilot" if args.pilot else ""
    output_dir = NS_ROOT / f"sweep{suffix}"

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
