#!/usr/bin/env python3
"""
B0.6 Adaptation Control Experiment Runner

Tests plasticity OFF→ON to distinguish genuine adaptation from robustification.
Disables plasticity during Phase 1, enables at switch point.

Usage:
  # Pilot (10 networks per stratum, heavy_pole only)
  python scripts/run_B0_6_adaptation.py --pilot --variant heavy_pole

  # Full run, both variants, OFF→ON mode
  python scripts/run_B0_6_adaptation.py --variant all

  # Full run with both modes (OFF→ON + always-on for direct comparison)
  python scripts/run_B0_6_adaptation.py --variant heavy_pole --mode both
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

from MorphoNAS_PL.experimentB0_6_adaptation import (
    DECAY_VALUES_FOCUSED,
    ETA_VALUES_FOCUSED,
    VARIANTS,
    DEFAULT_SWITCH_STEP,
    evaluate_network_adaptation,
    flatten_result_to_row,
    set_shutdown_event,
)
from MorphoNAS_PL.experimentB0_5_natural import load_network_from_file

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────
POOL_ROOT = Path("experiments/B0.5+/pool_subsample")
POOL_NETWORKS = POOL_ROOT / "networks"
OUTPUT_ROOT = Path("experiments/B0.6_adaptation")


def _load_network_ids(
    pilot: bool,
    pilot_per_stratum: int = 10,
    seed: int = 42,
    include_weak: bool = False,
    max_networks: int | None = None,
) -> list[int]:
    """Load network IDs from the B0.5+ pool, excluding Weak by default."""
    all_ids: list[int] = []
    for f in os.listdir(POOL_NETWORKS):
        if f.endswith(".json"):
            nid = int(f.replace("network_", "").replace(".json", ""))
            all_ids.append(nid)
    all_ids.sort()

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
            logger.info("  Excluded %d Weak networks", n_weak)

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
        logger.info("  Pilot: %s -> %d networks", stratum, n)

    return sorted(int(x) for x in selected)


def _worker_init(shutdown_event):
    set_shutdown_event(shutdown_event)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from MorphoNAS_PL.parallel_utils import configure_worker_threads
    configure_worker_threads()


def _evaluate_single(args: tuple) -> dict | None:
    network_path, eta, decay, variant, switch_step, rollouts, plasticity_mode = args
    try:
        genome, metadata = load_network_from_file(str(network_path))
        result = evaluate_network_adaptation(
            genome=genome,
            metadata=metadata,
            eta=eta,
            decay=decay,
            variant=variant,
            switch_step=switch_step,
            rollouts=rollouts,
            plasticity_mode=plasticity_mode,
        )
        return result
    except Exception as e:
        logger.error("Error evaluating %s: %s", network_path, e)
        return {"error": str(e), "network_path": str(network_path)}


CHECKPOINT_INTERVAL = 500


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
    modes: list[str],
) -> pd.DataFrame:
    """Run the adaptation sweep for one variant."""
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"B0_6_adaptation_{variant}.parquet"

    # Resume: load existing partial results
    existing_rows: list[dict] = []
    completed_keys: set[tuple] = set()
    if resume and parquet_path.exists():
        existing_df = pd.read_parquet(parquet_path)
        existing_rows = existing_df.to_dict("records")
        for _, row in existing_df.iterrows():
            completed_keys.add((
                int(row["network_id"]),
                float(row["eta"]),
                float(row["decay"]),
                str(row.get("plasticity_mode", "off_on")),
            ))
        print(f"  Resuming: {len(existing_rows):,} existing results loaded, "
              f"{len(completed_keys):,} unique keys done")

    # Build work items
    work_items: list[tuple] = []
    total_possible = 0
    for nid in network_ids:
        fpath = POOL_NETWORKS / f"network_{nid:05d}.json"
        if not fpath.exists():
            logger.warning("Network file not found: %s", fpath)
            continue
        for eta in ETA_VALUES_FOCUSED:
            for decay in DECAY_VALUES_FOCUSED:
                for mode in modes:
                    total_possible += 1
                    if (nid, eta, decay, mode) not in completed_keys:
                        work_items.append((
                            fpath, eta, decay, variant, switch_step,
                            rollouts, mode,
                        ))

    total = len(work_items)
    skipped = total_possible - total
    print(f"\n{'='*60}")
    print(f"B0.6 Adaptation Sweep: {variant}")
    print(f"  Networks: {len(network_ids)}")
    print(f"  Grid: {len(ETA_VALUES_FOCUSED)} eta x {len(DECAY_VALUES_FOCUSED)} decay "
          f"= {len(ETA_VALUES_FOCUSED)*len(DECAY_VALUES_FOCUSED)} points")
    print(f"  Modes: {modes}")
    print(f"  Total evaluations: {total:,} ({skipped:,} skipped from checkpoint)")
    print(f"  Workers: {n_workers}")
    print(f"  Switch step: {switch_step}")
    print(f"  Checkpoint every: {CHECKPOINT_INTERVAL} evals")
    print(f"{'='*60}\n")

    if total == 0:
        print("All evaluations already completed.")
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
                        f"  [{i+1:,}/{total:,}] "
                        f"{elapsed:.0f}s elapsed, "
                        f"{rate:.1f} eval/s, "
                        f"~{eta_remaining:.0f}s remaining"
                    )

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
    df.to_parquet(parquet_path, compression="zstd", index=False)
    print(f"Saved: {parquet_path} ({len(df):,} rows)")

    summary = _compute_summary(df, variant, elapsed, switch_step)
    summary_path = output_dir / f"B0_6_adaptation_{variant}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    return df


def _compute_summary(
    df: pd.DataFrame, variant: str, elapsed: float, switch_step: int,
) -> dict:
    from typing import Any
    summary: dict = {
        "variant": variant,
        "switch_step": switch_step,
        "n_networks": int(df["network_id"].nunique()),
        "n_evaluations": len(df),
        "elapsed_seconds": round(elapsed, 1),
        "modes": sorted(df["plasticity_mode"].unique().tolist()),
        "strata": {},
    }

    for mode in df["plasticity_mode"].unique():
        mode_df = df[df["plasticity_mode"] == mode]
        for stratum in mode_df["stratum"].unique():
            sdf = mode_df[mode_df["stratum"] == stratum]
            plastic_rows = sdf[sdf["eta"].abs() >= 1e-12]

            key = f"{stratum}_{mode}"
            stratum_summary: dict[str, Any] = {
                "n_networks": int(sdf["network_id"].nunique()),
                "n_evaluations": len(sdf),
            }

            if len(plastic_rows) > 0:
                stratum_summary["plastic"] = {
                    "avg_delta_total": round(
                        float(plastic_rows["delta_reward_total"].mean()), 2
                    ),
                    "avg_delta_phase2": round(
                        float(plastic_rows["delta_reward_phase2"].mean()), 2
                    ),
                    "avg_phase1_dw": round(
                        float(plastic_rows["phase1_mean_abs_dw"].mean()), 6
                    ),
                    "avg_phase2_dw": round(
                        float(plastic_rows["phase2_mean_abs_dw"].mean()), 6
                    ),
                }

            summary["strata"][key] = stratum_summary

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="B0.6 Adaptation Control Experiment"
    )
    parser.add_argument(
        "--variant",
        default="heavy_pole",
        choices=list(VARIANTS.keys()) + ["all"],
    )
    parser.add_argument(
        "--mode",
        default="off_on",
        choices=["off_on", "always_on", "both"],
        help="Plasticity mode: off_on (disabled Phase 1), always_on, or both",
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-per-stratum", type=int, default=10)
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

    network_ids = _load_network_ids(
        pilot=args.pilot,
        pilot_per_stratum=args.pilot_per_stratum,
        seed=args.seed,
        include_weak=args.include_weak,
        max_networks=args.max_networks,
    )
    print(f"Loaded {len(network_ids)} networks")

    modes = ["off_on", "always_on"] if args.mode == "both" else [args.mode]
    suffix = "_pilot" if args.pilot else ""
    output_dir = OUTPUT_ROOT / f"sweep{suffix}"

    variants = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    for variant in variants:
        run_sweep(
            network_ids,
            variant,
            switch_step=args.switch_step,
            rollouts=args.rollouts,
            n_workers=args.n_workers,
            output_dir=output_dir,
            modes=modes,
        )

    print(f"\nAll done! Results in: {output_dir}")


if __name__ == "__main__":
    main()
