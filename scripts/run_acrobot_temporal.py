#!/usr/bin/env python3
"""
Acrobot Temporal Trace Runner

Generates per-timestep traces (|Δw|, observations, actions, phase) for
selected Acrobot networks under non-stationary conditions.

Usage:
  # Small test (5 networks per stratum)
  python scripts/run_acrobot_temporal.py --per-stratum 5

  # Full run (50 per stratum = 200 networks)
  python scripts/run_acrobot_temporal.py

  # Custom output
  python scripts/run_acrobot_temporal.py --output-dir experiments/acrobot/temporal_profile
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

from MorphoNAS_PL.experiment_acrobot_temporal import (
    evaluate_network_temporal,
    set_shutdown_event,
)
from MorphoNAS_PL.experiment_acrobot import load_network_from_file
from MorphoNAS_PL.experiment_acrobot_nonstationary import DEFAULT_SWITCH_STEP

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────
POOL_ROOT = Path("experiments/acrobot/pool")
POOL_NETWORKS = POOL_ROOT / "networks"
NS_SWEEP_PATH = Path(
    "experiments/acrobot/sweep_nonstationary/sweep/acrobot_ns_heavy_link2_2x.parquet"
)
DEFAULT_OUTPUT_DIR = Path("experiments/acrobot/temporal_profile")

VARIANT = "heavy_link2_2x"
NON_WEAK_STRATA = ["low_mid", "high_mid", "near_perfect", "perfect"]


def _find_best_hyperparams(sweep_path: Path) -> dict[str, dict]:
    """Find the best (eta, decay) per stratum from the NS sweep.

    Returns dict mapping stratum -> {"eta": float, "decay": float}.
    Only considers non-zero eta values.
    """
    df = pd.read_parquet(sweep_path)
    best: dict[str, dict] = {}

    for stratum in NON_WEAK_STRATA:
        sdf = df[(df["stratum"] == stratum) & (df["eta"].abs() >= 1e-12)]
        if len(sdf) == 0:
            logger.warning("No plastic results for stratum %s", stratum)
            continue

        # Group by (eta, decay), take mean delta_reward_phase2
        agg = (
            sdf.groupby(["eta", "decay"])["delta_reward_phase2"]
            .mean()
            .reset_index()
        )
        best_row = agg.loc[agg["delta_reward_phase2"].idxmax()]
        best[stratum] = {
            "eta": float(best_row["eta"]),
            "decay": float(best_row["decay"]),
        }
        logger.info(
            "  %s: best eta=%.3f, decay=%.3f (mean delta_p2=%.2f)",
            stratum,
            best[stratum]["eta"],
            best[stratum]["decay"],
            float(best_row["delta_reward_phase2"]),
        )

    return best


def _select_networks(
    per_stratum: int, seed: int,
) -> dict[str, list[int]]:
    """Select networks per stratum from the Acrobot pool."""
    by_stratum: dict[str, list[int]] = {}
    for f in os.listdir(POOL_NETWORKS):
        if not f.endswith(".json"):
            continue
        fpath = POOL_NETWORKS / f
        with open(fpath) as fh:
            data = json.load(fh)
        stratum = data.get("stratum", "weak")
        if stratum in NON_WEAK_STRATA:
            nid = data["network_id"]
            by_stratum.setdefault(stratum, []).append(nid)

    rng = np.random.default_rng(seed)
    selected: dict[str, list[int]] = {}
    for stratum in NON_WEAK_STRATA:
        ids = sorted(by_stratum.get(stratum, []))
        n = min(per_stratum, len(ids))
        if n == 0:
            logger.warning("No networks in stratum %s", stratum)
            continue
        chosen = rng.choice(ids, size=n, replace=False).tolist()
        selected[stratum] = sorted(int(x) for x in chosen)
        logger.info("  Selected %d/%d networks from %s", n, len(ids), stratum)

    return selected


def _worker_init(shutdown_event):
    set_shutdown_event(shutdown_event)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from MorphoNAS_PL.parallel_utils import configure_worker_threads
    configure_worker_threads()


def _evaluate_single(args: tuple) -> dict | None:
    network_path, eta, decay, variant, switch_step, rollouts = args
    try:
        genome, metadata = load_network_from_file(str(network_path))
        result = evaluate_network_temporal(
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
        logger.error("Error evaluating %s: %s", network_path, e)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Acrobot Temporal Trace Runner"
    )
    parser.add_argument("--per-stratum", type=int, default=50)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--switch-step", type=int, default=DEFAULT_SWITCH_STEP,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    n_workers = args.n_workers or max(1, os.cpu_count() - 2)

    # Step 1: Find best hyperparams per stratum
    print(f"\n{'='*60}")
    print("ACROBOT TEMPORAL TRACE PROFILING")
    print(f"{'='*60}")
    print("\nStep 1: Finding best (eta, decay) per stratum...")
    best_hp = _find_best_hyperparams(NS_SWEEP_PATH)

    # Step 2: Select networks
    print(f"\nStep 2: Selecting {args.per_stratum} networks per stratum...")
    selected = _select_networks(args.per_stratum, args.seed)

    total_networks = sum(len(ids) for ids in selected.values())
    print(f"  Total: {total_networks} networks across {len(selected)} strata")

    # Step 3: Build work items
    work_items: list[tuple] = []
    for stratum, ids in selected.items():
        hp = best_hp.get(stratum)
        if hp is None:
            logger.warning("No hyperparams for stratum %s, skipping", stratum)
            continue
        for nid in ids:
            fpath = POOL_NETWORKS / f"network_{nid:05d}.json"
            if not fpath.exists():
                logger.warning("Network file not found: %s", fpath)
                continue
            work_items.append((
                fpath, hp["eta"], hp["decay"],
                VARIANT, args.switch_step, args.rollouts,
            ))

    print(f"\nStep 3: Running {len(work_items)} network evaluations "
          f"({args.rollouts} episodes each)...")
    print(f"  Workers: {n_workers}")

    # Step 4: Run evaluations
    shutdown_event = mp.Event()
    results: list[dict] = []
    t0 = time.time()

    try:
        with mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(shutdown_event,),
        ) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_evaluate_single, work_items, chunksize=1)
            ):
                if result is None:
                    continue
                if "error" in result:
                    continue
                results.append(result)

                if (i + 1) % 10 == 0 or (i + 1) == len(work_items):
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    remaining = (len(work_items) - i - 1) / rate if rate > 0 else 0
                    print(
                        f"  [{i+1}/{len(work_items)}] "
                        f"{elapsed:.0f}s elapsed, "
                        f"~{remaining:.0f}s remaining"
                    )

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        shutdown_event.set()

    elapsed = time.time() - t0
    print(f"\nCompleted {len(results)} network evaluations in {elapsed:.1f}s")

    # Step 5: Pack into NPZ + metadata
    print("\nStep 4: Saving traces...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz_arrays: dict[str, np.ndarray] = {}
    metadata_list: list[dict] = []

    for result in results:
        meta = result["metadata"]
        stratum = meta["stratum"]
        nid = meta["network_id"]
        metadata_list.append(meta)

        for ep_idx, trace in enumerate(result["traces"]):
            prefix = f"{stratum}_{nid}_{ep_idx}"
            npz_arrays[f"{prefix}_dw"] = trace["dw"]
            npz_arrays[f"{prefix}_obs"] = trace["obs"]
            npz_arrays[f"{prefix}_action"] = trace["action"]
            npz_arrays[f"{prefix}_phase"] = trace["phase"]

    npz_path = args.output_dir / f"traces_{VARIANT}.npz"
    np.savez_compressed(npz_path, **npz_arrays)
    print(f"  Saved {len(npz_arrays)} arrays to {npz_path}")

    meta_doc = {
        "experiment": "Acrobot Non-Stationary Temporal Profiling",
        "variant": VARIANT,
        "variant_params": best_hp,
        "per_stratum": args.per_stratum,
        "rollouts": args.rollouts,
        "switch_step": args.switch_step,
        "seed": args.seed,
        "n_networks": len(results),
        "elapsed_seconds": round(elapsed, 1),
        "networks": metadata_list,
    }
    meta_path = args.output_dir / "metadata_acrobot.json"
    with open(meta_path, "w") as f:
        json.dump(meta_doc, f, indent=2)
    print(f"  Saved metadata to {meta_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for stratum in NON_WEAK_STRATA:
        s_results = [r for r in results if r["metadata"]["stratum"] == stratum]
        if not s_results:
            continue
        p1_dws = [r["metadata"]["p1_mean_dw"] for r in s_results]
        p2_dws = [r["metadata"]["p2_mean_dw"] for r in s_results]
        ratios = [r["metadata"]["p2_p1_ratio"] for r in s_results]
        print(f"  {stratum} ({len(s_results)} nets):")
        print(f"    P1 mean |Δw|: {np.mean(p1_dws):.6f}")
        print(f"    P2 mean |Δw|: {np.mean(p2_dws):.6f}")
        print(f"    P2/P1 ratio:  {np.mean(ratios):.2f}")

    print(f"\nAll done! Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
