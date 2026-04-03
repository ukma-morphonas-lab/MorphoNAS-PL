#!/usr/bin/env python3
"""
Acrobot Static Plasticity Sweep

Run the 22-point eta x decay grid (11 eta x 2 decay) on all non-Weak
networks from the Acrobot pool. Saves results as parquet.

Usage:
  # Full sweep on all non-Weak networks
  python scripts/run_acrobot_sweep.py

  # Pilot on 50 networks
  python scripts/run_acrobot_sweep.py --max-networks 50

  # Resume interrupted sweep
  python scripts/run_acrobot_sweep.py --resume

  # Custom eta grid
  python scripts/run_acrobot_sweep.py --eta-csv "-0.10,-0.05,0.0,0.05,0.10"
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging.handlers import QueueListener
from typing import Iterable, Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("code"))

from MorphoNAS.genome import Genome
from MorphoNAS_PL.experiment_acrobot import (
    DECAY_VALUES,
    ETA_VALUES,
    Stratum,
    format_grid_point,
    load_network_from_file,
    run_grid_evaluation,
)
from MorphoNAS_PL.logging_config import setup_logging

logger = logging.getLogger(__name__)


def _worker_init(queue) -> None:
    from MorphoNAS_PL.parallel_utils import configure_worker_threads
    configure_worker_threads()
    setup_logging(queue=queue)


def _discover_networks(
    pool_dir: str,
    max_networks: Optional[int] = None,
    start_index: int = 0,
    stratum_filter: Optional[str] = None,
    exclude_weak: bool = True,
) -> list[str]:
    """Discover networks in the pool directory."""
    networks_dir = os.path.join(pool_dir, "networks")
    if not os.path.exists(networks_dir):
        logger.error(f"Networks directory not found: {networks_dir}")
        return []

    files = sorted([f for f in os.listdir(networks_dir) if f.endswith(".json")])

    # Filter by stratum
    if stratum_filter or exclude_weak:
        filtered = []
        for f in files:
            path = os.path.join(networks_dir, f)
            try:
                with open(path, "r") as fp:
                    data = json.load(fp)
                s = data.get("stratum", "weak")
                if stratum_filter and s != stratum_filter:
                    continue
                if exclude_weak and s == "weak":
                    continue
                filtered.append(f)
            except Exception:
                continue
        excluded = len(files) - len(filtered)
        files = filtered
        if excluded > 0:
            logger.info(f"Filtered: {len(files)} networks kept, {excluded} excluded")

    files = files[start_index:]
    if max_networks is not None:
        files = files[:max_networks]

    return [os.path.join(networks_dir, f) for f in files]


def _worker_task(
    *,
    network_path: str,
    eta: float,
    decay: float,
    episodes: int,
) -> Optional[dict]:
    """Run a single (network, eta, decay) evaluation on Acrobot."""
    genome, metadata = load_network_from_file(network_path)
    result = run_grid_evaluation(
        genome=genome,
        metadata=metadata,
        eta=float(eta),
        decay=float(decay),
        rollouts=int(episodes),
        reset_plastic_each_episode=True,
    )
    return result


def _flatten_result(result: dict) -> dict:
    """Flatten grid evaluation result into a row for parquet."""
    if "error" in result:
        return {
            "network_id": result.get("network_id"),
            "error": result.get("error"),
        }

    summary = result.get("summary", {})
    stats = result.get("network_stats", {})

    return {
        "network_id": result.get("network_id"),
        "stratum": result.get("stratum"),
        "eta": result.get("eta"),
        "decay": result.get("decay"),
        "num_neurons": stats.get("num_neurons"),
        "num_connections": stats.get("num_connections"),
        "baseline_reward": summary.get("baseline_reward"),
        "plastic_reward": summary.get("plastic_reward"),
        "delta_reward": summary.get("delta_reward"),
        "improved": summary.get("improved"),
        "mean_abs_delta_per_step": summary.get("mean_abs_delta_per_step", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acrobot Static Plasticity Sweep"
    )
    parser.add_argument(
        "--pool-dir", type=str, default="experiments/acrobot/pool",
        help="Pool directory (default: experiments/acrobot/pool)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/acrobot/sweep_static",
        help="Output directory (default: experiments/acrobot/sweep_static)",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-networks", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--stratum-filter", type=str, default=None,
        choices=["weak", "low_mid", "high_mid", "near_perfect", "perfect"],
    )
    parser.add_argument(
        "--include-weak", action="store_true",
        help="Include Weak-stratum networks (excluded by default)",
    )
    parser.add_argument("--eta-csv", type=str, default=None)
    parser.add_argument("--decay-csv", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-every", type=int, default=1000)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(
        console_level=getattr(logging, args.log_level),
        file_level=logging.DEBUG,
        log_dir=args.output_dir,
        log_file="sweep.log",
    )

    if args.eta_csv:
        eta_values = [float(x.strip()) for x in args.eta_csv.split(",")]
    else:
        eta_values = ETA_VALUES

    if args.decay_csv:
        decay_values = [float(x.strip()) for x in args.decay_csv.split(",")]
    else:
        decay_values = DECAY_VALUES

    logger.info("=" * 60)
    logger.info("ACROBOT STATIC PLASTICITY SWEEP")
    logger.info("=" * 60)
    logger.info(f"Pool: {args.pool_dir}")
    logger.info(f"Eta values: {len(eta_values)} ({eta_values})")
    logger.info(f"Decay values: {len(decay_values)} ({decay_values})")
    logger.info(f"Grid size: {len(eta_values) * len(decay_values)} combinations")
    logger.info("=" * 60)

    network_paths = _discover_networks(
        args.pool_dir,
        max_networks=args.max_networks,
        start_index=args.start_index,
        stratum_filter=args.stratum_filter,
        exclude_weak=not args.include_weak,
    )
    if not network_paths:
        logger.error("No networks found.")
        return

    logger.info(f"Networks: {len(network_paths)}")
    total_evaluations = len(network_paths) * len(eta_values) * len(decay_values)
    logger.info(f"Total evaluations: {total_evaluations:,}")

    # Build task list
    tasks: list[dict] = []
    for path in network_paths:
        for eta in eta_values:
            for decay in decay_values:
                tasks.append({
                    "network_path": path,
                    "eta": eta,
                    "decay": decay,
                    "episodes": int(args.episodes),
                })

    total_tasks = len(tasks)
    if total_tasks == 0:
        logger.warning("No tasks to run.")
        return

    max_workers = args.max_workers
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = max(1, cpu_count - 2)

    logger.info(f"Tasks: {total_tasks:,}, Workers: {max_workers}")

    manager = multiprocessing.Manager()
    log_queue = manager.Queue()

    root_logger = logging.getLogger()
    listener = QueueListener(
        log_queue, *root_logger.handlers, respect_handler_level=True
    )
    listener.start()

    completed = 0
    rows: list[dict] = []
    start_time = time.time()
    last_log_count = 0

    try:
        with ProcessPoolExecutor(
            max_workers=int(max_workers),
            initializer=_worker_init,
            initargs=(log_queue,),
        ) as executor:
            futures = {executor.submit(_worker_task, **task): i for i, task in enumerate(tasks)}
            logger.info(f"Starting {total_tasks:,} evaluations with {max_workers} workers...")

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result and "error" not in result:
                        rows.append(_flatten_result(result))
                    completed += 1

                    if args.log_every > 0 and (completed - last_log_count) >= args.log_every:
                        elapsed = time.time() - start_time
                        pct = completed / total_tasks * 100
                        rate = completed / elapsed if elapsed > 0 else 0
                        remaining = (total_tasks - completed) / rate if rate > 0 else 0

                        improved = sum(1 for r in rows if r.get("improved", False))
                        impr_rate = improved / len(rows) * 100 if rows else 0

                        msg = (
                            f"[{time.strftime('%H:%M:%S')}] {completed:,}/{total_tasks:,} "
                            f"({pct:.1f}%) | Improved: {impr_rate:.1f}% | "
                            f"ETA: {remaining/3600:.1f}h"
                        )
                        logger.info(msg)
                        print(msg, flush=True)
                        last_log_count = completed
                except Exception as exc:
                    logger.error(f"Task failed: {exc}")
                    completed += 1
    except KeyboardInterrupt:
        print("\nInterrupted! Saving partial results...")
    finally:
        listener.stop()

    if not rows:
        logger.warning("No results to save.")
        return

    df = pd.DataFrame(rows)

    # Save parquet
    parquet_path = os.path.join(args.output_dir, "acrobot_static_sweep.parquet")
    df.to_parquet(parquet_path, compression="zstd", index=False)
    logger.info(f"Saved: {parquet_path} ({len(df):,} rows)")

    # Save summary
    elapsed = time.time() - start_time
    improved_count = int(df["improved"].sum()) if "improved" in df.columns else 0
    mean_delta = float(df["delta_reward"].mean()) if "delta_reward" in df.columns else 0

    summary = {
        "pool_dir": args.pool_dir,
        "n_networks": len(network_paths),
        "n_grid_points": len(eta_values) * len(decay_values),
        "total_evaluations": completed,
        "total_rows": len(df),
        "improved_count": improved_count,
        "improvement_rate": improved_count / len(df) if len(df) > 0 else 0,
        "mean_delta_reward": mean_delta,
        "elapsed_seconds": round(elapsed, 1),
        "eta_values": eta_values,
        "decay_values": decay_values,
    }

    # Per-stratum summary
    if "stratum" in df.columns:
        for stratum in df["stratum"].unique():
            sdf = df[df["stratum"] == stratum]
            summary[f"stratum_{stratum}"] = {
                "n_evaluations": len(sdf),
                "n_networks": int(sdf["network_id"].nunique()),
                "improvement_rate": float(sdf["improved"].mean()) if "improved" in sdf.columns else 0,
                "mean_delta_reward": float(sdf["delta_reward"].mean()) if "delta_reward" in sdf.columns else 0,
            }

    with open(os.path.join(args.output_dir, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info("=" * 60)
    logger.info("ACROBOT STATIC SWEEP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Improved: {improved_count}/{len(df)} ({summary['improvement_rate']*100:.1f}%)")
    logger.info(f"Mean delta reward: {mean_delta:+.2f}")
    logger.info(f"Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
