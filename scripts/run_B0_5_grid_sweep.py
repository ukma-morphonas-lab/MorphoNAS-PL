#!/usr/bin/env python3
"""
B0.5 Grid Sweep

Run full eta × decay grid (75 combinations) on all networks in the natural pool.
For each (network, eta, decay) combination:
- Baseline: 20 episodes, plasticity OFF
- Plastic: 20 episodes, plasticity ON, same seeds
- Reset plastic weights each episode
- Track: delta_fitness, delta_reward, mean|Δw| per step
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

sys.path.append(os.path.abspath("code"))

from MorphoNAS.genome import Genome
from MorphoNAS_PL.experimentB0_5_natural import (
    DECAY_VALUES,
    ETA_VALUES,
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
) -> list[str]:
    """Discover networks in the pool directory."""
    networks_dir = os.path.join(pool_dir, "networks")
    if not os.path.exists(networks_dir):
        logger.error(f"Networks directory not found: {networks_dir}")
        return []

    files = sorted([f for f in os.listdir(networks_dir) if f.endswith(".json")])

    # Apply stratum filter if specified
    if stratum_filter:
        filtered = []
        for f in files:
            path = os.path.join(networks_dir, f)
            try:
                with open(path, "r") as fp:
                    data = json.load(fp)
                if data.get("stratum") == stratum_filter:
                    filtered.append(f)
            except Exception:
                continue
        files = filtered
        logger.info(f"Filtered to {len(files)} networks in stratum '{stratum_filter}'")

    # Apply start index and max networks
    files = files[start_index:]
    if max_networks is not None:
        files = files[:max_networks]

    return [os.path.join(networks_dir, f) for f in files]


def _iter_tasks(
    network_paths: Iterable[str],
    eta_values: Iterable[float],
    decay_values: Iterable[float],
) -> Iterable[tuple[str, float, float]]:
    """Generate all (network_path, eta, decay) combinations."""
    for path in network_paths:
        for eta in eta_values:
            for decay in decay_values:
                yield path, eta, decay


def _worker_task(
    *,
    network_path: str,
    eta: float,
    decay: float,
    out_path: str,
    episodes: int,
) -> Optional[dict]:
    """Run a single (network, eta, decay) evaluation."""
    genome, metadata = load_network_from_file(network_path)
    network_id = metadata.get("network_id", os.path.basename(network_path))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    result = run_grid_evaluation(
        genome=genome,
        metadata=metadata,
        eta=float(eta),
        decay=float(decay),
        rollouts=int(episodes),
        reset_plastic_each_episode=True,
    )

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    summary = result.get("summary", {})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B0.5 Grid Sweep: eta × decay on natural pool"
    )
    parser.add_argument(
        "--pool-dir",
        type=str,
        default="experiments/B0.5/pool_natural",
        help="Directory containing network pool",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/B0.5/sweep",
        help="Directory to store sweep outputs",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes per evaluation (default: 20)",
    )
    parser.add_argument(
        "--max-networks",
        type=int,
        default=None,
        help="Maximum networks to process (default: all)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index into network list (default: 0)",
    )
    parser.add_argument(
        "--stratum-filter",
        type=str,
        default=None,
        choices=["weak", "low_mid", "high_mid", "near_perfect", "perfect"],
        help="Filter to specific stratum (default: all)",
    )
    parser.add_argument(
        "--eta-csv",
        type=str,
        default=None,
        help="Comma-separated eta values (overrides default grid)",
    )
    parser.add_argument(
        "--decay-csv",
        type=str,
        default=None,
        help="Comma-separated decay values (overrides default grid)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel workers (default: cores - 2)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing outputs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (default: INFO)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5000,
        help="Log progress every N tasks (default: 5000)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(
        console_level=getattr(logging, args.log_level),
        file_level=logging.DEBUG,
        log_dir=args.output_dir,
        log_file="sweep.log",
    )

    # Parse eta and decay values
    if args.eta_csv:
        eta_values = [float(x.strip()) for x in args.eta_csv.split(",")]
    else:
        eta_values = ETA_VALUES

    if args.decay_csv:
        decay_values = [float(x.strip()) for x in args.decay_csv.split(",")]
    else:
        decay_values = DECAY_VALUES

    logger.info("=" * 60)
    logger.info("B0.5 GRID SWEEP")
    logger.info("=" * 60)
    logger.info(f"Pool: {args.pool_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Eta values: {len(eta_values)} ({eta_values[0]} to {eta_values[-1]})")
    logger.info(f"Decay values: {len(decay_values)} ({decay_values})")
    logger.info(f"Grid size: {len(eta_values) * len(decay_values)} combinations")
    logger.info("=" * 60)

    # Discover networks
    network_paths = _discover_networks(
        args.pool_dir,
        max_networks=args.max_networks,
        start_index=args.start_index,
        stratum_filter=args.stratum_filter,
    )
    if not network_paths:
        logger.error("No networks found to process.")
        return

    logger.info(f"Networks to process: {len(network_paths)}")
    total_evaluations = len(network_paths) * len(eta_values) * len(decay_values)
    logger.info(f"Total evaluations: {total_evaluations}")

    # Build task list
    tasks: list[dict] = []
    for path, eta, decay in _iter_tasks(network_paths, eta_values, decay_values):
        # Extract network ID from path
        basename = os.path.basename(path).replace(".json", "")
        grid_label = format_grid_point(eta, decay)
        out_dir = os.path.join(args.output_dir, grid_label)
        out_path = os.path.join(out_dir, f"{basename}.json")

        if args.resume and os.path.exists(out_path):
            continue

        tasks.append({
            "network_path": path,
            "eta": eta,
            "decay": decay,
            "out_path": out_path,
            "episodes": int(args.episodes),
        })

    total_tasks = len(tasks)
    if total_tasks == 0:
        logger.warning("No tasks to run (all outputs present or no tasks generated).")
        return

    logger.info(f"Tasks to run: {total_tasks}")
    if args.resume:
        logger.info(f"Skipped (existing): {total_evaluations - total_tasks}")

    max_workers = args.max_workers
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = max(1, cpu_count - 2)

    logger.info(f"Workers: {max_workers}")

    manager = multiprocessing.Manager()
    log_queue = manager.Queue()

    root_logger = logging.getLogger()
    listener = QueueListener(
        log_queue, *root_logger.handlers, respect_handler_level=True
    )
    listener.start()

    completed = 0
    improved_count = 0
    total_delta = 0.0
    last_log_count = 0
    start_time = time.time()

    try:
        with ProcessPoolExecutor(
            max_workers=int(max_workers),
            initializer=_worker_init,
            initargs=(log_queue,),
        ) as executor:
            futures = {executor.submit(_worker_task, **task): i for i, task in enumerate(tasks)}
            start_msg = f"Starting {total_tasks} evaluations with {max_workers} workers..."
            logger.info(start_msg)
            print(start_msg, flush=True)
            sys.stdout.flush()

            for future in as_completed(futures):
                try:
                    summary = future.result()
                    if summary:
                        if summary.get("improved", False):
                            improved_count += 1
                        total_delta += summary.get("delta_fitness", 0.0)
                    completed += 1

                    # Log progress at fixed intervals
                    if args.log_every > 0 and (completed - last_log_count) >= args.log_every:
                        elapsed = time.time() - start_time
                        pct = completed / total_tasks * 100
                        impr_rate = improved_count / completed * 100 if completed > 0 else 0
                        mean_delta = total_delta / completed if completed > 0 else 0

                        # Calculate ETA
                        if completed > 0:
                            rate = completed / elapsed  # tasks per second
                            remaining = total_tasks - completed
                            eta_seconds = remaining / rate if rate > 0 else 0
                            eta_hours = eta_seconds / 3600
                        else:
                            eta_hours = 0

                        progress_msg = (
                            f"[{time.strftime('%H:%M:%S')}] Progress: {completed}/{total_tasks} ({pct:.1f}%) | "
                            f"Improved: {improved_count}/{completed} ({impr_rate:.1f}%) | "
                            f"Mean Δ: {mean_delta:+.4f} | ETA: {eta_hours:.1f}h"
                        )
                        logger.info(progress_msg)
                        print(progress_msg, flush=True)
                        sys.stdout.flush()
                        last_log_count = completed
                except Exception as exc:
                    logger.error(f"Task failed: {exc}")
                    print(f"[{time.strftime('%H:%M:%S')}] ERROR: Task failed: {exc}", flush=True)
                    completed += 1
    finally:
        listener.stop()

    # Write sweep summary
    sweep_summary = {
        "pool_dir": args.pool_dir,
        "output_dir": args.output_dir,
        "eta_values": eta_values,
        "decay_values": decay_values,
        "n_networks": len(network_paths),
        "n_grid_points": len(eta_values) * len(decay_values),
        "total_evaluations": total_tasks,
        "completed": completed,
        "improved_count": improved_count,
        "improvement_rate": improved_count / completed if completed > 0 else 0,
        "mean_delta_fitness": total_delta / completed if completed > 0 else 0,
        "stratum_filter": args.stratum_filter,
    }

    with open(os.path.join(args.output_dir, "sweep_summary.json"), "w") as f:
        json.dump(sweep_summary, f, indent=2)

    total_elapsed = time.time() - start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("=" * 60)
    logger.info("SWEEP COMPLETE")
    logger.info("=" * 60)
    if completed > 0:
        logger.info(f"Completed: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")
        logger.info(f"Improved: {improved_count}/{completed} ({improved_count/completed*100:.1f}%)")
        logger.info(f"Mean Δfitness: {total_delta/completed:+.4f}")
        rate = completed / total_elapsed if total_elapsed > 0 else 0
        logger.info(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s ({rate:.2f} tasks/sec)")
    else:
        logger.info("No results")
    logger.info(f"Output: {args.output_dir}")

    if completed > 0:
        rate = completed / total_elapsed if total_elapsed > 0 else 0
        summary_msg = (
            f"COMPLETE: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%) | "
            f"Time: {int(hours)}h {int(minutes)}m | Rate: {rate:.2f} tasks/s"
        )
        print(summary_msg, flush=True)


if __name__ == "__main__":
    main()
