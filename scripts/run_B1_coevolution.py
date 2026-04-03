#!/usr/bin/env python3
"""
B1 Co-Evolution Runner

Co-evolves MorphoNAS developmental genomes with/without Hebbian plasticity.
Supports AWS-style partitioning: run subsets of independent GA runs per machine.

Example usage:
    # Run all 30 runs for Condition A on one machine
    python scripts/run_B1_coevolution.py --condition A --run-ids 0-29

    # Partition across 3 AWS instances
    python scripts/run_B1_coevolution.py --condition A --run-ids 0-9 --resume
    python scripts/run_B1_coevolution.py --condition A --run-ids 10-19 --resume
    python scripts/run_B1_coevolution.py --condition A --run-ids 20-29 --resume

    # Quick smoke test
    python scripts/run_B1_coevolution.py --condition A --run-ids 0 --pop-size 10 --max-gen 5
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import time
from logging.handlers import QueueListener
from pathlib import Path

sys.path.append(os.path.abspath("code"))

from MorphoNAS_PL.experimentB1_coevolution import (
    Condition,
    ENV_PRESETS,
    GAConfig,
    run_ga,
)
from MorphoNAS_PL.logging_config import setup_logging
from MorphoNAS_PL.parallel_utils import configure_worker_threads

logger = logging.getLogger(__name__)


def parse_run_ids(spec: str) -> list[int]:
    """Parse run ID specification.

    Supports:
        "0"         → [0]
        "0,1,5"     → [0, 1, 5]
        "0-9"       → [0, 1, ..., 9]
        "0-4,10-14" → [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    """
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def is_run_complete(run_dir: Path, max_gen: int) -> bool:
    """Check if a run has already completed all generations."""
    jsonl_path = run_dir / "generations.jsonl"
    if not jsonl_path.exists():
        return False
    with open(jsonl_path) as f:
        count = sum(1 for _ in f)
    # max_gen + 1 because we record generation 0 through max_gen inclusive
    return count >= max_gen + 1


def main():
    parser = argparse.ArgumentParser(
        description="B1 Co-Evolution: evolve MorphoNAS genomes with/without plasticity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["A", "B", "C"],
        help="Experiment condition: A=no plasticity, B=fixed plasticity, C=co-evolved",
    )
    parser.add_argument(
        "--run-ids",
        type=str,
        required=True,
        help='Run IDs to execute (e.g., "0-29", "0,1,5", "10-19")',
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=1000,
        help="Base seed; run i uses seed = base_seed + run_id (default: 1000)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="cartpole",
        choices=list(ENV_PRESETS.keys()),
        help="Environment preset (default: cartpole). Sets grid size, plasticity ranges, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory (default: experiments/B1 or experiments/B1_acrobot)",
    )
    parser.add_argument("--pop-size", type=int, default=50, help="Population size (default: 50)")
    parser.add_argument("--max-gen", type=int, default=200, help="Max generations (default: 200)")
    parser.add_argument(
        "--grid-size", type=int, default=None,
        help="Grid size (default: from env preset — 10 for cartpole, 20 for acrobot)",
    )
    parser.add_argument("--num-morphogens", type=int, default=3, help="Morphogens (default: 3)")
    parser.add_argument(
        "--max-growth-steps", type=int, default=200, help="Max growth steps (default: 200)"
    )
    parser.add_argument("--num-rollouts", type=int, default=20, help="Rollouts per eval (default: 20)")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel workers (default: cpu_count - 2)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed runs, resume partial runs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    condition = Condition(args.condition)
    run_ids = parse_run_ids(args.run_ids)
    env_config = ENV_PRESETS[args.env]

    # Resolve defaults from env preset
    grid_size = args.grid_size if args.grid_size is not None else env_config.default_grid_size
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "experiments/B1" if args.env == "cartpole" else f"experiments/B1_{args.env}"

    # Pin main process threads
    configure_worker_threads()

    # Logging
    base_dir = Path(output_dir) / f"condition_{condition.value}"
    base_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        log_dir=str(base_dir),
        log_file="b1_coevolution.log",
        console_level=getattr(logging, args.log_level),
    )

    # Log queue for workers
    log_queue = multiprocessing.Manager().Queue(-1)
    queue_listener = QueueListener(log_queue, *logging.getLogger().handlers, respect_handler_level=True)
    queue_listener.start()

    max_workers = args.max_workers
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 2)

    logger.info(
        f"B1 Co-Evolution | Condition {condition.value} | Env: {env_config.env_name} | "
        f"Runs: {run_ids} | Workers: {max_workers}"
    )
    logger.info(
        f"GA params: pop={args.pop_size} gen={args.max_gen} "
        f"grid={grid_size} morphogens={args.num_morphogens} "
        f"rollouts={args.num_rollouts}"
    )
    if condition == Condition.B:
        logger.info(f"Fixed plasticity: eta={env_config.fixed_eta}, decay={env_config.fixed_decay}")
    elif condition == Condition.C:
        logger.info(f"Eta range: {env_config.eta_range}, mutation std: {env_config.eta_mutation_std}")

    total_start = time.time()
    completed = 0
    skipped = 0

    try:
        for run_id in run_ids:
            run_dir = base_dir / f"run_{run_id:03d}"
            run_seed = args.base_seed + run_id

            # Check if already complete
            if args.resume and is_run_complete(run_dir, args.max_gen):
                logger.info(f"Run {run_id:03d}: already complete, skipping")
                skipped += 1
                continue

            logger.info(f"=== Run {run_id:03d} (seed={run_seed}) ===")

            config = GAConfig(
                condition=condition,
                run_seed=run_seed,
                env_config=env_config,
                pop_size=args.pop_size,
                max_gen=args.max_gen,
                grid_size=grid_size,
                num_morphogens=args.num_morphogens,
                max_growth_steps=args.max_growth_steps,
                num_rollouts=args.num_rollouts,
                max_workers=max_workers,
            )

            records = run_ga(
                config,
                run_dir,
                resume=args.resume,
                log_queue=log_queue,
            )

            completed += 1
            elapsed = time.time() - total_start
            logger.info(
                f"Run {run_id:03d} done ({len(records)} generations). "
                f"Total: {completed} completed, {skipped} skipped, "
                f"elapsed: {elapsed / 3600:.1f}h"
            )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Partial results saved.")
    finally:
        queue_listener.stop()

    total_elapsed = time.time() - total_start
    logger.info(
        f"B1 complete: {completed} runs finished, {skipped} skipped "
        f"in {total_elapsed / 3600:.1f}h"
    )


if __name__ == "__main__":
    main()
