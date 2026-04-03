#!/usr/bin/env python3
"""
B2 Random RNN Pool Generator

Generates random RNNs matching topology stats of competent MorphoNAS networks
from the B0.5 pool. Evaluates baselines (frozen weights) on CartPole.

Example usage:
    # Generate full pool (5 random RNNs per source network)
    python scripts/run_B2_random_rnn_pool.py

    # Partition across AWS instances
    python scripts/run_B2_random_rnn_pool.py --start-index 0 --max-networks 1000 --resume
    python scripts/run_B2_random_rnn_pool.py --start-index 1000 --max-networks 1000 --resume

    # Quick test
    python scripts/run_B2_random_rnn_pool.py --max-networks 10 --num-random 2
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
from pathlib import Path

sys.path.append(os.path.abspath("code"))

from MorphoNAS_PL.experimentB0_5_natural import VERIFICATION_SEEDS, get_stratum
from MorphoNAS_PL.experimentB2_random_rnn import (
    generate_random_rnn,
    evaluate_random_rnn,
    graph_to_dict,
    load_morphonas_pool_stats,
)
from MorphoNAS_PL.logging_config import setup_logging
from MorphoNAS_PL.parallel_utils import configure_worker_threads

logger = logging.getLogger(__name__)


def _worker_init(queue) -> None:
    configure_worker_threads()
    setup_logging(queue=queue)


def _worker_generate_and_eval(
    source_id,
    num_neurons: int,
    num_connections: int,
    random_seed: int,
    weight_range: tuple[float, float],
    num_rollouts: int,
    seeds: list[int],
) -> dict:
    """Worker: generate one random RNN and evaluate its baseline."""
    rng = np.random.default_rng(random_seed)

    G = generate_random_rnn(
        num_nodes=num_neurons,
        num_edges=num_connections,
        rng=rng,
        weight_range=weight_range,
    )

    if G is None:
        return {
            "source_network_id": source_id,
            "random_seed": random_seed,
            "num_neurons": num_neurons,
            "num_connections": num_connections,
            "valid": False,
            "error": "generation_failed",
        }

    # Evaluate baseline (no plasticity)
    result = evaluate_random_rnn(G, eta=0.0, decay=0.0, num_rollouts=num_rollouts, seeds=seeds)

    baseline_reward = result.get("avg_reward", 0.0)

    return {
        "source_network_id": source_id,
        "random_seed": random_seed,
        "num_neurons": num_neurons,
        "num_connections": num_connections,
        "actual_num_nodes": G.number_of_nodes(),
        "actual_num_edges": G.number_of_edges(),
        "valid": True,
        "baseline_reward": baseline_reward,
        "baseline_fitness": result.get("fitness", 0.0),
        "stratum": get_stratum(baseline_reward).value,
        "rewards": result.get("rewards", []),
        "graph": graph_to_dict(G),
    }


# Need numpy in worker scope
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="B2: Generate random RNN pool matching MorphoNAS topology stats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--b05-pool-dir",
        type=str,
        default="experiments/B0.5/pool_natural",
        help="Path to B0.5 pool directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/B2/pool",
        help="Output directory (default: experiments/B2/pool)",
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=5,
        help="Random RNNs per source network (default: 5)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=7000001,
        help="Base seed for random generation (default: 7000001)",
    )
    parser.add_argument(
        "--weight-min",
        type=float,
        default=0.01,
        help="Min weight (default: 0.01, matching MorphoNAS w=max(0.01,c/(1+d)))",
    )
    parser.add_argument(
        "--weight-max",
        type=float,
        default=1.0,
        help="Max weight (default: 1.0)",
    )
    parser.add_argument("--num-rollouts", type=int, default=20, help="Rollouts per eval (default: 20)")
    parser.add_argument("--start-index", type=int, default=0, help="Start index into source network list")
    parser.add_argument("--max-networks", type=int, default=None, help="Max source networks to process")
    parser.add_argument("--max-workers", type=int, default=None, help="Parallel workers")
    parser.add_argument("--resume", action="store_true", help="Skip existing results")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-every", type=int, default=500, help="Log progress every N tasks")

    args = parser.parse_args()

    configure_worker_threads()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        log_dir=str(output_dir),
        log_file="b2_pool_generation.log",
        console_level=getattr(logging, args.log_level),
    )

    # Load source network topology stats
    pool_stats = load_morphonas_pool_stats(args.b05_pool_dir, min_stratum="low_mid")
    if not pool_stats:
        logger.error("No competent networks found in pool. Check --b05-pool-dir.")
        sys.exit(1)

    # Apply partitioning
    pool_stats = pool_stats[args.start_index:]
    if args.max_networks is not None:
        pool_stats = pool_stats[:args.max_networks]

    logger.info(f"Processing {len(pool_stats)} source networks × {args.num_random} random RNNs each")

    # Load existing results for resume
    completed_keys = set()
    results_path = output_dir / "random_rnn_pool.jsonl"
    if args.resume and results_path.exists():
        with open(results_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed_keys.add((rec["source_network_id"], rec["random_seed"]))
                except Exception:
                    continue
        logger.info(f"Resume: {len(completed_keys)} already completed")

    # Build task list
    seeds = VERIFICATION_SEEDS[:args.num_rollouts]
    weight_range = (args.weight_min, args.weight_max)
    tasks = []

    for stat in pool_stats:
        for k in range(args.num_random):
            random_seed = args.base_seed + stat["network_id"] * 100 + k
            key = (stat["network_id"], random_seed)
            if key in completed_keys:
                continue
            tasks.append({
                "source_id": stat["network_id"],
                "num_neurons": stat["num_neurons"],
                "num_connections": stat["num_connections"],
                "random_seed": random_seed,
                "weight_range": weight_range,
                "num_rollouts": args.num_rollouts,
                "seeds": seeds,
            })

    if not tasks:
        logger.info("All tasks already completed.")
        return

    logger.info(f"{len(tasks)} tasks to process ({len(completed_keys)} skipped)")

    # Set up parallel execution
    max_workers = args.max_workers
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 2)

    log_queue = multiprocessing.Manager().Queue(-1)
    queue_listener = QueueListener(log_queue, *logging.getLogger().handlers, respect_handler_level=True)
    queue_listener.start()

    completed = 0
    start_time = time.time()

    try:
        with open(results_path, "a") as out_f:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_worker_init,
                initargs=(log_queue,),
            ) as executor:
                futures = {}
                for task in tasks:
                    future = executor.submit(
                        _worker_generate_and_eval,
                        task["source_id"],
                        task["num_neurons"],
                        task["num_connections"],
                        task["random_seed"],
                        task["weight_range"],
                        task["num_rollouts"],
                        task["seeds"],
                    )
                    futures[future] = task

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        out_f.write(json.dumps(result) + "\n")
                        out_f.flush()
                    except Exception as e:
                        task = futures[future]
                        logger.warning(
                            f"Task failed (source={task['source_id']}, "
                            f"seed={task['random_seed']}): {e}"
                        )
                        continue

                    completed += 1
                    if args.log_every > 0 and completed % args.log_every == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        eta_h = (len(tasks) - completed) / rate / 3600 if rate > 0 else 0
                        logger.info(
                            f"Progress: {completed}/{len(tasks)} ({100*completed/len(tasks):.1f}%) "
                            f"| {rate:.1f}/s | ETA: {eta_h:.1f}h"
                        )

    except KeyboardInterrupt:
        logger.warning("Interrupted. Partial results saved.")
    finally:
        queue_listener.stop()

    elapsed = time.time() - start_time
    logger.info(f"Pool generation complete: {completed} RNNs in {elapsed/3600:.1f}h")

    # Summary
    if results_path.exists():
        total = 0
        valid = 0
        strata_counts = {}
        with open(results_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    total += 1
                    if rec.get("valid", False):
                        valid += 1
                        s = rec.get("stratum", "weak")
                        strata_counts[s] = strata_counts.get(s, 0) + 1
                except Exception:
                    continue

        summary = {
            "total": total,
            "valid": valid,
            "strata": strata_counts,
            "source_networks": len(pool_stats),
            "random_per_source": args.num_random,
            "weight_range": list(weight_range),
        }
        with open(output_dir / "pool_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary: {valid}/{total} valid. Strata: {strata_counts}")


if __name__ == "__main__":
    main()
