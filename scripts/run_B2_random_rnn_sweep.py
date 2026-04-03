#!/usr/bin/env python3
"""
B2 Random RNN Plasticity Sweep

Applies the same η×λ grid as B0.5 to competent random RNNs from the B2 pool.

Example usage:
    # Full sweep
    python scripts/run_B2_random_rnn_sweep.py

    # Partition across AWS instances
    python scripts/run_B2_random_rnn_sweep.py --start-index 0 --max-networks 200 --resume
    python scripts/run_B2_random_rnn_sweep.py --start-index 200 --max-networks 200 --resume

    # Custom eta grid
    python scripts/run_B2_random_rnn_sweep.py --eta-csv "-0.05,-0.01,0,0.01,0.05"
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

import numpy as np

from MorphoNAS_PL.experimentB0_5_natural import (
    ETA_VALUES,
    DECAY_VALUES,
    VERIFICATION_SEEDS,
    get_stratum,
)
from MorphoNAS_PL.experimentB2_random_rnn import (
    evaluate_random_rnn,
    graph_from_dict,
)
from MorphoNAS_PL.logging_config import setup_logging
from MorphoNAS_PL.parallel_utils import configure_worker_threads

logger = logging.getLogger(__name__)


def _worker_init(queue) -> None:
    configure_worker_threads()
    setup_logging(queue=queue)


def _worker_sweep_single(
    graph_dict: dict,
    source_network_id,
    random_seed: int,
    eta: float,
    decay: float,
    baseline_reward: float,
    num_rollouts: int,
    seeds: list[int],
) -> dict:
    """Worker: evaluate one (random_rnn, eta, decay) combination."""
    G = graph_from_dict(graph_dict)

    result = evaluate_random_rnn(
        G, eta=eta, decay=decay, num_rollouts=num_rollouts, seeds=seeds,
    )

    plastic_reward = result.get("avg_reward", 0.0)
    delta_reward = plastic_reward - baseline_reward

    return {
        "source_network_id": source_network_id,
        "random_seed": random_seed,
        "num_neurons": graph_dict["num_nodes"],
        "num_connections": graph_dict["num_edges"],
        "eta": eta,
        "decay": decay,
        "baseline_reward": baseline_reward,
        "plastic_reward": plastic_reward,
        "delta_reward": delta_reward,
        "improved": delta_reward > 0,
        "plastic_fitness": result.get("fitness", 0.0),
        "mean_abs_delta_per_step": result.get("plasticity", {}).get("mean_abs_delta_per_step", 0.0),
    }


def load_competent_pool(pool_path: Path, min_stratum: str = "low_mid") -> list[dict]:
    """Load competent random RNNs from the B2 pool JSONL."""
    from MorphoNAS_PL.experimentB0_5_natural import Stratum

    stratum_order = [s.value for s in Stratum]
    min_idx = stratum_order.index(min_stratum)

    records = []
    with open(pool_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if not rec.get("valid", False):
                    continue
                stratum = rec.get("stratum", "weak")
                if stratum_order.index(stratum) < min_idx:
                    continue
                records.append(rec)
            except Exception:
                continue

    logger.info(f"Loaded {len(records)} competent random RNNs from {pool_path}")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="B2: Plasticity sweep on random RNN pool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pool-path",
        type=str,
        default="experiments/B2/pool/random_rnn_pool.jsonl",
        help="Path to B2 pool JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/B2/sweep",
        help="Output directory",
    )
    parser.add_argument(
        "--eta-csv",
        type=str,
        default=None,
        help="Custom eta values (comma-separated). Default: B0.5 grid.",
    )
    parser.add_argument(
        "--decay-csv",
        type=str,
        default=None,
        help="Custom decay values (comma-separated). Default: B0.5 grid.",
    )
    parser.add_argument("--num-rollouts", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0, help="Start index into network list")
    parser.add_argument("--max-networks", type=int, default=None, help="Max networks to process")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-every", type=int, default=5000)

    args = parser.parse_args()

    configure_worker_threads()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        log_dir=str(output_dir),
        log_file="b2_sweep.log",
        console_level=getattr(logging, args.log_level),
    )

    # Parse grids
    eta_values = ETA_VALUES
    decay_values = DECAY_VALUES
    if args.eta_csv:
        eta_values = [float(x) for x in args.eta_csv.split(",")]
    if args.decay_csv:
        decay_values = [float(x) for x in args.decay_csv.split(",")]

    logger.info(f"Grid: {len(eta_values)} eta × {len(decay_values)} decay = {len(eta_values)*len(decay_values)} points")

    # Load competent pool (auto-extract from tar.gz if needed)
    pool_path = Path(args.pool_path)
    if not pool_path.exists():
        tgz = pool_path.with_suffix(pool_path.suffix + ".tar.gz")
        if tgz.exists():
            import tarfile
            logger.info(f"Extracting {tgz} ...")
            with tarfile.open(tgz, "r:gz") as tf:
                tf.extract(pool_path.name, path=str(pool_path.parent))
        else:
            logger.error(f"Pool file not found: {pool_path}")
            sys.exit(1)

    pool = load_competent_pool(pool_path)
    pool = pool[args.start_index:]
    if args.max_networks is not None:
        pool = pool[:args.max_networks]

    if not pool:
        logger.error("No competent networks to process.")
        sys.exit(1)

    # Load existing results for resume
    results_path = output_dir / "random_rnn_sweep.jsonl"
    completed_keys = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = (rec["source_network_id"], rec["random_seed"], rec["eta"], rec["decay"])
                    completed_keys.add(key)
                except Exception:
                    continue
        logger.info(f"Resume: {len(completed_keys)} already completed")

    # Build task list
    seeds = VERIFICATION_SEEDS[:args.num_rollouts]
    tasks = []

    for rec in pool:
        graph_dict = rec["graph"]
        source_id = rec["source_network_id"]
        random_seed = rec["random_seed"]
        baseline_reward = rec["baseline_reward"]

        for eta in eta_values:
            for decay in decay_values:
                key = (source_id, random_seed, eta, decay)
                if key in completed_keys:
                    continue
                tasks.append({
                    "graph_dict": graph_dict,
                    "source_network_id": source_id,
                    "random_seed": random_seed,
                    "eta": eta,
                    "decay": decay,
                    "baseline_reward": baseline_reward,
                    "num_rollouts": args.num_rollouts,
                    "seeds": seeds,
                })

    if not tasks:
        logger.info("All tasks already completed.")
        return

    logger.info(f"{len(tasks)} sweep tasks to process ({len(pool)} networks × {len(eta_values)*len(decay_values)} grid)")

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
                        _worker_sweep_single,
                        task["graph_dict"],
                        task["source_network_id"],
                        task["random_seed"],
                        task["eta"],
                        task["decay"],
                        task["baseline_reward"],
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
                        logger.warning(f"Task failed: {e}")
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
    logger.info(f"Sweep complete: {completed} evaluations in {elapsed/3600:.1f}h")


if __name__ == "__main__":
    main()
