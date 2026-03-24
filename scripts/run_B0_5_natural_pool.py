#!/usr/bin/env python3
"""
B0.5 Natural Pool Generator

Generate 5,000 networks with single random network per seed (no best-of-N selection).
Each network is evaluated and assigned to a stratum based on baseline performance.

Key differences from B0.4:
- POPULATION_SIZE = 1 (no selection)
- 5,000 seeds for statistical power
- 5 strata instead of binary perfect/imperfect
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

sys.path.append(os.path.abspath("code"))

from MorphoNAS.genome import Genome
from MorphoNAS.grid import Grid
from MorphoNAS.neural_propagation import NeuralPropagator
from MorphoNAS_PL.experimentB0_5_natural import (
    STRATUM_BOUNDS,
    get_stratum,
    run_rollouts,
)
from MorphoNAS_PL.logging_config import setup_logging

logger = logging.getLogger(__name__)

_worker_env: Optional[gym.Env] = None
_worker_eval_seeds: Optional[list[int]] = None
_worker_eval_mode: str = "per-seed"
_worker_eval_rollouts: int = 20
_worker_min_neurons: int = 6
_worker_min_edges: int = 5


def _create_propagator(G) -> NeuralPropagator:
    return NeuralPropagator(
        G=G,
        input_dim=4,
        output_dim=2,
        activation_function=NeuralPropagator.tanh_activation,
        extra_thinking_time=2,
        additive_update=False,
    )


def _init_worker(
    env_name: str,
    eval_seed_mode: str,
    eval_seed: int,
    n_rollouts: int,
    min_neurons: int,
    min_edges: int,
) -> None:
    global _worker_env, _worker_eval_seeds, _worker_eval_mode, _worker_eval_rollouts
    global _worker_min_neurons, _worker_min_edges
    from MorphoNAS_PL.parallel_utils import configure_worker_threads
    configure_worker_threads()
    _worker_env = gym.make(env_name)
    _worker_eval_mode = eval_seed_mode
    _worker_eval_rollouts = int(n_rollouts)
    if eval_seed_mode == "fixed":
        _worker_eval_seeds = list(range(eval_seed, eval_seed + n_rollouts))
    else:
        _worker_eval_seeds = None
    _worker_min_neurons = int(min_neurons)
    _worker_min_edges = int(min_edges)


def _eval_single_network_worker(args: Tuple[int, int, dict]) -> dict:
    """
    Evaluate a single random network for a seed.

    Args:
        args: (network_index, seed, genome_params)

    Returns:
        dict with network evaluation results
    """
    network_idx, seed, genome_params = args

    # Generate single random genome for this seed
    rng = np.random.default_rng(seed)
    genome = Genome.random(
        rng=rng,
        size_x=genome_params["size_x"],
        size_y=genome_params["size_y"],
        max_growth_steps=genome_params["max_growth_steps"],
        num_morphogens=genome_params["num_morphogens"],
    )

    # Grow the network
    grid = Grid(genome)
    grid.run_simulation(verbose=False)
    G = grid.get_graph()

    num_neurons = int(G.number_of_nodes())
    num_edges = int(G.number_of_edges())

    # Check minimum requirements
    if num_neurons < _worker_min_neurons or num_edges < _worker_min_edges:
        return {
            "network_id": network_idx,
            "seed": int(seed),
            "valid": False,
            "error": "insufficient_neurons_or_edges",
            "network_stats": {"neurons": num_neurons, "connections": num_edges},
        }

    # Create propagator and evaluate
    propagator = _create_propagator(G)
    assert _worker_env is not None

    if _worker_eval_mode == "fixed":
        assert _worker_eval_seeds is not None
        eval_seeds = _worker_eval_seeds
    else:
        eval_seeds = list(range(int(seed), int(seed) + _worker_eval_rollouts))

    results = run_rollouts(
        propagator,
        len(eval_seeds),
        seeds=eval_seeds,
        reset_plastic_each_episode=True,
        env=_worker_env,
    )

    mean_reward = float(results.get("avg_reward", 0.0))
    stratum = get_stratum(mean_reward)

    return {
        "network_id": network_idx,
        "seed": int(seed),
        "valid": True,
        "baseline_reward": mean_reward,
        "baseline_fitness": mean_reward / 500.0,
        "stratum": stratum.value,
        "genome": genome.to_dict(),
        "network_stats": {"neurons": num_neurons, "connections": num_edges},
        "rollout_data": {
            "rewards": results.get("rewards", []),
            "lengths": results.get("lengths", []),
            "eval_seeds": eval_seeds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B0.5 Natural Pool: single random network per seed"
    )
    parser.add_argument(
        "--target-valid",
        type=int,
        default=50000,
        help="Target number of VALID networks to generate (default: 50000)",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=500000,
        help="Maximum seeds to try before giving up (default: 500000)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=3000001,
        help="Starting seed for generation (default: 3000001)",
    )
    parser.add_argument(
        "--eval-seed-mode",
        type=str,
        default="per-seed",
        choices=["fixed", "per-seed"],
        help="Use fixed seeds or per-seed rollouts (default: per-seed)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Seed for fixed evaluation mode (default: 42)",
    )
    parser.add_argument(
        "--eval-rollouts",
        type=int,
        default=20,
        help="Number of rollouts for evaluation (default: 20)",
    )
    parser.add_argument(
        "--min-neurons",
        type=int,
        default=6,
        help="Minimum neurons required (default: 6)",
    )
    parser.add_argument(
        "--min-edges",
        type=int,
        default=5,
        help="Minimum edges required (default: 5)",
    )
    parser.add_argument("--size-x", type=int, default=10, help="Grid size X")
    parser.add_argument("--size-y", type=int, default=10, help="Grid size Y")
    parser.add_argument(
        "--max-growth-steps", type=int, default=200, help="Max growth steps"
    )
    parser.add_argument(
        "--num-morphogens", type=int, default=3, help="Number of morphogens"
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="CartPole-v1",
        help="Gym environment name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/B0.5/pool_natural",
        help="Output directory for pool",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, cpu_count() - 2),
        help=f"Parallel workers (default: {max(1, cpu_count() - 2)})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log progress every N networks (default: 100)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    networks_dir = os.path.join(args.output_dir, "networks")
    os.makedirs(networks_dir, exist_ok=True)

    setup_logging(log_dir=args.output_dir, log_file="pool_build.log")

    from MorphoNAS_PL.parallel_utils import configure_worker_threads
    configure_worker_threads()

    genome_params = {
        "size_x": args.size_x,
        "size_y": args.size_y,
        "max_growth_steps": args.max_growth_steps,
        "num_morphogens": args.num_morphogens,
    }

    logger.info("=" * 60)
    logger.info("B0.5 NATURAL POOL GENERATION")
    logger.info("=" * 60)
    logger.info(f"Target VALID networks: {args.target_valid}")
    logger.info(f"Max seeds to try: {args.max_seeds}")
    logger.info(f"Starting seed: {args.start_seed}")
    logger.info(f"Population size per seed: 1 (NO SELECTION)")
    logger.info(
        f"Eval: {args.eval_rollouts} rollouts, mode={args.eval_seed_mode}"
    )
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Workers: {args.max_workers}")
    logger.info("=" * 60)

    # Track results by stratum
    stratum_counts = {s.value: 0 for s in STRATUM_BOUNDS.keys()}
    valid_count = 0
    invalid_count = 0
    all_rewards = []
    task_index = 1

    with Pool(
        args.max_workers,
        initializer=_init_worker,
        initargs=(
            args.env_name,
            args.eval_seed_mode,
            args.eval_seed,
            args.eval_rollouts,
            args.min_neurons,
            args.min_edges,
        ),
    ) as pool:
        # Generate and evaluate in batches until we reach target
        seed = args.start_seed
        while valid_count < args.target_valid and seed < args.start_seed + args.max_seeds:
            # Generate a batch of tasks
            batch_size = min(1000, args.max_seeds - (seed - args.start_seed))
            batch_tasks = [
                (task_index + i, seed + i, genome_params)
                for i in range(batch_size)
            ]
            task_index += batch_size
            seed += batch_size

            # Process batch
            for i, result in enumerate(
                pool.imap_unordered(_eval_single_network_worker, batch_tasks, chunksize=8),
                start=valid_count + invalid_count + 1,
            ):
                if result.get("valid", False):
                    valid_count += 1
                    stratum = result["stratum"]
                    stratum_counts[stratum] += 1
                    all_rewards.append(result["baseline_reward"])

                    # Save network to file
                    network_id = result["network_id"]
                    out_path = os.path.join(networks_dir, f"network_{network_id:05d}.json")
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2)
                else:
                    invalid_count += 1

                if args.log_every > 0 and (valid_count + invalid_count) % args.log_every == 0:
                    logger.info(
                        f"Progress: {valid_count + invalid_count} attempts | "
                        f"Valid: {valid_count}/{args.target_valid} ({valid_count/(valid_count+invalid_count)*100:.1f}%) | "
                        f"Invalid: {invalid_count}"
                    )
                    logger.info(f"  Strata: {stratum_counts}")

                # Check if we've reached target
                if valid_count >= args.target_valid:
                    logger.info(f"Target of {args.target_valid} valid networks reached!")
                    break

    # Compute summary statistics
    if all_rewards:
        reward_array = np.array(all_rewards)
        summary_stats = {
            "mean": float(np.mean(reward_array)),
            "std": float(np.std(reward_array)),
            "median": float(np.median(reward_array)),
            "min": float(np.min(reward_array)),
            "max": float(np.max(reward_array)),
            "percentiles": {
                "10": float(np.percentile(reward_array, 10)),
                "25": float(np.percentile(reward_array, 25)),
                "50": float(np.percentile(reward_array, 50)),
                "75": float(np.percentile(reward_array, 75)),
                "90": float(np.percentile(reward_array, 90)),
            },
        }
    else:
        summary_stats = {}

    # Write metadata
    metadata = {
        "experiment": "B0.5",
        "description": "Natural pool - single random network per seed",
        "target_valid": args.target_valid,
        "max_seeds": args.max_seeds,
        "start_seed": args.start_seed,
        "seeds_tried": valid_count + invalid_count,
        "population_size": 1,
        "selection_method": "none",
        "eval_seed_mode": args.eval_seed_mode,
        "eval_rollouts": args.eval_rollouts,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "validity_rate": valid_count / (valid_count + invalid_count) if (valid_count + invalid_count) > 0 else 0,
        "stratum_counts": stratum_counts,
        "stratum_bounds": {k.value: list(v) for k, v in STRATUM_BOUNDS.items()},
        "reward_summary": summary_stats,
        "genome_params": genome_params,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(args.output_dir, "pool_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Write strata summary
    strata_summary = {
        "total_valid": valid_count,
        "total_invalid": invalid_count,
        "strata": {},
    }
    for stratum, bounds in STRATUM_BOUNDS.items():
        count = stratum_counts[stratum.value]
        strata_summary["strata"][stratum.value] = {
            "count": count,
            "percentage": (count / valid_count * 100) if valid_count > 0 else 0,
            "bounds": list(bounds),
        }

    with open(os.path.join(args.output_dir, "strata_summary.json"), "w") as f:
        json.dump(strata_summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("POOL GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Valid networks: {valid_count}")
    logger.info(f"Invalid networks: {invalid_count}")
    logger.info("Stratum distribution:")
    for stratum, bounds in STRATUM_BOUNDS.items():
        count = stratum_counts[stratum.value]
        pct = (count / valid_count * 100) if valid_count > 0 else 0
        logger.info(f"  {stratum.value}: {count} ({pct:.1f}%)")
    if all_rewards:
        logger.info(f"Reward stats: mean={summary_stats['mean']:.1f}, "
                   f"median={summary_stats['median']:.1f}, "
                   f"std={summary_stats['std']:.1f}")
    logger.info(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
