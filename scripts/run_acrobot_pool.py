#!/usr/bin/env python3
"""
Acrobot Pool Generator

Generate 5,000 random-genome networks on 20x20 developmental grids,
evaluate each on Acrobot-v1 (20 episodes), and assign strata.

Usage:
  # Default: 5,000 valid networks, 20x20 grid
  python scripts/run_acrobot_pool.py

  # Quick pilot (100 networks)
  python scripts/run_acrobot_pool.py --target-valid 100

  # Custom grid size
  python scripts/run_acrobot_pool.py --size-x 15 --size-y 15
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
from MorphoNAS_PL.experiment_acrobot import (
    ENV_NAME,
    INPUT_DIM,
    MIN_NEURONS,
    OUTPUT_DIM,
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
_worker_min_neurons: int = MIN_NEURONS
_worker_min_edges: int = 5


def _create_propagator(G) -> NeuralPropagator:
    return NeuralPropagator(
        G=G,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
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
    """Evaluate a single random Acrobot network for a seed."""
    network_idx, seed, genome_params = args

    rng = np.random.default_rng(seed)
    genome = Genome.random(
        rng=rng,
        size_x=genome_params["size_x"],
        size_y=genome_params["size_y"],
        max_growth_steps=genome_params["max_growth_steps"],
        num_morphogens=genome_params["num_morphogens"],
    )

    grid = Grid(genome)
    grid.run_simulation(verbose=False)
    G = grid.get_graph()

    num_neurons = int(G.number_of_nodes())
    num_edges = int(G.number_of_edges())

    if num_neurons < _worker_min_neurons or num_edges < _worker_min_edges:
        return {
            "network_id": network_idx,
            "seed": int(seed),
            "valid": False,
            "error": "insufficient_neurons_or_edges",
            "network_stats": {"neurons": num_neurons, "connections": num_edges},
        }

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
        description="Acrobot Pool: single random network per seed"
    )
    parser.add_argument(
        "--target-valid", type=int, default=5000,
        help="Target number of VALID networks (default: 5000)",
    )
    parser.add_argument(
        "--max-seeds", type=int, default=100000,
        help="Maximum seeds to try (default: 100000)",
    )
    parser.add_argument(
        "--start-seed", type=int, default=5000001,
        help="Starting seed (default: 5000001, avoids CartPole pool collision)",
    )
    parser.add_argument(
        "--eval-seed-mode", type=str, default="per-seed",
        choices=["fixed", "per-seed"],
        help="Use fixed seeds or per-seed rollouts (default: per-seed)",
    )
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-rollouts", type=int, default=20)
    parser.add_argument(
        "--min-neurons", type=int, default=MIN_NEURONS,
        help=f"Minimum neurons required (default: {MIN_NEURONS})",
    )
    parser.add_argument("--min-edges", type=int, default=5)
    parser.add_argument("--size-x", type=int, default=20, help="Grid size X (default: 20)")
    parser.add_argument("--size-y", type=int, default=20, help="Grid size Y (default: 20)")
    parser.add_argument("--max-growth-steps", type=int, default=200)
    parser.add_argument("--num-morphogens", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=str, default="experiments/acrobot/pool",
        help="Output directory (default: experiments/acrobot/pool)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=max(1, cpu_count() - 2),
        help=f"Parallel workers (default: {max(1, cpu_count() - 2)})",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-every", type=int, default=100)

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
    logger.info("ACROBOT POOL GENERATION")
    logger.info("=" * 60)
    logger.info(f"Environment: {ENV_NAME}")
    logger.info(f"I/O dims: {INPUT_DIM} inputs, {OUTPUT_DIM} outputs")
    logger.info(f"Grid: {args.size_x}x{args.size_y}")
    logger.info(f"Target VALID networks: {args.target_valid}")
    logger.info(f"Starting seed: {args.start_seed}")
    logger.info(f"Eval: {args.eval_rollouts} rollouts, mode={args.eval_seed_mode}")
    logger.info(f"Workers: {args.max_workers}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)

    stratum_counts = {s.value: 0 for s in STRATUM_BOUNDS.keys()}
    valid_count = 0
    invalid_count = 0
    all_rewards = []
    task_index = 1

    with Pool(
        args.max_workers,
        initializer=_init_worker,
        initargs=(
            ENV_NAME,
            args.eval_seed_mode,
            args.eval_seed,
            args.eval_rollouts,
            args.min_neurons,
            args.min_edges,
        ),
    ) as pool:
        seed = args.start_seed
        while valid_count < args.target_valid and seed < args.start_seed + args.max_seeds:
            batch_size = min(1000, args.max_seeds - (seed - args.start_seed))
            batch_tasks = [
                (task_index + i, seed + i, genome_params)
                for i in range(batch_size)
            ]
            task_index += batch_size
            seed += batch_size

            for i, result in enumerate(
                pool.imap_unordered(_eval_single_network_worker, batch_tasks, chunksize=8),
                start=valid_count + invalid_count + 1,
            ):
                if result.get("valid", False):
                    valid_count += 1
                    stratum = result["stratum"]
                    stratum_counts[stratum] += 1
                    all_rewards.append(result["baseline_reward"])

                    network_id = result["network_id"]
                    out_path = os.path.join(networks_dir, f"network_{network_id:05d}.json")
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2)
                else:
                    invalid_count += 1

                if args.log_every > 0 and (valid_count + invalid_count) % args.log_every == 0:
                    logger.info(
                        f"Progress: {valid_count + invalid_count} attempts | "
                        f"Valid: {valid_count}/{args.target_valid} "
                        f"({valid_count/(valid_count+invalid_count)*100:.1f}%) | "
                        f"Invalid: {invalid_count}"
                    )
                    logger.info(f"  Strata: {stratum_counts}")

                if valid_count >= args.target_valid:
                    logger.info(f"Target of {args.target_valid} valid networks reached!")
                    break

    # Summary statistics
    if all_rewards:
        reward_array = np.array(all_rewards)
        summary_stats = {
            "mean": float(np.mean(reward_array)),
            "std": float(np.std(reward_array)),
            "median": float(np.median(reward_array)),
            "min": float(np.min(reward_array)),
            "max": float(np.max(reward_array)),
            "percentiles": {
                str(p): float(np.percentile(reward_array, p))
                for p in [5, 10, 25, 50, 75, 90, 95]
            },
        }
    else:
        summary_stats = {}

    metadata = {
        "experiment": "acrobot_pool",
        "env_name": ENV_NAME,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "target_valid": args.target_valid,
        "start_seed": args.start_seed,
        "seeds_tried": valid_count + invalid_count,
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

    # Write network IDs list for downstream scripts
    network_ids = sorted([
        int(f.replace("network_", "").replace(".json", ""))
        for f in os.listdir(networks_dir) if f.endswith(".json")
    ])
    with open(os.path.join(args.output_dir, "network_ids.txt"), "w") as f:
        for nid in network_ids:
            f.write(f"{nid}\n")

    logger.info("=" * 60)
    logger.info("ACROBOT POOL GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Valid networks: {valid_count}")
    logger.info(f"Invalid networks: {invalid_count}")
    logger.info(f"Validity rate: {metadata['validity_rate']*100:.1f}%")
    logger.info("Stratum distribution:")
    for stratum, bounds in STRATUM_BOUNDS.items():
        count = stratum_counts[stratum.value]
        pct = (count / valid_count * 100) if valid_count > 0 else 0
        logger.info(f"  {stratum.value}: {count} ({pct:.1f}%)")
    if all_rewards:
        logger.info(
            f"Reward stats: mean={summary_stats['mean']:.1f}, "
            f"median={summary_stats['median']:.1f}, "
            f"std={summary_stats['std']:.1f}, "
            f"range=[{summary_stats['min']:.0f}, {summary_stats['max']:.0f}]"
        )
    logger.info(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
