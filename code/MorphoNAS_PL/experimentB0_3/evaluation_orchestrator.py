"""
Evaluation orchestrator for B0.3 experiment.

Handles parallel evaluation of networks × plasticity configurations with
caching, early stopping, and progress tracking.
"""

import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib

import numpy as np

from MorphoNAS.genome import Genome
from MorphoNAS.grid import Grid
from MorphoNAS_PL.b1_fitness import B1FitnessFunction
from MorphoNAS_PL.plasticity_strategies import PlasticityGenome

from .config_schemas import (
    PlasticityConfig,
    NetworkSpec,
    EvaluationResult,
    PhaseResults,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationTask:
    """Single evaluation task."""
    network_spec: NetworkSpec
    plasticity_config: PlasticityConfig
    task_id: str


def evaluate_network_config(
    network_spec: NetworkSpec,
    plasticity_config: PlasticityConfig,
    n_rollouts: int,
    env_name: str,
) -> EvaluationResult:
    """
    Evaluate a single network with a plasticity configuration.

    This function is designed to run in a separate process.

    Args:
        network_spec: Network specification
        plasticity_config: Plasticity configuration
        n_rollouts: Number of rollout episodes
        env_name: Gym environment name

    Returns:
        EvaluationResult
    """
    start_time = time.time()

    # Reconstruct network
    genome = Genome.from_dict(network_spec.genome_dict)
    grid = Grid(genome)
    grid.run_simulation(verbose=False)
    G = grid.get_graph()

    # Create fitness function
    fitness_config = {
        "env_name": env_name,
        "passing_reward": 475.0,
        "seed": 42,
    }
    fitness_fn = B1FitnessFunction(
        targets=fitness_config,
        n_rollouts=n_rollouts,
    )

    # Evaluate frozen baseline
    _, frozen_rollout_data = fitness_fn.evaluate_baseline(grid, G=G)
    frozen_rewards = frozen_rollout_data.get("rewards", [])
    frozen_mean = float(np.mean(frozen_rewards)) if frozen_rewards else 0.0
    frozen_std = float(np.std(frozen_rewards)) if frozen_rewards else 0.0

    # Evaluate with plasticity
    pg = PlasticityGenome()
    pg.theta_10 = plasticity_config.theta_10
    pg.theta_11 = plasticity_config.theta_11
    pg.theta_01 = plasticity_config.theta_01
    pg.eta = plasticity_config.eta
    pg.tau_e = plasticity_config.tau_e
    pg.tau_x = plasticity_config.tau_x
    pg.baseline_decay = plasticity_config.baseline_decay

    _, plastic_rollout_data = fitness_fn.evaluate_with_plasticity(grid, pg, G=G)

    train_rewards = plastic_rollout_data.get("train_rewards", [])
    eval_rewards = plastic_rollout_data.get("eval_rewards", [])
    all_rewards = plastic_rollout_data.get("rewards", [])

    plastic_mean = float(np.mean(eval_rewards)) if eval_rewards else 0.0
    plastic_std = float(np.std(eval_rewards)) if eval_rewards else 0.0

    improvement = plastic_mean - frozen_mean

    # Extract plasticity metrics
    plasticity_metrics = {
        "mean_abs_dW": plastic_rollout_data.get("plasticity_update", {}).get("train", {}).get("mean_abs_dW", 0.0),
        "max_abs_dW": plastic_rollout_data.get("plasticity_update", {}).get("train", {}).get("max_abs_dW", 0.0),
        "mean_nonzero_frac": plastic_rollout_data.get("plasticity_update", {}).get("train", {}).get("mean_nonzero_frac", 0.0),
        "learning_delta": plastic_rollout_data.get("learning_delta", 0.0),
    }

    elapsed = time.time() - start_time

    result = EvaluationResult(
        network_id=network_spec.network_id,
        config_id=plasticity_config.config_id,
        frozen_mean=frozen_mean,
        frozen_std=frozen_std,
        frozen_rewards=frozen_rewards,
        plastic_mean=plastic_mean,
        plastic_std=plastic_std,
        train_rewards=train_rewards,
        eval_rewards=eval_rewards,
        improvement=improvement,
        plasticity_metrics=plasticity_metrics,
        evaluation_time=elapsed,
    )

    return result


class EvaluationOrchestrator:
    """Orchestrate parallel evaluation of networks × configs."""

    def __init__(
        self,
        n_rollouts: int = 20,
        env_name: str = "CartPole-v1",
        cache_dir: Optional[Path] = None,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize evaluation orchestrator.

        Args:
            n_rollouts: Number of rollout episodes per evaluation
            env_name: Gym environment name
            cache_dir: Directory for caching results (None = no caching)
            max_workers: Max parallel workers (None = CPU count)
        """
        self.n_rollouts = n_rollouts
        self.env_name = env_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_workers = max_workers

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using cache directory: {self.cache_dir}")

    def _get_cache_key(self, network_id: str, config_id: str) -> str:
        """Generate cache key for network × config pair."""
        key_str = f"{network_id}_{config_id}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_result(self, network_id: str, config_id: str) -> Optional[EvaluationResult]:
        """Try to load cached result."""
        if not self.cache_dir:
            return None

        cache_key = self._get_cache_key(network_id, config_id)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return EvaluationResult.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load cache for {network_id} × {config_id}: {e}")
                return None

        return None

    def _save_cached_result(self, result: EvaluationResult) -> None:
        """Save result to cache."""
        if not self.cache_dir:
            return

        cache_key = self._get_cache_key(result.network_id, result.config_id)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump(result.to_dict(), f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _should_early_stop(
        self,
        network_id: str,
        results_so_far: List[EvaluationResult],
        threshold: float = -20.0,
        min_configs_tested: int = 3,
    ) -> bool:
        """
        Check if we should early stop evaluation for this network.

        Early stop if network shows no plasticity benefit after testing
        min_configs_tested configurations.

        Args:
            network_id: Network ID
            results_so_far: Results for this network so far
            threshold: Improvement threshold
            min_configs_tested: Minimum configs to test before stopping

        Returns:
            True if should stop, False otherwise
        """
        if len(results_so_far) < min_configs_tested:
            return False

        # Check if all improvements are below threshold
        improvements = [r.improvement for r in results_so_far]
        if all(imp < threshold for imp in improvements):
            logger.info(
                f"Early stopping {network_id}: "
                f"all {len(improvements)} improvements < {threshold:.1f}"
            )
            return True

        return False

    def evaluate_phase(
        self,
        networks: List[NetworkSpec],
        configs: List[PlasticityConfig],
        phase_name: str = "phase1",
        early_stopping: bool = True,
        early_stop_threshold: float = -20.0,
        parallel: bool = True,
    ) -> PhaseResults:
        """
        Evaluate all networks × all configs for a phase.

        Args:
            networks: List of network specifications
            configs: List of plasticity configurations
            phase_name: Phase identifier
            early_stopping: Enable early stopping for non-responsive networks
            early_stop_threshold: Threshold for early stopping
            parallel: Use parallel evaluation

        Returns:
            PhaseResults object
        """
        logger.info(f"Starting {phase_name} evaluation:")
        logger.info(f"  Networks: {len(networks)}")
        logger.info(f"  Configs: {len(configs)}")
        logger.info(f"  Total evaluations: {len(networks) * len(configs)}")
        logger.info(f"  Early stopping: {early_stopping}")
        logger.info(f"  Parallel: {parallel} (max_workers={self.max_workers})")

        start_time = time.time()

        # Build task list
        tasks = []
        for network in networks:
            for config in configs:
                task = EvaluationTask(
                    network_spec=network,
                    plasticity_config=config,
                    task_id=f"{network.network_id}_{config.config_id}",
                )
                tasks.append(task)

        total_tasks = len(tasks)
        logger.info(f"Total tasks to evaluate: {total_tasks}")

        # Results storage: network_id -> config_id -> EvaluationResult
        results: Dict[str, Dict[str, EvaluationResult]] = {}

        successful = 0
        failed = 0
        cached = 0
        skipped = 0

        # Network-specific early stop tracking
        network_stopped = set()
        network_results_so_far: Dict[str, List[EvaluationResult]] = {
            net.network_id: [] for net in networks
        }

        if parallel and self.max_workers != 1:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {}

                for task in tasks:
                    # Check cache
                    cached_result = self._get_cached_result(
                        task.network_spec.network_id,
                        task.plasticity_config.config_id,
                    )

                    if cached_result:
                        # Use cached result
                        network_id = task.network_spec.network_id
                        config_id = task.plasticity_config.config_id

                        if network_id not in results:
                            results[network_id] = {}
                        results[network_id][config_id] = cached_result
                        network_results_so_far[network_id].append(cached_result)

                        cached += 1
                        continue

                    # Check early stop
                    if early_stopping and task.network_spec.network_id in network_stopped:
                        skipped += 1
                        continue

                    # Submit task
                    future = executor.submit(
                        evaluate_network_config,
                        task.network_spec,
                        task.plasticity_config,
                        self.n_rollouts,
                        self.env_name,
                    )
                    future_to_task[future] = task

                # Process results as they complete
                for i, future in enumerate(as_completed(future_to_task), 1):
                    task = future_to_task[future]

                    try:
                        result = future.result()

                        # Store result
                        network_id = result.network_id
                        config_id = result.config_id

                        if network_id not in results:
                            results[network_id] = {}
                        results[network_id][config_id] = result
                        network_results_so_far[network_id].append(result)

                        # Save to cache
                        self._save_cached_result(result)

                        successful += 1

                        # Log progress
                        total_done = successful + failed + cached + skipped
                        pct = total_done / total_tasks * 100
                        logger.info(
                            f"[{total_done}/{total_tasks} | {pct:.1f}%] "
                            f"{task.network_spec.network_id} × {task.plasticity_config.config_name}: "
                            f"Δ={result.improvement:+.1f} "
                            f"(frozen={result.frozen_mean:.1f}, plastic={result.plastic_mean:.1f}) "
                            f"[{result.evaluation_time:.1f}s]"
                        )

                        # Check early stop for this network
                        if (
                            early_stopping
                            and network_id not in network_stopped
                            and self._should_early_stop(
                                network_id,
                                network_results_so_far[network_id],
                                early_stop_threshold,
                            )
                        ):
                            network_stopped.add(network_id)

                    except Exception as e:
                        logger.error(f"Failed to evaluate {task.task_id}: {e}")
                        failed += 1

        else:
            # Sequential evaluation
            for i, task in enumerate(tasks, 1):
                # Check cache
                cached_result = self._get_cached_result(
                    task.network_spec.network_id,
                    task.plasticity_config.config_id,
                )

                if cached_result:
                    network_id = task.network_spec.network_id
                    config_id = task.plasticity_config.config_id

                    if network_id not in results:
                        results[network_id] = {}
                    results[network_id][config_id] = cached_result
                    network_results_so_far[network_id].append(cached_result)

                    cached += 1

                    pct = (i / total_tasks) * 100
                    logger.info(
                        f"[{i}/{total_tasks} | {pct:.1f}%] "
                        f"{task.network_spec.network_id} × {task.plasticity_config.config_name}: "
                        f"CACHED"
                    )
                    continue

                # Check early stop
                if early_stopping and task.network_spec.network_id in network_stopped:
                    skipped += 1
                    continue

                try:
                    result = evaluate_network_config(
                        task.network_spec,
                        task.plasticity_config,
                        self.n_rollouts,
                        self.env_name,
                    )

                    # Store result
                    network_id = result.network_id
                    config_id = result.config_id

                    if network_id not in results:
                        results[network_id] = {}
                    results[network_id][config_id] = result
                    network_results_so_far[network_id].append(result)

                    # Save to cache
                    self._save_cached_result(result)

                    successful += 1

                    # Log progress
                    pct = (i / total_tasks) * 100
                    logger.info(
                        f"[{i}/{total_tasks} | {pct:.1f}%] "
                        f"{task.network_spec.network_id} × {task.plasticity_config.config_name}: "
                        f"Δ={result.improvement:+.1f} "
                        f"(frozen={result.frozen_mean:.1f}, plastic={result.plastic_mean:.1f}) "
                        f"[{result.evaluation_time:.1f}s]"
                    )

                    # Check early stop
                    if (
                        early_stopping
                        and network_id not in network_stopped
                        and self._should_early_stop(
                            network_id,
                            network_results_so_far[network_id],
                            early_stop_threshold,
                        )
                    ):
                        network_stopped.add(network_id)

                except Exception as e:
                    logger.error(f"Failed to evaluate {task.task_id}: {e}")
                    failed += 1

        elapsed = time.time() - start_time

        logger.info(f"\n{phase_name} Evaluation Complete:")
        logger.info(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Cached: {cached}")
        logger.info(f"  Skipped (early stop): {skipped}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Networks early stopped: {len(network_stopped)}")

        # Build PhaseResults
        phase_results = PhaseResults(
            phase_name=phase_name,
            configs=configs,
            networks=networks,
            results=results,
            total_evaluations=total_tasks,
            successful_evaluations=successful + cached,
            failed_evaluations=failed,
            total_time=elapsed,
            experiment_config={
                "n_rollouts": self.n_rollouts,
                "env_name": self.env_name,
                "early_stopping": early_stopping,
                "early_stop_threshold": early_stop_threshold,
                "networks_early_stopped": list(network_stopped),
            },
        )

        return phase_results
