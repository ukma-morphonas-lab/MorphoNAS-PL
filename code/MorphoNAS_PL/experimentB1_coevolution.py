"""
B1 Co-Evolution Experiment Module

Co-evolves MorphoNAS developmental genomes with Hebbian plasticity parameters.
Three conditions:
  A — No plasticity: standard MorphoNAS GA, frozen weights
  B — Fixed plasticity: MorphoNAS GA, fitness evaluated with fixed η, λ
  C — Co-evolved plasticity: genome extended with η, λ; evolved alongside architecture

Supports multiple environments (CartPole, Acrobot) via EnvConfig presets.
Uses composition (Individual wraps Genome + plasticity params) to avoid modifying
the core Genome class. Reuses run_rollouts() from B0.5 for evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np

from MorphoNAS.genome import Genome
from MorphoNAS.grid import Grid
from MorphoNAS.neural_propagation import NeuralPropagator
from MorphoNAS_PL.plasticity_hooks import HebbianEdgeHook
from MorphoNAS_PL.experimentB0_5_natural import (
    VERIFICATION_SEEDS,
    Stratum,
    get_stratum,
    run_rollouts,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment and plasticity configuration
# ---------------------------------------------------------------------------


class Condition(str, Enum):
    A = "A"
    B = "B"
    C = "C"


@dataclass
class EnvConfig:
    """Environment and plasticity parameter configuration.

    Bundles environment-specific settings (gym env, network dims, fitness
    normalisation) with plasticity parameter ranges that depend on the task.
    """

    # Environment
    env_name: str = "CartPole-v1"
    input_dim: int = 4
    output_dim: int = 2
    max_reward: float = 500.0  # for fitness normalisation (fitness = reward / max_reward)
    min_neurons: int = 6  # skip networks smaller than this

    # Default grid size for this environment
    default_grid_size: int = 10

    # Condition B: fixed plasticity defaults
    fixed_eta: float = -0.01
    fixed_decay: float = 0.01

    # Condition C: plasticity parameter ranges and mutation
    eta_range: tuple[float, float] = (-0.5, 0.5)
    decay_range: tuple[float, float] = (0.0, 0.1)
    eta_mutation_std: float = 0.05
    decay_mutation_std: float = 0.005

    # Non-stationarity: mid-episode physics switch (None = static)
    perturbation: Optional[dict] = None  # e.g. {"LINK_MASS_2": 2.0} for Acrobot
    switch_step: int = 50  # step at which perturbation applies

    def to_dict(self) -> dict:
        return {
            "env_name": self.env_name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "max_reward": self.max_reward,
            "min_neurons": self.min_neurons,
            "default_grid_size": self.default_grid_size,
            "fixed_eta": self.fixed_eta,
            "fixed_decay": self.fixed_decay,
            "eta_range": list(self.eta_range),
            "decay_range": list(self.decay_range),
            "eta_mutation_std": self.eta_mutation_std,
            "decay_mutation_std": self.decay_mutation_std,
            "perturbation": self.perturbation,
            "switch_step": self.switch_step,
        }


# --- Presets ---

CARTPOLE_ENV = EnvConfig()  # all defaults are CartPole

ACROBOT_ENV = EnvConfig(
    env_name="Acrobot-v1",
    input_dim=6,
    output_dim=3,
    max_reward=500.0,  # Acrobot reward is negative; we use -reward, capped at 500
    min_neurons=9,  # 6 input + 3 output
    default_grid_size=20,
    fixed_eta=-0.001,
    fixed_decay=0.05,
    eta_range=(-0.1, 0.1),
    decay_range=(0.0, 0.1),
    eta_mutation_std=0.005,
    decay_mutation_std=0.005,
)

ACROBOT_NS_ENV = EnvConfig(
    env_name="Acrobot-v1",
    input_dim=6,
    output_dim=3,
    max_reward=500.0,
    min_neurons=9,
    default_grid_size=20,
    fixed_eta=-0.001,
    fixed_decay=0.05,
    eta_range=(-0.1, 0.1),
    decay_range=(0.0, 0.1),
    eta_mutation_std=0.005,
    decay_mutation_std=0.005,
    perturbation={"LINK_MASS_2": 2.0},  # 2× lower-link mass (matches characterisation)
    switch_step=50,  # matches characterisation: perturbation at step 50
)

ENV_PRESETS: dict[str, EnvConfig] = {
    "cartpole": CARTPOLE_ENV,
    "acrobot": ACROBOT_ENV,
    "acrobot_ns": ACROBOT_NS_ENV,
}


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------


@dataclass
class Individual:
    """A candidate solution: MorphoNAS genome + plasticity parameters."""

    genome: Genome
    eta: float = 0.0
    decay: float = 0.0
    fitness: float = 0.0

    def cache_key(self) -> tuple:
        """Key for fitness caching. Same genome + same plasticity = same fitness."""
        return (self.genome.to_bytes(), round(self.eta, 8), round(self.decay, 8))

    def to_dict(self) -> dict:
        return {
            "genome": self.genome.to_dict(),
            "eta": self.eta,
            "decay": self.decay,
            "fitness": self.fitness,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Individual:
        return cls(
            genome=Genome.from_dict(d["genome"]),
            eta=d["eta"],
            decay=d["decay"],
            fitness=d.get("fitness", 0.0),
        )


# ---------------------------------------------------------------------------
# Individual creation / mutation / crossover
# ---------------------------------------------------------------------------


def random_individual(
    rng: np.random.Generator,
    condition: Condition,
    env_config: EnvConfig,
    grid_size: int = 10,
    num_morphogens: int = 3,
    max_growth_steps: int = 200,
) -> Individual:
    """Create a random individual for the given condition."""
    genome = Genome.random(
        rng,
        size_x=grid_size,
        size_y=grid_size,
        num_morphogens=num_morphogens,
        max_growth_steps=max_growth_steps,
    )

    if condition == Condition.A:
        eta, decay = 0.0, 0.0
    elif condition == Condition.B:
        eta, decay = env_config.fixed_eta, env_config.fixed_decay
    elif condition == Condition.C:
        eta = float(rng.uniform(*env_config.eta_range))
        decay = float(rng.uniform(*env_config.decay_range))
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return Individual(genome=genome, eta=eta, decay=decay)


def crossover_individual(
    parent1: Individual,
    parent2: Individual,
    rng: np.random.Generator,
    condition: Condition,
    env_config: EnvConfig,
) -> Individual:
    """Crossover two individuals."""
    child_genome = Genome.crossover(parent1.genome, parent2.genome, rng)

    if condition == Condition.C:
        eta = parent1.eta if rng.random() < 0.5 else parent2.eta
        decay = parent1.decay if rng.random() < 0.5 else parent2.decay
    elif condition == Condition.B:
        eta, decay = env_config.fixed_eta, env_config.fixed_decay
    else:
        eta, decay = 0.0, 0.0

    return Individual(genome=child_genome, eta=eta, decay=decay)


def mutate_individual(
    individual: Individual,
    rng: np.random.Generator,
    condition: Condition,
    env_config: EnvConfig,
) -> Individual:
    """Mutate an individual (architecture + optionally plasticity params)."""
    mutated_genome = individual.genome.mutate(rng)

    eta = individual.eta
    decay = individual.decay

    if condition == Condition.C:
        # Mutate plasticity params with same probability as architecture
        # (the genome mutation already happened; plasticity mutation is independent)
        if rng.random() < 0.3:
            eta = float(np.clip(
                eta + rng.normal(0, env_config.eta_mutation_std),
                *env_config.eta_range,
            ))
        if rng.random() < 0.3:
            decay = float(np.clip(
                decay + rng.normal(0, env_config.decay_mutation_std),
                *env_config.decay_range,
            ))

    return Individual(genome=mutated_genome, eta=eta, decay=decay)


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------


def _create_propagator_for_eval(
    grid: Grid,
    eta: float,
    decay: float,
    env_config: EnvConfig,
) -> tuple[Optional[NeuralPropagator], dict]:
    """Create a NeuralPropagator from a grown grid, optionally with plasticity.

    Returns (propagator, metadata) or (None, metadata) if the network is too small.
    """
    G = grid.get_graph()
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes < env_config.min_neurons:
        return None, {"num_neurons": num_nodes, "num_connections": num_edges, "error": "too_small"}

    # Graph diameter (handles disconnected components)
    try:
        graph_diameter = int(nx.diameter(G))
    except nx.NetworkXError:
        max_length = 0
        for source in G.nodes():
            lengths = nx.shortest_path_length(G, source)
            max_length = max(max_length, max(lengths.values()))
        graph_diameter = int(max_length)

    # Create hook if plasticity is active
    edge_hook = None
    if abs(eta) > 1e-12:
        edge_hook = HebbianEdgeHook(learning_rate=float(eta), weight_decay=float(decay))

    propagator = NeuralPropagator(
        G=G,
        input_dim=env_config.input_dim,
        output_dim=env_config.output_dim,
        activation_function=NeuralPropagator.tanh_activation,
        extra_thinking_time=2,
        additive_update=False,
        graph_diameter=graph_diameter,
        edge_hook=edge_hook,
    )

    meta = {"num_neurons": num_nodes, "num_connections": num_edges}
    return propagator, meta


def evaluate_individual(
    individual: Individual,
    env_config: EnvConfig = CARTPOLE_ENV,
    num_rollouts: int = 20,
    seeds: Optional[list[int]] = None,
) -> float:
    """Evaluate an individual's fitness on the configured environment.

    Returns fitness in [0, 1].
    For CartPole: avg_reward / 500.
    For Acrobot: (-avg_reward) / 500, capped at 1.0.
    """
    import gymnasium as gym

    if seeds is None:
        seeds = VERIFICATION_SEEDS[:num_rollouts]

    try:
        grid = Grid(individual.genome)
        grid.run_simulation(verbose=False)
    except Exception as e:
        logger.debug(f"Grid simulation failed: {e}")
        return 0.0

    propagator, meta = _create_propagator_for_eval(
        grid, individual.eta, individual.decay, env_config
    )
    if propagator is None:
        return 0.0

    try:
        env = gym.make(env_config.env_name, render_mode=None)
        if env_config.perturbation is not None:
            if env_config.env_name == "Acrobot-v1":
                from MorphoNAS_PL.env_wrappers_acrobot import AcrobotMidEpisodeSwitchWrapper
                env = AcrobotMidEpisodeSwitchWrapper(
                    env, switch_step=env_config.switch_step,
                    target_params=env_config.perturbation,
                )
            else:
                from MorphoNAS_PL.env_wrappers import MidEpisodeSwitchWrapper
                env = MidEpisodeSwitchWrapper(
                    env, switch_step=env_config.switch_step,
                    target_params=env_config.perturbation,
                )
        result = run_rollouts(
            propagator,
            num_rollouts,
            seeds=seeds,
            reset_plastic_each_episode=True,
            env=env,
        )
        env.close()

        avg_reward = float(result["avg_reward"])

        # Normalise to [0, 1] fitness
        if env_config.env_name == "Acrobot-v1":
            # Acrobot reward is negative (closer to 0 = better, -500 = worst)
            # Convert: fitness = 1 + avg_reward/500 (maps -500→0, 0→1)
            fitness = max(0.0, min(1.0, 1.0 + avg_reward / env_config.max_reward))
        else:
            fitness = max(0.0, min(1.0, avg_reward / env_config.max_reward))

        return fitness
    except Exception as e:
        logger.debug(f"Rollout failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Worker for parallel evaluation
# ---------------------------------------------------------------------------

# Module-level state set by worker initializer
_worker_log_queue = None


def _worker_init(log_queue) -> None:
    """Initialize a worker process."""
    from MorphoNAS_PL.parallel_utils import configure_worker_threads
    from MorphoNAS_PL.logging_config import setup_logging

    configure_worker_threads()
    setup_logging(queue=log_queue)


def _worker_evaluate(
    individual_dict: dict, env_config_dict: dict, num_rollouts: int, seeds: list[int],
) -> float:
    """Worker function: deserialize individual, evaluate, return fitness."""
    ind = Individual.from_dict(individual_dict)
    ec = EnvConfig(**{
        k: tuple(v) if isinstance(v, list) else v
        for k, v in env_config_dict.items()
    })
    return evaluate_individual(ind, env_config=ec, num_rollouts=num_rollouts, seeds=seeds)


# ---------------------------------------------------------------------------
# GA loop
# ---------------------------------------------------------------------------


@dataclass
class GAConfig:
    """Configuration for a GA run."""

    condition: Condition
    run_seed: int
    env_config: EnvConfig = field(default_factory=lambda: CARTPOLE_ENV)
    pop_size: int = 50
    max_gen: int = 200
    grid_size: int = 10
    num_morphogens: int = 3
    max_growth_steps: int = 200
    selection_pressure: float = 0.2
    mutation_rate: float = 0.3
    num_elite: int = 2
    num_rollouts: int = 20
    max_workers: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "condition": self.condition.value,
            "run_seed": self.run_seed,
            "env_config": self.env_config.to_dict(),
            "pop_size": self.pop_size,
            "max_gen": self.max_gen,
            "grid_size": self.grid_size,
            "num_morphogens": self.num_morphogens,
            "max_growth_steps": self.max_growth_steps,
            "selection_pressure": self.selection_pressure,
            "mutation_rate": self.mutation_rate,
            "num_elite": self.num_elite,
            "num_rollouts": self.num_rollouts,
            "max_workers": self.max_workers,
        }


@dataclass
class GenerationRecord:
    """Stats for one generation, serialized to JSONL."""

    generation: int
    best_fitness: float
    avg_fitness: float
    median_fitness: float
    std_fitness: float
    best_eta: float
    best_decay: float
    best_reward: float
    num_viable: int  # networks with fitness > 0
    eta_mean: float  # population mean eta (meaningful for condition C)
    eta_std: float
    decay_mean: float
    decay_std: float
    eval_time_s: float
    gen_time_s: float

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "median_fitness": self.median_fitness,
            "std_fitness": self.std_fitness,
            "best_eta": self.best_eta,
            "best_decay": self.best_decay,
            "best_reward": self.best_reward,
            "num_viable": self.num_viable,
            "eta_mean": self.eta_mean,
            "eta_std": self.eta_std,
            "decay_mean": self.decay_mean,
            "decay_std": self.decay_std,
            "eval_time_s": self.eval_time_s,
            "gen_time_s": self.gen_time_s,
        }


def _evaluate_population(
    population: list[Individual],
    cache: dict,
    env_config: EnvConfig,
    num_rollouts: int,
    seeds: list[int],
    executor,
) -> list[float]:
    """Evaluate a population, using cache and parallel executor."""
    fitnesses = [0.0] * len(population)
    uncached_indices = []

    # Check cache first
    for i, ind in enumerate(population):
        key = ind.cache_key()
        if key in cache:
            fitnesses[i] = cache[key]
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        return fitnesses

    env_config_dict = env_config.to_dict()

    # Evaluate uncached in parallel
    if executor is not None:
        futures = {}
        for idx in uncached_indices:
            ind = population[idx]
            future = executor.submit(
                _worker_evaluate,
                ind.to_dict(),
                env_config_dict,
                num_rollouts,
                seeds,
            )
            futures[future] = idx

        from concurrent.futures import as_completed

        for future in as_completed(futures):
            idx = futures[future]
            try:
                fit = future.result()
            except Exception as e:
                logger.warning(f"Worker evaluation failed for individual {idx}: {e}")
                fit = 0.0
            fitnesses[idx] = fit
            cache[population[idx].cache_key()] = fit
    else:
        # Sequential fallback
        for idx in uncached_indices:
            fit = evaluate_individual(population[idx], env_config, num_rollouts, seeds)
            fitnesses[idx] = fit
            cache[population[idx].cache_key()] = fit

    return fitnesses


def _save_checkpoint(
    output_dir: Path,
    population: list[Individual],
    generation: int,
) -> None:
    """Save population checkpoint for resume."""
    checkpoint = {
        "generation": generation,
        "population": [ind.to_dict() for ind in population],
    }
    # Atomic write via temp file
    tmp = output_dir / "checkpoint.tmp"
    with open(tmp, "w") as f:
        json.dump(checkpoint, f)
    tmp.rename(output_dir / "checkpoint.json")


def _load_checkpoint(output_dir: Path) -> Optional[tuple[int, list[Individual]]]:
    """Load population checkpoint if it exists."""
    cp_path = output_dir / "checkpoint.json"
    if not cp_path.exists():
        return None
    try:
        with open(cp_path) as f:
            data = json.load(f)
        population = [Individual.from_dict(d) for d in data["population"]]
        return data["generation"], population
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def _count_completed_generations(output_dir: Path) -> int:
    """Count lines in generations.jsonl to determine resume point."""
    jsonl_path = output_dir / "generations.jsonl"
    if not jsonl_path.exists():
        return 0
    with open(jsonl_path) as f:
        return sum(1 for _ in f)


def run_ga(
    config: GAConfig,
    output_dir: Path,
    resume: bool = False,
    log_queue=None,
) -> list[dict]:
    """Run one GA experiment. Returns list of generation records.

    Args:
        config: GA configuration
        output_dir: Directory to write results
        resume: If True, resume from checkpoint
        log_queue: Logging queue for worker processes
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    seeds = VERIFICATION_SEEDS[: config.num_rollouts]
    rng = np.random.default_rng(config.run_seed)
    cache: dict = {}
    records: list[dict] = []

    # Determine starting generation and population
    start_gen = 0
    population: list[Individual] = []

    if resume:
        completed = _count_completed_generations(output_dir)
        checkpoint = _load_checkpoint(output_dir)
        if checkpoint is not None and completed > 0:
            start_gen = completed
            _, population = checkpoint
            # Restore fitness values from checkpoint
            for ind in population:
                cache[ind.cache_key()] = ind.fitness
            logger.info(f"Resuming from generation {start_gen} ({len(population)} individuals)")

            # Reload existing records
            jsonl_path = output_dir / "generations.jsonl"
            if jsonl_path.exists():
                with open(jsonl_path) as f:
                    for line in f:
                        records.append(json.loads(line))

            # Advance RNG to match state (consume the same random draws)
            # We need to re-derive the rng state. Simplest: use a separate rng
            # seeded from (run_seed, start_gen) for each generation.
            # This approach uses per-generation seeds for reproducibility.

    # Initialize population if not resuming
    env_config = config.env_config
    if not population:
        population = [
            random_individual(
                rng,
                config.condition,
                env_config,
                grid_size=config.grid_size,
                num_morphogens=config.num_morphogens,
                max_growth_steps=config.max_growth_steps,
            )
            for _ in range(config.pop_size)
        ]

    num_parents = max(2, int(config.pop_size * config.selection_pressure))

    # Open JSONL for appending
    jsonl_path = output_dir / "generations.jsonl"
    jsonl_mode = "a" if resume and start_gen > 0 else "w"

    # Set up parallel executor
    from concurrent.futures import ProcessPoolExecutor

    max_workers = config.max_workers
    if max_workers is None:
        import multiprocessing
        max_workers = max(1, multiprocessing.cpu_count() - 2)

    executor = None
    if max_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(log_queue,),
        )

    try:
        with open(jsonl_path, jsonl_mode) as jsonl_f:
            for gen in range(start_gen, config.max_gen + 1):
                gen_start = time.time()

                # Use per-generation RNG for reproducibility across resume
                gen_rng = np.random.default_rng(config.run_seed * 10000 + gen)

                # --- Evaluate ---
                eval_start = time.time()
                fitnesses = _evaluate_population(population, cache, env_config, config.num_rollouts, seeds, executor)
                eval_time = time.time() - eval_start

                # Update fitness on individuals
                for ind, fit in zip(population, fitnesses):
                    ind.fitness = fit

                # --- Record stats ---
                fit_arr = np.array(fitnesses)
                etas = np.array([ind.eta for ind in population])
                decays = np.array([ind.decay for ind in population])

                best_idx = int(np.argmax(fit_arr))
                best_ind = population[best_idx]

                rec = GenerationRecord(
                    generation=gen,
                    best_fitness=float(fit_arr[best_idx]),
                    avg_fitness=float(np.mean(fit_arr)),
                    median_fitness=float(np.median(fit_arr)),
                    std_fitness=float(np.std(fit_arr)),
                    best_eta=best_ind.eta,
                    best_decay=best_ind.decay,
                    best_reward=float(fit_arr[best_idx] * env_config.max_reward),
                    num_viable=int(np.sum(fit_arr > 0)),
                    eta_mean=float(np.mean(etas)),
                    eta_std=float(np.std(etas)),
                    decay_mean=float(np.mean(decays)),
                    decay_std=float(np.std(decays)),
                    eval_time_s=eval_time,
                    gen_time_s=0.0,  # filled below
                )

                if gen % 10 == 0 or gen == config.max_gen:
                    logger.info(
                        f"Gen {gen:3d} | best={rec.best_fitness:.4f} "
                        f"avg={rec.avg_fitness:.4f} viable={rec.num_viable}/{config.pop_size} "
                        f"eta={best_ind.eta:+.4f} decay={best_ind.decay:.4f} "
                        f"eval={eval_time:.1f}s"
                    )

                # Last generation — don't breed
                if gen == config.max_gen:
                    rec.gen_time_s = time.time() - gen_start
                    records.append(rec.to_dict())
                    jsonl_f.write(json.dumps(rec.to_dict()) + "\n")
                    jsonl_f.flush()
                    _save_checkpoint(output_dir, population, gen)
                    break

                # --- Selection ---
                sorted_indices = np.argsort(fit_arr)[::-1]
                parent_indices = sorted_indices[:num_parents]
                parents = [population[i] for i in parent_indices]

                # --- Elitism ---
                elite = [population[i] for i in sorted_indices[: config.num_elite]]

                # --- Offspring ---
                offspring: list[Individual] = []
                target_offspring = config.pop_size - config.num_elite
                while len(offspring) < target_offspring:
                    p1_idx, p2_idx = gen_rng.choice(len(parents), size=2, replace=False)
                    child = crossover_individual(
                        parents[p1_idx], parents[p2_idx], gen_rng, config.condition, env_config
                    )
                    if gen_rng.random() < config.mutation_rate:
                        child = mutate_individual(child, gen_rng, config.condition, env_config)
                    offspring.append(child)

                population = elite + offspring

                rec.gen_time_s = time.time() - gen_start
                records.append(rec.to_dict())
                jsonl_f.write(json.dumps(rec.to_dict()) + "\n")
                jsonl_f.flush()

                # Checkpoint every 10 generations
                if gen % 10 == 0:
                    _save_checkpoint(output_dir, population, gen)

    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    # Save final results
    best_idx = int(np.argmax([ind.fitness for ind in population]))
    best_ind = population[best_idx]

    def _fitness_to_reward(fitness: float) -> float:
        """Convert normalised fitness back to environment reward."""
        if env_config.env_name == "Acrobot-v1":
            return (fitness - 1.0) * env_config.max_reward
        return fitness * env_config.max_reward

    final_best = {
        "genome": best_ind.genome.to_dict(),
        "eta": best_ind.eta,
        "decay": best_ind.decay,
        "fitness": best_ind.fitness,
        "reward": _fitness_to_reward(best_ind.fitness),
        "stratum": get_stratum(best_ind.fitness * env_config.max_reward).value,
        "condition": config.condition.value,
        "run_seed": config.run_seed,
        "env": env_config.env_name,
    }
    with open(output_dir / "final_best.json", "w") as f:
        json.dump(final_best, f, indent=2)

    final_pop = {
        "condition": config.condition.value,
        "run_seed": config.run_seed,
        "env": env_config.env_name,
        "generation": config.max_gen,
        "individuals": [
            {
                "eta": ind.eta,
                "decay": ind.decay,
                "fitness": ind.fitness,
                "reward": _fitness_to_reward(ind.fitness),
                "genome": ind.genome.to_dict(),
            }
            for ind in population
        ],
    }
    with open(output_dir / "final_population.json", "w") as f:
        json.dump(final_pop, f, indent=2)

    logger.info(
        f"GA complete: best_fitness={best_ind.fitness:.4f} "
        f"reward={_fitness_to_reward(best_ind.fitness):.1f} eta={best_ind.eta:+.4f} "
        f"decay={best_ind.decay:.4f}"
    )

    return records
