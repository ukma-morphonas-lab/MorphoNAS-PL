from abc import ABC, abstractmethod
import os
import numpy as np
from threading import Lock
from concurrent.futures import ProcessPoolExecutor


def _worker_init():
    """Limit torch threading in workers to avoid CPU oversubscription."""
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        # Torch might not be installed in some environments
        pass

class Optimizer(ABC):
    """
    Base class for optimization algorithms.
    
    Args:
        fitness_fn (callable): Function that takes a Genome and returns its fitness score
        seed (int, optional): Random seed for reproducibility
        max_workers (int, optional): Maximum number of workers for parallel fitness evaluation
        convergence_strategy (ConvergenceStrategy, optional): Strategy for determining convergence
    """
    
    def __init__(self,
                 fitness_fn,
                 seed=None,
                 max_workers=None,
                 convergence_strategy=None,
                 rollout_handler=None):
        self.fitness_fn = fitness_fn
        self.max_workers = max_workers
        self.convergence_strategy = convergence_strategy
        # Optional callback to capture rollout data from evaluation results
        self.rollout_handler = rollout_handler
        
        # Initialize random number generator
        self.rng = np.random.default_rng(seed)
        
        # Initialize tracking variables
        self.generation = 0
        self._best_genome = None
        self._best_fitness = float('-inf')
        self._eval_count = 0
        self._current_fitness_scores = []  # Track current generation's fitness scores
        
        # Add fitness cache
        self.fitness_cache = {}
        
        # Add evaluation lock
        self._eval_lock = Lock()

        # Persistent executor to avoid per-generation process churn
        self._executor = None
        self._executor_workers = None

    def __del__(self):
        """Best-effort shutdown of the executor when optimizer is cleaned up."""
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
        except Exception:
            pass

    def _get_executor(self):
        """Create or reuse a process pool sized by max_workers."""
        if self._executor is None or self._executor_workers != self.max_workers:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=_worker_init
            )
            self._executor_workers = self.max_workers
        return self._executor

    def _extract_fitness_result(self, genome, result):
        """
        Normalize fitness results that may include rollout data tuples.

        Args:
            genome: Genome that was evaluated
            result: Fitness result (float or (fitness, rollout_data))

        Returns:
            float fitness value
        """
        rollout_data = None
        fitness = result

        # Unpack (fitness, rollout_data) tuples from multiprocessing workers
        if isinstance(result, tuple) and len(result) == 2:
            fitness, rollout_data = result

        # Record rollout data if provided and handler available
        if rollout_data is not None and self.rollout_handler:
            try:
                self.rollout_handler(genome, fitness, rollout_data)
            except Exception as e:
                # Don't fail optimization if rollout recording fails
                print(f"Warning: Failed to record rollout data: {e}")

        # Ensure downstream code always receives a float
        return float(fitness)
    
    def _evaluate_uncached_genomes(self, genomes):
        """Evaluate fitness for multiple genomes in parallel."""
        if self.max_workers == 1:
            # Avoid spawn overhead when only one worker is desired
            results = [self.fitness_fn(genome) for genome in genomes]
        else:
            executor = self._get_executor()
            # Batch tasks to reduce scheduling overhead for large populations
            worker_count = self.max_workers or os.cpu_count() or 1
            chunk_size = max(1, len(genomes) // (worker_count * 4)) if genomes else 1
            results = list(executor.map(self.fitness_fn, genomes, chunksize=chunk_size))
            
        # Update evaluation counter after batch processing
        with self._eval_lock:
            self._eval_count += len(genomes)
            
        return results
    
    def _evaluate_solutions(self, genomes):
        """Evaluate a batch of genomes in parallel."""
        # First check cache for all genomes
        fitness_scores = []
        uncached_genomes = []
        uncached_indices = []
        
        for i, genome in enumerate(genomes):
            genome_bytes = genome.to_bytes()
            if genome_bytes in self.fitness_cache:
                fitness_scores.append(self.fitness_cache[genome_bytes])
            else:
                uncached_genomes.append(genome)
                uncached_indices.append(i)
                fitness_scores.append(None)  # Placeholder
        
        # If there are uncached genomes, evaluate them in parallel
        if uncached_genomes:
            uncached_fitness_scores = self._evaluate_uncached_genomes(uncached_genomes)
            
            # Update cache and fitness scores
            for idx, genome, fitness in zip(uncached_indices, uncached_genomes, uncached_fitness_scores):
                fitness_value = self._extract_fitness_result(genome, fitness)
                genome_bytes = genome.to_bytes()
                self.fitness_cache[genome_bytes] = fitness_value
                fitness_scores[idx] = fitness_value
                
                # Update best solution if necessary
                if fitness_value > self._best_fitness:
                    self._best_fitness = fitness_value
                    self._best_genome = genome
        
        # Store current generation's fitness scores
        self._current_fitness_scores = fitness_scores
        return np.array(fitness_scores, dtype=float)
    
    @abstractmethod
    def step(self):
        """Perform one step of optimization."""
        pass
    
    def run(self, max_generations, callback=None):
        """
        Run the optimization process for max_generations.
        
        Args:
            max_generations (int): Maximum number of generations to run
            callback (callable, optional): Function called after each generation
                with (optimizer, generation, best_fitness, avg_fitness)
                Returns True to continue, False to stop
        """
        while self.generation < max_generations:
            if not self.step():
                break
            
            if callback:
                avg_fitness = np.mean(self._current_fitness_scores)
                if not callback(self, self.generation, self.best_fitness, avg_fitness):
                    break
    
    def get_mutation_rate(self):
        """Get the current mutation rate. Returns 0.0 for optimizers that don't use mutation."""
        return 0.0
    
    @property
    def best_solution(self):
        """Get the best genome found so far."""
        return self._best_genome
    
    @property
    def best_fitness(self):
        """Get the best fitness score found so far."""
        return self._best_fitness
    
    @property
    def current_fitness_scores(self):
        """Get fitness scores for current population."""
        return self._current_fitness_scores
    
    @property
    def evaluation_count(self):
        """Get the total number of fitness evaluations performed."""
        return self._eval_count 
