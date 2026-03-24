"""
Network pool generation and stratification for B0.3 experiment.

Generates 100 viable networks, evaluates frozen baselines, and stratifies
by performance into weak/mid/strong groups.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import networkx as nx

from MorphoNAS.genome import Genome
from MorphoNAS.grid import Grid
from MorphoNAS_PL.b1_fitness import B1FitnessFunction

from .config_schemas import NetworkSpec, NetworkFeatures

logger = logging.getLogger(__name__)


class NetworkPool:
    """Generate and manage a stratified pool of networks."""

    def __init__(
        self,
        grid_size_x: int = 10,
        grid_size_y: int = 10,
        max_growth_steps: int = 200,
        num_morphogens: int = 3,
        min_neurons: int = 6,
        min_edges: int = 5,
        frozen_baseline_rollouts: int = 20,
        env_name: str = "CartPole-v1",
        seed_base: int = 42,
    ):
        """
        Initialize network pool generator.

        Args:
            grid_size_x: Grid width for morphogenesis
            grid_size_y: Grid height for morphogenesis
            max_growth_steps: Maximum growth steps for network development
            num_morphogens: Number of morphogens
            min_neurons: Minimum neurons required
            min_edges: Minimum edges required
            frozen_baseline_rollouts: Episodes for baseline evaluation
            env_name: Gym environment name
            seed_base: Base random seed
        """
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.max_growth_steps = max_growth_steps
        self.num_morphogens = num_morphogens
        self.min_neurons = min_neurons
        self.min_edges = min_edges
        self.frozen_baseline_rollouts = frozen_baseline_rollouts
        self.env_name = env_name
        self.seed_base = seed_base

        # Initialize fitness function for baseline evaluation
        self.fitness_config = {
            "env_name": env_name,
            "passing_reward": 475.0,
            "seed": 42,
        }

    def generate_single_network(
        self, seed: int
    ) -> Optional[Tuple[Genome, Grid, nx.DiGraph, Dict]]:
        """
        Generate a single network and check if it's viable.

        Returns:
            (genome, grid, graph, stats) if viable, None otherwise
        """
        rng = np.random.default_rng(seed)

        try:
            genome = Genome.random(
                rng=rng,
                size_x=self.grid_size_x,
                size_y=self.grid_size_y,
                max_growth_steps=self.max_growth_steps,
                num_morphogens=self.num_morphogens,
            )

            grid = Grid(genome)
            grid.run_simulation(verbose=False)
            G = grid.get_graph()

            # Check viability
            num_neurons = G.number_of_nodes()
            num_edges = G.number_of_edges()

            if num_neurons < self.min_neurons or num_edges < self.min_edges:
                return None

            stats = {
                "num_neurons": num_neurons,
                "num_edges": num_edges,
                "num_inputs": sum(1 for n in G.nodes() if G.in_degree(n) == 0),
                "num_outputs": sum(1 for n in G.nodes() if G.out_degree(n) == 0),
            }

            return genome, grid, G, stats

        except Exception as e:
            logger.warning(f"Failed to generate network from seed {seed}: {e}")
            return None

    def compute_frozen_baseline(
        self, grid: Grid, G: nx.DiGraph
    ) -> Tuple[float, float, List[float]]:
        """
        Compute frozen baseline performance.

        Returns:
            (mean_reward, std_reward, rewards_list)
        """
        fitness_fn = B1FitnessFunction(
            targets=self.fitness_config,
            n_rollouts=self.frozen_baseline_rollouts,
        )

        try:
            _, rollout_data = fitness_fn.evaluate_baseline(grid, G=G)

            rewards = rollout_data.get("rewards", [])
            if len(rewards) == 0:
                return 0.0, 0.0, []

            mean_reward = float(np.mean(rewards))
            std_reward = float(np.std(rewards))

            return mean_reward, std_reward, rewards

        except Exception as e:
            logger.warning(f"Failed to evaluate frozen baseline: {e}")
            return 0.0, 0.0, []

    def generate_stratified_pool(
        self,
        n_total: int = 100,
        n_weak: int = 25,
        n_mid: int = 50,
        n_strong: int = 25,
        threshold_weak_mid: float = 150.0,
        threshold_mid_strong: float = 300.0,
        min_frozen_reward: float = 50.0,
        max_frozen_reward: float = 500.0,
        max_attempts: int = 10000,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> List[NetworkSpec]:
        """
        Generate stratified network pool.

        Args:
            n_total: Total networks to generate
            n_weak: Number of weak networks (<150 frozen reward)
            n_mid: Number of mid networks (150-300)
            n_strong: Number of strong networks (>=300)
            threshold_weak_mid: Threshold between weak and mid
            threshold_mid_strong: Threshold between mid and strong
            min_frozen_reward: Minimum acceptable frozen reward
            max_frozen_reward: Maximum acceptable frozen reward
            max_attempts: Maximum generation attempts
            parallel: Use parallel generation
            max_workers: Max parallel workers (None = CPU count)

        Returns:
            List of NetworkSpec objects
        """
        logger.info(f"Generating {n_total} networks stratified by performance...")
        logger.info(f"  Weak: {n_weak} (<{threshold_weak_mid})")
        logger.info(f"  Mid: {n_mid} ({threshold_weak_mid}-{threshold_mid_strong})")
        logger.info(f"  Strong: {n_strong} (>={threshold_mid_strong})")

        networks_weak = []
        networks_mid = []
        networks_strong = []

        start_time = time.time()
        attempts = 0
        successful = 0

        while (
            len(networks_weak) < n_weak
            or len(networks_mid) < n_mid
            or len(networks_strong) < n_strong
        ):
            if attempts >= max_attempts:
                logger.warning(
                    f"Reached max attempts ({max_attempts}). "
                    f"Generated {successful}/{n_total} networks."
                )
                break

            seed = self.seed_base + attempts
            attempts += 1

            # Generate network
            result = self.generate_single_network(seed)
            if result is None:
                continue

            genome, grid, G, stats = result

            # Compute frozen baseline
            mean_reward, std_reward, rewards = self.compute_frozen_baseline(grid, G)

            if mean_reward < min_frozen_reward or mean_reward >= max_frozen_reward:
                continue

            # Create NetworkSpec
            network_spec = NetworkSpec(
                network_id=f"network_{seed:05d}",
                seed=seed,
                genome_dict=genome.to_dict(),
                num_neurons=stats["num_neurons"],
                num_edges=stats["num_edges"],
                num_inputs=stats["num_inputs"],
                num_outputs=stats["num_outputs"],
                grid_size_x=self.grid_size_x,
                grid_size_y=self.grid_size_y,
                max_growth_steps=self.max_growth_steps,
                num_morphogens=self.num_morphogens,
                frozen_baseline=mean_reward,
                frozen_baseline_std=std_reward,
            )

            # Stratify
            if mean_reward < threshold_weak_mid and len(networks_weak) < n_weak:
                network_spec.performance_group = "weak"
                networks_weak.append(network_spec)
                successful += 1
                logger.info(
                    f"  Weak #{len(networks_weak)}/{n_weak}: "
                    f"seed={seed}, reward={mean_reward:.1f}, "
                    f"neurons={stats['num_neurons']}, edges={stats['num_edges']}"
                )
            elif (
                threshold_weak_mid <= mean_reward < threshold_mid_strong
                and len(networks_mid) < n_mid
            ):
                network_spec.performance_group = "mid"
                networks_mid.append(network_spec)
                successful += 1
                logger.info(
                    f"  Mid #{len(networks_mid)}/{n_mid}: "
                    f"seed={seed}, reward={mean_reward:.1f}, "
                    f"neurons={stats['num_neurons']}, edges={stats['num_edges']}"
                )
            elif mean_reward >= threshold_mid_strong and len(networks_strong) < n_strong:
                network_spec.performance_group = "strong"
                networks_strong.append(network_spec)
                successful += 1
                logger.info(
                    f"  Strong #{len(networks_strong)}/{n_strong}: "
                    f"seed={seed}, reward={mean_reward:.1f}, "
                    f"neurons={stats['num_neurons']}, edges={stats['num_edges']}"
                )

        elapsed = time.time() - start_time
        logger.info(
            f"Generated {successful} networks in {elapsed:.1f}s "
            f"({attempts} attempts, {successful/attempts*100:.1f}% success rate)"
        )

        # Combine and return
        all_networks = networks_weak + networks_mid + networks_strong
        return all_networks

    def extract_network_features(
        self, network_spec: NetworkSpec
    ) -> NetworkFeatures:
        """
        Extract network features for regression analysis.

        Args:
            network_spec: Network specification

        Returns:
            NetworkFeatures object
        """
        # Reconstruct network to extract features
        genome = Genome.from_dict(network_spec.genome_dict)
        grid = Grid(genome)
        grid.run_simulation(verbose=False)
        G = grid.get_graph()

        # Basic topology
        num_neurons = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = num_edges / (num_neurons ** 2) if num_neurons > 0 else 0.0

        # Connectivity
        degrees = [G.degree(n) for n in G.nodes()]
        avg_degree = float(np.mean(degrees)) if degrees else 0.0
        max_degree = int(np.max(degrees)) if degrees else 0

        input_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
        output_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

        input_connectivity = (
            float(np.mean([G.out_degree(n) for n in input_nodes]))
            if input_nodes
            else 0.0
        )
        output_connectivity = (
            float(np.mean([G.in_degree(n) for n in output_nodes]))
            if output_nodes
            else 0.0
        )

        # Advanced features (optional)
        try:
            clustering_coefficient = float(nx.average_clustering(G.to_undirected()))
        except:
            clustering_coefficient = None

        try:
            if nx.is_weakly_connected(G):
                avg_path_length = float(nx.average_shortest_path_length(G))
            else:
                avg_path_length = None
        except:
            avg_path_length = None

        return NetworkFeatures(
            network_id=network_spec.network_id,
            num_neurons=num_neurons,
            num_edges=num_edges,
            density=density,
            avg_degree=avg_degree,
            max_degree=max_degree,
            input_connectivity=input_connectivity,
            output_connectivity=output_connectivity,
            frozen_baseline=network_spec.frozen_baseline,
            clustering_coefficient=clustering_coefficient,
            avg_path_length=avg_path_length,
        )

    def save_pool(
        self,
        networks: List[NetworkSpec],
        output_dir: Path,
    ) -> None:
        """
        Save network pool to disk.

        Args:
            networks: List of network specifications
            output_dir: Directory to save to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save networks
        networks_file = output_dir / "networks.json"
        networks_data = [n.to_dict() for n in networks]
        with open(networks_file, 'w') as f:
            json.dump(networks_data, f, indent=2)

        logger.info(f"Saved {len(networks)} networks to {networks_file}")

        # Save features
        features = [self.extract_network_features(n) for n in networks]
        features_file = output_dir / "network_features.json"
        features_data = [f.to_dict() for f in features]
        with open(features_file, 'w') as f:
            json.dump(features_data, f, indent=2)

        logger.info(f"Saved network features to {features_file}")

        # Save stratification summary
        stratification = {
            "total": len(networks),
            "weak": len([n for n in networks if n.performance_group == "weak"]),
            "mid": len([n for n in networks if n.performance_group == "mid"]),
            "strong": len([n for n in networks if n.performance_group == "strong"]),
            "frozen_baseline_stats": {
                "mean": float(np.mean([n.frozen_baseline for n in networks])),
                "std": float(np.std([n.frozen_baseline for n in networks])),
                "min": float(np.min([n.frozen_baseline for n in networks])),
                "max": float(np.max([n.frozen_baseline for n in networks])),
            },
        }

        stratification_file = output_dir / "stratification.json"
        with open(stratification_file, 'w') as f:
            json.dump(stratification, f, indent=2)

        logger.info(f"Saved stratification summary to {stratification_file}")

    @staticmethod
    def load_pool(pool_dir: Path) -> List[NetworkSpec]:
        """
        Load network pool from disk.

        Args:
            pool_dir: Directory containing network pool

        Returns:
            List of NetworkSpec objects
        """
        pool_dir = Path(pool_dir)
        networks_file = pool_dir / "networks.json"

        with open(networks_file, 'r') as f:
            networks_data = json.load(f)

        networks = [NetworkSpec.from_dict(n) for n in networks_data]
        logger.info(f"Loaded {len(networks)} networks from {networks_file}")

        return networks
