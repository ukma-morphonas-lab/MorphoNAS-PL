"""
Parameter sampling for B0.3 experiment.

Implements Latin Hypercube Sampling for Phase 1 and adaptive sampling
for Phase 2 based on Phase 1 results.
"""

import logging
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from sklearn.cluster import KMeans

from .config_schemas import (
    PlasticityConfig,
    PhaseResults,
    PLASTICITY_PARAM_BOUNDS,
    DEFAULT_PLASTICITY_PARAMS,
)

logger = logging.getLogger(__name__)


class ParameterSampler:
    """Generate plasticity parameter configurations."""

    def __init__(self, seed: int = 100):
        """
        Initialize parameter sampler.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_lhs_configs(
        self,
        n_samples: int = 25,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> List[PlasticityConfig]:
        """
        Generate parameter configurations using Latin Hypercube Sampling.

        Args:
            n_samples: Number of configurations to generate
            bounds: Parameter bounds (uses default if None)

        Returns:
            List of PlasticityConfig objects
        """
        if bounds is None:
            bounds = PLASTICITY_PARAM_BOUNDS

        logger.info(f"Generating {n_samples} LHS configurations...")

        # Extract parameter names and bounds
        param_names = list(bounds.keys())
        lower_bounds = np.array([bounds[p][0] for p in param_names])
        upper_bounds = np.array([bounds[p][1] for p in param_names])

        # Generate LHS samples in [0, 1]
        sampler = qmc.LatinHypercube(d=len(param_names), seed=self.seed)
        unit_samples = sampler.random(n=n_samples)

        # Transform to parameter bounds
        # Special handling for log-scale parameter (eta)
        configs = []
        for i, unit_sample in enumerate(unit_samples):
            params = {}
            for j, param_name in enumerate(param_names):
                if param_name == "eta":
                    # Log scale for eta
                    log_lower = np.log10(bounds["eta"][0])
                    log_upper = np.log10(bounds["eta"][1])
                    log_value = log_lower + unit_sample[j] * (log_upper - log_lower)
                    params["eta"] = 10 ** log_value
                else:
                    # Linear scale for other parameters
                    params[param_name] = (
                        lower_bounds[j] + unit_sample[j] * (upper_bounds[j] - lower_bounds[j])
                    )

            config = PlasticityConfig(
                **params,
                config_name=f"lhs_{i+1:03d}",
            )
            configs.append(config)

        logger.info(f"Generated {len(configs)} LHS configurations")
        return configs

    def generate_ablation_configs(
        self,
        baseline_params: Optional[Dict[str, float]] = None,
        variation_levels: int = 3,
    ) -> List[PlasticityConfig]:
        """
        Generate ablation configurations (one-factor-at-a-time).

        Args:
            baseline_params: Baseline parameter values (uses default if None)
            variation_levels: Number of variation levels per parameter

        Returns:
            List of PlasticityConfig objects (1 baseline + 7 params × variation_levels)
        """
        if baseline_params is None:
            baseline_params = DEFAULT_PLASTICITY_PARAMS

        logger.info(f"Generating ablation configurations...")

        configs = []

        # Baseline configuration
        baseline_config = PlasticityConfig(
            **baseline_params,
            config_name="baseline",
        )
        configs.append(baseline_config)

        # One-at-a-time variations
        param_names = list(PLASTICITY_PARAM_BOUNDS.keys())

        for param_name in param_names:
            lower, upper = PLASTICITY_PARAM_BOUNDS[param_name]

            # Generate variation values
            if param_name == "eta":
                # Log scale for eta
                log_lower = np.log10(lower)
                log_upper = np.log10(upper)
                log_values = np.linspace(log_lower, log_upper, variation_levels + 2)[1:-1]
                values = 10 ** log_values
            else:
                # Linear scale
                values = np.linspace(lower, upper, variation_levels + 2)[1:-1]

            # Create configs
            for i, value in enumerate(values):
                params = baseline_params.copy()
                params[param_name] = float(value)

                config = PlasticityConfig(
                    **params,
                    config_name=f"ablation_{param_name}_{i+1}",
                )
                configs.append(config)

        logger.info(f"Generated {len(configs)} ablation configurations (1 baseline + {len(configs)-1} variations)")
        return configs

    def analyze_phase1_for_adaptive(
        self,
        phase1_results: PhaseResults,
        n_top_configs: int = 5,
        n_clusters: int = 3,
    ) -> Dict:
        """
        Analyze Phase 1 results to inform adaptive sampling.

        Args:
            phase1_results: Results from Phase 1
            n_top_configs: Number of top configurations to identify
            n_clusters: Number of network clusters to identify

        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing Phase 1 results for adaptive sampling...")

        # Get improvement matrix
        improvement_matrix = phase1_results.get_improvement_matrix()

        # Compute per-config mean improvements
        config_improvements = {}
        for config in phase1_results.configs:
            improvements = phase1_results.get_config_improvements(config.config_id)
            if improvements:
                config_improvements[config.config_id] = {
                    "mean": float(np.mean(improvements)),
                    "std": float(np.std(improvements)),
                    "median": float(np.median(improvements)),
                    "min": float(np.min(improvements)),
                    "max": float(np.max(improvements)),
                    "config": config,
                }

        # Sort by mean improvement
        sorted_configs = sorted(
            config_improvements.items(),
            key=lambda x: x[1]["mean"],
            reverse=True,
        )

        top_configs = [item[1]["config"] for item in sorted_configs[:n_top_configs]]
        top_config_ids = [c.config_id for c in top_configs]

        logger.info(f"Top {n_top_configs} configurations by mean improvement:")
        for i, (config_id, stats) in enumerate(sorted_configs[:n_top_configs]):
            logger.info(f"  #{i+1}: {config_id} - mean Δ = {stats['mean']:.1f}")

        # Cluster networks by optimal parameter preferences
        # Build feature matrix: network_id -> [best_config_params]
        network_best_params = []
        network_ids = []

        for network in phase1_results.networks:
            network_improvements = phase1_results.get_network_improvements(network.network_id)
            if not network_improvements:
                continue

            # Find best config for this network
            best_config_id = max(network_improvements, key=network_improvements.get)
            best_config = next(c for c in phase1_results.configs if c.config_id == best_config_id)

            # Extract parameters as feature vector
            params_vector = [
                best_config.theta_10,
                best_config.theta_11,
                best_config.theta_01,
                np.log10(best_config.eta),  # Log scale
                best_config.tau_e,
                best_config.tau_x,
                best_config.baseline_decay,
            ]

            network_best_params.append(params_vector)
            network_ids.append(network.network_id)

        # Cluster networks
        if len(network_best_params) >= n_clusters:
            X = np.array(network_best_params)
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            cluster_centers = kmeans.cluster_centers_

            logger.info(f"Identified {n_clusters} network clusters")
        else:
            cluster_labels = None
            cluster_centers = None
            logger.warning(f"Not enough networks for clustering (need >= {n_clusters})")

        # Analyze parameter variance
        # High variance params are good candidates for refinement
        param_variances = {}
        param_names = ["theta_10", "theta_11", "theta_01", "eta", "tau_e", "tau_x", "baseline_decay"]

        for i, param_name in enumerate(param_names):
            if param_name == "eta":
                values = [np.log10(c.eta) for c in phase1_results.configs]
            else:
                values = [getattr(c, param_name) for c in phase1_results.configs]

            # Weight by improvement
            weighted_values = []
            weights = []
            for config in phase1_results.configs:
                improvements = phase1_results.get_config_improvements(config.config_id)
                if improvements:
                    mean_improvement = np.mean(improvements)
                    if mean_improvement > 0:  # Only positive improvements
                        if param_name == "eta":
                            weighted_values.append(np.log10(config.eta))
                        else:
                            weighted_values.append(getattr(config, param_name))
                        weights.append(mean_improvement)

            if weighted_values:
                param_variances[param_name] = {
                    "variance": float(np.var(values)),
                    "weighted_mean": float(np.average(weighted_values, weights=weights)),
                    "weighted_std": float(np.sqrt(np.average((np.array(weighted_values) - np.average(weighted_values, weights=weights))**2, weights=weights))),
                }

        logger.info("Parameter variance analysis:")
        for param, stats in sorted(param_variances.items(), key=lambda x: x[1]["variance"], reverse=True):
            logger.info(f"  {param}: variance={stats['variance']:.4f}, weighted_mean={stats['weighted_mean']:.4f}")

        return {
            "top_configs": top_configs,
            "top_config_ids": top_config_ids,
            "config_improvements": config_improvements,
            "sorted_configs": sorted_configs,
            "network_clusters": {
                "labels": cluster_labels.tolist() if cluster_labels is not None else None,
                "centers": cluster_centers.tolist() if cluster_centers is not None else None,
                "network_ids": network_ids,
            },
            "param_variances": param_variances,
        }

    def generate_adaptive_configs(
        self,
        phase1_analysis: Dict,
        n_samples: int = 20,
        local_refinement_fraction: float = 0.3,
        network_targeting_fraction: float = 0.4,
        exploration_fraction: float = 0.3,
    ) -> List[PlasticityConfig]:
        """
        Generate adaptive configurations based on Phase 1 analysis.

        Args:
            phase1_analysis: Analysis results from analyze_phase1_for_adaptive()
            n_samples: Total number of adaptive configurations to generate
            local_refinement_fraction: Fraction for local refinement around top configs
            network_targeting_fraction: Fraction for network-specific targeting
            exploration_fraction: Fraction for exploration of high-variance regions

        Returns:
            List of PlasticityConfig objects
        """
        logger.info(f"Generating {n_samples} adaptive configurations...")

        n_local = int(n_samples * local_refinement_fraction)
        n_network = int(n_samples * network_targeting_fraction)
        n_explore = n_samples - n_local - n_network

        logger.info(f"  Local refinement: {n_local}")
        logger.info(f"  Network targeting: {n_network}")
        logger.info(f"  Exploration: {n_explore}")

        configs = []

        # 1. Local refinement around top configs
        top_configs = phase1_analysis["top_configs"]
        for i in range(n_local):
            # Pick a random top config
            base_config = self.rng.choice(top_configs)

            # Add small Gaussian noise
            params = {
                "theta_10": np.clip(base_config.theta_10 + self.rng.normal(0, 0.1), -1.0, 1.0),
                "theta_11": np.clip(base_config.theta_11 + self.rng.normal(0, 0.1), -1.0, 1.0),
                "theta_01": np.clip(base_config.theta_01 + self.rng.normal(0, 0.1), -1.0, 1.0),
                "eta": np.clip(
                    10 ** (np.log10(base_config.eta) + self.rng.normal(0, 0.3)),
                    1e-5, 1e-1
                ),
                "tau_e": np.clip(base_config.tau_e + self.rng.normal(0, 0.5), 0.1, 5.0),
                "tau_x": np.clip(base_config.tau_x + self.rng.normal(0, 0.02), 0.8, 0.99),
                "baseline_decay": np.clip(base_config.baseline_decay + self.rng.normal(0, 0.1), 0.0, 0.99),
            }

            config = PlasticityConfig(
                **params,
                config_name=f"adaptive_local_{i+1:03d}",
            )
            configs.append(config)

        # 2. Network-specific targeting (if clusters available)
        if phase1_analysis["network_clusters"]["centers"] is not None:
            cluster_centers = np.array(phase1_analysis["network_clusters"]["centers"])
            param_names = ["theta_10", "theta_11", "theta_01", "eta", "tau_e", "tau_x", "baseline_decay"]

            for i in range(n_network):
                # Pick a random cluster center
                center = cluster_centers[i % len(cluster_centers)]

                # Convert back from feature space
                params = {}
                for j, param_name in enumerate(param_names):
                    if param_name == "eta":
                        # Convert from log scale
                        params["eta"] = np.clip(10 ** center[j], 1e-5, 1e-1)
                    else:
                        lower, upper = PLASTICITY_PARAM_BOUNDS[param_name]
                        params[param_name] = np.clip(float(center[j]), lower, upper)

                config = PlasticityConfig(
                    **params,
                    config_name=f"adaptive_network_{i+1:03d}",
                )
                configs.append(config)
        else:
            # Fallback: sample uniformly
            logger.warning("No clusters available, using uniform sampling for network targeting")
            fallback_configs = self.generate_lhs_configs(n_samples=n_network)
            for i, config in enumerate(fallback_configs):
                config.config_name = f"adaptive_network_{i+1:03d}"
                configs.append(config)

        # 3. Exploration of high-variance regions
        param_variances = phase1_analysis["param_variances"]
        # Sort params by variance (descending)
        high_var_params = sorted(param_variances.keys(), key=lambda p: param_variances[p]["variance"], reverse=True)[:3]

        logger.info(f"Exploring high-variance parameters: {high_var_params}")

        for i in range(n_explore):
            params = DEFAULT_PLASTICITY_PARAMS.copy()

            # Vary high-variance parameters
            for param_name in high_var_params:
                lower, upper = PLASTICITY_PARAM_BOUNDS[param_name]
                if param_name == "eta":
                    # Log scale
                    log_lower = np.log10(lower)
                    log_upper = np.log10(upper)
                    log_value = log_lower + self.rng.uniform() * (log_upper - log_lower)
                    params["eta"] = 10 ** log_value
                else:
                    params[param_name] = lower + self.rng.uniform() * (upper - lower)

            config = PlasticityConfig(
                **params,
                config_name=f"adaptive_explore_{i+1:03d}",
            )
            configs.append(config)

        logger.info(f"Generated {len(configs)} adaptive configurations")
        return configs

    def save_configs(self, configs: List[PlasticityConfig], output_path: Path) -> None:
        """Save configurations to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        configs_data = [c.to_dict() for c in configs]
        with open(output_path, 'w') as f:
            json.dump(configs_data, f, indent=2)

        logger.info(f"Saved {len(configs)} configurations to {output_path}")

    @staticmethod
    def load_configs(config_path: Path) -> List[PlasticityConfig]:
        """Load configurations from JSON file."""
        config_path = Path(config_path)

        with open(config_path, 'r') as f:
            configs_data = json.load(f)

        configs = [PlasticityConfig.from_dict(c) for c in configs_data]
        logger.info(f"Loaded {len(configs)} configurations from {config_path}")

        return configs
