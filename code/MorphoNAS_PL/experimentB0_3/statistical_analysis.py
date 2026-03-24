"""
Statistical analysis for B0.3 experiment.

Implements paired t-tests, effect sizes, confidence intervals, and regression
analysis to rigorously characterize plasticity mechanism.
"""

import logging
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from .config_schemas import (
    PhaseResults,
    PlasticityConfig,
    NetworkSpec,
    NetworkFeatures,
    StatisticalTestResult,
)

logger = logging.getLogger(__name__)


class StatisticalAnalysis:
    """Statistical analysis of plasticity experiments."""

    def __init__(self, n_bootstrap: int = 1000, random_seed: int = 102):
        """
        Initialize statistical analysis.

        Args:
            n_bootstrap: Number of bootstrap iterations for CIs
            random_seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(random_seed)

    def compute_paired_tests(
        self,
        phase_results: PhaseResults,
    ) -> Dict[str, StatisticalTestResult]:
        """
        Compute paired statistical tests for each configuration.

        Compares frozen vs plastic performance within each network (paired test).

        Args:
            phase_results: Results from a phase

        Returns:
            Dict mapping config_id to StatisticalTestResult
        """
        logger.info("Computing paired statistical tests...")

        test_results = {}

        for config in phase_results.configs:
            config_id = config.config_id

            # Get improvements for this config across all networks
            improvements = []
            frozen_means = []
            plastic_means = []

            for network in phase_results.networks:
                network_results = phase_results.results.get(network.network_id, {})
                if config_id in network_results:
                    result = network_results[config_id]
                    improvements.append(result.improvement)
                    frozen_means.append(result.frozen_mean)
                    plastic_means.append(result.plastic_mean)

            if len(improvements) < 2:
                logger.warning(f"Not enough data for {config_id}, skipping")
                continue

            # Paired t-test (frozen vs plastic within-network)
            t_stat, p_value = stats.ttest_rel(plastic_means, frozen_means)

            # Check normality (Shapiro-Wilk test)
            if len(improvements) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(improvements)
                is_normal = shapiro_p > 0.05
            else:
                shapiro_stat, shapiro_p = None, None
                is_normal = None

            # If not normal, also compute Wilcoxon signed-rank test
            if is_normal is False and len(improvements) >= 5:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(improvements)
            else:
                wilcoxon_stat, wilcoxon_p = None, None

            # Cohen's d effect size
            mean_diff = float(np.mean(improvements))
            std_diff = float(np.std(improvements, ddof=1))
            cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

            # Bootstrap CI for Cohen's d
            d_ci_lower, d_ci_upper = self.bootstrap_cohens_d_ci(
                plastic_means, frozen_means
            )

            test_result = StatisticalTestResult(
                test_name="paired_ttest",
                config_id=config_id,
                statistic=float(t_stat),
                p_value=float(p_value),
                effect_size=cohens_d,
                effect_size_ci_lower=d_ci_lower,
                effect_size_ci_upper=d_ci_upper,
                n_samples=len(improvements),
                metadata={
                    "mean_improvement": mean_diff,
                    "std_improvement": std_diff,
                    "median_improvement": float(np.median(improvements)),
                    "shapiro_stat": float(shapiro_stat) if shapiro_stat is not None else None,
                    "shapiro_p": float(shapiro_p) if shapiro_p is not None else None,
                    "is_normal": is_normal,
                    "wilcoxon_stat": float(wilcoxon_stat) if wilcoxon_stat is not None else None,
                    "wilcoxon_p": float(wilcoxon_p) if wilcoxon_p is not None else None,
                },
            )

            test_results[config_id] = test_result

        logger.info(f"Computed paired tests for {len(test_results)} configurations")
        return test_results

    def bootstrap_cohens_d_ci(
        self,
        group1: List[float],
        group2: List[float],
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for Cohen's d.

        Args:
            group1: First group (plastic)
            group2: Second group (frozen)
            confidence: Confidence level

        Returns:
            (lower_bound, upper_bound)
        """
        group1 = np.array(group1)
        group2 = np.array(group2)
        n = len(group1)

        if n < 2:
            return (0.0, 0.0)

        bootstrap_ds = []

        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = self.rng.choice(n, size=n, replace=True)
            g1_boot = group1[indices]
            g2_boot = group2[indices]

            # Compute Cohen's d
            diff = g1_boot - g2_boot
            mean_diff = np.mean(diff)
            std_diff = np.std(diff, ddof=1)
            d = mean_diff / std_diff if std_diff > 0 else 0.0
            bootstrap_ds.append(d)

        # Compute percentile CI
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_ds, alpha / 2 * 100)
        upper = np.percentile(bootstrap_ds, (1 - alpha / 2) * 100)

        return (float(lower), float(upper))

    def compute_aggregate_statistics(
        self,
        phase_results: PhaseResults,
    ) -> Dict:
        """
        Compute aggregate statistics for configurations and networks.

        Args:
            phase_results: Results from a phase

        Returns:
            Dictionary with aggregate statistics
        """
        logger.info("Computing aggregate statistics...")

        stats_dict = {
            "per_config": {},
            "per_network": {},
            "overall": {},
        }

        # Per-config statistics
        for config in phase_results.configs:
            improvements = phase_results.get_config_improvements(config.config_id)

            if not improvements:
                continue

            improvements = np.array(improvements)

            stats_dict["per_config"][config.config_id] = {
                "config_name": config.config_name,
                "n_networks": len(improvements),
                "mean": float(np.mean(improvements)),
                "median": float(np.median(improvements)),
                "std": float(np.std(improvements)),
                "iqr": float(np.percentile(improvements, 75) - np.percentile(improvements, 25)),
                "min": float(np.min(improvements)),
                "max": float(np.max(improvements)),
                "p25": float(np.percentile(improvements, 25)),
                "p75": float(np.percentile(improvements, 75)),
                "success_rate": float(np.mean(improvements > 10)),  # % with Δ > 10
                "failure_rate": float(np.mean(improvements < -10)),  # % with Δ < -10
            }

        # Per-network statistics
        for network in phase_results.networks:
            network_improvements = phase_results.get_network_improvements(network.network_id)

            if not network_improvements:
                continue

            improvements = np.array(list(network_improvements.values()))
            config_ids = list(network_improvements.keys())

            best_config_id = max(network_improvements, key=network_improvements.get)
            best_improvement = network_improvements[best_config_id]

            stats_dict["per_network"][network.network_id] = {
                "frozen_baseline": network.frozen_baseline,
                "performance_group": network.performance_group,
                "n_configs_tested": len(network_improvements),
                "mean_improvement": float(np.mean(improvements)),
                "median_improvement": float(np.median(improvements)),
                "std_improvement": float(np.std(improvements)),
                "best_config_id": best_config_id,
                "best_improvement": float(best_improvement),
                "worst_improvement": float(np.min(improvements)),
                "responsiveness": float(np.max(improvements) - np.min(improvements)),  # Range
            }

        # Overall statistics
        all_improvements = []
        for network in phase_results.networks:
            network_improvements = phase_results.get_network_improvements(network.network_id)
            all_improvements.extend(network_improvements.values())

        if all_improvements:
            all_improvements = np.array(all_improvements)
            stats_dict["overall"] = {
                "total_evaluations": len(all_improvements),
                "mean_improvement": float(np.mean(all_improvements)),
                "median_improvement": float(np.median(all_improvements)),
                "std_improvement": float(np.std(all_improvements)),
                "success_rate": float(np.mean(all_improvements > 10)),
                "failure_rate": float(np.mean(all_improvements < -10)),
            }

        logger.info("Aggregate statistics computed")
        return stats_dict

    def analyze_parameter_importance(
        self,
        phase_results: PhaseResults,
        baseline_config_id: Optional[str] = None,
    ) -> Dict:
        """
        Analyze parameter importance using ablation results.

        Args:
            phase_results: Results from a phase
            baseline_config_id: ID of baseline config (auto-detect if None)

        Returns:
            Dictionary with parameter importance rankings
        """
        logger.info("Analyzing parameter importance...")

        # Find baseline config (look for config_name == "baseline")
        if baseline_config_id is None:
            baseline_config = next(
                (c for c in phase_results.configs if c.config_name == "baseline"),
                None
            )
            if baseline_config:
                baseline_config_id = baseline_config.config_id
            else:
                logger.warning("No baseline configuration found")
                return {}

        # Get baseline improvements
        baseline_improvements = phase_results.get_config_improvements(baseline_config_id)
        if not baseline_improvements:
            logger.warning(f"No data for baseline config {baseline_config_id}")
            return {}

        baseline_mean = np.mean(baseline_improvements)

        # Find ablation configs
        ablation_configs = [
            c for c in phase_results.configs
            if c.config_name and c.config_name.startswith("ablation_")
        ]

        param_importance = {}

        for config in ablation_configs:
            # Extract parameter name from config_name (e.g., "ablation_eta_1" -> "eta")
            parts = config.config_name.split("_")
            if len(parts) >= 3:
                param_name = parts[1]
            else:
                continue

            improvements = phase_results.get_config_improvements(config.config_id)
            if not improvements:
                continue

            mean_improvement = np.mean(improvements)
            delta_from_baseline = mean_improvement - baseline_mean

            if param_name not in param_importance:
                param_importance[param_name] = {
                    "variations": [],
                    "mean_delta": 0.0,
                    "max_delta": 0.0,
                }

            param_importance[param_name]["variations"].append({
                "config_id": config.config_id,
                "config_name": config.config_name,
                "mean_improvement": float(mean_improvement),
                "delta_from_baseline": float(delta_from_baseline),
            })

        # Compute aggregate importance per parameter
        for param_name, data in param_importance.items():
            deltas = [v["delta_from_baseline"] for v in data["variations"]]
            data["mean_delta"] = float(np.mean(np.abs(deltas)))
            data["max_delta"] = float(np.max(np.abs(deltas)))

        # Sort by mean absolute delta (descending)
        sorted_params = sorted(
            param_importance.items(),
            key=lambda x: x[1]["mean_delta"],
            reverse=True
        )

        result = {
            "baseline_config_id": baseline_config_id,
            "baseline_mean_improvement": float(baseline_mean),
            "parameter_importance": dict(sorted_params),
            "importance_ranking": [p[0] for p in sorted_params],
        }

        logger.info(f"Parameter importance ranking: {result['importance_ranking']}")
        return result

    def fit_regression_models(
        self,
        phase_results: PhaseResults,
        network_features: List[NetworkFeatures],
    ) -> Dict:
        """
        Fit regression models to predict improvement from network features.

        Args:
            phase_results: Results from a phase
            network_features: List of network features

        Returns:
            Dictionary with regression results
        """
        logger.info("Fitting regression models...")

        # Build feature matrix and target vector
        X_list = []
        y_list = []
        network_ids = []

        for network in phase_results.networks:
            # Get features for this network
            features = next(
                (f for f in network_features if f.network_id == network.network_id),
                None
            )
            if not features:
                continue

            # Get best improvement for this network
            network_improvements = phase_results.get_network_improvements(network.network_id)
            if not network_improvements:
                continue

            best_improvement = max(network_improvements.values())

            # Build feature vector
            feature_vector = [
                features.num_neurons,
                features.num_edges,
                features.density,
                features.avg_degree,
                features.max_degree,
                features.input_connectivity,
                features.output_connectivity,
                features.frozen_baseline,
            ]

            X_list.append(feature_vector)
            y_list.append(best_improvement)
            network_ids.append(network.network_id)

        if len(X_list) < 5:
            logger.warning("Not enough data for regression (need >= 5 networks)")
            return {}

        X = np.array(X_list)
        y = np.array(y_list)

        feature_names = [
            "num_neurons",
            "num_edges",
            "density",
            "avg_degree",
            "max_degree",
            "input_connectivity",
            "output_connectivity",
            "frozen_baseline",
        ]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit Ridge regression with cross-validation
        alphas = [0.1, 1.0, 10.0, 100.0]
        best_alpha = None
        best_score = -np.inf

        for alpha in alphas:
            model = Ridge(alpha=alpha)
            scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X)), scoring='r2')
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

        # Fit final model
        model = Ridge(alpha=best_alpha)
        model.fit(X_scaled, y)

        # Get R² score
        r2 = model.score(X_scaled, y)

        # Get coefficients
        coefficients = {}
        for i, name in enumerate(feature_names):
            coefficients[name] = {
                "coefficient": float(model.coef_[i]),
                "abs_coefficient": float(np.abs(model.coef_[i])),
            }

        # Sort by absolute coefficient (descending)
        sorted_coefficients = sorted(
            coefficients.items(),
            key=lambda x: x[1]["abs_coefficient"],
            reverse=True
        )

        result = {
            "model_type": "Ridge",
            "best_alpha": float(best_alpha),
            "r2_score": float(r2),
            "cv_r2_score": float(best_score),
            "n_samples": len(X),
            "coefficients": dict(sorted_coefficients),
            "feature_importance_ranking": [c[0] for c in sorted_coefficients],
            "intercept": float(model.intercept_),
        }

        logger.info(f"Regression R²={r2:.3f}, top features: {result['feature_importance_ranking'][:3]}")
        return result

    def save_analysis(self, analysis_dict: Dict, output_path: Path) -> None:
        """Save analysis results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def convert_to_serializable(obj):
            """Recursively convert numpy types and custom objects to JSON-serializable types."""
            # Handle numpy types
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                 np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle objects with to_dict method
            elif hasattr(obj, 'to_dict'):
                return convert_to_serializable(obj.to_dict())
            # Recursively handle dicts
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            # Recursively handle lists
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_dict = convert_to_serializable(analysis_dict)

        with open(output_path, 'w') as f:
            json.dump(serializable_dict, f, indent=2)

        logger.info(f"Saved analysis results to {output_path}")

    @staticmethod
    def load_analysis(analysis_path: Path) -> Dict:
        """Load analysis results from JSON file."""
        analysis_path = Path(analysis_path)

        with open(analysis_path, 'r') as f:
            data = json.load(f)

        logger.info(f"Loaded analysis results from {analysis_path}")
        return data
