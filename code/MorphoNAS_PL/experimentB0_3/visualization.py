"""
Visualization suite for B0.3 experiment results.

Generates publication-quality figures for mechanism characterization.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
import seaborn as sns

from .config_schemas import PhaseResults, NetworkFeatures

logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class Visualization:
    """Generate analysis figures."""

    def __init__(self, output_dir: Path):
        """
        Initialize visualization suite.

        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving figures to {self.output_dir}")

    def plot_improvement_heatmap(
        self,
        phase_results: PhaseResults,
        sort_by: str = "frozen_baseline",
    ) -> Path:
        """
        Plot heatmap of improvements: networks × configs.

        Args:
            phase_results: Results from a phase
            sort_by: How to sort networks ("frozen_baseline" or "best_improvement")

        Returns:
            Path to saved figure
        """
        logger.info("Generating improvement heatmap...")

        # Build improvement matrix
        network_ids = [n.network_id for n in phase_results.networks]
        config_ids = [c.config_id for c in phase_results.configs]

        matrix = np.zeros((len(network_ids), len(config_ids)))
        matrix[:] = np.nan

        for i, network_id in enumerate(network_ids):
            for j, config_id in enumerate(config_ids):
                if network_id in phase_results.results and config_id in phase_results.results[network_id]:
                    matrix[i, j] = phase_results.results[network_id][config_id].improvement

        # Sort networks
        if sort_by == "frozen_baseline":
            sort_indices = np.argsort([n.frozen_baseline for n in phase_results.networks])
        elif sort_by == "best_improvement":
            best_imps = [np.nanmax(matrix[i, :]) for i in range(len(network_ids))]
            sort_indices = np.argsort(best_imps)[::-1]
        else:
            sort_indices = np.arange(len(network_ids))

        matrix = matrix[sort_indices, :]
        sorted_networks = [phase_results.networks[i] for i in sort_indices]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot heatmap
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-50, vmax=50)

        # Set ticks
        ax.set_xticks(np.arange(len(config_ids)))
        ax.set_yticks(np.arange(len(network_ids)))
        ax.set_xticklabels([c.config_name or c.config_id[:8] for c in phase_results.configs], rotation=45, ha='right')
        ax.set_yticklabels([f"{n.network_id} ({n.frozen_baseline:.0f})" for n in sorted_networks])

        ax.set_xlabel("Configuration")
        ax.set_ylabel("Network (frozen baseline)")
        ax.set_title(f"Improvement Heatmap: {phase_results.phase_name}")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Improvement (Δ reward)")

        plt.tight_layout()

        output_path = self.output_dir / f"{phase_results.phase_name}_heatmap_improvement.pdf"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved heatmap to {output_path}")
        return output_path

    def plot_config_boxplots(
        self,
        phase_results: PhaseResults,
        top_k: int = 15,
    ) -> Path:
        """
        Plot box plots of improvement distribution per config.

        Args:
            phase_results: Results from a phase
            top_k: Show top-k configs by mean improvement

        Returns:
            Path to saved figure
        """
        logger.info("Generating config box plots...")

        # Compute mean improvements
        config_means = []
        for config in phase_results.configs:
            improvements = phase_results.get_config_improvements(config.config_id)
            if improvements:
                config_means.append((config, np.mean(improvements)))

        # Sort and take top-k
        config_means.sort(key=lambda x: x[1], reverse=True)
        top_configs = [c for c, _ in config_means[:top_k]]

        # Gather data
        data = []
        labels = []
        for config in top_configs:
            improvements = phase_results.get_config_improvements(config.config_id)
            data.append(improvements)
            labels.append(config.config_name or config.config_id[:8])

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # Color boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Improvement (Δ reward)")
        ax.set_title(f"Top {top_k} Configurations by Mean Improvement")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / f"{phase_results.phase_name}_boxplot_configs.pdf"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved box plots to {output_path}")
        return output_path

    def plot_baseline_vs_improvement(
        self,
        phase_results: PhaseResults,
    ) -> Path:
        """
        Scatter plot: frozen baseline vs best improvement.

        Args:
            phase_results: Results from a phase

        Returns:
            Path to saved figure
        """
        logger.info("Generating baseline vs improvement scatter...")

        baselines = []
        best_improvements = []
        groups = []

        for network in phase_results.networks:
            network_improvements = phase_results.get_network_improvements(network.network_id)
            if network_improvements:
                best_imp = max(network_improvements.values())
                baselines.append(network.frozen_baseline)
                best_improvements.append(best_imp)
                groups.append(network.performance_group or "unknown")

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot by group
        group_colors = {"weak": "red", "mid": "orange", "strong": "green", "unknown": "gray"}
        for group in set(groups):
            mask = np.array(groups) == group
            ax.scatter(
                np.array(baselines)[mask],
                np.array(best_improvements)[mask],
                c=group_colors.get(group, "gray"),
                label=group,
                alpha=0.6,
                s=50,
            )

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Frozen Baseline Reward")
        ax.set_ylabel("Best Improvement (Δ reward)")
        ax.set_title("Network Baseline vs Best Improvement")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / f"{phase_results.phase_name}_scatter_baseline_improvement.pdf"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved scatter plot to {output_path}")
        return output_path

    def plot_summary_report(
        self,
        phase_results: PhaseResults,
        aggregate_stats: Dict,
    ) -> Path:
        """
        Generate multi-panel summary figure.

        Args:
            phase_results: Results from a phase
            aggregate_stats: Aggregate statistics

        Returns:
            Path to saved figure
        """
        logger.info("Generating summary report figure...")

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Panel 1: Overall distribution
        ax1 = fig.add_subplot(gs[0, :])
        all_improvements = []
        for network in phase_results.networks:
            network_improvements = phase_results.get_network_improvements(network.network_id)
            all_improvements.extend(network_improvements.values())

        ax1.hist(all_improvements, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel("Improvement (Δ reward)")
        ax1.set_ylabel("Count")
        ax1.set_title("Distribution of All Improvements")
        ax1.grid(alpha=0.3)

        # Panel 2: Per-group means
        ax2 = fig.add_subplot(gs[1, 0])
        group_means = {}
        for network in phase_results.networks:
            group = network.performance_group or "unknown"
            network_improvements = phase_results.get_network_improvements(network.network_id)
            if network_improvements:
                best_imp = max(network_improvements.values())
                if group not in group_means:
                    group_means[group] = []
                group_means[group].append(best_imp)

        groups = list(group_means.keys())
        means = [np.mean(group_means[g]) for g in groups]
        stds = [np.std(group_means[g]) for g in groups]

        ax2.bar(groups, means, yerr=stds, capsize=5, alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_ylabel("Mean Best Improvement")
        ax2.set_title("Improvement by Network Group")
        ax2.grid(alpha=0.3, axis='y')

        # Panel 3: Config success rates
        ax3 = fig.add_subplot(gs[1, 1])
        top_configs = sorted(
            aggregate_stats['per_config'].items(),
            key=lambda x: x[1]['mean'],
            reverse=True
        )[:10]

        config_names = [stats['config_name'] or cid[:8] for cid, stats in top_configs]
        success_rates = [stats['success_rate'] * 100 for _, stats in top_configs]

        ax3.barh(config_names, success_rates, alpha=0.7)
        ax3.set_xlabel("Success Rate (%)")
        ax3.set_title("Top 10 Configs: Success Rate (Δ > 10)")
        ax3.grid(alpha=0.3, axis='x')

        # Panel 4: Network responsiveness
        ax4 = fig.add_subplot(gs[2, :])
        responsiveness = []
        frozen_baselines = []
        for network in phase_results.networks:
            network_improvements = phase_results.get_network_improvements(network.network_id)
            if network_improvements:
                resp = max(network_improvements.values()) - min(network_improvements.values())
                responsiveness.append(resp)
                frozen_baselines.append(network.frozen_baseline)

        ax4.scatter(frozen_baselines, responsiveness, alpha=0.6)
        ax4.set_xlabel("Frozen Baseline Reward")
        ax4.set_ylabel("Responsiveness (range of improvements)")
        ax4.set_title("Network Responsiveness to Parameter Variation")
        ax4.grid(alpha=0.3)

        plt.suptitle(f"{phase_results.phase_name} - Summary Report", fontsize=14, y=0.995)

        output_path = self.output_dir / f"{phase_results.phase_name}_summary_report.pdf"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved summary report to {output_path}")
        return output_path
