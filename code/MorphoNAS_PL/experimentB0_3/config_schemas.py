"""
Configuration schemas and dataclasses for B0.3 experiment.

Defines structured data types for plasticity configurations, network specifications,
evaluation results, and analysis outputs.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
import hashlib


@dataclass
class PlasticityConfig:
    """Plasticity parameter configuration."""

    # Hebbian coefficients
    theta_10: float  # Pure Hebbian term (pre-activity only)
    theta_11: float  # Interaction term (pre × fluctuation)
    theta_01: float  # Fluctuation-only term

    # Learning parameters
    eta: float  # Learning rate (log scale)

    # Time constants
    tau_e: float  # Eligibility trace decay (seconds)
    tau_x: float  # Running average decay (EMA coefficient)

    # Reward baseline
    baseline_decay: float  # Reward baseline adaptation

    # Optional metadata
    config_id: Optional[str] = None
    config_name: Optional[str] = None

    def __post_init__(self):
        """Generate config_id if not provided."""
        if self.config_id is None:
            self.config_id = self.generate_id()

    def generate_id(self) -> str:
        """Generate unique ID from parameter values."""
        params_str = f"{self.theta_10:.6f}_{self.theta_11:.6f}_{self.theta_01:.6f}_" \
                     f"{self.eta:.6e}_{self.tau_e:.3f}_{self.tau_x:.4f}_{self.baseline_decay:.4f}"
        return hashlib.md5(params_str.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlasticityConfig":
        """Create from dictionary."""
        return cls(**data)

    def to_genome_dict(self) -> Dict[str, float]:
        """Convert to PlasticityGenome-compatible dict."""
        return {
            "theta_10": self.theta_10,
            "theta_11": self.theta_11,
            "theta_01": self.theta_01,
            "eta": self.eta,
            "tau_e": self.tau_e,
            "tau_x": self.tau_x,
            "baseline_decay": self.baseline_decay,
        }


@dataclass
class NetworkSpec:
    """Network specification with genome and metadata."""

    network_id: str
    seed: int
    genome_dict: Dict[str, Any]

    # Network properties
    num_neurons: int
    num_edges: int
    num_inputs: int
    num_outputs: int

    # Grid properties
    grid_size_x: int
    grid_size_y: int
    max_growth_steps: int
    num_morphogens: int

    # Frozen baseline performance
    frozen_baseline: Optional[float] = None
    frozen_baseline_std: Optional[float] = None

    # Stratification group
    performance_group: Optional[str] = None  # "weak", "mid", "strong"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkSpec":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class NetworkFeatures:
    """Extracted network features for regression analysis."""

    network_id: str

    # Basic topology
    num_neurons: int
    num_edges: int
    density: float  # edges / (neurons^2)

    # Connectivity
    avg_degree: float
    max_degree: int
    input_connectivity: float  # avg edges from inputs
    output_connectivity: float  # avg edges to outputs

    # Performance
    frozen_baseline: float

    # Optional advanced features
    clustering_coefficient: Optional[float] = None
    avg_path_length: Optional[float] = None
    modularity: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkFeatures":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EvaluationResult:
    """Single network × config evaluation result."""

    network_id: str
    config_id: str

    # Frozen baseline (for this specific evaluation)
    frozen_mean: float
    frozen_std: float
    frozen_rewards: List[float]

    # Plastic performance
    plastic_mean: float
    plastic_std: float
    train_rewards: List[float]
    eval_rewards: List[float]

    # Improvement
    improvement: float  # plastic_mean - frozen_mean

    # Plasticity metrics
    plasticity_metrics: Dict[str, Any] = field(default_factory=dict)

    # Timing
    evaluation_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PhaseResults:
    """Results from a full phase (all networks × all configs)."""

    phase_name: str  # "phase1" or "phase2"

    # Configurations tested
    configs: List[PlasticityConfig]

    # Networks tested
    networks: List[NetworkSpec]

    # Results: network_id -> config_id -> EvaluationResult
    results: Dict[str, Dict[str, EvaluationResult]]

    # Metadata
    total_evaluations: int
    successful_evaluations: int
    failed_evaluations: int
    total_time: float

    # Configuration
    experiment_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "phase_name": self.phase_name,
            "configs": [c.to_dict() for c in self.configs],
            "networks": [n.to_dict() for n in self.networks],
            "results": {
                net_id: {
                    cfg_id: result.to_dict()
                    for cfg_id, result in configs.items()
                }
                for net_id, configs in self.results.items()
            },
            "total_evaluations": self.total_evaluations,
            "successful_evaluations": self.successful_evaluations,
            "failed_evaluations": self.failed_evaluations,
            "total_time": self.total_time,
            "experiment_config": self.experiment_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseResults":
        """Create from dictionary."""
        return cls(
            phase_name=data["phase_name"],
            configs=[PlasticityConfig.from_dict(c) for c in data["configs"]],
            networks=[NetworkSpec.from_dict(n) for n in data["networks"]],
            results={
                net_id: {
                    cfg_id: EvaluationResult.from_dict(result)
                    for cfg_id, result in configs.items()
                }
                for net_id, configs in data["results"].items()
            },
            total_evaluations=data["total_evaluations"],
            successful_evaluations=data["successful_evaluations"],
            failed_evaluations=data["failed_evaluations"],
            total_time=data["total_time"],
            experiment_config=data.get("experiment_config", {}),
        )

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PhaseResults":
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_improvement_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get network_id -> config_id -> improvement matrix."""
        return {
            net_id: {
                cfg_id: result.improvement
                for cfg_id, result in configs.items()
            }
            for net_id, configs in self.results.items()
        }

    def get_config_improvements(self, config_id: str) -> List[float]:
        """Get all improvements for a specific config across networks."""
        improvements = []
        for net_results in self.results.values():
            if config_id in net_results:
                improvements.append(net_results[config_id].improvement)
        return improvements

    def get_network_improvements(self, network_id: str) -> Dict[str, float]:
        """Get all improvements for a specific network across configs."""
        if network_id not in self.results:
            return {}
        return {
            cfg_id: result.improvement
            for cfg_id, result in self.results[network_id].items()
        }


@dataclass
class StatisticalTestResult:
    """Result from a statistical test."""

    test_name: str  # "paired_ttest", "wilcoxon", "cohens_d"
    config_id: str

    # Test statistic
    statistic: float
    p_value: Optional[float] = None

    # Effect size
    effect_size: Optional[float] = None  # Cohen's d
    effect_size_ci_lower: Optional[float] = None
    effect_size_ci_upper: Optional[float] = None

    # Sample info
    n_samples: int = 0

    # Additional info
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatisticalTestResult":
        """Create from dictionary."""
        return cls(**data)


# Parameter bounds for sampling
PLASTICITY_PARAM_BOUNDS = {
    "theta_10": (-1.0, 1.0),
    "theta_11": (-1.0, 1.0),
    "theta_01": (-1.0, 1.0),
    "eta": (1e-5, 1e-1),  # log scale
    "tau_e": (0.1, 5.0),
    "tau_x": (0.8, 0.99),
    "baseline_decay": (0.0, 0.99),
}

# Default plasticity parameters (baseline configuration)
DEFAULT_PLASTICITY_PARAMS = {
    "theta_10": 0.1,
    "theta_11": 0.1,
    "theta_01": 0.0,
    "eta": 1e-3,
    "tau_e": 1.0,
    "tau_x": 0.9,
    "baseline_decay": 0.9,
}
