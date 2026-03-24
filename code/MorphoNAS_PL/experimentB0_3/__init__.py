"""
B0.3: Comprehensive Mechanism Characterization

Definitive manual exploration of three-factor plasticity parameter space
to understand the mechanism before attempting evolution (B1+).

This module closes out the B0 baseline experiments with rigorous
characterization at scale:
- 100 networks stratified by performance
- 50+ parameter configurations (LHS + adaptive)
- Statistical rigor (paired tests, effect sizes, CIs)
"""

from .config_schemas import (
    PlasticityConfig,
    NetworkSpec,
    NetworkFeatures,
    EvaluationResult,
    PhaseResults,
)
from .network_pool import NetworkPool
from .parameter_sampler import ParameterSampler
from .evaluation_orchestrator import EvaluationOrchestrator
from .statistical_analysis import StatisticalAnalysis
from .visualization import Visualization

__all__ = [
    "PlasticityConfig",
    "NetworkSpec",
    "NetworkFeatures",
    "EvaluationResult",
    "PhaseResults",
    "NetworkPool",
    "ParameterSampler",
    "EvaluationOrchestrator",
    "StatisticalAnalysis",
    "Visualization",
]
