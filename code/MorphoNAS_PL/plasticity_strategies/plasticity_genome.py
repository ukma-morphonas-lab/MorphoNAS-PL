"""
PlasticityGenome: Evolvable parameters for three-factor plasticity rules.

This module defines the genome for B1 meta-plasticity experiments, where
evolution discovers optimal plasticity parameters alongside network architecture.

Based on Maoutsa 2025 and Gerstner et al. 2018 three-factor learning rules.

Extended with theta_10 (pure Hebbian term) to provide a stronger learning signal
for rate-coded networks where fluctuation terms are near-zero.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import json


@dataclass
class PlasticityGenome:
    """
    Evolvable plasticity parameters for three-factor learning rules.

    The genome encodes parameters that control how synaptic weights change
    based on pre/post activity and reward signals.

    Parameters:
        theta_10: Pure Hebbian coefficient - scales pre-synaptic activity directly
        theta_11: Interaction coefficient - scales pre * fluctuation term
        theta_01: Fluctuation coefficient - scales postsynaptic fluctuation term
        eta: Learning rate for weight updates (log scale, typically 1e-5 to 1e-1)
        tau_e: Eligibility trace decay time constant (seconds)
        tau_x: Running average decay for postsynaptic activity (EMA coefficient)
        baseline_decay: EMA decay for reward baseline (0.0 = no baseline, 1.0 = frozen)

    The eligibility trace dynamics follow:
        de_ij/dt = H_θ(r_j, x_i) - e_ij/τ_e

    Where H_θ is:
        H_θ = θ₁₀ · r_j + θ₁₁ · r_j · (x̄_i - x_i) + θ₀₁ · (x̄_i - x_i)

    Weight updates occur at episode end:
        Δw_ij = η · e_ij · (R - R̄)
    """

    theta_10: float = 0.0  # NEW: pure Hebbian term (pre-activity only)
    theta_11: float = 0.0
    theta_01: float = 0.0
    eta: float = 1e-3  # Increased default from 1e-5
    tau_e: float = 1.0
    tau_x: float = 0.9
    baseline_decay: float = 0.9  # NEW: controls reward baseline adaptation

    # Extended theta range to allow stronger signals
    THETA_RANGE: tuple = field(default=(-1.0, 1.0), repr=False)
    ETA_LOG_RANGE: tuple = field(default=(-5, -1), repr=False)  # 1e-5 to 1e-1
    TAU_E_RANGE: tuple = field(default=(0.1, 5.0), repr=False)
    TAU_X_RANGE: tuple = field(default=(0.8, 0.99), repr=False)
    BASELINE_DECAY_RANGE: tuple = field(default=(0.0, 0.99), repr=False)

    THETA_MUTATION_STD: float = field(default=0.1, repr=False)  # Increased from 0.002
    ETA_LOG_MUTATION_STD: float = field(default=0.5, repr=False)
    TAU_E_MUTATION_STD: float = field(default=0.5, repr=False)
    TAU_X_MUTATION_STD: float = field(default=0.02, repr=False)
    BASELINE_DECAY_MUTATION_STD: float = field(default=0.1, repr=False)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        self.theta_10 = np.clip(self.theta_10, *self.THETA_RANGE)
        self.theta_11 = np.clip(self.theta_11, *self.THETA_RANGE)
        self.theta_01 = np.clip(self.theta_01, *self.THETA_RANGE)
        log_eta = np.log10(max(self.eta, 1e-10))
        log_eta = np.clip(log_eta, *self.ETA_LOG_RANGE)
        self.eta = 10**log_eta
        self.tau_e = np.clip(self.tau_e, *self.TAU_E_RANGE)
        self.tau_x = np.clip(self.tau_x, *self.TAU_X_RANGE)
        self.baseline_decay = np.clip(self.baseline_decay, *self.BASELINE_DECAY_RANGE)

    @classmethod
    def random(cls, rng: Optional[np.random.Generator] = None) -> "PlasticityGenome":
        if rng is None:
            rng = np.random.default_rng()

        return cls(
            theta_10=rng.uniform(*cls.THETA_RANGE),
            theta_11=rng.uniform(*cls.THETA_RANGE),
            theta_01=rng.uniform(*cls.THETA_RANGE),
            eta=10 ** rng.uniform(*cls.ETA_LOG_RANGE),
            tau_e=rng.uniform(*cls.TAU_E_RANGE),
            tau_x=rng.uniform(*cls.TAU_X_RANGE),
            baseline_decay=rng.uniform(*cls.BASELINE_DECAY_RANGE),
        )

    @classmethod
    def disabled(cls) -> "PlasticityGenome":
        return cls(
            theta_10=0.0,
            theta_11=0.0,
            theta_01=0.0,
            eta=0.0,
            tau_e=1.0,
            tau_x=0.9,
            baseline_decay=0.9,
        )

    def mutate(
        self,
        rng: Optional[np.random.Generator] = None,
        mutation_rate: float = 0.3,
    ) -> "PlasticityGenome":
        if rng is None:
            rng = np.random.default_rng()

        new_theta_10 = self.theta_10
        new_theta_11 = self.theta_11
        new_theta_01 = self.theta_01
        new_eta = self.eta
        new_tau_e = self.tau_e
        new_tau_x = self.tau_x
        new_baseline_decay = self.baseline_decay

        if rng.random() < mutation_rate:
            new_theta_10 = np.clip(
                self.theta_10 + rng.normal(0, self.THETA_MUTATION_STD),
                *self.THETA_RANGE,
            )

        if rng.random() < mutation_rate:
            new_theta_11 = np.clip(
                self.theta_11 + rng.normal(0, self.THETA_MUTATION_STD),
                *self.THETA_RANGE,
            )

        if rng.random() < mutation_rate:
            new_theta_01 = np.clip(
                self.theta_01 + rng.normal(0, self.THETA_MUTATION_STD),
                *self.THETA_RANGE,
            )

        if rng.random() < mutation_rate:
            log_eta = np.log10(self.eta)
            log_eta = np.clip(
                log_eta + rng.normal(0, self.ETA_LOG_MUTATION_STD),
                *self.ETA_LOG_RANGE,
            )
            new_eta = 10**log_eta

        if rng.random() < mutation_rate:
            new_tau_e = np.clip(
                self.tau_e + rng.normal(0, self.TAU_E_MUTATION_STD),
                *self.TAU_E_RANGE,
            )

        if rng.random() < mutation_rate:
            new_tau_x = np.clip(
                self.tau_x + rng.normal(0, self.TAU_X_MUTATION_STD),
                *self.TAU_X_RANGE,
            )

        if rng.random() < mutation_rate:
            new_baseline_decay = np.clip(
                self.baseline_decay + rng.normal(0, self.BASELINE_DECAY_MUTATION_STD),
                *self.BASELINE_DECAY_RANGE,
            )

        return PlasticityGenome(
            theta_10=new_theta_10,
            theta_11=new_theta_11,
            theta_01=new_theta_01,
            eta=new_eta,
            tau_e=new_tau_e,
            tau_x=new_tau_x,
            baseline_decay=new_baseline_decay,
        )

    @classmethod
    def crossover(
        cls,
        parent1: "PlasticityGenome",
        parent2: "PlasticityGenome",
        rng: Optional[np.random.Generator] = None,
    ) -> "PlasticityGenome":
        if rng is None:
            rng = np.random.default_rng()

        return cls(
            theta_10=parent1.theta_10 if rng.random() < 0.5 else parent2.theta_10,
            theta_11=parent1.theta_11 if rng.random() < 0.5 else parent2.theta_11,
            theta_01=parent1.theta_01 if rng.random() < 0.5 else parent2.theta_01,
            eta=parent1.eta if rng.random() < 0.5 else parent2.eta,
            tau_e=parent1.tau_e if rng.random() < 0.5 else parent2.tau_e,
            tau_x=parent1.tau_x if rng.random() < 0.5 else parent2.tau_x,
            baseline_decay=parent1.baseline_decay
            if rng.random() < 0.5
            else parent2.baseline_decay,
        )

    def to_dict(self) -> dict:
        return {
            "theta_10": float(self.theta_10),
            "theta_11": float(self.theta_11),
            "theta_01": float(self.theta_01),
            "eta": float(self.eta),
            "tau_e": float(self.tau_e),
            "tau_x": float(self.tau_x),
            "baseline_decay": float(self.baseline_decay),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlasticityGenome":
        return cls(
            theta_10=data.get("theta_10", 0.0),
            theta_11=data["theta_11"],
            theta_01=data["theta_01"],
            eta=data["eta"],
            tau_e=data["tau_e"],
            tau_x=data["tau_x"],
            baseline_decay=data.get("baseline_decay", 0.9),
        )

    def to_json(self, filepath: Optional[str] = None) -> Optional[str]:
        data = self.to_dict()
        if filepath:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            return None
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(
        cls, json_str: Optional[str] = None, filepath: Optional[str] = None
    ) -> "PlasticityGenome":
        if json_str is not None and filepath is not None:
            raise ValueError("Provide either json_str or filepath, not both")

        if filepath:
            with open(filepath, "r") as f:
                data = json.load(f)
        elif json_str is not None:
            data = json.loads(json_str)
        else:
            raise ValueError("Provide either json_str or filepath")

        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"PlasticityGenome("
            f"θ₁₀={self.theta_10:.4f}, "
            f"θ₁₁={self.theta_11:.4f}, "
            f"θ₀₁={self.theta_01:.4f}, "
            f"η={self.eta:.2e}, "
            f"τ_e={self.tau_e:.2f}, "
            f"τ_x={self.tau_x:.3f}, "
            f"baseline={self.baseline_decay:.2f})"
        )
