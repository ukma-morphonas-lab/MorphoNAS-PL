"""
PolynomialPlasticityGenome: Evolvable polynomial plasticity parameters.

Implements the polynomial H_theta from Maoutsa 2025:
    H_theta = SUM_{0 <= k,l <= d} theta_{k,l} * pre^k * fluct^l

With d=2 (degree 2), we have 9 coefficients for H_theta.
Total parameters: 9 (polynomial) + 1 (eta) + 1 (tau_e) + 1 (tau_x) = 12

The polynomial allows learning of:
- Thresholding effects (quadratic terms)
- Saturation (higher-order terms)
- Complex pre/post interactions
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import json


@dataclass
class PolynomialPlasticityGenome:
    """
    Evolvable polynomial plasticity parameters following Maoutsa 2025.

    The genome encodes a degree-d polynomial over (pre, fluct) where:
    - pre: presynaptic firing rate r_j
    - fluct: postsynaptic fluctuation (x_bar_i - x_i)

    H_theta = SUM_{k,l=0}^{d} theta[k,l] * pre^k * fluct^l

    Parameters:
        theta: 2D array of polynomial coefficients, shape (d+1, d+1)
        eta: Learning rate for weight updates (log scale)
        tau_e: Eligibility trace decay time constant (seconds)
        tau_x: Running average decay for postsynaptic activity (EMA coefficient)

    For d=2 (default), theta is 3x3 = 9 coefficients:
        theta[0,0]: bias
        theta[1,0]: pre (pure Hebbian-like)
        theta[0,1]: fluct
        theta[1,1]: pre * fluct (classic Hebbian)
        theta[2,0]: pre^2
        theta[0,2]: fluct^2
        theta[2,1]: pre^2 * fluct
        theta[1,2]: pre * fluct^2
        theta[2,2]: pre^2 * fluct^2
    """

    degree: int = 2
    theta: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    eta: float = 1e-3
    tau_e: float = 1.0
    tau_x: float = 0.9

    THETA_RANGE: tuple = field(default=(-1.0, 1.0), repr=False)
    ETA_LOG_RANGE: tuple = field(default=(-5, -1), repr=False)
    TAU_E_RANGE: tuple = field(default=(0.1, 5.0), repr=False)
    TAU_X_RANGE: tuple = field(default=(0.8, 0.99), repr=False)

    THETA_MUTATION_STD: float = field(default=0.1, repr=False)
    ETA_LOG_MUTATION_STD: float = field(default=0.5, repr=False)
    TAU_E_MUTATION_STD: float = field(default=0.5, repr=False)
    TAU_X_MUTATION_STD: float = field(default=0.02, repr=False)

    def __post_init__(self):
        expected_size = self.degree + 1
        if isinstance(self.theta, list):
            self.theta = np.array(self.theta)
        if self.theta.shape != (expected_size, expected_size):
            self.theta = np.zeros((expected_size, expected_size))
        self._validate()

    def _validate(self):
        self.theta = np.clip(self.theta, *self.THETA_RANGE)
        log_eta = np.log10(max(self.eta, 1e-10))
        log_eta = np.clip(log_eta, *self.ETA_LOG_RANGE)
        self.eta = 10**log_eta
        self.tau_e = np.clip(self.tau_e, *self.TAU_E_RANGE)
        self.tau_x = np.clip(self.tau_x, *self.TAU_X_RANGE)

    @property
    def num_coefficients(self) -> int:
        """Number of polynomial coefficients."""
        return (self.degree + 1) ** 2

    @classmethod
    def random(
        cls, degree: int = 2, rng: Optional[np.random.Generator] = None
    ) -> "PolynomialPlasticityGenome":
        if rng is None:
            rng = np.random.default_rng()

        size = degree + 1
        theta = rng.uniform(*cls.THETA_RANGE, size=(size, size))

        return cls(
            degree=degree,
            theta=theta,
            eta=10 ** rng.uniform(*cls.ETA_LOG_RANGE),
            tau_e=rng.uniform(*cls.TAU_E_RANGE),
            tau_x=rng.uniform(*cls.TAU_X_RANGE),
        )

    @classmethod
    def disabled(cls, degree: int = 2) -> "PolynomialPlasticityGenome":
        """Create a genome with all zeros (no plasticity)."""
        size = degree + 1
        return cls(
            degree=degree,
            theta=np.zeros((size, size)),
            eta=0.0,
            tau_e=1.0,
            tau_x=0.9,
        )

    def mutate(
        self,
        rng: Optional[np.random.Generator] = None,
        mutation_rate: float = 0.3,
    ) -> "PolynomialPlasticityGenome":
        if rng is None:
            rng = np.random.default_rng()

        new_theta = self.theta.copy()
        new_eta = self.eta
        new_tau_e = self.tau_e
        new_tau_x = self.tau_x

        size = self.degree + 1
        for k in range(size):
            for l in range(size):
                if rng.random() < mutation_rate:
                    new_theta[k, l] = np.clip(
                        self.theta[k, l] + rng.normal(0, self.THETA_MUTATION_STD),
                        *self.THETA_RANGE,
                    )

        if rng.random() < mutation_rate:
            log_eta = np.log10(max(self.eta, 1e-10))
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

        return PolynomialPlasticityGenome(
            degree=self.degree,
            theta=new_theta,
            eta=new_eta,
            tau_e=new_tau_e,
            tau_x=new_tau_x,
        )

    @classmethod
    def crossover(
        cls,
        parent1: "PolynomialPlasticityGenome",
        parent2: "PolynomialPlasticityGenome",
        rng: Optional[np.random.Generator] = None,
    ) -> "PolynomialPlasticityGenome":
        if rng is None:
            rng = np.random.default_rng()

        assert parent1.degree == parent2.degree, "Parents must have same degree"

        size = parent1.degree + 1
        new_theta = np.zeros((size, size))

        for k in range(size):
            for l in range(size):
                new_theta[k, l] = (
                    parent1.theta[k, l] if rng.random() < 0.5 else parent2.theta[k, l]
                )

        return cls(
            degree=parent1.degree,
            theta=new_theta,
            eta=parent1.eta if rng.random() < 0.5 else parent2.eta,
            tau_e=parent1.tau_e if rng.random() < 0.5 else parent2.tau_e,
            tau_x=parent1.tau_x if rng.random() < 0.5 else parent2.tau_x,
        )

    def to_dict(self) -> dict:
        return {
            "degree": self.degree,
            "theta": self.theta.tolist(),
            "eta": float(self.eta),
            "tau_e": float(self.tau_e),
            "tau_x": float(self.tau_x),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolynomialPlasticityGenome":
        return cls(
            degree=data.get("degree", 2),
            theta=np.array(data["theta"]),
            eta=data["eta"],
            tau_e=data["tau_e"],
            tau_x=data["tau_x"],
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
    ) -> "PolynomialPlasticityGenome":
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
        t = self.theta
        return (
            f"PolynomialPlasticityGenome(d={self.degree}, "
            f"theta[1,0]={t[1, 0]:.3f}, theta[1,1]={t[1, 1]:.3f}, "
            f"eta={self.eta:.2e}, tau_e={self.tau_e:.2f})"
        )
