"""
B0.6 Dose-Response Experiment Module

Measures whether giving plasticity more time after the switch improves
Phase 2 performance. Tests plasticity OFF→ON at multiple switch times
(100, 200, 300, 400) on the anti-Hebbian range.

This is a thin wrapper around the Exp 1 adaptation module.
"""

from __future__ import annotations

from MorphoNAS_PL.experimentB0_6_adaptation import (
    evaluate_network_adaptation,
    flatten_result_to_row,
    set_shutdown_event,
)

# Re-export for runner script
__all__ = [
    "ETA_VALUES_DOSE",
    "DECAY_VALUES_DOSE",
    "SWITCH_TIMES",
    "VARIANT_DOSE",
    "evaluate_network_dose_response",
    "flatten_result_to_row",
    "set_shutdown_event",
]

# ── Dose-response specific grid ─────────────────────────────────────

# Anti-Hebbian range + control
ETA_VALUES_DOSE = [-0.20, -0.10, -0.05, -0.03, -0.01, 0.0]

# Single decay value (known sweet spot)
DECAY_VALUES_DOSE = [0.01]

# Switch times to test
SWITCH_TIMES = [100, 200, 300, 400]

# Single variant (cheaper; gravity_2x can follow if results are promising)
VARIANT_DOSE = "heavy_pole"


def evaluate_network_dose_response(
    *,
    genome,
    metadata: dict,
    eta: float,
    decay: float,
    switch_step: int,
    rollouts: int = 20,
) -> dict:
    """Evaluate one (network, eta, decay, switch_step) for dose-response.

    Always uses plasticity_mode='off_on' and variant='heavy_pole'.
    """
    return evaluate_network_adaptation(
        genome=genome,
        metadata=metadata,
        eta=eta,
        decay=decay,
        variant=VARIANT_DOSE,
        switch_step=switch_step,
        rollouts=rollouts,
        plasticity_mode="off_on",
    )
