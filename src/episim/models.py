from __future__ import annotations

from collections.abc import Callable

import numpy as np

StateFn = Callable[[float, np.ndarray], np.ndarray]


def apply_intervention(
    beta: float,
    t: float,
    intervention_day: float | None = None,
    intervention_strength: float = 0.0,
) -> float:
    """Return the effective transmission rate at time t."""
    if intervention_day is None or t < intervention_day:
        return beta
    return beta * (1.0 - intervention_strength)


def sir_derivatives(
    t: float,
    state: np.ndarray,
    beta: float,
    gamma: float,
    population: float,
    intervention_day: float | None = None,
    intervention_strength: float = 0.0,
) -> np.ndarray:
    s, i, r = state
    effective_beta = apply_intervention(
        beta=beta,
        t=t,
        intervention_day=intervention_day,
        intervention_strength=intervention_strength,
    )
    force = effective_beta * s * i / population
    return np.array([-force, force - gamma * i, gamma * i], dtype=float)


def seir_derivatives(
    t: float,
    state: np.ndarray,
    beta: float,
    sigma: float,
    gamma: float,
    population: float,
    intervention_day: float | None = None,
    intervention_strength: float = 0.0,
) -> np.ndarray:
    s, e, i, r = state
    effective_beta = apply_intervention(
        beta=beta,
        t=t,
        intervention_day=intervention_day,
        intervention_strength=intervention_strength,
    )
    force = effective_beta * s * i / population
    return np.array(
        [-force, force - sigma * e, sigma * e - gamma * i, gamma * i],
        dtype=float,
    )
