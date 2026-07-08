from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .models import seir_derivatives, sir_derivatives


@dataclass(slots=True)
class SimulationResult:
    model_name: str
    time: np.ndarray
    compartments: dict[str, np.ndarray]
    parameters: dict[str, float]
    population: float

    @property
    def dataframe(self):
        import pandas as pd

        data = {"day": self.time}
        data.update(self.compartments)
        return pd.DataFrame(data)


def _rk4_integrate(
    derivative_fn: Callable[[float, np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    days: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if days <= 0:
        raise ValueError("days must be positive.")

    steps = int(np.ceil(days / dt))
    time = np.linspace(0.0, steps * dt, steps + 1, dtype=float)
    states = np.zeros((steps + 1, len(initial_state)), dtype=float)
    states[0] = initial_state.astype(float)

    for idx in range(steps):
        t = time[idx]
        y = states[idx]
        k1 = derivative_fn(t, y)
        k2 = derivative_fn(t + dt / 2.0, y + dt * k1 / 2.0)
        k3 = derivative_fn(t + dt / 2.0, y + dt * k2 / 2.0)
        k4 = derivative_fn(t + dt, y + dt * k3)
        next_state = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states[idx + 1] = np.clip(next_state, a_min=0.0, a_max=None)

    return time, states


def run_sir(
    population: int,
    initial_infected: int,
    initial_recovered: int = 0,
    beta: float = 0.3,
    gamma: float = 0.1,
    days: float = 160.0,
    dt: float = 0.25,
    intervention_day: float | None = None,
    intervention_strength: float = 0.0,
) -> SimulationResult:
    initial_susceptible = population - initial_infected - initial_recovered
    if initial_susceptible < 0:
        raise ValueError("Initial S + I + R must not exceed population.")

    initial_state = np.array(
        [initial_susceptible, initial_infected, initial_recovered], dtype=float
    )

    def derivative(t: float, state: np.ndarray) -> np.ndarray:
        return sir_derivatives(
            t=t,
            state=state,
            beta=beta,
            gamma=gamma,
            population=population,
            intervention_day=intervention_day,
            intervention_strength=intervention_strength,
        )

    time, states = _rk4_integrate(derivative, initial_state, days=days, dt=dt)
    return SimulationResult(
        model_name="SIR",
        time=time,
        compartments={
            "Susceptible": states[:, 0],
            "Infectious": states[:, 1],
            "Recovered": states[:, 2],
        },
        parameters={
            "beta": beta,
            "gamma": gamma,
            "r0": beta / gamma if gamma else np.inf,
            "intervention_day": float(intervention_day)
            if intervention_day is not None
            else np.nan,
            "intervention_strength": intervention_strength,
            "dt": dt,
            "days": days,
        },
        population=float(population),
    )


def run_seir(
    population: int,
    initial_infected: int,
    initial_exposed: int = 0,
    initial_recovered: int = 0,
    beta: float = 0.3,
    sigma: float = 0.2,
    gamma: float = 0.1,
    days: float = 160.0,
    dt: float = 0.25,
    intervention_day: float | None = None,
    intervention_strength: float = 0.0,
) -> SimulationResult:
    initial_susceptible = (
        population - initial_exposed - initial_infected - initial_recovered
    )
    if initial_susceptible < 0:
        raise ValueError("Initial S + E + I + R must not exceed population.")

    initial_state = np.array(
        [
            initial_susceptible,
            initial_exposed,
            initial_infected,
            initial_recovered,
        ],
        dtype=float,
    )

    def derivative(t: float, state: np.ndarray) -> np.ndarray:
        return seir_derivatives(
            t=t,
            state=state,
            beta=beta,
            sigma=sigma,
            gamma=gamma,
            population=population,
            intervention_day=intervention_day,
            intervention_strength=intervention_strength,
        )

    time, states = _rk4_integrate(derivative, initial_state, days=days, dt=dt)
    return SimulationResult(
        model_name="SEIR",
        time=time,
        compartments={
            "Susceptible": states[:, 0],
            "Exposed": states[:, 1],
            "Infectious": states[:, 2],
            "Recovered": states[:, 3],
        },
        parameters={
            "beta": beta,
            "sigma": sigma,
            "gamma": gamma,
            "r0": beta / gamma if gamma else np.inf,
            "intervention_day": float(intervention_day)
            if intervention_day is not None
            else np.nan,
            "intervention_strength": intervention_strength,
            "dt": dt,
            "days": days,
        },
        population=float(population),
    )
