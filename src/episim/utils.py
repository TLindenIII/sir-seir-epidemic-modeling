from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .simulation import SimulationResult, run_seir, run_sir


@dataclass(slots=True)
class SummaryMetrics:
    peak_infectious: float
    peak_day: float
    final_outbreak_size: float
    final_outbreak_share: float
    outbreak_duration: float
    time_to_extinction: float | None


def summarize_simulation(
    result: SimulationResult,
    extinction_threshold: float = 1.0,
) -> SummaryMetrics:
    infectious = result.compartments["Infectious"]
    susceptible = result.compartments["Susceptible"]
    peak_idx = int(np.argmax(infectious))
    remaining = np.where(infectious <= extinction_threshold)[0]
    extinction_idx = next((idx for idx in remaining if idx >= peak_idx), None)
    final_outbreak_size = float(result.population - susceptible[-1])
    return SummaryMetrics(
        peak_infectious=float(infectious[peak_idx]),
        peak_day=float(result.time[peak_idx]),
        final_outbreak_size=final_outbreak_size,
        final_outbreak_share=final_outbreak_size / result.population,
        outbreak_duration=float(result.time[-1]),
        time_to_extinction=(
            float(result.time[extinction_idx]) if extinction_idx is not None else None
        ),
    )


def compare_models(sir_result: SimulationResult, seir_result: SimulationResult) -> pd.DataFrame:
    sir_summary = summarize_simulation(sir_result)
    seir_summary = summarize_simulation(seir_result)
    return pd.DataFrame(
        [
            {
                "Metric": "Peak infectious",
                "SIR": sir_summary.peak_infectious,
                "SEIR": seir_summary.peak_infectious,
            },
            {
                "Metric": "Peak day",
                "SIR": sir_summary.peak_day,
                "SEIR": seir_summary.peak_day,
            },
            {
                "Metric": "Final outbreak share",
                "SIR": sir_summary.final_outbreak_share,
                "SEIR": seir_summary.final_outbreak_share,
            },
            {
                "Metric": "Time to extinction",
                "SIR": sir_summary.time_to_extinction,
                "SEIR": seir_summary.time_to_extinction,
            },
        ]
    )


def sensitivity_grid(
    model_name: str,
    beta_values: np.ndarray,
    secondary_values: np.ndarray,
    population: int,
    initial_infected: int,
    gamma: float,
    sigma: float = 0.2,
    days: float = 160.0,
    dt: float = 0.5,
) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for beta in beta_values:
        for secondary in secondary_values:
            if model_name == "SIR":
                result = run_sir(
                    population=population,
                    initial_infected=initial_infected,
                    beta=float(beta),
                    gamma=float(secondary),
                    days=days,
                    dt=dt,
                )
                label = "gamma"
            elif model_name == "SEIR":
                result = run_seir(
                    population=population,
                    initial_infected=initial_infected,
                    beta=float(beta),
                    sigma=float(secondary),
                    gamma=gamma,
                    days=days,
                    dt=dt,
                )
                label = "sigma"
            else:
                raise ValueError("model_name must be either 'SIR' or 'SEIR'.")

            summary = summarize_simulation(result)
            records.append(
                {
                    "beta": float(beta),
                    label: float(secondary),
                    "peak_infectious": summary.peak_infectious,
                    "peak_day": summary.peak_day,
                    "final_outbreak_share": summary.final_outbreak_share,
                }
            )
    return pd.DataFrame.from_records(records)
