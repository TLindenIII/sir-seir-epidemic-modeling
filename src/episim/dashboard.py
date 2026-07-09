from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from plotly.utils import PlotlyJSONEncoder

from .plotting import compartment_figure, compartment_figure_dict
from .simulation import SimulationResult, run_seir, run_sir
from .utils import SummaryMetrics, summarize_simulation

ModelName = Literal["SIR", "SEIR"]
PROFILE_ORDER = [
    "baseline",
    "step1_no_live_table",
    "step2_split_callbacks",
    "step3_lightweight_figure",
    "step4_clientside",
]
MODEL_COLUMNS = {
    "SIR": ["day", "Susceptible", "Infectious", "Recovered"],
    "SEIR": ["day", "Susceptible", "Exposed", "Infectious", "Recovered"],
}
METRIC_SPECS = [
    ("Peak infectious", "peak_infectious"),
    ("Peak day", "peak_day"),
    ("Final outbreak size", "final_outbreak_size"),
    ("Final outbreak share", "final_outbreak_share"),
    ("Time to extinction", "time_to_extinction"),
]
PARAMETER_LABELS = {
    "beta": "Beta: β",
    "gamma": "Gamma: γ",
    "sigma": "Sigma: σ",
    "r0": "Basic reproduction number: R₀",
    "intervention_day": "Intervention day: tᵢ",
    "intervention_strength": "Intervention strength: Δβ",
    "dt": "Time step: Δt",
    "days": "Simulation horizon: T",
}


@dataclass(frozen=True, slots=True)
class OptimizationProfile:
    name: str
    label: str
    slider_updatemode: str
    manual_table_refresh: bool
    split_live_and_tables: bool
    lightweight_figure: bool
    clientside_live: bool
    description: str


PROFILE_MAP = {
    "baseline": OptimizationProfile(
        name="baseline",
        label="Baseline",
        slider_updatemode="drag",
        manual_table_refresh=False,
        split_live_and_tables=False,
        lightweight_figure=False,
        clientside_live=False,
        description="Single Python callback updates the graph, metrics, and tables on every drag tick.",
    ),
    "step1_no_live_table": OptimizationProfile(
        name="step1_no_live_table",
        label="Step 1",
        slider_updatemode="drag",
        manual_table_refresh=True,
        split_live_and_tables=False,
        lightweight_figure=False,
        clientside_live=False,
        description="Simulation tables leave the live path and refresh only when requested.",
    ),
    "step2_split_callbacks": OptimizationProfile(
        name="step2_split_callbacks",
        label="Step 2",
        slider_updatemode="mouseup",
        manual_table_refresh=False,
        split_live_and_tables=True,
        lightweight_figure=False,
        clientside_live=False,
        description="Live graph and metrics read drag_value while heavy tables refresh on release.",
    ),
    "step3_lightweight_figure": OptimizationProfile(
        name="step3_lightweight_figure",
        label="Step 3",
        slider_updatemode="mouseup",
        manual_table_refresh=False,
        split_live_and_tables=True,
        lightweight_figure=True,
        clientside_live=False,
        description="The live path uses a lighter figure payload and fewer points per redraw.",
    ),
    "step4_clientside": OptimizationProfile(
        name="step4_clientside",
        label="Step 4",
        slider_updatemode="mouseup",
        manual_table_refresh=False,
        split_live_and_tables=True,
        lightweight_figure=True,
        clientside_live=True,
        description="Live graph and metric updates run entirely in the browser; Python handles release-only tables.",
    ),
}


def get_profile(name: str | None = None) -> OptimizationProfile:
    if name is None:
        return PROFILE_MAP["step4_clientside"]
    if name not in PROFILE_MAP:
        raise ValueError(f"Unknown optimization profile: {name}")
    return PROFILE_MAP[name]


def run_model(model_name: ModelName, **params) -> SimulationResult:
    if model_name == "SIR":
        return run_sir(
            population=int(params["population"]),
            initial_infected=int(params["initial_infected"]),
            beta=float(params["beta"]),
            gamma=float(params["gamma"]),
            days=float(params["days"]),
            dt=0.25,
            intervention_day=float(params["intervention_day"]),
            intervention_strength=float(params["intervention_strength"]),
        )
    return run_seir(
        population=int(params["population"]),
        initial_infected=int(params["initial_infected"]),
        initial_exposed=int(params["initial_exposed"]),
        beta=float(params["beta"]),
        sigma=float(params["sigma"]),
        gamma=float(params["gamma"]),
        days=float(params["days"]),
        dt=0.25,
        intervention_day=float(params["intervention_day"]),
        intervention_strength=float(params["intervention_strength"]),
    )


def metric_strings(summary: SummaryMetrics) -> tuple[str, str, str, str, str]:
    return (
        f"{summary.peak_infectious:,.0f}",
        f"{summary.peak_day:.1f}",
        f"{summary.final_outbreak_size:,.0f}",
        f"{summary.final_outbreak_share:.1%}",
        (
            f"{summary.time_to_extinction:.1f} days"
            if summary.time_to_extinction is not None
            else "Not reached"
        ),
    )


def parameter_rows(parameters: dict[str, float]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, value in parameters.items():
        label = PARAMETER_LABELS.get(name, name.replace("_", " ").capitalize())
        if pd.isna(value):
            display = "None"
        elif name in {"beta", "gamma", "sigma", "intervention_strength", "dt"}:
            display = f"{value:.2f}"
        elif name in {"days", "intervention_day"}:
            display = f"{value:.0f}"
        elif name == "r0":
            display = f"{value:.2f}"
        else:
            display = str(value)
        rows.append({"Parameter": label, "Value": display})
    return rows


def result_records(result: SimulationResult, digits: int = 2) -> list[dict[str, float]]:
    return result.dataframe.round(digits).to_dict("records")


def live_figure_payload(
    result: SimulationResult,
    *,
    lightweight: bool,
) -> dict:
    if lightweight:
        return compartment_figure_dict(result, max_points=181, use_webgl=True)
    return compartment_figure(result).to_plotly_json()


def live_bundle(
    model_name: ModelName,
    *,
    lightweight: bool,
    **params,
) -> dict:
    result = run_model(model_name, **params)
    summary = summarize_simulation(result)
    return {
        "figure": live_figure_payload(result, lightweight=lightweight),
        "metrics": metric_strings(summary),
        "parameters": parameter_rows(result.parameters),
        "simulation": result_records(result),
    }


def live_only_bundle(
    model_name: ModelName,
    *,
    lightweight: bool,
    **params,
) -> dict:
    result = run_model(model_name, **params)
    summary = summarize_simulation(result)
    return {
        "figure": live_figure_payload(result, lightweight=lightweight),
        "metrics": metric_strings(summary),
    }


def table_bundle(model_name: ModelName, **params) -> dict:
    result = run_model(model_name, **params)
    return {
        "parameters": parameter_rows(result.parameters),
        "simulation": result_records(result),
    }


def payload_size_bytes(payload) -> int:
    return len(PlotlyJSONEncoder().encode(payload).encode("utf-8"))
