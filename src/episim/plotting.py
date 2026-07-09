from __future__ import annotations

from collections.abc import Mapping

import plotly.graph_objects as go

from .simulation import SimulationResult
from .utils import sensitivity_grid, summarize_simulation


COLOR_MAP = {
    "Susceptible": "#1d4ed8",
    "Exposed": "#f59e0b",
    "Infectious": "#dc2626",
    "Recovered": "#16a34a",
}

SCENARIO_COLORS = {
    "No intervention": "#b91c1c",
    "Weak intervention": "#ea580c",
    "Moderate intervention": "#0284c7",
    "Strong intervention": "#0f766e",
}


def _figure_layout(title: str, height: int = 500) -> dict:
    return {
        "template": "plotly_white",
        "height": height,
        "margin": {"l": 20, "r": 20, "t": 56, "b": 20},
        "title": {"text": title, "x": 0.02},
        "paper_bgcolor": "#fffdf7",
        "plot_bgcolor": "#fffdf7",
        "font": {"family": "IBM Plex Sans, sans-serif", "color": "#172033"},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
        "xaxis": {"gridcolor": "rgba(15, 23, 42, 0.08)", "zeroline": False},
        "yaxis": {"gridcolor": "rgba(15, 23, 42, 0.08)", "zeroline": False},
    }


def compartment_figure(result: SimulationResult) -> go.Figure:
    figure = go.Figure()
    for name, values in result.compartments.items():
        figure.add_trace(
            go.Scatter(
                x=result.time,
                y=values,
                mode="lines",
                name=name,
                line={"width": 3, "color": COLOR_MAP.get(name, "#334155")},
            )
        )

    summary = summarize_simulation(result)
    figure.add_vline(
        x=summary.peak_day,
        line_dash="dash",
        line_color="#0f172a",
        annotation_text=f"Peak day {summary.peak_day:.1f}",
    )
    figure.update_layout(**_figure_layout(f"{result.model_name} compartment dynamics"))
    figure.update_xaxes(title="Day")
    figure.update_yaxes(title="People")
    return figure


def scenario_comparison_figure(
    scenarios: Mapping[str, SimulationResult],
) -> go.Figure:
    figure = go.Figure()
    for label, result in scenarios.items():
        figure.add_trace(
            go.Scatter(
                x=result.time,
                y=result.compartments["Infectious"],
                mode="lines",
                name=label,
                line={"width": 3, "color": SCENARIO_COLORS.get(label, "#475569")},
            )
        )

    figure.update_layout(**_figure_layout("Intervention scenario comparison"))
    figure.update_xaxes(title="Day")
    figure.update_yaxes(title="Infectious people")
    return figure


def sensitivity_heatmap(
    model_name: str,
    population: int,
    initial_infected: int,
    gamma: float,
    sigma: float,
    metric: str,
    beta_values,
    secondary_values,
):
    frame = sensitivity_grid(
        model_name=model_name,
        beta_values=beta_values,
        secondary_values=secondary_values,
        population=population,
        initial_infected=initial_infected,
        gamma=gamma,
        sigma=sigma,
    )
    secondary_label = "gamma" if model_name == "SIR" else "sigma"
    pivot = frame.pivot(index=secondary_label, columns="beta", values=metric)
    figure = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Sunset",
            colorbar={"title": metric.replace("_", " ").title()},
        )
    )
    figure.update_layout(**_figure_layout(f"{model_name} sensitivity heatmap", height=460))
    figure.update_xaxes(title="beta")
    figure.update_yaxes(title=secondary_label)
    return figure
