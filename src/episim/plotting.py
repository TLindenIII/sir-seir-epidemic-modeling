from __future__ import annotations

import plotly.graph_objects as go

from .simulation import SimulationResult
from .utils import sensitivity_grid, summarize_simulation


COLOR_MAP = {
    "Susceptible": "#1d4ed8",
    "Exposed": "#f59e0b",
    "Infectious": "#dc2626",
    "Recovered": "#16a34a",
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
    figure.update_layout(
        template="plotly_white",
        height=500,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        title=f"{result.model_name} compartment dynamics",
        xaxis_title="Day",
        yaxis_title="People",
        legend_title="Compartment",
    )
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
            colorscale="YlOrRd",
            colorbar={"title": metric.replace("_", " ").title()},
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=450,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        title=f"{model_name} sensitivity heatmap",
        xaxis_title="beta",
        yaxis_title=secondary_label,
    )
    return figure
