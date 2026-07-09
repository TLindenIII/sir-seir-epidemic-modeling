from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go

from .simulation import SimulationResult
from .utils import sensitivity_grid, summarize_simulation


COLOR_MAP = {
    "Susceptible": "#f4efe8",
    "Exposed": "#ff9f43",
    "Infectious": "#ff3b4d",
    "Recovered": "#5ed6ff",
}

SCENARIO_COLORS = {
    "No intervention": "#ff3b4d",
    "Weak intervention": "#ff7a36",
    "Moderate intervention": "#ffd166",
    "Strong intervention": "#5ed6ff",
}


def _figure_layout(title: str, height: int = 500) -> dict:
    return {
        "template": "plotly_dark",
        "height": height,
        "margin": {"l": 20, "r": 20, "t": 92, "b": 24},
        "title": {"text": title, "x": 0.02, "y": 0.97, "pad": {"b": 22}},
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "font": {"family": "IBM Plex Sans, sans-serif", "color": "#f5f0ea"},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0.0,
        },
        "xaxis": {"gridcolor": "rgba(255, 255, 255, 0.08)", "zeroline": False},
        "yaxis": {"gridcolor": "rgba(255, 255, 255, 0.08)", "zeroline": False},
    }


def _downsample_indices(length: int, max_points: int | None) -> np.ndarray:
    if max_points is None or length <= max_points:
        return np.arange(length)

    step = int(np.ceil(length / max_points))
    indices = np.arange(0, length, step, dtype=int)
    if indices[-1] != length - 1:
        indices = np.append(indices, length - 1)
    return indices


def compartment_figure_dict(
    result: SimulationResult,
    *,
    max_points: int | None = None,
    use_webgl: bool = False,
) -> dict:
    indices = _downsample_indices(len(result.time), max_points)
    summary = summarize_simulation(result)
    trace_type = "scattergl" if use_webgl else "scatter"
    data = []

    for name, values in result.compartments.items():
        data.append(
            {
                "type": trace_type,
                "x": result.time[indices].tolist(),
                "y": values[indices].tolist(),
                "mode": "lines",
                "name": name,
                "line": {"width": 3, "color": COLOR_MAP.get(name, "#e2d8cd")},
            }
        )

    layout = _figure_layout(f"{result.model_name} compartment dynamics")
    layout["xaxis"]["title"] = "Day"
    layout["yaxis"]["title"] = "People"
    layout["shapes"] = [
        {
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": summary.peak_day,
            "x1": summary.peak_day,
            "y0": 0,
            "y1": 1,
            "line": {"dash": "dash", "color": "#f4efe8", "width": 1},
        }
    ]
    layout["annotations"] = [
        {
            "x": summary.peak_day,
            "y": 1.03,
            "xref": "x",
            "yref": "paper",
            "text": f"Peak day {summary.peak_day:.1f}",
            "showarrow": False,
            "font": {"size": 12, "color": "#f4efe8"},
        }
    ]
    return {"data": data, "layout": layout}


def compartment_figure(result: SimulationResult) -> go.Figure:
    return go.Figure(compartment_figure_dict(result))


def scenario_comparison_figure(
    scenarios: Mapping[str, SimulationResult],
) -> go.Figure:
    figure = go.Figure()
    for label, result in scenarios.items():
        color_key = next(
            (name for name in SCENARIO_COLORS if label.startswith(name)),
            None,
        )
        figure.add_trace(
            go.Scatter(
                x=result.time,
                y=result.compartments["Infectious"],
                mode="lines",
                name=label,
                line={
                    "width": 3,
                    "color": SCENARIO_COLORS.get(color_key or "", "#475569"),
                },
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
