from __future__ import annotations

from functools import lru_cache

import pandas as pd
from dash import Dash, Input, Output, callback, dash_table, dcc, html

from episim.plotting import (
    compartment_figure,
    scenario_comparison_figure,
    sensitivity_heatmap,
)
from episim.simulation import run_seir, run_sir
from episim.utils import compare_models, summarize_simulation


EXTERNAL_STYLESHEETS = [
    (
        "https://fonts.googleapis.com/css2?"
        "family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap"
    )
]
GRAPH_CONFIG = {"displaylogo": False, "responsive": True}
ASSUMPTION_ROWS = [
    {"Question": "Has an incubation delay?", "SIR": "No", "SEIR": "Yes"},
    {"Question": "More realistic for latent spread?", "SIR": "Less", "SEIR": "More"},
    {"Question": "Easiest to explain?", "SIR": "Yes", "SEIR": "Slightly harder"},
    {"Question": "Additional parameter", "SIR": "None", "SEIR": "sigma"},
]


def slider_block(
    label: str,
    component_id: str,
    min_value: float,
    max_value: float,
    step: float,
    value: float,
) -> html.Div:
    return html.Div(
        className="control-block",
        children=[
            html.Div(label, className="control-label"),
            dcc.Slider(
                id=component_id,
                min=min_value,
                max=max_value,
                step=step,
                value=value,
                marks=None,
                updatemode="drag",
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
    )


def metric_cards(summary) -> list[html.Div]:
    items = [
        ("Peak infectious", f"{summary.peak_infectious:,.0f}"),
        ("Peak day", f"{summary.peak_day:.1f}"),
        ("Final outbreak size", f"{summary.final_outbreak_size:,.0f}"),
        ("Final outbreak share", f"{summary.final_outbreak_share:.1%}"),
        (
            "Time to extinction",
            f"{summary.time_to_extinction:.1f} days"
            if summary.time_to_extinction is not None
            else "Not reached",
        ),
    ]
    return [
        html.Div(
            className="metric-card",
            children=[
                html.Div(label, className="metric-label"),
                html.Div(value, className="metric-value"),
            ],
        )
        for label, value in items
    ]


def parameter_rows(parameters: dict[str, float]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, value in parameters.items():
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
        rows.append({"Parameter": name, "Value": display})
    return rows


def comparison_rows(frame: pd.DataFrame) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row in frame.to_dict("records"):
        formatted = {"Metric": row["Metric"]}
        for key in ("SIR", "SEIR"):
            value = row[key]
            if value is None or pd.isna(value):
                formatted[key] = "Not reached"
            elif "share" in row["Metric"].lower():
                formatted[key] = f"{value:.1%}"
            else:
                formatted[key] = f"{value:,.1f}"
        records.append(formatted)
    return records


def scenario_summary_rows(results: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, result in results.items():
        summary = summarize_simulation(result)
        rows.append(
            {
                "Scenario": name,
                "Peak infectious": f"{summary.peak_infectious:,.0f}",
                "Peak day": f"{summary.peak_day:.1f}",
                "Final outbreak share": f"{summary.final_outbreak_share:.1%}",
            }
        )
    return rows


def result_records(result, digits: int = 2) -> list[dict[str, float]]:
    frame = result.dataframe.round(digits)
    return frame.to_dict("records")


@lru_cache(maxsize=512)
def cached_sir_result(
    population: int,
    initial_infected: int,
    beta: float,
    gamma: float,
    days: int,
    intervention_day: int,
    intervention_strength: float,
) :
    return run_sir(
        population=population,
        initial_infected=initial_infected,
        beta=beta,
        gamma=gamma,
        days=float(days),
        dt=0.25,
        intervention_day=float(intervention_day),
        intervention_strength=intervention_strength,
    )


@lru_cache(maxsize=512)
def cached_seir_result(
    population: int,
    initial_infected: int,
    initial_exposed: int,
    beta: float,
    sigma: float,
    gamma: float,
    days: int,
    intervention_day: int,
    intervention_strength: float,
) :
    return run_seir(
        population=population,
        initial_infected=initial_infected,
        initial_exposed=initial_exposed,
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        days=float(days),
        dt=0.25,
        intervention_day=float(intervention_day),
        intervention_strength=intervention_strength,
    )


def data_table(component_id: str, columns: list[str], page_size: int = 8) -> dash_table.DataTable:
    return dash_table.DataTable(
        id=component_id,
        columns=[{"name": column, "id": column} for column in columns],
        data=[],
        page_size=page_size,
        sort_action="native",
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#f2efe6",
            "fontWeight": 600,
            "border": "none",
            "color": "#172033",
        },
        style_cell={
            "padding": "12px 14px",
            "backgroundColor": "#fffdf7",
            "border": "none",
            "color": "#314056",
            "fontFamily": "IBM Plex Sans, sans-serif",
            "textAlign": "left",
        },
    )


def build_home_tab() -> html.Div:
    sir = cached_sir_result(10_000, 10, 0.3, 0.1, 160, 30, 0.4)
    seir = cached_seir_result(10_000, 10, 20, 0.3, 0.2, 0.1, 160, 30, 0.4)
    comparison = comparison_rows(compare_models(sir, seir))
    return html.Div(
        className="tab-panel",
        children=[
            html.Div(
                className="hero-card",
                children=[
                    html.Div("Compartmental modeling, without dead UI.", className="eyebrow"),
                    html.H1(
                        "Interactive Epidemiological Modeling with SIR and SEIR Simulations",
                        className="hero-title",
                    ),
                    html.P(
                        "Adjust transmission, recovery, incubation, and interventions with live "
                        "slider feedback. The dashboard is built for explanation first, but the "
                        "simulation core is fast enough to stay fluid while you drag.",
                        className="hero-copy",
                    ),
                ],
            ),
            html.Div(
                className="home-grid",
                children=[
                    html.Div(
                        className="info-card",
                        children=[
                            html.H3("What this project demonstrates"),
                            html.Ul(
                                className="feature-list",
                                children=[
                                    html.Li("Deterministic SIR and SEIR outbreak simulation"),
                                    html.Li("Intervention scenarios that flatten and delay the peak"),
                                    html.Li("Sensitivity analysis across core epidemiological parameters"),
                                    html.Li("Assumption-aware comparison between SIR and SEIR dynamics"),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="info-card",
                        children=[
                            html.H3("Core equations"),
                            html.Div("dS/dt = -beta * S * I / N", className="equation-line"),
                            html.Div("dI/dt = beta * S * I / N - gamma * I", className="equation-line"),
                            html.Div("dR/dt = gamma * I", className="equation-line"),
                            html.Div("dE/dt = beta * S * I / N - sigma * E", className="equation-line"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="card-stack",
                children=[
                    html.H3("Baseline SIR vs SEIR"),
                    data_table("home-comparison-table", ["Metric", "SIR", "SEIR"], page_size=6),
                    dcc.Store(id="home-comparison-store", data=comparison),
                ],
            ),
        ],
    )


def build_model_tab(model_name: str) -> html.Div:
    prefix = model_name.lower()
    controls = [
        slider_block(f"{model_name} population", f"{prefix}-population", 1_000, 500_000, 1_000, 10_000),
        slider_block("Initial infected", f"{prefix}-infected", 1, 1_000, 1, 10),
        slider_block("Transmission rate (beta)", f"{prefix}-beta", 0.05, 1.0, 0.01, 0.30),
        slider_block("Recovery rate (gamma)", f"{prefix}-gamma", 0.02, 0.5, 0.01, 0.10),
        slider_block("Simulation horizon (days)", f"{prefix}-days", 30, 365, 1, 160),
        slider_block("Intervention day", f"{prefix}-intervention-day", 0, 180, 1, 30),
        slider_block(
            "Intervention strength",
            f"{prefix}-intervention-strength",
            0.0,
            0.9,
            0.05,
            0.4,
        ),
    ]
    if model_name == "SEIR":
        controls.insert(2, slider_block("Initial exposed", "seir-exposed", 0, 2_000, 1, 20))
        controls.insert(4, slider_block("Incubation rate (sigma)", "seir-sigma", 0.05, 1.0, 0.01, 0.20))

    columns = ["day", "Susceptible", "Infectious", "Recovered"]
    if model_name == "SEIR":
        columns = ["day", "Susceptible", "Exposed", "Infectious", "Recovered"]

    return html.Div(
        className="tab-panel split-layout",
        children=[
            html.Div(
                className="control-card",
                children=[
                    html.H2(f"{model_name} model"),
                    html.P(
                        "Each slider fires while you drag, so the graph and summary cards move "
                        "continuously instead of waiting for a release event.",
                        className="section-note",
                    ),
                    *controls,
                ],
            ),
            html.Div(
                className="content-stack",
                children=[
                    dcc.Graph(id=f"{prefix}-graph", config=GRAPH_CONFIG),
                    html.Div(id=f"{prefix}-metrics", className="metric-grid"),
                    html.Div(
                        className="table-card",
                        children=[
                            html.H3("Parameter summary"),
                            data_table(f"{prefix}-parameter-table", ["Parameter", "Value"], page_size=8),
                        ],
                    ),
                    html.Details(
                        className="details-card",
                        children=[
                            html.Summary("Simulation table"),
                            data_table(f"{prefix}-simulation-table", columns, page_size=12),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_scenario_tab() -> html.Div:
    return html.Div(
        className="tab-panel split-layout",
        children=[
            html.Div(
                className="control-card",
                children=[
                    html.H2("Scenario Simulator"),
                    html.P(
                        "Compare how intervention timing changes the infectious curve under the same "
                        "baseline outbreak conditions.",
                        className="section-note",
                    ),
                    slider_block("Population", "scenario-population", 1_000, 100_000, 1_000, 10_000),
                    slider_block("Initial infected", "scenario-infected", 1, 500, 1, 10),
                    slider_block("Transmission rate (beta)", "scenario-beta", 0.05, 1.0, 0.01, 0.30),
                    slider_block("Recovery rate (gamma)", "scenario-gamma", 0.02, 0.5, 0.01, 0.10),
                    slider_block("Intervention day", "scenario-day", 0, 120, 1, 30),
                ],
            ),
            html.Div(
                className="content-stack",
                children=[
                    dcc.Graph(id="scenario-graph", config=GRAPH_CONFIG),
                    html.Div(
                        className="table-card",
                        children=[
                            html.H3("Scenario summary"),
                            data_table(
                                "scenario-table",
                                ["Scenario", "Peak infectious", "Peak day", "Final outbreak share"],
                                page_size=6,
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_sensitivity_tab() -> html.Div:
    return html.Div(
        className="tab-panel split-layout",
        children=[
            html.Div(
                className="control-card",
                children=[
                    html.H2("Parameter Sensitivity"),
                    html.P(
                        "Heatmaps summarize how peak size, timing, and final outbreak share move "
                        "as model parameters change.",
                        className="section-note",
                    ),
                    html.Div(
                        className="control-block",
                        children=[
                            html.Div("Model", className="control-label"),
                            dcc.RadioItems(
                                id="sensitivity-model",
                                options=[
                                    {"label": "SIR", "value": "SIR"},
                                    {"label": "SEIR", "value": "SEIR"},
                                ],
                                value="SIR",
                                className="radio-grid",
                                inputClassName="radio-input",
                                labelClassName="radio-label",
                            ),
                        ],
                    ),
                    html.Div(
                        className="control-block",
                        children=[
                            html.Div("Metric", className="control-label"),
                            dcc.Dropdown(
                                id="sensitivity-metric",
                                options=[
                                    {"label": "Peak infectious", "value": "peak_infectious"},
                                    {"label": "Peak day", "value": "peak_day"},
                                    {"label": "Final outbreak share", "value": "final_outbreak_share"},
                                ],
                                value="peak_infectious",
                                clearable=False,
                            ),
                        ],
                    ),
                    slider_block("Population", "sensitivity-population", 1_000, 100_000, 1_000, 10_000),
                    slider_block("Initial infected", "sensitivity-infected", 1, 300, 1, 10),
                    slider_block("Recovery rate (gamma)", "sensitivity-gamma", 0.02, 0.5, 0.01, 0.10),
                    slider_block("Incubation rate (sigma)", "sensitivity-sigma", 0.05, 1.0, 0.01, 0.20),
                ],
            ),
            html.Div(
                className="content-stack",
                children=[
                    dcc.Graph(id="sensitivity-graph", config=GRAPH_CONFIG),
                    html.Div(
                        "For SIR, the vertical axis represents gamma. For SEIR, it represents sigma.",
                        className="mini-note",
                    ),
                ],
            ),
        ],
    )


def build_comparison_tab() -> html.Div:
    return html.Div(
        className="tab-panel split-layout",
        children=[
            html.Div(
                className="control-card",
                children=[
                    html.H2("SIR vs SEIR"),
                    html.P(
                        "Use shared parameters to compare how a latent compartment shifts the outbreak curve.",
                        className="section-note",
                    ),
                    slider_block("Population", "comparison-population", 1_000, 100_000, 1_000, 10_000),
                    slider_block("Initial infected", "comparison-infected", 1, 500, 1, 10),
                    slider_block("Initial exposed", "comparison-exposed", 0, 2_000, 1, 20),
                    slider_block("Transmission rate (beta)", "comparison-beta", 0.05, 1.0, 0.01, 0.30),
                    slider_block("Incubation rate (sigma)", "comparison-sigma", 0.05, 1.0, 0.01, 0.20),
                    slider_block("Recovery rate (gamma)", "comparison-gamma", 0.02, 0.5, 0.01, 0.10),
                    slider_block("Simulation horizon (days)", "comparison-days", 30, 365, 1, 160),
                ],
            ),
            html.Div(
                className="content-stack",
                children=[
                    html.Div(
                        className="two-up",
                        children=[
                            dcc.Graph(id="comparison-sir-graph", config=GRAPH_CONFIG),
                            dcc.Graph(id="comparison-seir-graph", config=GRAPH_CONFIG),
                        ],
                    ),
                    html.Div(
                        className="table-card",
                        children=[
                            html.H3("Metric comparison"),
                            data_table("comparison-table", ["Metric", "SIR", "SEIR"], page_size=6),
                        ],
                    ),
                    html.Div(
                        className="table-card",
                        children=[
                            html.H3("Assumption matrix"),
                            data_table("assumption-table", ["Question", "SIR", "SEIR"], page_size=6),
                            dcc.Store(id="assumption-store", data=ASSUMPTION_ROWS),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_about_tab() -> html.Div:
    return html.Div(
        className="tab-panel about-grid",
        children=[
            html.Div(
                className="info-card",
                children=[
                    html.H3("SIR intuition"),
                    html.P(
                        "SIR assumes people move directly from susceptible to infectious to recovered. "
                        "It is the cleanest model for explaining how beta and gamma compete."
                    ),
                    html.Div("R0 = beta / gamma", className="equation-line"),
                ],
            ),
            html.Div(
                className="info-card",
                children=[
                    html.H3("SEIR intuition"),
                    html.P(
                        "SEIR inserts an exposed compartment before infectiousness. That delay makes "
                        "the curve more realistic when there is a latent period."
                    ),
                    html.Div("Latent period ~= 1 / sigma", className="equation-line"),
                    html.Div("Recovery period ~= 1 / gamma", className="equation-line"),
                ],
            ),
            html.Div(
                className="info-card wide-card",
                children=[
                    html.H3("How to read the controls"),
                    html.Ul(
                        className="feature-list",
                        children=[
                            html.Li("Higher beta pushes the outbreak upward and earlier."),
                            html.Li("Lower gamma keeps people infectious longer."),
                            html.Li("Higher sigma moves exposed people into the infectious pool sooner."),
                            html.Li("A stronger intervention reduces beta after the chosen day."),
                        ],
                    ),
                ],
            ),
        ],
    )


def create_app() -> Dash:
    application = Dash(
        __name__,
        external_stylesheets=EXTERNAL_STYLESHEETS,
        title="Interactive Epidemiological Modeling",
    )
    application.layout = html.Div(
        className="page-shell",
        children=[
            html.Div(
                className="topbar",
                children=[
                    html.Div("EpiSim", className="brand"),
                    html.Div(
                        "Live deterministic outbreak modeling with drag-updating controls",
                        className="tagline",
                    ),
                ],
            ),
            dcc.Tabs(
                id="main-tabs",
                value="home",
                className="tab-strip",
                children=[
                    dcc.Tab(label="Home", value="home", className="tab-chip", selected_className="tab-chip--selected", children=build_home_tab()),
                    dcc.Tab(label="SIR Model", value="sir", className="tab-chip", selected_className="tab-chip--selected", children=build_model_tab("SIR")),
                    dcc.Tab(label="SEIR Model", value="seir", className="tab-chip", selected_className="tab-chip--selected", children=build_model_tab("SEIR")),
                    dcc.Tab(label="Scenario Simulator", value="scenario", className="tab-chip", selected_className="tab-chip--selected", children=build_scenario_tab()),
                    dcc.Tab(label="Parameter Sensitivity", value="sensitivity", className="tab-chip", selected_className="tab-chip--selected", children=build_sensitivity_tab()),
                    dcc.Tab(label="SIR vs SEIR", value="comparison", className="tab-chip", selected_className="tab-chip--selected", children=build_comparison_tab()),
                    dcc.Tab(label="About the Math", value="about", className="tab-chip", selected_className="tab-chip--selected", children=build_about_tab()),
                ],
            ),
        ],
    )
    return application


app = create_app()
server = app.server


@callback(Output("home-comparison-table", "data"), Input("home-comparison-store", "data"))
def hydrate_home_table(rows):
    return rows


@callback(Output("assumption-table", "data"), Input("assumption-store", "data"))
def hydrate_assumption_table(rows):
    return rows


@callback(
    Output("sir-graph", "figure"),
    Output("sir-metrics", "children"),
    Output("sir-parameter-table", "data"),
    Output("sir-simulation-table", "data"),
    Input("sir-population", "value"),
    Input("sir-infected", "value"),
    Input("sir-beta", "value"),
    Input("sir-gamma", "value"),
    Input("sir-days", "value"),
    Input("sir-intervention-day", "value"),
    Input("sir-intervention-strength", "value"),
)
def update_sir_tab(
    population: int,
    initial_infected: int,
    beta: float,
    gamma: float,
    days: int,
    intervention_day: int,
    intervention_strength: float,
):
    result = cached_sir_result(
        population,
        initial_infected,
        beta,
        gamma,
        days,
        intervention_day,
        intervention_strength,
    )
    summary = summarize_simulation(result)
    return (
        compartment_figure(result),
        metric_cards(summary),
        parameter_rows(result.parameters),
        result_records(result),
    )


@callback(
    Output("seir-graph", "figure"),
    Output("seir-metrics", "children"),
    Output("seir-parameter-table", "data"),
    Output("seir-simulation-table", "data"),
    Input("seir-population", "value"),
    Input("seir-infected", "value"),
    Input("seir-exposed", "value"),
    Input("seir-beta", "value"),
    Input("seir-sigma", "value"),
    Input("seir-gamma", "value"),
    Input("seir-days", "value"),
    Input("seir-intervention-day", "value"),
    Input("seir-intervention-strength", "value"),
)
def update_seir_tab(
    population: int,
    initial_infected: int,
    initial_exposed: int,
    beta: float,
    sigma: float,
    gamma: float,
    days: int,
    intervention_day: int,
    intervention_strength: float,
):
    result = cached_seir_result(
        population,
        initial_infected,
        initial_exposed,
        beta,
        sigma,
        gamma,
        days,
        intervention_day,
        intervention_strength,
    )
    summary = summarize_simulation(result)
    return (
        compartment_figure(result),
        metric_cards(summary),
        parameter_rows(result.parameters),
        result_records(result),
    )


@callback(
    Output("scenario-graph", "figure"),
    Output("scenario-table", "data"),
    Input("scenario-population", "value"),
    Input("scenario-infected", "value"),
    Input("scenario-beta", "value"),
    Input("scenario-gamma", "value"),
    Input("scenario-day", "value"),
)
def update_scenario_tab(
    population: int,
    initial_infected: int,
    beta: float,
    gamma: float,
    intervention_day: int,
):
    scenarios = {
        "No intervention": cached_sir_result(
            population, initial_infected, beta, gamma, 160, intervention_day, 0.0
        ),
        "Weak intervention": cached_sir_result(
            population, initial_infected, beta, gamma, 160, intervention_day, 0.2
        ),
        "Moderate intervention": cached_sir_result(
            population, initial_infected, beta, gamma, 160, intervention_day, 0.4
        ),
        "Strong intervention": cached_sir_result(
            population, initial_infected, beta, gamma, 160, intervention_day, 0.6
        ),
    }
    return scenario_comparison_figure(scenarios), scenario_summary_rows(scenarios)


@callback(
    Output("sensitivity-graph", "figure"),
    Input("sensitivity-model", "value"),
    Input("sensitivity-metric", "value"),
    Input("sensitivity-population", "value"),
    Input("sensitivity-infected", "value"),
    Input("sensitivity-gamma", "value"),
    Input("sensitivity-sigma", "value"),
)
def update_sensitivity_tab(
    model_name: str,
    metric: str,
    population: int,
    initial_infected: int,
    gamma: float,
    sigma: float,
):
    secondary_values = (
        pd.Series([round(0.05 + idx * 0.025, 3) for idx in range(15)]).to_numpy()
        if model_name == "SIR"
        else pd.Series([round(0.05 + idx * 0.039, 3) for idx in range(15)]).to_numpy()
    )
    beta_values = pd.Series([round(0.1 + idx * 0.04, 3) for idx in range(16)]).to_numpy()
    return sensitivity_heatmap(
        model_name=model_name,
        population=population,
        initial_infected=initial_infected,
        gamma=gamma,
        sigma=sigma,
        metric=metric,
        beta_values=beta_values,
        secondary_values=secondary_values,
    )


@callback(
    Output("comparison-sir-graph", "figure"),
    Output("comparison-seir-graph", "figure"),
    Output("comparison-table", "data"),
    Input("comparison-population", "value"),
    Input("comparison-infected", "value"),
    Input("comparison-exposed", "value"),
    Input("comparison-beta", "value"),
    Input("comparison-sigma", "value"),
    Input("comparison-gamma", "value"),
    Input("comparison-days", "value"),
)
def update_comparison_tab(
    population: int,
    initial_infected: int,
    initial_exposed: int,
    beta: float,
    sigma: float,
    gamma: float,
    days: int,
):
    sir_result = run_sir(
        population=population,
        initial_infected=initial_infected,
        beta=beta,
        gamma=gamma,
        days=float(days),
        dt=0.25,
    )
    seir_result = run_seir(
        population=population,
        initial_infected=initial_infected,
        initial_exposed=initial_exposed,
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        days=float(days),
        dt=0.25,
    )
    return (
        compartment_figure(sir_result),
        compartment_figure(seir_result),
        comparison_rows(compare_models(sir_result, seir_result)),
    )


if __name__ == "__main__":
    app.run(debug=False)
