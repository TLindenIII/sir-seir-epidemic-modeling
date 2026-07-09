from __future__ import annotations

import os

import numpy as np
import pandas as pd
from dash import (
    Dash,
    ClientsideFunction,
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    dash_table,
    dcc,
    html,
    no_update,
)

from episim.dashboard import (
    MODEL_COLUMNS,
    get_profile,
    live_bundle,
    live_only_bundle,
    result_records,
    run_model,
    table_bundle,
)
from episim.plotting import compartment_figure, scenario_comparison_figure, sensitivity_heatmap
from episim.presets import DISEASE_PRESETS, PRESET_BY_KEY, preset_options
from episim.simulation import run_seir, run_sir
from episim.utils import compare_models, summarize_simulation


EXTERNAL_STYLESHEETS = [
    (
        "https://fonts.googleapis.com/css2?"
        "family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;700&display=swap"
    )
]
GRAPH_CONFIG = {"displaylogo": False, "responsive": True}
PROFILE = get_profile(os.getenv("EPISIM_OPTIMIZATION_MODE"))
ASSUMPTION_ROWS = [
    {"Question": "Has an incubation delay?", "SIR": "No", "SEIR": "Yes"},
    {"Question": "More realistic for latent spread?", "SIR": "Less", "SEIR": "More"},
    {"Question": "Easiest to explain?", "SIR": "Yes", "SEIR": "Slightly harder"},
    {"Question": "Additional parameter", "SIR": "None", "SEIR": "sigma"},
]
METRIC_FIELDS = [
    ("Peak infectious", "peak"),
    ("Peak day", "peak-day"),
    ("Final outbreak size", "final-size"),
    ("Final outbreak share", "final-share"),
    ("Time to extinction", "extinction"),
]


def preset_parameter_text(preset) -> str:
    values = [f"beta {preset.beta:.3f}", f"gamma {preset.gamma:.3f}"]
    if preset.sigma is not None:
        values.append(f"sigma {preset.sigma:.3f}")
    return " · ".join(values)


def preset_links(citations) -> list[html.Component]:
    children: list[html.Component] = []
    for index, citation in enumerate(citations):
        if index:
            children.append(html.Span(" · ", className="citation-separator"))
        children.append(
            html.A(
                citation.label,
                href=citation.url,
                target="_blank",
                rel="noreferrer",
                className="citation-link",
            )
        )
    return children


def preset_card(preset) -> html.Div:
    return html.Div(
        className="preset-card",
        children=[
            html.Div(preset.model_hint, className="preset-kicker"),
            html.H4(preset.disease, className="preset-title"),
            html.Div(preset_parameter_text(preset), className="preset-parameters"),
            html.P(preset.note, className="preset-note"),
            html.Div(className="preset-links", children=preset_links(preset.citations)),
        ],
    )


def preset_control(model_name: str) -> html.Div:
    prefix = model_name.lower()
    note = (
        "Loads literature-backed beta and gamma values from the selected disease preset."
        if model_name == "SIR"
        else "Loads literature-backed beta, sigma, and gamma values from the selected disease preset."
    )
    return html.Div(
        className="control-block",
        children=[
            html.Div("Literature preset", className="control-label"),
            dcc.Dropdown(
                id=f"{prefix}-preset",
                options=preset_options(),
                placeholder="Select a disease preset",
                clearable=True,
                className="preset-dropdown",
            ),
            html.Div(note, className="mini-note"),
        ],
    )


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
                updatemode=PROFILE.slider_updatemode,
                allow_direct_input=True,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
    )


def data_table(component_id: str, columns: list[str], page_size: int = 8) -> dash_table.DataTable:
    return dash_table.DataTable(
        id=component_id,
        columns=[{"name": column, "id": column} for column in columns],
        data=[],
        page_size=page_size,
        sort_action="none",
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#151212",
            "fontWeight": 600,
            "border": "1px solid rgba(255, 255, 255, 0.14)",
            "color": "#f4efe8",
        },
        style_cell={
            "padding": "12px 14px",
            "backgroundColor": "#141010",
            "border": "1px solid rgba(255, 255, 255, 0.08)",
            "color": "#d7d0cb",
            "fontFamily": "IBM Plex Sans, sans-serif",
            "textAlign": "left",
        },
    )


def metric_grid(prefix: str) -> html.Div:
    cards = []
    for label, suffix in METRIC_FIELDS:
        cards.append(
            html.Div(
                className="metric-card",
                children=[
                    html.Div(label, className="metric-label"),
                    html.Div("--", id=f"{prefix}-metric-{suffix}", className="metric-value"),
                ],
            )
        )
    return html.Div(id=f"{prefix}-metrics", className="metric-grid", children=cards)


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


def build_home_tab() -> html.Div:
    sir_result = run_sir(
        population=10_000,
        initial_infected=10,
        beta=0.3,
        gamma=0.1,
        days=160,
        dt=0.25,
        intervention_day=30,
        intervention_strength=0.4,
    )
    seir_result = run_seir(
        population=10_000,
        initial_infected=10,
        initial_exposed=20,
        beta=0.3,
        sigma=0.2,
        gamma=0.1,
        days=160,
        dt=0.25,
        intervention_day=30,
        intervention_strength=0.4,
    )
    comparison = comparison_rows(compare_models(sir_result, seir_result))

    return html.Div(
        className="tab-panel home-panel",
        children=[
            html.Div(
                className="hero-shell",
                children=[
                    html.Div(
                        className="hero-copy-stack",
                        children=[
                            html.Div("Interactive epidemiology lab", className="eyebrow"),
                            html.H1(
                                className="hero-title",
                                children=[
                                    html.Span("SIR & SEIR"),
                                    html.Br(),
                                    html.Span("MODELING"),
                                    html.Br(),
                                    html.Span("WITH LIVE"),
                                    html.Br(),
                                    html.Span("SLIDER FEEDBACK"),
                                ],
                            ),
                            html.Div(className="hero-slash"),
                            html.P(
                                "Study outbreak dynamics with a fast deterministic core, compare "
                                "interventions, and move between SIR, SEIR, sensitivity, and "
                                "scenario views without leaving the same interface.",
                                className="hero-copy",
                            ),
                            html.Div(
                                className="hero-actions",
                                children=[
                                    html.Div("Use the tabs below to explore the models", className="cta-tag"),
                                    html.Div(
                                        className="watch-pill",
                                        children=[
                                            html.Div("PY", className="watch-avatar"),
                                            html.Span("Python simulation core, live browser updates"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="hero-visual-card",
                        children=[
                            html.Div(className="tunnel-grid"),
                            html.Div(
                                className="hero-visual-caption",
                                children=[
                                    html.Div("Interactive view", className="visual-kicker"),
                                    html.Div("Epidemic curves in motion", className="visual-title"),
                                    html.Div(
                                        "Drag the model parameters and watch the outbreak reshape in real time.",
                                        className="visual-copy",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="home-footer-strip",
                children=[
                    html.Div(
                        className="footer-chip",
                        children=[
                            html.Div("Models", className="footer-label"),
                            html.Div("SIR, SEIR, comparison, sensitivity", className="footer-value"),
                        ],
                    ),
                    html.Div(
                        className="footer-chip",
                        children=[
                            html.Div("Views", className="footer-label"),
                            html.Div("Models, scenarios, sensitivity, comparison", className="footer-value"),
                        ],
                    ),
                    html.Div(
                        className="footer-chip",
                        children=[
                            html.Div("Interface", className="footer-label"),
                            html.Div("Dash app with live slider-driven updates", className="footer-value"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="card-grid card-grid--home",
                children=[
                    html.Div(
                        className="panel-card",
                        children=[
                            html.H3("What this project demonstrates"),
                            html.Ul(
                                className="feature-list",
                                children=[
                                    html.Li("Differential-equation based epidemic simulation"),
                                    html.Li("Intervention scenarios with time-varying transmission"),
                                    html.Li("Lightweight interactive controls tuned for fast parameter sweeps"),
                                    html.Li("A portfolio-ready interface instead of notebook-only output"),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="panel-card",
                        children=[
                            html.H3("Core equations"),
                            dcc.Markdown(
                                r"$$\frac{dS}{dt} = -\beta \frac{SI}{N}$$",
                                mathjax=True,
                                className="equation-line",
                            ),
                            dcc.Markdown(
                                r"$$\frac{dI}{dt} = \beta \frac{SI}{N} - \gamma I$$",
                                mathjax=True,
                                className="equation-line",
                            ),
                            dcc.Markdown(
                                r"$$\frac{dR}{dt} = \gamma I$$",
                                mathjax=True,
                                className="equation-line",
                            ),
                            dcc.Markdown(
                                r"$$\frac{dE}{dt} = \beta \frac{SI}{N} - \sigma E$$",
                                mathjax=True,
                                className="equation-line",
                            ),
                        ],
                    ),
                    html.Div(
                        className="panel-card panel-card--wide",
                        children=[
                            html.H3("Baseline SIR vs SEIR"),
                            data_table("home-comparison-table", ["Metric", "SIR", "SEIR"], page_size=6),
                            dcc.Store(id="home-comparison-store", data=comparison),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="card-grid card-grid--lab",
                children=[
                    html.Div(
                        className="panel-card",
                        children=[
                            html.H3("How to explore"),
                            html.Ul(
                                className="feature-list",
                                children=[
                                    html.Li("Use the SIR and SEIR tabs to inspect individual compartment dynamics."),
                                    html.Li("Open Scenario Simulator to compare intervention timing and strength."),
                                    html.Li("Use Parameter Sensitivity for heatmap-level pattern scanning."),
                                    html.Li("Compare SIR vs SEIR directly to see how a latent compartment changes timing."),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="panel-card panel-card--wide",
                        children=[
                            html.H3("Literature-backed disease presets"),
                            html.P(
                                "Use the preset dropdown inside the SIR or SEIR tabs to load published starting "
                                "parameters. When a value is derived for the simplified teaching model, that is "
                                "called out explicitly below.",
                                className="section-note",
                            ),
                            html.Div(
                                className="preset-grid",
                                children=[preset_card(preset) for preset in DISEASE_PRESETS],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_model_tab(model_name: str) -> html.Div:
    prefix = model_name.lower()
    controls = [
        preset_control(model_name),
        slider_block(f"{model_name} population", f"{prefix}-population", 1_000, 500_000, 1_000, 10_000),
        slider_block("Initial infected", f"{prefix}-infected", 1, 1_000, 1, 10),
        slider_block("Transmission rate (beta)", f"{prefix}-beta", 0.05, 1.0, 0.01, 0.30),
        slider_block("Recovery rate (gamma)", f"{prefix}-gamma", 0.02, 0.5, 0.01, 0.10),
        slider_block("Simulation horizon (days)", f"{prefix}-days", 30, 365, 1, 160),
        slider_block("Intervention day", f"{prefix}-intervention-day", 0, 180, 1, 30),
        slider_block("Intervention strength", f"{prefix}-intervention-strength", 0.0, 0.9, 0.05, 0.4),
    ]
    if model_name == "SEIR":
        controls.insert(2, slider_block("Initial exposed", "seir-exposed", 0, 2_000, 1, 20))
        controls.insert(4, slider_block("Incubation rate (sigma)", "seir-sigma", 0.05, 1.0, 0.01, 0.20))

    helper_text = (
        "Drag the sliders or type exact values to update the model. Use the literature preset "
        "dropdown to load paper-backed parameters as a starting point."
    )

    extra_controls = []
    if PROFILE.manual_table_refresh:
        extra_controls.append(
            html.Button("Refresh table", id=f"{prefix}-table-refresh", className="ghost-button")
        )

    return html.Div(
        className="tab-panel split-layout",
        children=[
            html.Div(
                className="control-card",
                children=[
                    html.H2(f"{model_name} model"),
                    html.P(helper_text, className="section-note"),
                    *controls,
                    *extra_controls,
                ],
            ),
            html.Div(
                className="content-stack",
                children=[
                    html.Div(
                        className="graph-card",
                        children=[dcc.Graph(id=f"{prefix}-graph", config=GRAPH_CONFIG)],
                    ),
                    metric_grid(prefix),
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
                            data_table(
                                f"{prefix}-simulation-table",
                                MODEL_COLUMNS[model_name],
                                page_size=12,
                            ),
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
                        "Compare infectious curves across four intervention strengths under the same baseline outbreak.",
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
                    html.Div(className="graph-card", children=[dcc.Graph(id="scenario-graph", config=GRAPH_CONFIG)]),
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
                        "Heatmaps summarize how peak size, timing, and total outbreak share move when parameters shift.",
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
                    html.Div(className="graph-card", children=[dcc.Graph(id="sensitivity-graph", config=GRAPH_CONFIG)]),
                    html.Div(
                        className="table-card",
                        children=[
                            html.H3("How to read the heatmap"),
                            html.Ul(
                                className="feature-list",
                                children=[
                                    html.Li("The horizontal axis is beta, so moving right means higher transmission."),
                                    html.Li("For SIR the vertical axis is gamma, while for SEIR it switches to sigma."),
                                    html.Li("Brighter regions mark larger values for the selected metric, whether that means a higher peak, a later peak day, or a larger final outbreak share."),
                                ],
                            ),
                        ],
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
                        "Use shared inputs to see how adding a latent compartment delays and reshapes the outbreak.",
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
                            html.Div(className="graph-card", children=[dcc.Graph(id="comparison-sir-graph", config=GRAPH_CONFIG)]),
                            html.Div(className="graph-card", children=[dcc.Graph(id="comparison-seir-graph", config=GRAPH_CONFIG)]),
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
                className="panel-card",
                children=[
                    html.H3("SIR intuition"),
                    html.P(
                        "SIR moves people directly from susceptible to infectious to recovered. "
                        "It is the cleanest model for showing how beta and gamma compete."
                    ),
                    html.Div("R0 = beta / gamma", className="equation-line"),
                ],
            ),
            html.Div(
                className="panel-card",
                children=[
                    html.H3("SEIR intuition"),
                    html.P(
                        "SEIR inserts an exposed compartment before infectiousness, which makes the timing more realistic when spread has a latent phase."
                    ),
                    html.Div("Latent period ~= 1 / sigma", className="equation-line"),
                    html.Div("Recovery period ~= 1 / gamma", className="equation-line"),
                ],
            ),
            html.Div(
                className="panel-card panel-card--wide",
                children=[
                    html.H3("How to read the controls"),
                    html.Ul(
                        className="feature-list",
                        children=[
                            html.Li("Higher beta pushes the outbreak upward and earlier."),
                            html.Li("Lower gamma keeps infectious people in circulation longer."),
                            html.Li("Higher sigma moves exposed people into the infectious compartment sooner."),
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
        suppress_callback_exceptions=False,
    )
    application.layout = html.Div(
        className="page-shell",
        children=[
            html.Div(
                className="frame-shell",
                children=[
                    html.Div(
                        className="topbar",
                        children=[
                            html.Div(
                                className="brand-lockup",
                                children=[
                                    html.Div("E", className="brand-mark"),
                                    html.Div("EpiSim", className="brand"),
                                ],
                            ),
                            html.Div(
                                className="topbar-copy",
                                children=[
                                    html.Div("Fast, paper-backed epidemic modeling", className="topbar-title"),
                                    html.Div(
                                        "Use the tabs below to switch between SIR, SEIR, scenarios, sensitivity, and direct model comparison.",
                                        className="topbar-note",
                                    ),
                                ],
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
            ),
        ],
    )
    return application


def register_data_callbacks():
    @callback(Output("home-comparison-table", "data"), Input("home-comparison-store", "data"))
    def hydrate_home_table(rows):
        return rows

    @callback(Output("assumption-table", "data"), Input("assumption-store", "data"))
    def hydrate_assumption_table(rows):
        return rows


def register_preset_callbacks():
    @callback(
        Output("sir-beta", "value"),
        Output("sir-gamma", "value"),
        Input("sir-preset", "value"),
        prevent_initial_call=True,
    )
    def load_sir_preset(preset_key: str | None):
        if not preset_key:
            return no_update, no_update
        preset = PRESET_BY_KEY[preset_key]
        return preset.beta, preset.gamma

    @callback(
        Output("seir-beta", "value"),
        Output("seir-sigma", "value"),
        Output("seir-gamma", "value"),
        Input("seir-preset", "value"),
        prevent_initial_call=True,
    )
    def load_seir_preset(preset_key: str | None):
        if not preset_key:
            return no_update, no_update, no_update
        preset = PRESET_BY_KEY[preset_key]
        return (
            preset.beta,
            preset.sigma if preset.sigma is not None else no_update,
            preset.gamma,
        )


def live_value(drag_value, release_value):
    return release_value if drag_value is None else drag_value


def register_model_callbacks():
    if PROFILE.clientside_live:
        clientside_callback(
            ClientsideFunction(namespace="episim", function_name="updateSirLive"),
            Output("sir-graph", "figure"),
            Output("sir-metric-peak", "children"),
            Output("sir-metric-peak-day", "children"),
            Output("sir-metric-final-size", "children"),
            Output("sir-metric-final-share", "children"),
            Output("sir-metric-extinction", "children"),
            Input("sir-population", "drag_value"),
            Input("sir-population", "value"),
            Input("sir-infected", "drag_value"),
            Input("sir-infected", "value"),
            Input("sir-beta", "drag_value"),
            Input("sir-beta", "value"),
            Input("sir-gamma", "drag_value"),
            Input("sir-gamma", "value"),
            Input("sir-days", "drag_value"),
            Input("sir-days", "value"),
            Input("sir-intervention-day", "drag_value"),
            Input("sir-intervention-day", "value"),
            Input("sir-intervention-strength", "drag_value"),
            Input("sir-intervention-strength", "value"),
        )
        clientside_callback(
            ClientsideFunction(namespace="episim", function_name="updateSeirLive"),
            Output("seir-graph", "figure"),
            Output("seir-metric-peak", "children"),
            Output("seir-metric-peak-day", "children"),
            Output("seir-metric-final-size", "children"),
            Output("seir-metric-final-share", "children"),
            Output("seir-metric-extinction", "children"),
            Input("seir-population", "drag_value"),
            Input("seir-population", "value"),
            Input("seir-infected", "drag_value"),
            Input("seir-infected", "value"),
            Input("seir-exposed", "drag_value"),
            Input("seir-exposed", "value"),
            Input("seir-beta", "drag_value"),
            Input("seir-beta", "value"),
            Input("seir-sigma", "drag_value"),
            Input("seir-sigma", "value"),
            Input("seir-gamma", "drag_value"),
            Input("seir-gamma", "value"),
            Input("seir-days", "drag_value"),
            Input("seir-days", "value"),
            Input("seir-intervention-day", "drag_value"),
            Input("seir-intervention-day", "value"),
            Input("seir-intervention-strength", "drag_value"),
            Input("seir-intervention-strength", "value"),
        )
    elif PROFILE.split_live_and_tables:
        @callback(
            Output("sir-graph", "figure"),
            Output("sir-metric-peak", "children"),
            Output("sir-metric-peak-day", "children"),
            Output("sir-metric-final-size", "children"),
            Output("sir-metric-final-share", "children"),
            Output("sir-metric-extinction", "children"),
            Input("sir-population", "drag_value"),
            Input("sir-population", "value"),
            Input("sir-infected", "drag_value"),
            Input("sir-infected", "value"),
            Input("sir-beta", "drag_value"),
            Input("sir-beta", "value"),
            Input("sir-gamma", "drag_value"),
            Input("sir-gamma", "value"),
            Input("sir-days", "drag_value"),
            Input("sir-days", "value"),
            Input("sir-intervention-day", "drag_value"),
            Input("sir-intervention-day", "value"),
            Input("sir-intervention-strength", "drag_value"),
            Input("sir-intervention-strength", "value"),
        )
        def update_sir_live(*args):
            population = live_value(args[0], args[1])
            initial_infected = live_value(args[2], args[3])
            beta = live_value(args[4], args[5])
            gamma = live_value(args[6], args[7])
            days = live_value(args[8], args[9])
            intervention_day = live_value(args[10], args[11])
            intervention_strength = live_value(args[12], args[13])
            payload = live_only_bundle(
                "SIR",
                lightweight=PROFILE.lightweight_figure,
                population=population,
                initial_infected=initial_infected,
                beta=beta,
                gamma=gamma,
                days=days,
                intervention_day=intervention_day,
                intervention_strength=intervention_strength,
                initial_exposed=0,
                sigma=0.2,
            )
            return (payload["figure"], *payload["metrics"])

        @callback(
            Output("seir-graph", "figure"),
            Output("seir-metric-peak", "children"),
            Output("seir-metric-peak-day", "children"),
            Output("seir-metric-final-size", "children"),
            Output("seir-metric-final-share", "children"),
            Output("seir-metric-extinction", "children"),
            Input("seir-population", "drag_value"),
            Input("seir-population", "value"),
            Input("seir-infected", "drag_value"),
            Input("seir-infected", "value"),
            Input("seir-exposed", "drag_value"),
            Input("seir-exposed", "value"),
            Input("seir-beta", "drag_value"),
            Input("seir-beta", "value"),
            Input("seir-sigma", "drag_value"),
            Input("seir-sigma", "value"),
            Input("seir-gamma", "drag_value"),
            Input("seir-gamma", "value"),
            Input("seir-days", "drag_value"),
            Input("seir-days", "value"),
            Input("seir-intervention-day", "drag_value"),
            Input("seir-intervention-day", "value"),
            Input("seir-intervention-strength", "drag_value"),
            Input("seir-intervention-strength", "value"),
        )
        def update_seir_live(*args):
            population = live_value(args[0], args[1])
            initial_infected = live_value(args[2], args[3])
            initial_exposed = live_value(args[4], args[5])
            beta = live_value(args[6], args[7])
            sigma = live_value(args[8], args[9])
            gamma = live_value(args[10], args[11])
            days = live_value(args[12], args[13])
            intervention_day = live_value(args[14], args[15])
            intervention_strength = live_value(args[16], args[17])
            payload = live_only_bundle(
                "SEIR",
                lightweight=PROFILE.lightweight_figure,
                population=population,
                initial_infected=initial_infected,
                initial_exposed=initial_exposed,
                beta=beta,
                sigma=sigma,
                gamma=gamma,
                days=days,
                intervention_day=intervention_day,
                intervention_strength=intervention_strength,
            )
            return (payload["figure"], *payload["metrics"])
    else:
        @callback(
            Output("sir-graph", "figure"),
            Output("sir-metric-peak", "children"),
            Output("sir-metric-peak-day", "children"),
            Output("sir-metric-final-size", "children"),
            Output("sir-metric-final-share", "children"),
            Output("sir-metric-extinction", "children"),
            Output("sir-parameter-table", "data"),
            Input("sir-population", "value"),
            Input("sir-infected", "value"),
            Input("sir-beta", "value"),
            Input("sir-gamma", "value"),
            Input("sir-days", "value"),
            Input("sir-intervention-day", "value"),
            Input("sir-intervention-strength", "value"),
        )
        def update_sir_baseline(
            population: int,
            initial_infected: int,
            beta: float,
            gamma: float,
            days: int,
            intervention_day: int,
            intervention_strength: float,
        ):
            payload = live_bundle(
                "SIR",
                lightweight=PROFILE.lightweight_figure,
                population=population,
                initial_infected=initial_infected,
                beta=beta,
                gamma=gamma,
                days=days,
                intervention_day=intervention_day,
                intervention_strength=intervention_strength,
                initial_exposed=0,
                sigma=0.2,
            )
            return (payload["figure"], *payload["metrics"], payload["parameters"])

        @callback(
            Output("seir-graph", "figure"),
            Output("seir-metric-peak", "children"),
            Output("seir-metric-peak-day", "children"),
            Output("seir-metric-final-size", "children"),
            Output("seir-metric-final-share", "children"),
            Output("seir-metric-extinction", "children"),
            Output("seir-parameter-table", "data"),
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
        def update_seir_baseline(
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
            payload = live_bundle(
                "SEIR",
                lightweight=PROFILE.lightweight_figure,
                population=population,
                initial_infected=initial_infected,
                initial_exposed=initial_exposed,
                beta=beta,
                sigma=sigma,
                gamma=gamma,
                days=days,
                intervention_day=intervention_day,
                intervention_strength=intervention_strength,
            )
            return (payload["figure"], *payload["metrics"], payload["parameters"])

    if PROFILE.name == "baseline":
        @callback(
            Output("sir-simulation-table", "data"),
            Input("sir-population", "value"),
            Input("sir-infected", "value"),
            Input("sir-beta", "value"),
            Input("sir-gamma", "value"),
            Input("sir-days", "value"),
            Input("sir-intervention-day", "value"),
            Input("sir-intervention-strength", "value"),
        )
        def update_sir_baseline_table(
            population: int,
            initial_infected: int,
            beta: float,
            gamma: float,
            days: int,
            intervention_day: int,
            intervention_strength: float,
        ):
            return result_records(
                run_model(
                    "SIR",
                    population=population,
                    initial_infected=initial_infected,
                    beta=beta,
                    gamma=gamma,
                    days=days,
                    intervention_day=intervention_day,
                    intervention_strength=intervention_strength,
                    initial_exposed=0,
                    sigma=0.2,
                )
            )

        @callback(
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
        def update_seir_baseline_table(
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
            return result_records(
                run_model(
                    "SEIR",
                    population=population,
                    initial_infected=initial_infected,
                    initial_exposed=initial_exposed,
                    beta=beta,
                    sigma=sigma,
                    gamma=gamma,
                    days=days,
                    intervention_day=intervention_day,
                    intervention_strength=intervention_strength,
                )
            )
    elif PROFILE.manual_table_refresh:
        @callback(
            Output("sir-simulation-table", "data"),
            Input("sir-table-refresh", "n_clicks"),
            State("sir-population", "value"),
            State("sir-infected", "value"),
            State("sir-beta", "value"),
            State("sir-gamma", "value"),
            State("sir-days", "value"),
            State("sir-intervention-day", "value"),
            State("sir-intervention-strength", "value"),
        )
        def refresh_sir_table(
            _clicks: int | None,
            population: int,
            initial_infected: int,
            beta: float,
            gamma: float,
            days: int,
            intervention_day: int,
            intervention_strength: float,
        ):
            return table_bundle(
                "SIR",
                population=population,
                initial_infected=initial_infected,
                beta=beta,
                gamma=gamma,
                days=days,
                intervention_day=intervention_day,
                intervention_strength=intervention_strength,
                initial_exposed=0,
                sigma=0.2,
            )["simulation"]

        @callback(
            Output("seir-simulation-table", "data"),
            Input("seir-table-refresh", "n_clicks"),
            State("seir-population", "value"),
            State("seir-infected", "value"),
            State("seir-exposed", "value"),
            State("seir-beta", "value"),
            State("seir-sigma", "value"),
            State("seir-gamma", "value"),
            State("seir-days", "value"),
            State("seir-intervention-day", "value"),
            State("seir-intervention-strength", "value"),
        )
        def refresh_seir_table(
            _clicks: int | None,
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
            return table_bundle(
                "SEIR",
                population=population,
                initial_infected=initial_infected,
                initial_exposed=initial_exposed,
                beta=beta,
                sigma=sigma,
                gamma=gamma,
                days=days,
                intervention_day=intervention_day,
                intervention_strength=intervention_strength,
            )["simulation"]
    else:
        @callback(
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
        def update_sir_tables(
            population: int,
            initial_infected: int,
            beta: float,
            gamma: float,
            days: int,
            intervention_day: int,
            intervention_strength: float,
        ):
            bundle = table_bundle(
                "SIR",
                population=population,
                initial_infected=initial_infected,
                beta=beta,
                gamma=gamma,
                days=days,
                intervention_day=intervention_day,
                intervention_strength=intervention_strength,
                initial_exposed=0,
                sigma=0.2,
            )
            return bundle["parameters"], bundle["simulation"]

        @callback(
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
        def update_seir_tables(
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
            bundle = table_bundle(
                "SEIR",
                population=population,
                initial_infected=initial_infected,
                initial_exposed=initial_exposed,
                beta=beta,
                sigma=sigma,
                gamma=gamma,
                days=days,
                intervention_day=intervention_day,
                intervention_strength=intervention_strength,
            )
            return bundle["parameters"], bundle["simulation"]


def register_remaining_callbacks():
    @callback(
        Output("scenario-graph", "figure"),
        Output("scenario-table", "data"),
        Input("scenario-population", "drag_value"),
        Input("scenario-population", "value"),
        Input("scenario-infected", "drag_value"),
        Input("scenario-infected", "value"),
        Input("scenario-beta", "drag_value"),
        Input("scenario-beta", "value"),
        Input("scenario-gamma", "drag_value"),
        Input("scenario-gamma", "value"),
        Input("scenario-day", "drag_value"),
        Input("scenario-day", "value"),
    )
    def update_scenario_tab(*args):
        population = live_value(args[0], args[1])
        initial_infected = live_value(args[2], args[3])
        beta = live_value(args[4], args[5])
        gamma = live_value(args[6], args[7])
        intervention_day = live_value(args[8], args[9])
        scenarios = {
            "No intervention (0% beta reduction)": run_sir(population=population, initial_infected=initial_infected, beta=beta, gamma=gamma, days=160, dt=0.25, intervention_day=intervention_day, intervention_strength=0.0),
            "Weak intervention (20% beta reduction)": run_sir(population=population, initial_infected=initial_infected, beta=beta, gamma=gamma, days=160, dt=0.25, intervention_day=intervention_day, intervention_strength=0.2),
            "Moderate intervention (40% beta reduction)": run_sir(population=population, initial_infected=initial_infected, beta=beta, gamma=gamma, days=160, dt=0.25, intervention_day=intervention_day, intervention_strength=0.4),
            "Strong intervention (60% beta reduction)": run_sir(population=population, initial_infected=initial_infected, beta=beta, gamma=gamma, days=160, dt=0.25, intervention_day=intervention_day, intervention_strength=0.6),
        }
        return scenario_comparison_figure(scenarios), scenario_summary_rows(scenarios)

    @callback(
        Output("sensitivity-graph", "figure"),
        Input("sensitivity-model", "value"),
        Input("sensitivity-metric", "value"),
        Input("sensitivity-population", "drag_value"),
        Input("sensitivity-population", "value"),
        Input("sensitivity-infected", "drag_value"),
        Input("sensitivity-infected", "value"),
        Input("sensitivity-gamma", "drag_value"),
        Input("sensitivity-gamma", "value"),
        Input("sensitivity-sigma", "drag_value"),
        Input("sensitivity-sigma", "value"),
    )
    def update_sensitivity_tab(model_name: str, metric: str, *args):
        population = live_value(args[0], args[1])
        initial_infected = live_value(args[2], args[3])
        gamma = live_value(args[4], args[5])
        sigma = live_value(args[6], args[7])
        beta_values = np.round(np.linspace(0.1, 0.7, 16), 3)
        secondary_values = np.round(np.linspace(0.05, 0.4, 15), 3)
        if model_name == "SEIR":
            secondary_values = np.round(np.linspace(0.05, 0.6, 15), 3)
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
        Input("comparison-population", "drag_value"),
        Input("comparison-population", "value"),
        Input("comparison-infected", "drag_value"),
        Input("comparison-infected", "value"),
        Input("comparison-exposed", "drag_value"),
        Input("comparison-exposed", "value"),
        Input("comparison-beta", "drag_value"),
        Input("comparison-beta", "value"),
        Input("comparison-sigma", "drag_value"),
        Input("comparison-sigma", "value"),
        Input("comparison-gamma", "drag_value"),
        Input("comparison-gamma", "value"),
        Input("comparison-days", "drag_value"),
        Input("comparison-days", "value"),
    )
    def update_comparison_tab(*args):
        population = live_value(args[0], args[1])
        initial_infected = live_value(args[2], args[3])
        initial_exposed = live_value(args[4], args[5])
        beta = live_value(args[6], args[7])
        sigma = live_value(args[8], args[9])
        gamma = live_value(args[10], args[11])
        days = live_value(args[12], args[13])
        sir_result = run_sir(population=population, initial_infected=initial_infected, beta=beta, gamma=gamma, days=float(days), dt=0.25)
        seir_result = run_seir(population=population, initial_infected=initial_infected, initial_exposed=initial_exposed, beta=beta, sigma=sigma, gamma=gamma, days=float(days), dt=0.25)
        return (
            compartment_figure(sir_result),
            compartment_figure(seir_result),
            comparison_rows(compare_models(sir_result, seir_result)),
        )


app = create_app()
server = app.server
register_data_callbacks()
register_preset_callbacks()
register_model_callbacks()
register_remaining_callbacks()


if __name__ == "__main__":
    app.run(debug=False)
