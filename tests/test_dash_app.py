from dash import dcc
import pandas as pd

from app.dash_app import (
    comparison_rows,
    create_app,
    metric_cards,
    parameter_rows,
)
from episim.simulation import run_sir
from episim.utils import summarize_simulation


def test_create_app_has_expected_tabs():
    app = create_app()
    tabs = next(child for child in app.layout.children if isinstance(child, dcc.Tabs))
    labels = [tab.label for tab in tabs.children]
    assert labels == [
        "Home",
        "SIR Model",
        "SEIR Model",
        "Scenario Simulator",
        "Parameter Sensitivity",
        "SIR vs SEIR",
        "About the Math",
    ]


def test_metric_cards_render_all_summary_metrics():
    result = run_sir(
        population=10_000,
        initial_infected=10,
        beta=0.3,
        gamma=0.1,
        days=160,
    )
    cards = metric_cards(summarize_simulation(result))
    assert len(cards) == 5


def test_parameter_rows_formats_missing_intervention_day():
    result = run_sir(
        population=10_000,
        initial_infected=10,
        beta=0.3,
        gamma=0.1,
        days=160,
        intervention_day=None,
    )
    rows = parameter_rows(result.parameters)
    intervention_row = next(row for row in rows if row["Parameter"] == "intervention_day")
    assert intervention_row["Value"] == "None"


def test_comparison_rows_formats_percentages():
    rows = comparison_rows(
        pd.DataFrame(
            [
                {"Metric": "Final outbreak share", "SIR": 0.75, "SEIR": 0.65},
                {"Metric": "Peak day", "SIR": 42.0, "SEIR": 55.0},
            ]
        )
    )
    assert rows[0]["SIR"] == "75.0%"
    assert rows[1]["SEIR"] == "55.0"
