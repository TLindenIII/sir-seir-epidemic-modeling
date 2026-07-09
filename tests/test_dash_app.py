from dash import dcc
import pandas as pd

from app.dash_app import comparison_rows, create_app, metric_grid
from episim.dashboard import PROFILE_MAP, parameter_rows
from episim.simulation import run_sir


def test_create_app_has_expected_tabs():
    app = create_app()
    frame = app.layout.children[0]
    tabs = next(child for child in frame.children if isinstance(child, dcc.Tabs))
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


def test_metric_grid_contains_five_value_slots():
    grid = metric_grid("sir")
    assert len(grid.children) == 5
    ids = [card.children[1].id for card in grid.children]
    assert ids == [
        "sir-metric-peak",
        "sir-metric-peak-day",
        "sir-metric-final-size",
        "sir-metric-final-share",
        "sir-metric-extinction",
    ]


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


def test_profile_map_contains_all_optimization_steps():
    assert set(PROFILE_MAP) == {
        "baseline",
        "step1_no_live_table",
        "step2_split_callbacks",
        "step3_lightweight_figure",
        "step4_clientside",
    }
