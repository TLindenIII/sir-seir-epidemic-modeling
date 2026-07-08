import numpy as np

from episim.models import apply_intervention
from episim.simulation import run_seir, run_sir
from episim.utils import summarize_simulation


def test_sir_conserves_population():
    result = run_sir(
        population=10_000,
        initial_infected=10,
        beta=0.35,
        gamma=0.1,
        days=120,
        dt=0.25,
    )
    total = sum(result.compartments.values())
    assert np.allclose(total, result.population, atol=1e-4)


def test_seir_conserves_population():
    result = run_seir(
        population=10_000,
        initial_infected=10,
        initial_exposed=20,
        beta=0.3,
        sigma=0.2,
        gamma=0.1,
        days=120,
        dt=0.25,
    )
    total = sum(result.compartments.values())
    assert np.allclose(total, result.population, atol=1e-4)


def test_intervention_reduces_peak_infectious():
    baseline = run_sir(
        population=10_000,
        initial_infected=10,
        beta=0.35,
        gamma=0.1,
        days=160,
        dt=0.25,
    )
    intervention = run_sir(
        population=10_000,
        initial_infected=10,
        beta=0.35,
        gamma=0.1,
        days=160,
        dt=0.25,
        intervention_day=25,
        intervention_strength=0.45,
    )
    assert intervention.compartments["Infectious"].max() < baseline.compartments[
        "Infectious"
    ].max()


def test_summary_metrics_are_well_formed():
    result = run_seir(
        population=5_000,
        initial_infected=5,
        initial_exposed=12,
        beta=0.32,
        sigma=0.22,
        gamma=0.12,
        days=150,
        dt=0.25,
    )
    summary = summarize_simulation(result)
    assert summary.peak_infectious > 0
    assert 0 <= summary.final_outbreak_share <= 1
    assert summary.peak_day >= 0


def test_apply_intervention_leaves_beta_unchanged_before_start():
    assert apply_intervention(0.3, t=10.0, intervention_day=20.0, intervention_strength=0.5) == 0.3
