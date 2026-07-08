from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from episim.plotting import compartment_figure, sensitivity_heatmap
from episim.simulation import run_seir, run_sir
from episim.utils import compare_models, summarize_simulation


st.set_page_config(
    page_title="Interactive Epidemiological Modeling",
    page_icon="📈",
    layout="wide",
)


def metric_frame(summary) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metric": [
                "Peak infectious",
                "Peak day",
                "Final outbreak size",
                "Final outbreak share",
                "Time to extinction",
            ],
            "Value": [
                f"{summary.peak_infectious:,.0f}",
                f"{summary.peak_day:.1f}",
                f"{summary.final_outbreak_size:,.0f}",
                f"{summary.final_outbreak_share:.1%}",
                (
                    f"{summary.time_to_extinction:.1f} days"
                    if summary.time_to_extinction is not None
                    else "Not reached"
                ),
            ],
        }
    )


@st.cache_data(show_spinner=False)
def cached_sir(**kwargs):
    return run_sir(**kwargs)


@st.cache_data(show_spinner=False)
def cached_seir(**kwargs):
    return run_seir(**kwargs)


def home_tab():
    st.title("Interactive Epidemiological Modeling with SIR and SEIR Simulations")
    st.write(
        "Explore how compartmental epidemic models behave, how interventions change the curve, "
        "and why adding an exposed compartment changes timing."
    )
    left, right = st.columns([1.2, 1.0], gap="large")
    with left:
        st.subheader("What you can do here")
        st.markdown(
            """
            - Compare SIR and SEIR assumptions
            - Adjust transmission, recovery, and incubation parameters
            - Model interventions that reduce transmission after a chosen day
            - Inspect sensitivity heatmaps for peak size and outbreak timing
            """
        )
    with right:
        st.subheader("Core equations")
        st.latex(r"\frac{dS}{dt} = -\beta \frac{SI}{N}")
        st.latex(r"\frac{dI}{dt} = \beta \frac{SI}{N} - \gamma I")
        st.latex(r"\frac{dR}{dt} = \gamma I")
        st.latex(r"\frac{dE}{dt} = \beta \frac{SI}{N} - \sigma E")


def shared_controls(include_exposed: bool):
    population = st.sidebar.slider("Population", 1_000, 500_000, 10_000, step=1_000)
    initial_infected = st.sidebar.slider("Initial infected", 1, 1_000, 10)
    beta = st.sidebar.slider("Transmission rate (beta)", 0.05, 1.00, 0.30, 0.01)
    gamma = st.sidebar.slider("Recovery rate (gamma)", 0.02, 0.50, 0.10, 0.01)
    days = st.sidebar.slider("Simulation horizon (days)", 30, 365, 160)
    intervention_day = st.sidebar.slider("Intervention day", 0, 180, 30)
    intervention_strength = st.sidebar.slider(
        "Intervention strength", 0.0, 0.9, 0.4, 0.05
    )

    controls = {
        "population": population,
        "initial_infected": initial_infected,
        "beta": beta,
        "gamma": gamma,
        "days": days,
        "dt": 0.25,
        "intervention_day": float(intervention_day),
        "intervention_strength": intervention_strength,
    }
    if include_exposed:
        controls["initial_exposed"] = st.sidebar.slider("Initial exposed", 0, 2_000, 20)
        controls["sigma"] = st.sidebar.slider(
            "Incubation rate (sigma)", 0.05, 1.00, 0.20, 0.01
        )
    return controls


def render_model_tab(model_name: str):
    st.sidebar.header(f"{model_name} controls")
    controls = shared_controls(include_exposed=model_name == "SEIR")
    result = cached_sir(**controls) if model_name == "SIR" else cached_seir(**controls)
    summary = summarize_simulation(result)

    primary, secondary = st.columns([2.2, 1.0], gap="large")
    with primary:
        st.plotly_chart(compartment_figure(result), use_container_width=True)
    with secondary:
        st.subheader("Key metrics")
        st.dataframe(metric_frame(summary), hide_index=True, use_container_width=True)
        st.subheader("Parameter summary")
        st.dataframe(
            pd.DataFrame(
                {"Parameter": list(result.parameters.keys()), "Value": result.parameters.values()}
            ),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Simulation table"):
        st.dataframe(result.dataframe.round(3), use_container_width=True, height=320)


def scenario_tab():
    st.subheader("Scenario Simulator")
    population = st.slider("Population", 1_000, 100_000, 10_000, step=1_000, key="scenario_population")
    initial_infected = st.slider("Initial infected", 1, 500, 10, key="scenario_infected")
    beta = st.slider("Transmission rate (beta)", 0.05, 1.00, 0.30, 0.01, key="scenario_beta")
    gamma = st.slider("Recovery rate (gamma)", 0.02, 0.50, 0.10, 0.01, key="scenario_gamma")
    intervention_day = st.slider("Intervention day", 0, 120, 30, key="scenario_day")

    scenarios = {
        "No intervention": 0.0,
        "Weak intervention": 0.2,
        "Moderate intervention": 0.4,
        "Strong intervention": 0.6,
    }

    comparison = []
    for name, strength in scenarios.items():
        result = cached_sir(
            population=population,
            initial_infected=initial_infected,
            beta=beta,
            gamma=gamma,
            days=160,
            dt=0.25,
            intervention_day=float(intervention_day),
            intervention_strength=strength,
        )
        df = result.dataframe.assign(Scenario=name)
        comparison.append(df)

    comparison_df = pd.concat(comparison, ignore_index=True)
    st.line_chart(
        comparison_df.pivot(index="day", columns="Scenario", values="Infectious"),
        height=420,
    )
    st.caption("Higher intervention strength lowers beta after the selected intervention day.")


def sensitivity_tab():
    st.subheader("Parameter Sensitivity")
    model_name = st.radio("Model", ["SIR", "SEIR"], horizontal=True)
    metric = st.selectbox(
        "Metric",
        ["peak_infectious", "peak_day", "final_outbreak_share"],
        format_func=lambda value: value.replace("_", " ").title(),
    )

    beta_values = np.round(np.linspace(0.1, 0.7, 16), 3)
    secondary_values = np.round(np.linspace(0.05, 0.4, 15), 3)
    if model_name == "SEIR":
        secondary_values = np.round(np.linspace(0.05, 0.6, 15), 3)

    st.plotly_chart(
        sensitivity_heatmap(
            model_name=model_name,
            population=10_000,
            initial_infected=10,
            gamma=0.1,
            sigma=0.2,
            metric=metric,
            beta_values=beta_values,
            secondary_values=secondary_values,
        ),
        use_container_width=True,
    )


def comparison_tab():
    st.subheader("SIR vs SEIR")
    sir_result = cached_sir(
        population=10_000,
        initial_infected=10,
        beta=0.30,
        gamma=0.10,
        days=160,
        dt=0.25,
    )
    seir_result = cached_seir(
        population=10_000,
        initial_infected=10,
        initial_exposed=20,
        beta=0.30,
        sigma=0.20,
        gamma=0.10,
        days=160,
        dt=0.25,
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(compartment_figure(sir_result), use_container_width=True)
    with right:
        st.plotly_chart(compartment_figure(seir_result), use_container_width=True)
    st.dataframe(compare_models(sir_result, seir_result), hide_index=True, use_container_width=True)


def about_tab():
    st.subheader("About the Math")
    st.markdown(
        """
        `SIR` assumes people move directly from susceptible to infectious to recovered.
        `SEIR` inserts an exposed compartment, which delays the infectious surge and is often
        a better fit when a disease has a latent period.

        A useful mental model:

        - higher `beta` means more transmission pressure
        - lower `gamma` means people remain infectious longer
        - higher `sigma` means exposed people become infectious sooner
        - lowering `beta` after an intervention flattens and delays the peak
        """
    )


tab_home, tab_sir, tab_seir, tab_scenarios, tab_sensitivity, tab_about = st.tabs(
    ["Home", "SIR Model", "SEIR Model", "Scenario Simulator", "Parameter Sensitivity", "About the Math"]
)

with tab_home:
    home_tab()

with tab_sir:
    render_model_tab("SIR")

with tab_seir:
    render_model_tab("SEIR")

with tab_scenarios:
    scenario_tab()

with tab_sensitivity:
    sensitivity_tab()

with tab_about:
    about_tab()
