# Interactive Epidemiological Modeling with SIR and SEIR Simulations

An educational epidemiology project that explains compartmental modeling, simulates SIR and SEIR outbreaks, compares intervention assumptions, and exposes the results through a lightweight Streamlit app.

## Why this project exists

This repository is designed to be both a learning resource and a polished portfolio artifact. It emphasizes:

- clear mathematical framing
- fast deterministic simulations suitable for interactive sliders
- reusable analysis helpers for scenario comparison
- test coverage around the simulation core

## Features

- Fast fixed-step RK4 solvers for SIR and SEIR dynamics
- Intervention modeling through time-varying transmission reduction
- Summary metrics such as peak infections, peak day, outbreak duration, and attack rate
- Parameter sensitivity analysis utilities
- Plotly visualizations embedded in Streamlit
- Unit tests covering conservation, interventions, and metric calculations

## Repository layout

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   └── sample_cases.csv
├── figures/
├── notebooks/
├── src/
│   └── episim/
│       ├── __init__.py
│       ├── models.py
│       ├── plotting.py
│       ├── simulation.py
│       └── utils.py
└── tests/
    └── test_models.py
```

## Mathematical models

### SIR

\[
\frac{dS}{dt} = -\beta \frac{SI}{N}, \quad
\frac{dI}{dt} = \beta \frac{SI}{N} - \gamma I, \quad
\frac{dR}{dt} = \gamma I
\]

### SEIR

\[
\frac{dS}{dt} = -\beta \frac{SI}{N}, \quad
\frac{dE}{dt} = \beta \frac{SI}{N} - \sigma E, \quad
\frac{dI}{dt} = \sigma E - \gamma I, \quad
\frac{dR}{dt} = \gamma I
\]

Where:

- `beta` is the transmission rate
- `gamma` is the recovery rate
- `sigma` is the exposed-to-infectious transition rate
- `R0 = beta / gamma` for the basic SIR framing

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
streamlit run app/streamlit_app.py
```

## Planned extensions

- Fit model parameters to real or synthetic case data
- Add stochastic outbreak simulation for small populations
- Compare deterministic and network-based epidemic spread
