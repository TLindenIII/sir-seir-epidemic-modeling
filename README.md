# Interactive Epidemiological Modeling with SIR and SEIR Simulations

An educational epidemiology project that explains compartmental modeling, simulates SIR and SEIR outbreaks, compares intervention assumptions, and exposes the results through a lightweight Dash app with live slider updates.

## Why this project exists

This repository is designed to be both a learning resource and a polished portfolio artifact. It emphasizes:

- clear mathematical framing
- fast deterministic simulations suitable for drag-updating sliders
- measurable performance profiling across multiple optimization stages
- reusable analysis helpers for scenario comparison
- test coverage around the simulation core

## Features

- Fast fixed-step RK4 solvers for SIR and SEIR dynamics
- Intervention modeling through time-varying transmission reduction
- Summary metrics such as peak infections, peak day, outbreak duration, and attack rate
- Parameter sensitivity analysis utilities
- Plotly visualizations embedded in Dash
- Clientside live slider updates for the SIR and SEIR model tabs
- Benchmark script that compares baseline, split-callback, lightweight-figure, and clientside profiles
- Unit tests covering conservation, interventions, and metric calculations

## Repository layout

```text
.
├── app/
│   ├── assets/
│   │   ├── episim-clientside.js
│   │   └── theme.css
│   └── dash_app.py
├── data/
│   └── sample_cases.csv
├── figures/
├── notebooks/
├── scripts/
│   └── benchmark_profiles.py
├── src/
│   └── episim/
│       ├── __init__.py
│       ├── dashboard.py
│       ├── models.py
│       ├── plotting.py
│       ├── simulation.py
│       └── utils.py
└── tests/
    ├── test_dash_app.py
    ├── test_dashboard_profiles.py
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
python app/dash_app.py
```

To inspect the performance ladder directly:

```bash
python scripts/benchmark_profiles.py
```

To force a specific optimization stage at runtime:

```bash
EPISIM_OPTIMIZATION_MODE=baseline python app/dash_app.py
EPISIM_OPTIMIZATION_MODE=step4_clientside python app/dash_app.py
```

## Planned extensions

- Fit model parameters to real or synthetic case data
- Add stochastic outbreak simulation for small populations
- Compare deterministic and network-based epidemic spread
