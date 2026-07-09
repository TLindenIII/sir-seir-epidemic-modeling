from __future__ import annotations

import json
import subprocess
from pathlib import Path
from statistics import mean
from time import perf_counter

from episim.dashboard import (
    PROFILE_ORDER,
    get_profile,
    live_bundle,
    live_only_bundle,
    payload_size_bytes,
    table_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CLIENTSIDE_PATH = ROOT / "app" / "assets" / "episim-clientside.js"
BENCHMARK_PARAMS = {
    "population": 10_000,
    "initial_infected": 10,
    "initial_exposed": 20,
    "beta": 0.30,
    "sigma": 0.20,
    "gamma": 0.10,
    "days": 160,
    "intervention_day": 30,
    "intervention_strength": 0.40,
}


def timed_mean(fn, iterations: int = 100) -> tuple[float, object]:
    samples = []
    payload = None
    for _ in range(iterations):
        start = perf_counter()
        payload = fn()
        samples.append((perf_counter() - start) * 1000)
    return mean(samples), payload


def benchmark_python_profile(profile_name: str) -> dict[str, float | str]:
    profile = get_profile(profile_name)
    if profile.name == "baseline":
        live_ms, payload = timed_mean(
            lambda: live_bundle("SIR", lightweight=False, **BENCHMARK_PARAMS)
        )
        return {
            "profile": profile.name,
            "path": "python-live-all",
            "live_ms": round(live_ms, 3),
            "live_kb": round(payload_size_bytes(payload) / 1024, 2),
            "table_ms": 0.0,
            "table_kb": 0.0,
        }

    if profile.name == "step1_no_live_table":
        live_ms, live_payload = timed_mean(
            lambda: (lambda payload: {
                "figure": payload["figure"],
                "metrics": payload["metrics"],
                "parameters": payload["parameters"],
            })(live_bundle("SIR", lightweight=False, **BENCHMARK_PARAMS))
        )
        table_ms, table_payload = timed_mean(
            lambda: table_bundle("SIR", **BENCHMARK_PARAMS)["simulation"]
        )
        return {
            "profile": profile.name,
            "path": "python-live-graph-metrics-params",
            "live_ms": round(live_ms, 3),
            "live_kb": round(payload_size_bytes(live_payload) / 1024, 2),
            "table_ms": round(table_ms, 3),
            "table_kb": round(payload_size_bytes(table_payload) / 1024, 2),
        }

    live_ms, live_payload = timed_mean(
        lambda: live_only_bundle(
            "SIR",
            lightweight=profile.lightweight_figure,
            **BENCHMARK_PARAMS,
        )
    )
    table_ms, tables = timed_mean(lambda: table_bundle("SIR", **BENCHMARK_PARAMS))
    return {
        "profile": profile.name,
        "path": "python-live-graph-metrics" if not profile.clientside_live else "browser-live-graph-metrics",
        "live_ms": round(live_ms, 3),
        "live_kb": round(payload_size_bytes(live_payload) / 1024, 2),
        "table_ms": round(table_ms, 3),
        "table_kb": round(payload_size_bytes(tables) / 1024, 2),
    }


def benchmark_clientside_js(iterations: int = 200) -> dict[str, float | str]:
    node_script = f"""
const fs = require('fs');
const {{ performance }} = require('perf_hooks');
global.window = {{ dash_clientside: {{ no_update: null }}, episimPerf: {{}} }};
eval(fs.readFileSync({json.dumps(str(CLIENTSIDE_PATH))}, 'utf8'));
const fn = window.episimPerf.buildSirLivePayload;
const params = {json.dumps(BENCHMARK_PARAMS)};
const samples = [];
let payload = null;
for (let i = 0; i < {iterations}; i += 1) {{
  const start = performance.now();
  payload = fn(params.population, params.initial_infected, params.beta, params.gamma, params.days, params.intervention_day, params.intervention_strength);
  samples.push(performance.now() - start);
}}
const mean = samples.reduce((acc, value) => acc + value, 0) / samples.length;
const size = Buffer.byteLength(JSON.stringify(payload), 'utf8');
console.log(JSON.stringify({{ mean, size }}));
"""
    completed = subprocess.run(
        ["node", "-e", node_script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    return {
        "profile": "step4_clientside",
        "path": "browser-live-graph-metrics",
        "live_ms": round(result["mean"], 3),
        "live_kb": round(result["size"] / 1024, 2),
    }


def main() -> int:
    records = [benchmark_python_profile(name) for name in PROFILE_ORDER[:-1]]
    clientside = benchmark_clientside_js()
    records.append(
        {
            "profile": clientside["profile"],
            "path": clientside["path"],
            "live_ms": clientside["live_ms"],
            "live_kb": clientside["live_kb"],
            "table_ms": records[-1]["table_ms"],
            "table_kb": records[-1]["table_kb"],
        }
    )

    print("profile,path,live_ms,live_kb,table_ms,table_kb")
    for record in records:
        print(
            ",".join(
                str(
                    record.get(
                        key,
                        "",
                    )
                )
                for key in (
                    "profile",
                    "path",
                    "live_ms",
                    "live_kb",
                    "table_ms",
                    "table_kb",
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
