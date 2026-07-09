from episim.dashboard import (
    get_profile,
    live_bundle,
    live_only_bundle,
    payload_size_bytes,
    table_bundle,
)


TEST_PARAMS = {
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


def test_lightweight_live_bundle_has_smaller_payload_than_full_bundle():
    full_bundle = live_only_bundle("SIR", lightweight=False, **TEST_PARAMS)
    lightweight_bundle = live_only_bundle("SIR", lightweight=True, **TEST_PARAMS)
    assert payload_size_bytes(lightweight_bundle) < payload_size_bytes(full_bundle)


def test_table_bundle_contains_parameter_and_simulation_rows():
    bundle = table_bundle("SEIR", **TEST_PARAMS)
    assert bundle["parameters"]
    assert bundle["simulation"]
    assert "Exposed" in bundle["simulation"][0]


def test_baseline_bundle_contains_simulation_table_data():
    bundle = live_bundle("SIR", lightweight=False, **TEST_PARAMS)
    assert bundle["simulation"]
    assert bundle["parameters"]


def test_get_profile_returns_clientside_default():
    assert get_profile().name == "step4_clientside"
