import json
from pathlib import Path

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "regression_baselines.json"
)

REQUIRED = (
    "point_source_phi_fd",
    "band_centers_mhz",
    "q_oscillation_period_khz",
    "unpolarized_transit_leakage_raw",
    "unpolarized_transit_leakage_ortho",
    "zenith_null_ortho_max",
    "zoom_recovery_real",
    "zoom_recovery_ideal",
    "parent_stokes_over_i_30mhz",
)


def test_fixture_has_every_required_baseline():
    data = json.loads(FIXTURE.read_text())
    missing = [key for key in REQUIRED if key not in data]
    assert missing == [], missing


def test_every_baseline_records_its_source():
    data = json.loads(FIXTURE.read_text())
    for key, value in data.items():
        if key.startswith("_") or not isinstance(value, dict):
            continue
        assert "source" in value, key
