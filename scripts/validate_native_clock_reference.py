"""Reproduce the native engine before switching to the pinned clock."""

import common  # noqa: F401
import json

import numpy as np

from common import REPO
from chromatic_leakage import setup
from measured_system import native_means
from paper_common import digest, load_record, write_json
from lusee_faraday.config import SIDEREAL_DAY_S


def compute():
    import croissant as cro
    from croissant.constants import sidereal_day

    config = json.loads((REPO / "configs/paper/leakage.json").read_text())
    receiver, _, key, directory = setup(config)
    record = load_record(REPO / "docs/chromatic-leakage-results.json", key)
    row = next(
        r
        for r in record["results"]
        if r["response"] == "asbuilt" and r["center_mhz"] == 30
    )
    row = dict(
        row, arrays=[f for f in row["arrays"] if f["frequency_mhz"] == 30]
    )
    with np.load(directory / "sky_components.npz") as data:
        sky = {k: data[k] for k in data.files}
    lmax = sky["alms"].shape[-2] - 1
    angles = 2 * np.pi * np.asarray(config["sidereal_turns"])
    phases = np.exp(-1j * angles[:, None] * np.arange(-lmax, lmax + 1))
    _, _, error = native_means(row, config, receiver, sky, phases)
    original = cro.simulator.rot_alm_z
    caught = False
    try:
        # Memory-only test double; upstream files are never modified.
        cro.simulator.rot_alm_z = lambda *args, **kwargs: phases
        native_means(row, config, receiver, sky, phases)
    except ValueError as exc:
        caught = "does not reproduce" in str(exc)
    finally:
        cro.simulator.rot_alm_z = original
    write_json(
        REPO / "docs/native-clock-reference-validation.json",
        dict(
            passed=error <= 1e-11 and caught,
            native_reference_relative_error=error,
            incorrect_clock_fault_caught=caught,
            upstream_period_s=float(sidereal_day["moon"]),
            pinned_period_s=SIDEREAL_DAY_S,
            period_difference_s=float(sidereal_day["moon"] - SIDEREAL_DAY_S),
            source_sha256={
                name: digest(REPO / name)
                for name in (
                    "scripts/measured_system.py",
                    "scripts/validate_native_clock_reference.py",
                )
            },
            scope=(
                "The saved native snapshot uses the upstream engine clock "
                "and "
                "its actual TDB elapsed timestamps. That geometry is "
                "reproduced "
                "before averaging on the common pinned FFT-ready clock; both "
                "periods remain explicit and upstream checkouts are untouched."
            ),
        ),
    )
    print(
        "native error", error, "wrong-clock fault caught", caught, flush=True
    )
    if error > 1e-11 or not caught:
        raise SystemExit(1)


if __name__ == "__main__":
    compute()
