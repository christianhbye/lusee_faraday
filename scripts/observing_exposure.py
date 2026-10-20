"""Record actual-site nightside exposure for fixed sidereal phase bins."""

import common  # noqa: F401
import argparse
import json
from pathlib import Path

from common import REPO, moon_location, times
from paper_common import manifest, write_json
from lusee_faraday.config import N_TIMES, SIDEREAL_DAY_S
from lusee_faraday.observing import phase_exposures, site_solar_altitudes


def compute(config, path):
    base = json.loads((REPO / "configs/paper/covariance.json").read_text())
    start, location = times()[0], moon_location()
    complete = dict(
        config,
        start_utc=start.utc.isot,
        longitude_deg=float(location.lon.deg),
        latitude_deg=float(location.lat.deg),
        sidereal_turns=base["sidereal_turns"],
        bin_seconds=SIDEREAL_DAY_S / N_TIMES,
    )
    provenance, key = manifest(
        complete,
        [
            "scripts/observing_exposure.py",
            "scripts/common.py",
            "scripts/paper_common.py",
            "src/lusee_faraday/config.py",
            "src/lusee_faraday/observing.py",
        ],
    )

    def solar_altitude(seconds):
        return site_solar_altitudes(start, seconds, location)

    rows = []
    for count in config["lunations"]:
        values = phase_exposures(
            count * config["adopted_lunation_days"] * 86400,
            base["sidereal_turns"],
            solar_altitude,
            config["electronics_duty_cycle"],
            night_limit_deg=config["night_limit_deg"],
        )
        rows.extend(dict(lunations=count, **v) for v in values)
    write_json(
        path, dict(fingerprint=key, provenance=provenance, results=rows)
    )
    for row in rows:
        print(
            row["lunations"],
            row["sidereal_turn"],
            row["eligible_visits"],
            row["exposure_s"],
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/paper/observing.json"
    )
    parser.add_argument(
        "--record", type=Path, default=REPO / "docs/observing-exposure.json"
    )
    args = parser.parse_args()
    compute(json.loads(args.config.read_text()), args.record)
