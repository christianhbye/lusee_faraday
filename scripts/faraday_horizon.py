"""Regenerate the conditional Faraday horizon table; see --help."""

import common  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np

from common import REPO
from paper_common import manifest, write_json
from lusee_faraday.faraday_horizon import (
    ROTATION_PER_PC,
    faraday_rate,
    horizon_depth,
    unrotated_brightness,
)

SOURCES = [
    "scripts/common.py",
    "scripts/paper_common.py",
    "scripts/faraday_horizon.py",
    "src/lusee_faraday/faraday_horizon.py",
    "src/lusee_faraday/conventions.py",
    "src/lusee_faraday/config.py",
]


def compute(config):
    freq = config["frequencies_mhz"]
    projections = np.asarray(config["absolute_field_projection"], float)
    if not np.all(
        np.isfinite(projections) & (projections >= 0) & (projections <= 1)
    ):
        raise ValueError("absolute field projections must lie in [0,1]")
    rows = []
    for scenario in config["scenarios"]:
        for projection in projections:
            field = float(scenario["field_ug"] * projection)
            for limit in config["phase_limits_rad"]:
                depth = horizon_depth(freq, scenario["ne_cm3"], field, limit)
                brightness = unrotated_brightness(
                    freq,
                    depth,
                    config["emitting_path_pc"],
                    config["emissivity_ref_k_per_pc"],
                    config["emissivity_ref_mhz"],
                    config["emissivity_beta"],
                )
                for i, f in enumerate(freq):
                    rows.append(
                        dict(
                            scenario=scenario["name"],
                            freq_mhz=f,
                            phase_limit_rad=limit,
                            absolute_field_projection=float(projection),
                            rate_rad_m2_pc=float(
                                faraday_rate(scenario["ne_cm3"], field)
                            ),
                            horizon_pc=(
                                float(depth[i])
                                if np.isfinite(depth[i])
                                else None
                            ),
                            horizon_unbounded=bool(np.isinf(depth[i])),
                            unrotated_brightness_k=float(brightness[i]),
                            longer_than_100_pc_at_30_mhz=bool(
                                f == 30 and depth[i] > 100
                            ),
                        )
                    )
    provenance, key = manifest(config, SOURCES)
    return dict(
        fingerprint=key,
        manifest=provenance,
        rows=rows,
        rotation_coefficient_rad_m2_pc=ROTATION_PER_PC,
        scope=config["scope"],
        limitations=[
            "A uniform signed rotation rate defines the horizon convention.",
            "Direction, finite emitting volume and emissivity are assumed.",
            "No angular or stochastic depth statistics are inferred.",
            "Brightness and its spectrum are illustrative and unrotated.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/paper/horizon.json"
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPO / "docs/faraday-horizon-results.json",
    )
    args = parser.parse_args()
    record = compute(json.loads(args.config.read_text()))
    write_json(args.record, record)
    print(f"wrote {args.record} ({len(record['rows'])} scenarios/frequencies)")


if __name__ == "__main__":
    main()
