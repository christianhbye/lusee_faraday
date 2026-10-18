"""Supplementary local path/emissivity and conditional lunar-plasma bounds."""

import common  # noqa: F401
import argparse
import json
from pathlib import Path

import astropy.units as u
import numpy as np

from common import REPO
from paper_common import manifest, write_json
from lusee_faraday.conventions import lambda_squared
from lusee_faraday.faraday_horizon import ROTATION_PER_PC, faraday_rate


def compute(config, record):
    freq = np.array([10.0, 30.0, 50.0])
    psr, emissivity, lunar = (
        config[name] for name in ("pulsar", "emissivity", "lunar_plasma")
    )
    plasma = []
    for density in lunar["density_cm3"]:
        path_pc = (lunar["path_km"] * u.km).to_value(u.pc)
        field_ug = (lunar["aligned_field_nt"] * u.nT).to_value(u.uG)
        depth = float(faraday_rate(density, field_ug) * path_pc)
        plasma.append(
            dict(
                density_cm3=density,
                path_pc=path_pc,
                field_ug=field_ug,
                faraday_bound_rad_m2=depth,
                complex_polarization_phase_bound_rad=(
                    2 * depth * lambda_squared(freq)
                ).tolist(),
            )
        )
    brightness = []
    for value in emissivity["total_intensity_k_per_pc"]:
        for fraction in emissivity["assumed_polarized_fractions"]:
            brightness.append(
                dict(
                    total_intensity_at_ref_k_per_pc=value,
                    assumed_polarized_fraction=fraction,
                    polarized_emissivity_k_per_pc=(
                        value
                        * fraction
                        * (freq / emissivity["reference_mhz"])
                        ** emissivity["spectral_index"]
                    ).tolist(),
                )
            )
    provenance, key = manifest(
        config,
        [
            "scripts/local_input_audit.py",
            "scripts/paper_common.py",
            "src/lusee_faraday/conventions.py",
            "src/lusee_faraday/faraday_horizon.py",
        ],
    )
    write_json(
        record,
        dict(
            fingerprint=key,
            provenance=provenance,
            frequencies_mhz=freq.tolist(),
            path_average_ne_cm3=psr["dm_pc_cm3"] / psr["distance_pc"],
            historical_path_average_field_ug=(
                psr["historical_rm_rad_m2"]
                / ROTATION_PER_PC
                / psr["dm_pc_cm3"]
            ),
            lunar_plasma=plasma,
            emissivity=brightness,
            scope=(
                "Conditional unit-checked arithmetic. Historical pulsar path "
                "averages and kpc-scale total emissivity do not identify "
                "local "
                "polarized variance. Lunar envelope is assumed, not a "
                "measured "
                "universal nightside limit. No launch date is assumed."
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/paper/local-input-audit.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPO / "docs/local-input-audit-results.json",
    )
    args = parser.parse_args()
    compute(json.loads(args.config.read_text()), args.record)
