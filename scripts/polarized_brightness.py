"""Beam-integrated polarized brightness in mK and as a fraction of Stokes I.

Reads the local-medium sensitivity cases and the as-built leakage record and
reports, per band and sidereal phase, the raw (before the smooth fit)
ensemble RMS of calibrated pseudo-Stokes Q and U in parent channels, in mK
and as a fraction of the beam-weighted Stokes I, with the as-built
I -> Q,U leakage on the same footing. This re-expresses existing results
for comparison with external wide-beam estimates; it is not a new
prediction.
"""

import common  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np

from common import REPO
from paper_common import digest, manifest, write_json

PARENT = np.arange(3)  # bins 0-2 are parent channels, 3-194 zoom bins
QU = np.array([1, 2])


def load(row):
    path = REPO / row["path"]
    if digest(path) != row["sha256"]:
        raise ValueError(f"corrupt input {row['path']}")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def variance_rms(variance, phase, bins=PARENT):
    """Q, U RMS from per-bin variances laid out as (bin, 4*phase+Stokes)."""
    v = np.asarray(variance)[bins][:, 4 * phase + QU]
    return np.sqrt(np.maximum(v.mean(axis=0), 0.0))


def linear_rms(stokes, phase, bins=PARENT):
    """Q, U RMS over bins of a linear (bin, 4*phase+Stokes) spectrum."""
    s = np.asarray(stokes)[bins][:, 4 * phase + QU]
    return np.sqrt((s**2).mean(axis=0))


def stokes_i(stokes, phase, bins=PARENT):
    return float(np.asarray(stokes)[bins][:, 4 * phase].mean())


def profile_label(profile):
    if profile["kind"] == "screen":
        return f"screen f={profile['fraction']:g}"
    return profile["kind"]


def summary(values):
    v = np.asarray(values)
    return dict(
        min=float(v.min()), median=float(np.median(v)), max=float(v.max())
    )


def compute(path):
    sens_path = REPO / "docs/local-medium-sensitivity-results.json"
    system_path = REPO / "docs/measured-system-results.json"
    sep_path = REPO / "docs/local-medium-separability-results.json"
    sens = json.loads(sens_path.read_text())
    system = json.loads(system_path.read_text())
    sep = json.loads(sep_path.read_text())
    if not (sens["all_passed"] and sep["all_passed"]):
        raise ValueError("input numerical gates failed")
    provenance, fp = manifest(
        dict(
            input_records={
                str(p.relative_to(REPO)): digest(p)
                for p in (sens_path, system_path, sep_path)
            }
        ),
        ["scripts/polarized_brightness.py", "scripts/paper_common.py"],
    )
    phases = len(sep["axes"]["phase"])
    bands = {}
    for center in sorted({r["center_mhz"] for r in sens["results"]}):
        row = next(
            r
            for r in system["leakage"]
            if r["response"] == "asbuilt" and r["center_mhz"] == center
        )
        stokes = load(row)["linear_stokes_k"]
        bands[center] = [
            dict(
                stokes_i_k=stokes_i(stokes, t),
                leakage_rms_k=linear_rms(stokes, t),
            )
            for t in range(phases)
        ]
    cases = []
    for row in sens["results"]:
        raw = load(row)["raw_total"]
        for t in range(phases):
            rms = variance_rms(raw, t)
            ref = bands[row["center_mhz"]][t]
            cases.append(
                dict(
                    case_id=row["case_id"],
                    center_mhz=row["center_mhz"],
                    phase=sep["axes"]["phase"][t],
                    profile=profile_label(row["profile"]),
                    sigma_depth_rad_m2=row["sigma_depth_rad_m2"],
                    mean_depth_rad_m2=row["mean_depth_rad_m2"],
                    model=row["model"],
                    q_rms_mk=float(rms[0] * 1e3),
                    u_rms_mk=float(rms[1] * 1e3),
                    q_over_i=float(rms[0] / ref["stokes_i_k"]),
                    u_over_i=float(rms[1] / ref["stokes_i_k"]),
                )
            )
    groups = {}
    for c in cases:
        key = (c["center_mhz"], c["profile"], c["sigma_depth_rad_m2"])
        groups.setdefault(key, []).append(c)
    by_profile = [
        dict(
            center_mhz=k[0],
            profile=k[1],
            sigma_depth_rad_m2=k[2],
            cases=len(v),
            q_rms_mk=summary([c["q_rms_mk"] for c in v]),
            u_rms_mk=summary([c["u_rms_mk"] for c in v]),
            q_over_i=summary([c["q_over_i"] for c in v]),
            u_over_i=summary([c["u_over_i"] for c in v]),
        )
        for k, v in sorted(groups.items())
    ]
    leakage = [
        dict(
            center_mhz=center,
            phase=sep["axes"]["phase"][t],
            stokes_i_k=b["stokes_i_k"],
            q_rms_k=float(b["leakage_rms_k"][0]),
            u_rms_k=float(b["leakage_rms_k"][1]),
            q_over_i=float(b["leakage_rms_k"][0] / b["stokes_i_k"]),
            u_over_i=float(b["leakage_rms_k"][1] / b["stokes_i_k"]),
        )
        for center, rows in bands.items()
        for t, b in enumerate(rows)
    ]
    write_json(
        path,
        dict(
            fingerprint=fp,
            provenance=provenance,
            all_passed=True,
            intrinsic_rms_k=sens["intrinsic_rms_k"],
            channel_type="parent",
            leakage=leakage,
            by_profile=by_profile,
            cases=cases,
            emissivity_amplitude_factors=[
                dict(
                    total_intensity_at_ref_k_per_pc=e[
                        "total_intensity_at_ref_k_per_pc"
                    ],
                    assumed_polarized_fraction=e["assumed_polarized_fraction"],
                    amplitude_factor=e["amplitude_factor"],
                )
                for e in sep["emissivity_normalization_sensitivity"]
            ],
            scope=(
                "Raw (before the smooth fit) ensemble RMS of calibrated "
                "pseudo-Stokes Q and U in parent channels, smooth plus "
                "speckle, for every local-medium case at the 5 K reference "
                "normalization; rescale linearly by an emissivity amplitude "
                "factor. Stokes I and leakage are the as-built linear "
                "spectra in the same channels. Ensemble RMS, not a band RMS "
                "of one realization; conditional on the same assumptions as "
                "the local-medium grid."
            ),
        ),
    )
    unrot = [p for p in by_profile if p["profile"] == "screen f=0"]
    for p in unrot:
        print(
            f"{p['center_mhz']:g} MHz unrotated: Q rms "
            f"{p['q_rms_mk']['median']:.0f} mK, Q/I "
            f"{p['q_over_i']['median']:.2e}"
        )
    for b in leakage:
        print(
            f"{b['center_mhz']:g} MHz phase {b['phase']}: I "
            f"{b['stokes_i_k']:.0f} K, leakage Q/I {b['q_over_i']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "docs/polarized-brightness-results.json",
    )
    compute(parser.parse_args().out)
