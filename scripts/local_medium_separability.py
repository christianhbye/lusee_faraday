"""Tabulate every signed local-mean case against the common leakage/noise."""

import common  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np

from common import CACHE_DIR, REPO
from local_medium_sensitivity import setup
from paper_common import digest, load_record, manifest, write_json


def load(row):
    path = REPO / row["path"]
    if digest(path) != row["sha256"]:
        raise ValueError("corrupt sensitivity input")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def compute(config, path):
    sep, horizon, _, base, _, _, key, _ = setup(config)
    input_path = REPO / "docs/local-medium-sensitivity-results.json"
    record = load_record(input_path, key)
    if not record["all_passed"]:
        raise ValueError("local-medium numerical gates failed")
    system_path = REPO / "docs/measured-system-results.json"
    system = json.loads(system_path.read_text())
    exposure = json.loads((REPO / sep["exposure_record"]).read_text())
    audit_path = REPO / "docs/local-input-audit-results.json"
    audit = json.loads(audit_path.read_text())
    provenance, fp = manifest(
        dict(
            local_config=config,
            input_records={
                str(p.relative_to(REPO)): digest(p)
                for p in (
                    input_path,
                    system_path,
                    REPO / sep["exposure_record"],
                    audit_path,
                )
            },
        ),
        [
            "scripts/local_medium_separability.py",
            "scripts/local_medium_sensitivity.py",
            "scripts/paper_common.py",
        ],
    )
    directory = CACHE_DIR / "local_medium_separability" / fp
    directory.mkdir(parents=True, exist_ok=True)
    groups = [np.arange(3), np.arange(3, 195)]
    phases = len(base["sidereal_turns"])
    lunations = sorted(set(row["lunations"] for row in exposure["results"]))
    noise_models = sorted(
        set(
            (r["voltage_nv_rt_hz"], r["current_pa_rt_hz"])
            for r in system["noise"]
        )
    )
    tau = np.array(
        [
            [
                next(
                    e["exposure_s"]
                    for e in exposure["results"]
                    if e["lunations"] == count and e["sidereal_turn"] == turn
                )
                for count in lunations
            ]
            for turn in base["sidereal_turns"]
        ]
    )
    noise_rms = {}
    leakage = {}
    intensity = {}
    for center in sorted(set(row["center_mhz"] for row in record["results"])):
        row = next(
            r
            for r in system["leakage"]
            if r["response"] == "asbuilt" and r["center_mhz"] == center
        )
        data = load(row)
        leakage[center] = np.array(
            [
                [
                    [
                        np.sqrt(
                            np.mean(
                                data["linear_residual_k"][indices, 4 * t + c]
                                ** 2
                            )
                        )
                        for c in (1, 2)
                    ]
                    for indices in groups
                ]
                for t in range(phases)
            ]
        )
        intensity[center] = float(np.min(data["linear_stokes_k"][:, ::4]))
        for e, current in noise_models:
            row = next(
                r
                for r in system["noise"]
                if r["center_mhz"] == center
                and r["voltage_nv_rt_hz"] == e
                and r["current_pa_rt_hz"] == current
            )
            data = load(row)
            noise_rms[center, e, current] = np.array(
                [
                    [
                        [
                            np.sqrt(
                                max(
                                    np.diag(
                                        data["residual_covariance_per_second"][
                                            t, :, c, :, c
                                        ]
                                    )[indices].mean(),
                                    0,
                                )
                            )
                            for c in (1, 2)
                        ]
                        for indices in groups
                    ]
                    for t in range(phases)
                ]
            )
    shape = (len(record["results"]), phases, 2, 2)
    raw, speckle, smooth = [np.zeros(shape) for _ in range(3)]
    noise_ratio = np.full(shape + (len(lunations), len(noise_models)), np.nan)
    total_ratio = np.full_like(noise_ratio, np.nan)
    leakage_ratio = np.full(shape, np.nan)
    coordinate = []
    largest_stokes_fraction = 0.0
    for i, row in enumerate(record["results"]):
        data = load(row)
        largest_stokes_fraction = max(
            largest_stokes_fraction,
            max(
                float(
                    np.sqrt(
                        np.maximum(
                            data["raw_total"]
                            .reshape(-1, phases, 4)[:, t]
                            .sum(axis=1),
                            0,
                        )
                    ).max()
                )
                / intensity[row["center_mhz"]]
                for t in range(phases)
            ),
        )
        for t in range(phases):
            for g, indices in enumerate(groups):
                for j, c in enumerate((1, 2)):
                    raw[i, t, g, j] = np.sqrt(
                        max(data["raw_speckle"][indices, 4 * t + c].mean(), 0)
                    )
                    speckle[i, t, g, j] = np.sqrt(
                        max(
                            data["residual_speckle"][
                                indices, 4 * t + c
                            ].mean(),
                            0,
                        )
                    )
                    smooth[i, t, g, j] = np.sqrt(
                        max(
                            data["residual_smooth"][indices, 4 * t + c].mean(),
                            0,
                        )
                    )
        for ni, (e, current) in enumerate(noise_models):
            for li in range(len(lunations)):
                for t in range(phases):
                    if tau[t, li] > 0:
                        noise = noise_rms[row["center_mhz"], e, current][
                            t
                        ] / np.sqrt(tau[t, li])
                        noise_ratio[i, t, :, :, li, ni] = speckle[i, t] / noise
                        total_ratio[i, t, :, :, li, ni] = (
                            np.hypot(speckle[i, t], smooth[i, t]) / noise
                        )
        np.divide(
            speckle[i],
            leakage[row["center_mhz"]],
            out=leakage_ratio[i],
            where=leakage[row["center_mhz"]] > 0,
        )
        coordinate.append(
            {
                k: v
                for k, v in row.items()
                if k
                in (
                    "case_id",
                    "center_mhz",
                    "mean_depth_rad_m2",
                    "mean_sources",
                    "sigma_depth_rad_m2",
                    "screen_index",
                    "profile_index",
                    "profile",
                    "model",
                    "intrinsic_rms_k",
                )
            }
        )
    transmission = np.divide(
        speckle**2, raw**2, out=np.full_like(raw, np.nan), where=raw > 0
    )
    array_path = directory / "decision_grid.npz"
    np.savez_compressed(
        array_path,
        raw_speckle_rms_k=raw,
        residual_speckle_rms_k=speckle,
        residual_smooth_rms_k=smooth,
        retained_speckle_power=transmission,
        speckle_to_noise_amplitude=noise_ratio,
        total_to_noise_amplitude=total_ratio,
        speckle_to_leakage_amplitude=leakage_ratio,
        exposure_s=tau,
    )
    freq = np.asarray(audit["frequencies_mhz"])
    index = np.flatnonzero(
        np.isclose(freq, horizon["emissivity_ref_mhz"], rtol=0, atol=1e-12)
    )
    if len(index) != 1:
        raise ValueError("emissivity audit lacks the reference frequency")
    scales = []
    for row in audit["emissivity"]:
        value = row["polarized_emissivity_k_per_pc"][index[0]]
        factor = value / horizon["emissivity_ref_k_per_pc"]
        scales.append(
            dict(
                **row,
                polarized_emissivity_at_reference_k_per_pc=value,
                amplitude_factor=factor,
                scaled_stokes_rms_fraction=largest_stokes_fraction * factor,
                linear_noise_approximation_supported=(
                    largest_stokes_fraction * factor <= 0.05
                ),
                largest_speckle_to_noise_amplitude=float(
                    np.nanmax(noise_ratio) * factor
                ),
                largest_total_polarization_to_noise_amplitude=float(
                    np.nanmax(total_ratio) * factor
                ),
            )
        )
    write_json(
        path,
        dict(
            fingerprint=fp,
            provenance=provenance,
            all_passed=True,
            coordinates=coordinate,
            arrays=dict(
                path=str(array_path.relative_to(REPO)),
                sha256=digest(array_path),
            ),
            axes=dict(
                case="coordinates in row order",
                phase=base["sidereal_turns"],
                channel_type=["parent", "zoom"],
                component=["Q", "U"],
                lunations=lunations,
                noise_model=[
                    dict(voltage_nv_rt_hz=e, current_pa_rt_hz=c)
                    for e, c in noise_models
                ],
            ),
            array_axes=[
                "case",
                "phase",
                "channel_type",
                "component",
                "lunations",
                "noise_model",
            ],
            unavailable_measurements=(
                "NaN in binary arrays; zero exposure has no noise ratio"
            ),
            intrinsic_rms_k=record["intrinsic_rms_k"],
            largest_speckle_to_noise_amplitude=float(np.nanmax(noise_ratio)),
            largest_total_polarization_to_noise_amplitude=float(
                np.nanmax(total_ratio)
            ),
            emissivity_normalization_sensitivity=scales,
            largest_polarized_stokes_rms_fraction=largest_stokes_fraction,
            scope=(
                "All signed mean columns implied by the adopted source "
                "pairs and "
                "path, all five profiles, both conditional dispersions, both "
                "phase distributions, both bands and parent/zoom types. "
                "The full "
                "emitting normalization is epsilon times path; depolarization "
                "comes from the depth integral. HII emissivity cases only "
                "rescale this normalization; local polarized emission is "
                "not measured. "
                "No calibrated separation significance is claimed."
            ),
        ),
    )
    print(
        "Cases",
        len(coordinate),
        "max speckle/noise",
        np.nanmax(noise_ratio),
        "max total/noise",
        np.nanmax(total_ratio),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/paper/local-medium-sensitivity.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPO / "docs/local-medium-separability-results.json",
    )
    args = parser.parse_args()
    compute(json.loads(args.config.read_text()), args.record)
