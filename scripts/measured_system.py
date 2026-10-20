"""Chromatic I-only means and complete loaded radiometer predictions."""

import common  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from common import CACHE_DIR, REPO, moon_location, times
from chromatic_leakage import setup as leakage_setup
from paper_common import digest, load_record, manifest, write_json
from residual_signal import common_fit
from lusee_faraday import engine
from lusee_faraday.conventions import PORT_PAIRS
from lusee_faraday.config import SIDEREAL_DAY_S
from lusee_faraday.observing import sidereal_bin_window
from lusee_faraday.receiver_noise import (
    channel_radiometer_covariance,
    check_spectral_covariance,
    diagonal_amplifier_covariance,
    stokes_quadratic_matrices,
)

SOURCES = [
    "scripts/measured_system.py",
    "scripts/chromatic_leakage.py",
    "scripts/residual_signal.py",
    "scripts/paper_common.py",
    "scripts/common.py",
    "src/lusee_faraday/config.py",
    "src/lusee_faraday/conventions.py",
    "src/lusee_faraday/engine.py",
    "src/lusee_faraday/instrument.py",
    "src/lusee_faraday/polarimeter.py",
    "src/lusee_faraday/receiver_noise.py",
    "src/lusee_faraday/observing.py",
    "src/lusee_faraday/separability.py",
    "src/lusee_faraday/channelization.py",
]


def matrices_at_native(path, frequency, beam):
    """Small FITS blocks; same monopole-derived Rmoon as target_matrices."""
    import fitsio

    with fitsio.FITS(str(path)) as hdus:
        freq = hdus["freq"].read()
        selected = np.flatnonzero(
            np.isclose(freq, frequency, rtol=0, atol=1e-10)
        )
        if len(selected) != 1:
            raise ValueError("loading frequency must be a unique native bin")
        index = selected[0]

        def read(name):
            return (
                hdus[name + "_real"].read()[index]
                + 1j * hdus[name + "_imag"].read()[index]
            )

        za, rloss = read("ZA"), read("Rloss")
    lmax = beam.shape[-2] - 1
    pair = beam[:, 0, 0, lmax] * np.sqrt(4 * np.pi) / 4
    rsky = np.zeros((4, 4), complex)
    for value, (a, b) in zip(pair, PORT_PAIRS):
        rsky[a, b], rsky[b, a] = value, value.conjugate()
    rmoon = (za + za.conj().T) / 2 - rsky - rloss
    return za, rmoon, rloss


def loaded_covariance(pair, za, zl, rmoon, rloss, tm=0.0, ta=0.0):
    from lusee.Covariance import (
        apply_receiver_loading,
        assemble_open_covariance,
        project_hermitian,
    )

    opened = assemble_open_covariance(
        pair[:, None, :], rmoon[None], rloss[None], T_moon=tm, T_ant=ta
    )
    return np.asarray(
        project_hermitian(apply_receiver_loading(opened, za[None], zl)[0])
    )[:, 0]


def native_means(row, config, receiver, sky, phases):
    import astropy.units as u
    import croissant as cro

    values, loading = [], None
    reference_error = 0.0
    # FaradaySky stores one spectrum per dual block; this is I-only.
    beta, ref = sky["beta"][:, 0], sky["ref"][:, 0]
    rotated = np.asarray(
        cro.rotations.gal2mepa(
            sky["alms"], et=cro.rotations.jd_to_et(times()[0].tdb.jd)
        )
    )
    native_times = times()[0] + (
        np.asarray(config["sidereal_turns"]) * SIDEREAL_DAY_S * u.s
    )
    elapsed = (native_times.tdb - native_times[0].tdb).to_value("s")
    reference_phases = np.asarray(
        cro.simulator.rot_alm_z(sky["alms"].shape[-2] - 1, times=elapsed)
    )
    for file in row["arrays"]:
        path = REPO / file["path"]
        if digest(path) != file["sha256"]:
            raise ValueError("corrupt native leakage artifact")
        with np.load(path) as data:
            original = data["pair_alms"]
            za, rm, rl = matrices_at_native(
                REPO / config["responses"][row["response"]],
                file["frequency_mhz"],
                original,
            )
            beam = engine.response_in_mepa(
                original, times()[0], moon_location(), original.shape[-2] - 1
            )
            coeff = (file["frequency_mhz"] / ref) ** beta

            # I-only: dual block 0, independent synchrotron and CMB spectra.
            def contract(window, sampling_phases=phases):
                contracted = engine.contract_blocks(
                    beam * window, rotated, sampling_phases
                )
                return np.einsum("k,ktp->tp", coeff, contracted[:, 0])

            zl = receiver.Z([file["frequency_mhz"]])
            point = loaded_covariance(
                contract(1.0, reference_phases), za, zl, rm, rl
            )
            difference = float(
                abs(point - data["covariance"]).max()
                / abs(data["covariance"]).max()
            )
            reference_error = max(reference_error, difference)
            values.append(
                loaded_covariance(
                    contract(sidereal_bin_window(original.shape[-2] - 1)),
                    za,
                    zl,
                    rm,
                    rl,
                )
            )
            if file["frequency_mhz"] == row["center_mhz"]:
                loading = (za, rm, rl)
    if reference_error > 1e-11 or loading is None:
        raise ValueError("native loading/phase reference does not reproduce")
    return np.stack(values), loading, reference_error


def compute(config, record_path):
    from croissant.constants import sidereal_day

    leakage = json.loads((REPO / config["leakage_config"]).read_text())
    receiver, _, lk, ld = leakage_setup(leakage)
    record = load_record(REPO / config["leakage_record"], lk)
    if not record["all_passed"]:
        raise ValueError("native chromatic leakage gate failed")
    noise = json.loads((REPO / config["noise_config"]).read_text())
    complete = dict(
        config,
        leakage_fingerprint=lk,
        noise=noise,
        exposure_record_sha256=digest(REPO / config["exposure_record"]),
        reference_sidereal_day_s=float(sidereal_day["moon"]),
        production_sidereal_day_s=SIDEREAL_DAY_S,
    )
    provenance, key = manifest(complete, SOURCES)
    directory = CACHE_DIR / "measured_system" / key
    directory.mkdir(parents=True, exist_ok=True)
    with np.load(ld / "sky_components.npz") as data:
        sky = {name: data[name] for name in data.files}
    lmax = sky["alms"].shape[-2] - 1
    angles = 2 * np.pi * np.asarray(leakage["sidereal_turns"])
    phases = np.exp(-1j * angles[:, None] * np.arange(-lmax, lmax + 1))
    rows, noise_rows = [], []
    for center in leakage["centers_mhz"]:
        reference = next(
            r
            for r in record["results"]
            if r["center_mhz"] == center and r["response"] == "asbuilt"
        )
        x = np.array(reference["x_real"]) + 1j * np.array(reference["x_imag"])
        y = np.array(reference["y_real"]) + 1j * np.array(reference["y_imag"])
        readout = stokes_quadratic_matrices(
            x, y, reference["normalization_v2_per_hz_per_k"]
        )
        fine, w, nominal, residual, fits = common_fit(
            center, config["smooth_degree"]
        )
        df = (fine[1] - fine[0]) * 1e6
        for row in [r for r in record["results"] if r["center_mhz"] == center]:
            path = directory / f"{row['response']}_{center:g}.npz"
            if path.exists():
                check = load_record(path.with_suffix(".json"), key)
                if digest(path) != check["sha256"]:
                    raise ValueError("corrupt averaged leakage checkpoint")
                result = check["result"]
                print("verified", path.name, flush=True)
            else:
                print("averaging", row["response"], center, flush=True)
                native, loading, error = native_means(
                    row, leakage, receiver, sky, phases
                )
                functions = {
                    "linear": interp1d(row["native_mhz"], native, axis=0),
                    "pchip": PchipInterpolator(
                        row["native_mhz"], native.real, axis=0
                    ),
                }
                # Interpolate real/imag separately; preserve Hermiticity.
                imaginary = PchipInterpolator(
                    row["native_mhz"], native.imag, axis=0
                )
                arrays = {}
                for name, func in functions.items():
                    covariance = func(fine)
                    if name == "pchip":
                        covariance = covariance + 1j * imaginary(fine)
                    check_spectral_covariance(covariance)
                    stokes = np.einsum(
                        "pab,ftba->ftp", readout, covariance
                    ).real
                    arrays[name + "_stokes_k"] = w @ stokes.reshape(
                        len(fine), -1
                    )
                    arrays[name + "_residual_k"] = (
                        residual @ arrays[name + "_stokes_k"]
                    )
                    if row["response"] == "asbuilt" and name == "linear":
                        arrays["fine_sky_covariance"] = covariance
                np.savez_compressed(
                    path,
                    nominal_mhz=nominal,
                    operator=residual,
                    readout=readout,
                    native_covariance=native,
                    za=loading[0],
                    rmoon=loading[1],
                    rloss=loading[2],
                    **arrays,
                )
                result = dict(
                    response=row["response"],
                    center_mhz=center,
                    native_mhz=row["native_mhz"],
                    reference_relative_error=error,
                    path=str(path.relative_to(REPO)),
                    sha256=digest(path),
                )
                write_json(
                    path.with_suffix(".json"),
                    dict(
                        fingerprint=key,
                        sha256=digest(path),
                        result=result,
                    ),
                )
            rows.append(result)
        asbuilt = next(
            r
            for r in rows
            if r["center_mhz"] == center and r["response"] == "asbuilt"
        )
        with np.load(REPO / asbuilt["path"]) as data:
            za, rm, rl = data["za"], data["rmoon"], data["rloss"]
            system_sky = data["fine_sky_covariance"]
        zl = receiver.Z([center])
        thermal = loaded_covariance(
            np.zeros((1, 10), complex),
            za,
            zl,
            rm,
            rl,
            noise["moon_temperature_k"],
            noise["antenna_temperature_k"],
        )[0]
        for e in noise["voltage_asd_nv_rt_hz"]:
            for current in noise["current_asd_pa_rt_hz"]:
                path = directory / f"noise_{center:g}_{e:g}_{current:g}.npz"
                if path.exists():
                    check = load_record(path.with_suffix(".json"), key)
                    if digest(path) != check["sha256"]:
                        raise ValueError("corrupt radiometer checkpoint")
                    result = check["result"]
                else:
                    print("noise", center, e, current, flush=True)
                    amp = diagonal_amplifier_covariance(
                        za, zl[0], e * 1e-9, current * 1e-12
                    )
                    system = system_sky + thermal + amp
                    raw, filtered = [], []
                    for phase in range(len(angles)):
                        raw.append(
                            channel_radiometer_covariance(
                                system[:, phase], w, df, 1.0, readout
                            )
                        )
                        filtered.append(
                            channel_radiometer_covariance(
                                system[:, phase],
                                residual @ w,
                                df,
                                1.0,
                                readout,
                            )
                        )
                    np.savez_compressed(
                        path,
                        covariance_per_second=raw,
                        residual_covariance_per_second=filtered,
                        system_center_psd=system[len(fine) // 2],
                        thermal_psd=thermal,
                        amplifier_psd=amp,
                        bias_k=np.einsum(
                            "pab,ba->p", readout, thermal + amp
                        ).real,
                        nominal_mhz=nominal,
                        bandwidth_hz=df / np.sum(w * w, axis=1),
                    )
                    result = dict(
                        center_mhz=center,
                        voltage_nv_rt_hz=e,
                        current_pa_rt_hz=current,
                        path=str(path.relative_to(REPO)),
                        sha256=digest(path),
                    )
                    write_json(
                        path.with_suffix(".json"),
                        dict(
                            fingerprint=key, sha256=digest(path), result=result
                        ),
                    )
                noise_rows.append(result)
    write_json(
        record_path,
        dict(
            fingerprint=key,
            provenance=provenance,
            leakage=rows,
            noise=noise_rows,
            all_passed=True,
            scope=(
                "Native nside=512 Haslam+CMB; full chromatic beam/loading; "
                "as-built calibration and common MEPA boxcar for all "
                "variants. "
                "Point references use the upstream engine's own rotation "
                "clock; production uses the pinned FFT-ready sidereal clock. "
                "Both periods are recorded without modifying upstream. "
                "Linear and PCHIP interpolation are model choices. Noise uses "
                "full chromatic I-only power plus separate frozen Moon, loss "
                "and voltage/current terms. Per-second covariance is divided "
                "by available exposure; zero exposure is unavailable. Bias "
                "spectra are exported and assumed calibrated for sky-I "
                "ratios. "
                "Polarized sky power is omitted from system noise and must be "
                "small compared with I for these conditional examples."
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/paper/separability.json"
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPO / "docs/measured-system-results.json",
    )
    args = parser.parse_args()
    compute(json.loads(args.config.read_text()), args.record)
