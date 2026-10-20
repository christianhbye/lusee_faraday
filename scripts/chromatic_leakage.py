"""Native-frequency I-only response, model ablations and loaded covariances."""

import common  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np

from common import CACHE_DIR, REPO, moon_location, times
from paper_common import digest, load_record, manifest, write_json
from lusee_faraday import engine, instrument, polarimeter, response
from lusee_faraday.config import BETA_I, FREQ_REF_I, SIDEREAL_DAY_S, T_CMB
from lusee_faraday.sky import FaradaySky

SOURCES = [
    "scripts/common.py",
    "scripts/paper_common.py",
    "scripts/chromatic_leakage.py",
    "src/lusee_faraday/config.py",
    "src/lusee_faraday/conventions.py",
    "src/lusee_faraday/engine.py",
    "src/lusee_faraday/instrument.py",
    "src/lusee_faraday/polarimeter.py",
    "src/lusee_faraday/response.py",
    "src/lusee_faraday/sky.py",
]


def setup(config):
    from lusee.ReceiverImpedance import JFETReceiver

    receiver = JFETReceiver(**config["receiver_parameters"])
    parameters = dict(
        config,
        resolved_receiver_parameters={
            k: np.asarray(v).tolist() for k, v in receiver.params.items()
        },
        receiver_channel_map=list(receiver.channel_map),
        input_sha256={
            name: digest(REPO / name)
            for name in [config["intensity_map"]]
            + list(config["responses"].values())
        },
        spectral_index=BETA_I,
        reference_mhz=FREQ_REF_I,
        cmb_k=T_CMB,
    )
    provenance, key = manifest(parameters, SOURCES)
    directory = CACHE_DIR / "chromatic_leakage" / key
    return receiver, provenance, key, directory


def load_intensity(path):
    from astropy.io import fits

    with fits.open(path) as hdus:
        header = hdus[1].header
        if (
            header["ORDERING"].strip().upper() != "RING"
            or header["NSIDE"] != 512
        ):
            raise ValueError("intensity input must be native nside=512 RING")
        values = hdus[1].data["TEMPERATURE"].ravel().astype(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite Haslam intensity")
    return values - T_CMB


def sky_components(config, directory):
    path = directory / "sky_components.npz"
    metadata = path.with_suffix(".json")
    if path.exists():
        if digest(path) != json.loads(metadata.read_text())["sha256"]:
            raise ValueError("corrupt I-only sky cache")
        with np.load(path) as data:
            return FaradaySky(
                data["alms"],
                phi_fd=[0.0, 0.0],
                beta=data["beta"],
                ref_freq_mhz=data["ref"],
            )
    values = load_intensity(REPO / config["intensity_map"])
    sky = FaradaySky.i_only(
        values, config["lmax"], beta_i=BETA_I, ref_freq_i=FREQ_REF_I
    )
    cmb = FaradaySky.i_only(np.full_like(values, T_CMB), config["lmax"])
    result = FaradaySky(
        np.r_[sky.component_alms, cmb.component_alms],
        phi_fd=[0.0, 0.0],
        beta=np.r_[sky.beta, cmb.beta],
        ref_freq_mhz=np.r_[sky.ref_freq_mhz, cmb.ref_freq_mhz],
    )
    np.savez_compressed(
        path,
        alms=result.component_alms,
        beta=result.beta,
        ref=result.ref_freq_mhz,
    )
    write_json(metadata, dict(sha256=digest(path)))
    return result


def truncate(alms, lmax):
    full = alms.shape[-2] - 1
    start, stop = full - lmax, full + lmax + 1
    ell_stop = lmax + 1
    return alms[..., :ell_stop, start:stop]


def compute(config, record_path):
    import astropy.units as u

    receiver, provenance, key, directory = setup(config)
    directory.mkdir(parents=True, exist_ok=True)
    sky = sky_components(config, directory)
    tt = (
        times()[0]
        + np.asarray(config["sidereal_turns"]) * SIDEREAL_DAY_S * u.s
    )
    rows = []
    for name, response_path in config["responses"].items():
        print("loading response", name, flush=True)
        resp = response.load_response(REPO / response_path)
        for center in config["centers_mhz"]:
            native = np.asarray(resp.freq, float)
            index = response.native_channel_index(resp, center)
            count = config["native_neighbors_each_side"]
            first, last = max(0, index - count), min(
                len(native), index + count + 1
            )
            selected = native[first:last]
            if len(selected) != 2 * count + 1:
                raise ValueError("native frequency bracket does not fit")
            x, y, _ = polarimeter.zenith_vectors(
                resp, receiver, center, mode="ortho"
            )
            norm = float(
                polarimeter.pseudo_stokes(
                    instrument.blackbody_normalization(
                        resp, receiver, [center], impedance_freq_mhz=center
                    ),
                    x,
                    y,
                )[0, 0]
            )
            files = []
            for frequency in selected:
                path = directory / f"{name}_{center:g}_{frequency:.9g}.npz"
                if path.exists():
                    check = load_record(path.with_suffix(".json"), key)
                    if digest(path) != check["sha256"]:
                        raise ValueError(
                            "corrupt chromatic-leakage checkpoint"
                        )
                    print("verified", path.name, flush=True)
                else:
                    print("transform", name, center, frequency, flush=True)
                    beam = response.four_port_pair_alms(
                        resp, frequency, config["lmax"]
                    )
                    pair = engine.expand(
                        engine.contract(
                            beam,
                            sky.component_alms,
                            tt,
                            moon_location(),
                            config["lmax"],
                        ),
                        sky.coeffs([frequency]),
                    )
                    matrices = instrument.covariance(
                        pair,
                        resp,
                        receiver,
                        [frequency],
                        T_moon=0.0,
                        T_ant=0.0,
                        impedance_freq_mhz=frequency,
                    )[:, 0]
                    low = config["check_lmax"]
                    low_pair = engine.expand(
                        engine.contract(
                            truncate(beam, low),
                            truncate(sky.component_alms, low),
                            tt,
                            moon_location(),
                            low,
                        ),
                        sky.coeffs([frequency]),
                    )
                    low_matrices = instrument.covariance(
                        low_pair,
                        resp,
                        receiver,
                        [frequency],
                        T_moon=0.0,
                        T_ant=0.0,
                        impedance_freq_mhz=frequency,
                    )[:, 0]
                    high_s = polarimeter.pseudo_stokes(matrices, x, y) / norm
                    low_s = (
                        polarimeter.pseudo_stokes(low_matrices, x, y) / norm
                    )
                    fractional_error = abs(high_s - low_s) / abs(
                        high_s[:, 0, None]
                    )
                    minimum = np.linalg.eigvalsh(matrices)[:, 0]
                    scale = np.max(abs(matrices), axis=(-1, -2))
                    passed = fractional_error.max() <= config[
                        "beam_error_over_i"
                    ] and np.all(minimum >= -1e-9 * scale)
                    np.savez_compressed(
                        path,
                        covariance=matrices,
                        stokes_k=high_s,
                        pair_alms=beam,
                        low_stokes_k=low_s,
                    )
                    check = dict(
                        fingerprint=key,
                        sha256=digest(path),
                        passed=bool(passed),
                        maximum_beam_error_over_i=float(
                            fractional_error.max()
                        ),
                        min_covariance_eigenvalue=float(minimum.min()),
                    )
                    write_json(path.with_suffix(".json"), check)
                files.append(
                    dict(
                        frequency_mhz=float(frequency),
                        path=str(path.relative_to(REPO)),
                        sha256=digest(path),
                        diagnostics=check,
                    )
                )
            rows.append(
                dict(
                    response=name,
                    center_mhz=center,
                    native_mhz=selected.tolist(),
                    x_real=x.real.tolist(),
                    x_imag=x.imag.tolist(),
                    y_real=y.real.tolist(),
                    y_imag=y.imag.tolist(),
                    normalization_v2_per_hz_per_k=norm,
                    arrays=files,
                )
            )
        del resp
    passed = all(
        file["diagnostics"]["passed"] for row in rows for file in row["arrays"]
    )
    write_json(
        record_path,
        dict(
            fingerprint=key,
            provenance=provenance,
            results=rows,
            all_passed=passed,
            scope=(
                "Native-frequency I-only Haslam+CMB harmonic predictions. "
                "The beam and loading are chromatic; readout calibration "
                "is fixed per band. Raw loaded matrices support a common "
                "as-built calibration for mismatch comparisons. Thermal "
                "and amplifier terms are added separately."
            ),
        ),
    )
    if not passed:
        raise SystemExit("Retained chromatic leakage gate failures.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/paper/leakage.json"
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPO / "docs/chromatic-leakage-results.json",
    )
    args = parser.parse_args()
    compute(json.loads(args.config.read_text()), args.record)


if __name__ == "__main__":
    main()
