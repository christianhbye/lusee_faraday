"""Angular spectra and a common masked sidereal observing operator."""

import common  # noqa: F401
import argparse
import json

import numpy as np

from common import CACHE_DIR, REPO, moon_location, times
from chromatic_leakage import setup as leakage_setup
from faraday_covariance import (
    angular_transform,
    intrinsic_spectra,
    spectral_amplitude,
)
from measured_system import loaded_covariance, matrices_at_native
from paper_common import digest, load_record, manifest, write_json
from paper_kernels import load as load_kernel
from lusee_faraday import engine
from lusee_faraday.config import SIDEREAL_DAY_S
from lusee_faraday.conventions import lambda_squared
from lusee_faraday.depth_channel_covariance import depth_phase_coefficients
from lusee_faraday.depth_depolarization import DepthProfile
from lusee_faraday.observing import (
    phase_exposures,
    site_solar_altitudes,
    sidereal_bin_window,
)
from lusee_faraday.receiver_noise import (
    diagonal_amplifier_covariance,
    quadratic_radiometer_covariance,
    stokes_quadratic_matrices,
)
from lusee_faraday.spin_covariance import AngularScreen, helicity_covariance
from lusee_faraday.visibility_covariance import (
    evaluate_mode_moments,
    pair_mode_moments,
)


def compute():
    import croissant as cro

    base = json.loads((REPO / "configs/paper/covariance.json").read_text())
    leakage = json.loads((REPO / "configs/paper/leakage.json").read_text())
    noise = json.loads(
        (REPO / "configs/paper/receiver-noise.json").read_text()
    )
    kr = json.loads((REPO / "docs/paper-kernel-results.json").read_text())
    receiver, _, key, ld = leakage_setup(leakage)
    count = 256
    width = SIDEREAL_DAY_S / count
    turns = np.arange(count) / count
    exposure = phase_exposures(
        SIDEREAL_DAY_S,
        turns,
        lambda seconds: site_solar_altitudes(
            times()[0], seconds, moon_location()
        ),
        0.5,
        bin_seconds=width,
    )
    mask = np.array([row["available"] for row in exposure])
    modes = np.fft.fftfreq(count) * count
    observing = (
        np.exp(-2j * np.pi * modes[:, None] * turns)
        * mask[None, :]
        / mask.sum()
    )
    complete = dict(
        covariance=base,
        noise=noise,
        leakage_fingerprint=key,
        kernel_record_sha256=digest(REPO / "docs/paper-kernel-results.json"),
        sidereal_samples=count,
        bin_seconds=width,
        duty=0.5,
        night_margin_deg=-1.0,
    )
    provenance, fp = manifest(
        complete,
        [
            "scripts/common_observing_control.py",
            "scripts/chromatic_leakage.py",
            "scripts/measured_system.py",
            "scripts/faraday_covariance.py",
            "scripts/paper_common.py",
            "src/lusee_faraday/depth_channel_covariance.py",
            "src/lusee_faraday/spin_covariance.py",
            "src/lusee_faraday/visibility_covariance.py",
            "src/lusee_faraday/observing.py",
            "src/lusee_faraday/engine.py",
            "src/lusee_faraday/receiver_noise.py",
        ],
    )
    directory = CACHE_DIR / "observing_control" / fp
    directory.mkdir(parents=True, exist_ok=True)
    sky_file = ld / "sky_components.npz"
    with np.load(sky_file) as data:
        sky = {k: data[k] for k in data.files}
    if (
        digest(sky_file)
        != json.loads(sky_file.with_suffix(".json").read_text())["sha256"]
    ):
        raise ValueError("corrupt I-only sky components")
    rotated = np.asarray(
        cro.rotations.gal2mepa(
            sky["alms"], et=cro.rotations.jd_to_et(times()[0].tdb.jd)
        )
    )
    rows = []
    for center in (30.0, 50.0):
        path = directory / f"{center:g}.npz"
        if path.exists():
            saved = load_record(path.with_suffix(".json"), fp)
            if digest(path) != saved["sha256"]:
                raise ValueError("corrupt observing control")
            rows.append(saved["result"])
            continue
        item = kr["arrays"][f"{center:g}"]
        data = load_kernel(REPO / item["path"], kr["fingerprint"])
        kernels = data["stokes_kernels"]
        lmax = kernels.shape[-2] - 1
        window = sidereal_bin_window(lmax, width)
        kernels = kernels * window
        transform = angular_transform(lmax, base["angular"], refinement=2)
        intrinsic = transform.correlations(
            **intrinsic_spectra(lmax, base["intrinsic"])
        )
        screen = AngularScreen(**base["screens"][1]["parameters"])
        rho = screen.exponent(transform.theta)
        x = lambda_squared(center)
        common_lmax = min(lmax, rotated.shape[-2] - 1)
        first_m, stop_m = lmax - common_lmax, lmax + common_lmax + 1
        common_kernels = kernels[..., : common_lmax + 1, first_m:stop_m]
        phase_time = np.exp(
            -2j
            * np.pi
            * turns[:, None]
            * np.arange(-common_lmax, common_lmax + 1)
        )
        spectra, signal_cov, fringe_cov, fringe_relation = [], [], [], []
        for profile in (
            DepthProfile(fraction=0),
            DepthProfile(),
            DepthProfile(kind="slab"),
        ):
            coefficients = depth_phase_coefficients(x, screen, profile)[
                "coefficients"
            ][0]
            power = np.exp(-np.arange(len(coefficients))[:, None] * rho)
            # Transform only the centered phase term. The broad smooth
            # intrinsic spectrum is known exactly; transforming it through
            # a core-oriented trapezoid introduces false negative high-ell
            # power, most visibly in the exactly unrotated control.
            plus = (abs(coefficients[1:]) ** 2) @ power[1:]
            minus = (coefficients[1:] ** 2) @ power[1:]
            cl = transform.spectra(
                intrinsic["plus"] * plus, intrinsic["minus"] * minus
            )
            initial = intrinsic_spectra(lmax, base["intrinsic"])
            cl["plus"] += abs(coefficients[0]) ** 2 * initial["plus"]
            cl["minus"] += coefficients[0] ** 2 * initial["minus"]
            amp = float(
                spectral_amplitude(np.array([center]), base["intrinsic"])[0]
            )
            cl = {k: v * amp**2 for k, v in cl.items()}
            spectra.append(np.stack([cl["plus"], cl["minus"]]))
            harmonic = pair_mode_moments(
                common_kernels[1:3],
                helicity_covariance(
                    **{k: v[: common_lmax + 1] for k, v in cl.items()}
                ),
            )
            temporal = evaluate_mode_moments(harmonic, phase_time)[
                "covariance"
            ].real
            q = temporal[:, 0, :, 0]
            signal_cov.append(q)
            fringe_cov.append(observing @ q @ observing.conj().T)
            fringe_relation.append(observing @ q @ observing.T)
        native_path = ld / f"asbuilt_{center:g}_{center:g}.npz"
        native_check = load_record(native_path.with_suffix(".json"), key)
        if digest(native_path) != native_check["sha256"]:
            raise ValueError("corrupt native response slice")
        with np.load(native_path) as native:
            original = native["pair_alms"]
        za, rm, rl = matrices_at_native(
            REPO / leakage["responses"]["asbuilt"], center, original
        )
        beam = engine.response_in_mepa(
            original, times()[0], moon_location(), original.shape[-2] - 1
        )
        low_lmax = beam.shape[-2] - 1
        low_phases = np.exp(
            -2j * np.pi * turns[:, None] * np.arange(-low_lmax, low_lmax + 1)
        )
        contracted = engine.contract_blocks(
            beam * sidereal_bin_window(low_lmax, width), rotated, low_phases
        )
        coeff = (center / sky["ref"][:, 0]) ** sky["beta"][:, 0]
        pairs = np.einsum("k,ktp->tp", coeff, contracted[:, 0])
        zl = receiver.Z([center])
        sky_cov = loaded_covariance(pairs, za, zl, rm, rl)
        readout = stokes_quadratic_matrices(
            data["x_vec"], data["y_vec"], float(data["normalization"])
        )
        stokes = np.einsum("pab,tba->tp", readout, sky_cov).real
        thermal = loaded_covariance(
            np.zeros((1, 10), complex),
            za,
            zl,
            rm,
            rl,
            noise["moon_temperature_k"],
            noise["antenna_temperature_k"],
        )[0]
        amp = diagonal_amplifier_covariance(
            za,
            zl[0],
            noise["voltage_reference_nv_rt_hz"] * 1e-9,
            noise["current_reference_pa_rt_hz"] * 1e-12,
        )
        # A declared 25 kHz top-hat for this spatial-only control; parent/zoom
        # noise with their measured overlaps is handled by measured_system.
        temporal_noise = quadratic_radiometer_covariance(
            sky_cov + thermal + amp, readout, 25000 * width * 0.5
        )[:, 1, 1]
        noise_cov = (observing * temporal_noise) @ observing.conj().T
        leak = observing @ stokes[:, 1]
        eigen = np.linalg.eigvalsh(np.asarray(signal_cov))
        scale = np.max(np.abs(signal_cov))
        passed = bool(eigen.min() >= -1e-8 * scale)
        np.savez_compressed(
            path,
            turns=turns,
            mask=mask,
            modes=modes,
            fringe_rate_microhz=modes / SIDEREAL_DAY_S * 1e6,
            observing_operator=observing,
            spectra=spectra,
            signal_covariance=signal_cov,
            fringe_covariance=fringe_cov,
            fringe_relation=fringe_relation,
            leakage_fringe=leak,
            noise_fringe_covariance=noise_cov,
            intensity_stokes_k=stokes,
        )
        row = dict(
            center_mhz=center,
            passed=passed,
            minimum_signal_eigenvalue=float(eigen.min()),
            observed_bins=int(mask.sum()),
            total_bins=count,
            common_observed_lmax=common_lmax,
            theory_lmax=lmax,
            path=str(path.relative_to(REPO)),
            sha256=digest(path),
        )
        rows.append(row)
        write_json(
            path.with_suffix(".json"),
            dict(fingerprint=fp, sha256=digest(path), result=row),
        )
        print(row, flush=True)
    write_json(
        REPO / "docs/common-observing-results.json",
        dict(
            fingerprint=fp,
            provenance=provenance,
            results=rows,
            all_passed=all(r["passed"] for r in rows),
            scope=(
                "Native-band angular and data fringe spectra through the "
                "same "
                "finite bin, actual-site nightside mask and Fourier operator. "
                "The mask mixes fringe modes; no angular map is "
                "reconstructed. "
                "Noise uses an explicit 25 kHz top-hat and one 50%-duty phase "
                "bin, separately from the measured parent/zoom forecast. "
                "No white-input or map-level sensitivity claim is made."
            ),
        ),
    )
    if not all(r["passed"] for r in rows):
        raise SystemExit("Retained common-observing gate failure")


def plot():
    import matplotlib.pyplot as plt
    from plot_measurement_predictions import limits, save

    record_path = REPO / "docs/common-observing-results.json"
    record = json.loads(record_path.read_text())
    if not record["all_passed"]:
        raise ValueError("common-observing gates failed")
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.3), layout="constrained")
    for column, row in enumerate(record["results"]):
        with np.load(REPO / row["path"]) as data:
            for index, label in enumerate(("Unrotated", "Foreground", "Slab")):
                axes[0, column].semilogy(
                    np.arange(data["spectra"].shape[-1])[2:],
                    np.maximum(data["spectra"][index, 0, 2:].real, 1e-30),
                    label=label,
                )
                frequency = data["fringe_rate_microhz"]
                selected = frequency >= 0
                power = np.diag(data["fringe_covariance"][index]).real
                axes[1, column].semilogy(
                    frequency[selected],
                    np.maximum(power[selected], 1e-30),
                    label=label,
                )
            axes[1, column].semilogy(
                frequency[selected],
                abs(data["leakage_fringe"][selected]) ** 2,
                "--",
                color="black",
                label="I leakage",
            )
            axes[1, column].semilogy(
                frequency[selected],
                np.diag(data["noise_fringe_covariance"]).real[selected],
                ":",
                color="black",
                label="Radiometer",
            )
            axes[0, column].set(
                title=f"{row['center_mhz']:g} MHz",
                xlabel="Multipole ell",
                ylabel="EE + BB (K²)",
            )
            axes[1, column].set(
                xlabel="Fringe rate (microHz)",
                ylabel="Observed Q fringe power (K²)",
            )
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7)
    limits(axes[0], 0)
    limits(axes[1], 0)
    upper = axes[0, 0].get_ylim()[1]
    for ax in axes[0]:
        ax.set_ylim(upper * 1e-12, upper)
    save(
        fig,
        "faraday_angular_and_fringe",
        "Conditional unrotated, foreground and uniform-slab polarized angular "
        "spectra (top), and their Q fringe power through a common instrument "
        "and observing operator (bottom). All use the local 0.02 rad/m^2 "
        "dispersion illustration and 1 K intrinsic RMS at 30 MHz. "
        "The operator "
        "averages 256 sidereal bins, selects complete actual-site nightside "
        "bins and applies their normalized Fourier transform. The same mask "
        "and finite-bin average, with common harmonic cutoff "
        f"ell={record['results'][0]['common_observed_lmax']}, "
        "act on polarized covariance, deterministic "
        "Haslam I leakage and explicit loaded receiver/radiometer "
        "noise. "
        "Noise here uses a 25 kHz top-hat at 50% duty per retained bin. The "
        "window mixes modes: low-ell input shape alone does not establish "
        "separability, and no angular-map sensitivity is inferred. Full "
        "complex fringe covariance and relation are exported. Angular power "
        "below the displayed twelve-decade range is omitted.",
        [record_path],
        generator=__file__,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["compute", "plot"])
    args = parser.parse_args()
    compute() if args.stage == "compute" else plot()
