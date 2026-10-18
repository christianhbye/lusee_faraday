"""Conditional spin-2 channel moments through the real instrument."""

import common  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np

from common import CACHE_DIR, FIG_DIR, REPO
from paper_common import digest, load_record, manifest, write_json
from paper_kernels import compute as compute_kernels
from paper_kernels import load as load_kernel
from paper_kernels import setup as kernel_setup
from lusee_faraday import channelization as ch
from lusee_faraday.config import BETA_QU, fine_freqs, parent_centers
from lusee_faraday.conventions import lambda_squared
from lusee_faraday.frequency_covariance import (
    gaussian_visibility_draw,
    interpolated_channel_operator,
    project_real_frequency_kernel,
)
from lusee_faraday.spin_covariance import (
    AngularScreen,
    SpinTransform,
    eb_spectra,
)
from lusee_faraday.visibility_covariance import AngularReadoutProjector

SOURCES = [
    "scripts/common.py",
    "scripts/paper_common.py",
    "scripts/faraday_covariance.py",
    "scripts/paper_kernels.py",
    "src/lusee_faraday/config.py",
    "src/lusee_faraday/conventions.py",
    "src/lusee_faraday/spin_covariance.py",
    "src/lusee_faraday/visibility_covariance.py",
    "src/lusee_faraday/frequency_covariance.py",
    "src/lusee_faraday/channelization.py",
]


def setup(config):
    kp = REPO / config["kernel_config"]
    parameters = json.loads(kp.read_text())
    _, _, _, key, _ = kernel_setup(parameters)
    kernel_record = load_record(REPO / config["kernel_record"], key)
    complete = dict(config, kernel_fingerprint=key, beta_qu=BETA_QU)
    provenance, fingerprint = manifest(complete, SOURCES)
    directory = CACHE_DIR / "faraday_covariance" / fingerprint
    return parameters, kernel_record, provenance, fingerprint, directory


def intrinsic_spectra(lmax, settings):
    ell = np.arange(lmax + 1)
    power = np.exp(-ell * (ell + 1) / settings["ell_scale"] ** 2)
    power[:2] = 0
    power *= settings["p_ref_k"] ** 2 / np.sum(
        (2 * ell + 1) * power / (4 * np.pi)
    )
    efrac, ebfrac = settings["ee_fraction"], settings["eb_fraction"]
    return eb_spectra(efrac * power, (1 - efrac) * power, ebfrac * power)


def spectral_amplitude(freq, settings):
    ref, turnover = settings["ref_mhz"], settings["turnover_mhz"]
    return (freq / ref) ** BETA_QU * np.exp(
        -((turnover / freq) ** 2.1) + (turnover / ref) ** 2.1
    )


def angular_transform(lmax, settings, refinement=1):
    theta = np.r_[
        0.0,
        np.geomspace(
            settings["theta_min_rad"],
            0.01,
            refinement * settings["core_nodes"] + 1,
        ),
        np.linspace(0.01, np.pi, refinement * settings["body_nodes"] + 1)[1:],
    ]
    return SpinTransform.theta_grid(theta, lmax)


def projector(kernels, config, lmax=None, refinement=1):
    full_lmax = kernels.shape[-2] - 1
    lmax = full_lmax if lmax is None else lmax
    mstart = full_lmax - lmax
    stop = full_lmax + lmax + 1
    beam = kernels[:, :, : lmax + 1, mstart:stop]
    angles = 2 * np.pi * np.asarray(config["sidereal_turns"])
    phases = np.exp(-1j * angles[:, None] * np.arange(-lmax, lmax + 1))
    transform = angular_transform(lmax, config["angular"], refinement)
    return AngularReadoutProjector(
        beam, phases, transform, intrinsic_spectra(lmax, config["intrinsic"])
    )


def channel_map(center):
    fine = fine_freqs(center)
    parents = parent_centers(center)
    # Three parents and all 64 zoom bins of the central parent. FFT order
    # is retained in arrays; nominal frequencies are sorted only for plots.
    w = np.r_[
        ch.channel_operator(fine, parents),
        ch.channel_operator(fine, [center], "zoom"),
    ]
    frequency = np.r_[parents, center + ch.zoom_bin_offsets_hz() * 1e-6]
    return fine, w, frequency


def channel_moments(projected, center, screen, count, config, term="total"):
    fine, w, _ = channel_map(center)
    nodes = np.linspace(fine[0], fine[-1], count)
    compressed = interpolated_channel_operator(fine, nodes, w)

    def kernel(fi, fj):
        x, y = lambda_squared(fi)[:, None], lambda_squared(fj)[None, :]
        value = projected.screen_covariance(x, y, screen, term=term)
        a = spectral_amplitude(fi, config["intrinsic"])
        b = spectral_amplitude(fj, config["intrinsic"])
        return value * (a[:, None] * b[None, :])[..., None, None]

    return project_real_frequency_kernel(
        nodes, compressed, kernel, block_size=config["frequency_block_size"]
    )


def scaled_difference(a, b, reference):
    """Worst covariance change in units of the two off RMS amplitudes."""
    scale = np.sqrt(np.outer(np.diag(reference), np.diag(reference)))
    return float(np.max(abs(a - b) / np.maximum(scale, np.finfo(float).tiny)))


def compute(config, record_path):
    parameters, kr, provenance, key, directory = setup(config)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for center in config["centers_mhz"]:
        item = kr["arrays"][f"{center:g}"]
        if digest(REPO / item["path"]) != item["sha256"]:
            raise ValueError("kernel record hash differs from array")
        data = load_kernel(REPO / item["path"], kr["fingerprint"])
        beam = data["stokes_kernels"]
        base = projector(beam, config)
        refined = projector(beam, config, refinement=2)
        cut = projector(beam, config, lmax=config["check_lmax"])
        for case_index, case in enumerate(config["screens"]):
            path = directory / f"{center:g}mhz_case{case_index}.npz"
            if path.exists():
                check = load_record(path.with_suffix(".json"), key)
                if digest(path) != check["sha256"]:
                    raise ValueError("corrupt channel moment cache")
                rows.append(check["result"])
                print(
                    f"verified moments {center:g} MHz case{case_index}",
                    flush=True,
                )
                continue
            print(f"moments {center:g} MHz case{case_index}", flush=True)
            screen = AngularScreen(**case["parameters"])
            n1, n2 = case["frequency_nodes"]
            fine, w, frequency = channel_map(center)
            bins = len(frequency)
            off = channel_moments(
                base,
                center,
                AngularScreen(sigma_depth=0, mean_depth=0),
                n1,
                config,
            ).reshape(bins * base.size, -1)
            coarse = channel_moments(base, center, screen, n1, config)
            covariance = channel_moments(base, center, screen, n2, config)
            angular = channel_moments(refined, center, screen, n2, config)
            bandlimit = channel_moments(cut, center, screen, n2, config)
            smooth = channel_moments(
                base, center, screen, n2, config, term="smooth"
            )
            dimension = bins * base.size
            cov = covariance.reshape(dimension, dimension)
            checks = dict(
                frequency=scaled_difference(
                    coarse.reshape(cov.shape), cov, off
                ),
                angular=scaled_difference(
                    angular.reshape(cov.shape), cov, off
                ),
                bandlimit=scaled_difference(
                    bandlimit.reshape(cov.shape), cov, off
                ),
            )
            # Relative checks use on-power for nonzero components, with a
            # declared absolute allowance near zero in off-power units.
            on_scale = np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
            off_scale = np.sqrt(np.outer(np.diag(off), np.diag(off)))
            allowed = config["relative_tolerance"] * on_scale + (
                config["absolute_off_tolerance"] * off_scale
            )
            for name, comparison in (
                ("frequency", coarse),
                ("angular", angular),
                ("bandlimit", bandlimit),
            ):
                error = abs(comparison.reshape(cov.shape) - cov)
                checks[name + "_worst_gate_ratio"] = float(
                    np.max(error / allowed)
                )
            passed = all(
                v <= 1 for k, v in checks.items() if k.endswith("gate_ratio")
            )
            draw_seed = [config["seed"], case_index, int(round(center * 1e6))]
            draw = gaussian_visibility_draw(
                cov, draw_seed, relative_tolerance=1e-8
            )
            np.savez_compressed(
                path,
                covariance=covariance,
                smooth=smooth,
                speckle=covariance - smooth,
                off=off,
                frequency_mhz=frequency,
                draw=draw["sample"].reshape(bins, base.size),
                noise_overlap_per_hz=ch.white_spectrum_noise_overlap(
                    w, (fine[1] - fine[0]) * 1e6
                ),
            )
            row = dict(
                center_mhz=center,
                case=case,
                checks=checks,
                passed=passed,
                min_eigenvalue=draw["min_eigenvalue"],
                clipped_roundoff_eigenvalues=draw[
                    "clipped_roundoff_eigenvalues"
                ],
                draw_tolerance=draw["tolerance"],
                draw_seed=draw_seed,
                rms_k=np.sqrt(np.diag(cov)).reshape(bins, -1).tolist(),
                path=str(path.relative_to(REPO)),
                sha256=digest(path),
            )
            write_json(
                path.with_suffix(".json"),
                dict(fingerprint=key, sha256=digest(path), result=row),
            )
            rows.append(row)
            print(
                f"saved, gates {'PASS' if passed else 'FAIL'}: {checks}",
                flush=True,
            )
    record = dict(
        fingerprint=key,
        provenance=provenance,
        results=rows,
        all_numerical_gates_passed=all(row["passed"] for row in rows),
        scope="Conditional foreground-screen second moments and Gaussian "
        "visibility illustrations. Intrinsic amplitude, E/B spectra, "
        "local angular screen and low-frequency turnover are assumed. "
        "Mixed-depth real-beam projection, leakage and loaded receiver "
        "noise are pending; no LuSEE sensitivity claim is made.",
    )
    write_json(record_path, record)
    if not record["all_numerical_gates_passed"]:
        raise SystemExit("Numerical gates failed; retained all results.")


def plot(config, record_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    _, _, _, key, _ = setup(config)
    record = load_record(record_path, key)
    if not record["all_numerical_gates_passed"]:
        raise ValueError("figure needs passing numerical gates")
    selected = [
        r
        for r in record["results"]
        if r["case"]["label"] == config["figure_case"]
    ]
    fig, axes = plt.subplots(
        2,
        len(selected),
        figsize=(6.4, 5.4),
        squeeze=False,
        layout="constrained",
    )
    amplitude_limit = 0.0
    for column, row in enumerate(selected):
        if digest(REPO / row["path"]) != row["sha256"]:
            raise ValueError("corrupt figure input")
        with np.load(REPO / row["path"]) as data:
            cov = data["covariance"]
            # Index D as (selected sidereal phase, I/Q/U/V).
            q = cov[3:, 1, 3:, 1]
            scale = np.sqrt(np.diag(q))
            corr = q / np.outer(scale, scale)
            freq = data["frequency_mhz"][3:]
            order = np.argsort(freq)
            offset = (freq[order] - row["center_mhz"]) * 1e3
            half_bin = (offset[1] - offset[0]) / 2
            limits = [offset[0] - half_bin, offset[-1] + half_bin]
            ax = axes[0, column]
            image = ax.imshow(
                np.maximum(1 - corr[np.ix_(order, order)], 1e-12),
                origin="lower",
                extent=limits * 2,
                norm=LogNorm(vmin=1e-12, vmax=1e-4),
                cmap="viridis",
                aspect="equal",
            )
            ax.set(
                xlabel="Frequency offset (kHz)",
                ylabel="Frequency offset (kHz)",
                title=f"{row['center_mhz']:g} MHz",
            )
            ax = axes[1, column]
            for component, color in ((1, "#0072B2"), (2, "#D55E00")):
                rms = np.sqrt(np.diag(cov[3:, component, 3:, component]))
                amplitude_limit = max(
                    amplitude_limit,
                    float(np.max(rms)),
                    float(np.max(abs(data["draw"][3:, component]))),
                )
                ax.plot(
                    offset,
                    data["draw"][3:, component][order],
                    color=color,
                    lw=0.9,
                    label=("Q" if component == 1 else "U") + " draw",
                )
                ax.fill_between(
                    offset, -rms[order], rms[order], color=color, alpha=0.12
                )
            ax.set(xlabel="Frequency offset (kHz)", ylabel="Pseudo-Stokes (K)")
            ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_ylim(-1.1 * amplitude_limit, 1.1 * amplitude_limit)
    fig.colorbar(
        image, ax=axes[0].tolist(), label="1 − Q correlation", shrink=0.8
    )
    name = FIG_DIR / "faraday_channel_covariance"
    fig.savefig(name.with_suffix(".pdf"))
    fig.savefig(name.with_suffix(".png"), dpi=180)
    plt.close(fig)
    case = next(
        c for c in config["screens"] if c["label"] == config["figure_case"]
    )
    screen = case["parameters"]
    caption = (
        "Conditional as-built four-port foreground-screen predictions at "
        "the MEPA reference sidereal phase. Top: central-parent zoom-bin Q "
        "loss of correlation, 1-r, retaining folded images (display floor "
        "1e-12, including the unit-correlation diagonal). Bottom: one "
        "conditional "
        "Gaussian Q/U visibility draw and marginal ±1 RMS bands. The "
        f"intrinsic spin-2 RMS is {config['intrinsic']['p_ref_k']:g} K at "
        f"{config['intrinsic']['ref_mhz']:g} MHz before beam averaging, with "
        "the assumed E/B spectrum and turnover in the configuration. "
        f"The column has mean {screen['mean_depth']:g} rad/m², dispersion "
        f"{screen['sigma_depth']:g} rad/m², theta0={screen['theta0_rad']:g} "
        f"rad and alpha={screen['alpha']:.6g}. The two bands are separate "
        "draws; cross-band realization covariance has not been modeled. "
        "This is signal-only: mixed emitting depths, I leakage, radiometer "
        "noise and Gaussianity validation remain open. Future LuSEE-Night "
        "data must be compared after the same loading and channel response."
    )
    name.with_suffix(".caption.txt").write_text(caption + "\n")
    write_json(
        name.with_suffix(".manifest.json"),
        dict(
            fingerprint=key,
            record_sha256=digest(record_path),
            pdf_sha256=digest(name.with_suffix(".pdf")),
            caption=caption,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["kernels", "moments", "validate", "plot"]
    )
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/paper/covariance.json"
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPO / "docs/faraday-covariance-results.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage == "kernels":
        compute_kernels(
            json.loads((REPO / config["kernel_config"]).read_text()),
            REPO / config["kernel_record"],
        )
    elif args.stage == "moments":
        compute(config, args.record)
    elif args.stage == "plot":
        plot(config, args.record)
    else:
        _, _, _, key, _ = setup(config)
        record = load_record(args.record, key)
        for row in record["results"]:
            if digest(REPO / row["path"]) != row["sha256"]:
                raise ValueError("corrupt moment array")
        if not record["all_numerical_gates_passed"]:
            raise SystemExit("Retained failed numerical gates.")
        print(f"Verified {len(record['results'])} conditional cases.")


if __name__ == "__main__":
    main()
