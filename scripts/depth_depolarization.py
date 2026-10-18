"""Depth-profile benchmark with checked caches and compute/plot stages."""

import common  # noqa: F401
import argparse
import json
from dataclasses import asdict, replace
from itertools import cycle
from pathlib import Path

import numpy as np

from common import CACHE_DIR, FIG_DIR, REPO
from paper_common import digest, load_record, manifest, write_json
from lusee_faraday.depth_depolarization import (
    DepthProfile,
    DepthQuadrature,
    depth_qu_moments,
)
from lusee_faraday.gaussian_screen import ScreenModel, SeparationQuadrature

SOURCES = [
    "scripts/common.py",
    "scripts/paper_common.py",
    "scripts/depth_depolarization.py",
    "src/lusee_faraday/depth_depolarization.py",
    "src/lusee_faraday/gaussian_screen.py",
    "src/lusee_faraday/config.py",
    "src/lusee_faraday/conventions.py",
]


def setup(config):
    model = ScreenModel(**config["screen_model"])
    if (
        not config["sigma_depth_values"]
        or not config["profiles"]
        or not config["frequencies_mhz"]
        or model.p_ref_k <= 0
    ):
        raise ValueError("benchmark needs frequencies, profiles and nonzero P")
    numerical = DepthQuadrature(
        **config["depth_numerical"],
        separation=SeparationQuadrature(**config["separation_numerical"]),
    )
    complete = dict(config, resolved_screen_model=asdict(model))
    provenance, key = manifest(complete, SOURCES)
    return model, numerical, provenance, key


def numerical_checks(freq, model, profile, numerical, result):
    """Independently refine depth nodes and separation tolerance/support."""
    order = max(numerical.order, result["depth_order"])
    depth = replace(
        numerical,
        order=order,
        max_order=max(numerical.max_order, 2 * order),
        epsabs=numerical.epsabs / 10,
        epsrel=numerical.epsrel / 10,
    )
    separation = replace(
        numerical,
        separation=replace(
            numerical.separation,
            epsabs=numerical.separation.epsabs / 10,
            epsrel=numerical.separation.epsrel / 10,
            upper_u=numerical.separation.upper_u + 8,
        ),
    )
    diagnostics = {}
    for name, settings in (("depth", depth), ("separation", separation)):
        refined = depth_qu_moments(freq, model, profile, settings)
        for component in ("smooth", "speckle", "total"):
            baseline = result[component + "_power"] / result["off_power"]
            comparison = refined[component + "_power"] / refined["off_power"]
            change = abs(comparison - baseline)
            if np.any(change > 2e-9 + 0.01 * abs(comparison)):
                raise ValueError(f"{name} refinement failed for {component}")
            diagnostics[name + "_" + component + "_max_fraction_shift"] = (
                float(change.max())
            )
        scale = np.sqrt(
            result["off_power"][:, None] * result["off_power"][None, :]
        )
        for moment in ("complex_covariance", "complex_relation"):
            change = abs(refined[moment] - result[moment]) / scale
            if change.max() > 2e-9:
                raise ValueError(f"{name} refinement failed for {moment}")
            diagnostics[name + "_" + moment + "_max_shift"] = float(
                change.max()
            )
    return diagnostics


def compute(config, record_path):
    model, numerical, provenance, key = setup(config)
    output = CACHE_DIR / "depth_depolarization" / key
    output.mkdir(parents=True, exist_ok=True)
    freq = np.asarray(config["frequencies_mhz"], float)
    rows, files = [], {}
    for sigma in config["sigma_depth_values"]:
        for p, parameters in enumerate(config["profiles"]):
            profile = DepthProfile(**parameters)
            path = output / f"sigma{sigma:.12g}_profile{p}.npz"
            check_path = path.with_suffix(".json")
            if path.exists() and check_path.exists():
                check = load_record(check_path, key)
                if digest(path) != check["array_sha256"]:
                    raise ValueError(f"corrupt cached result: {path}")
            else:
                case_model = replace(model, sigma_depth=sigma)
                result = depth_qu_moments(
                    freq,
                    case_model,
                    profile,
                    numerical,
                )
                checks = numerical_checks(
                    freq, case_model, profile, numerical, result
                )
                # The new result has one common depth rule for every pair.
                eigen = np.linalg.eigvalsh(result["covariance"])
                result["min_real_covariance_eigenvalue"] = float(eigen.min())
                if (
                    eigen.min()
                    < -10 * numerical.epsabs * result["off_power"].max()
                ):
                    raise ValueError("depth covariance failed positivity")
                temp = path.with_suffix(".tmp")
                with temp.open("wb") as handle:
                    np.savez(handle, freqs_mhz=freq, **result)
                temp.replace(path)
                check = dict(
                    fingerprint=key,
                    array_sha256=digest(path),
                    numerical_refinements=checks,
                )
                write_json(check_path, check)
            with np.load(path, allow_pickle=False) as result:
                off = result["off_power"]
                if np.any(off <= 0):
                    raise ValueError("power fractions need positive off power")
                row = dict(
                    profile=asdict(profile),
                    sigma_depth_rad_m2=sigma,
                    depth_order=int(result["depth_order"]),
                    depth_error=float(result["depth_error"]),
                    separation_error_estimate=float(
                        result["separation_error_estimate"]
                    ),
                    min_real_covariance_eigenvalue=float(
                        result["min_real_covariance_eigenvalue"]
                    ),
                    numerical_refinements=check["numerical_refinements"],
                )
                for name in ("smooth", "speckle", "total", "off"):
                    power = result[name + "_power"]
                    row[name + "_power_k2"] = power.tolist()
                    row[name + "_rms_k"] = np.sqrt(
                        np.maximum(power, 0)
                    ).tolist()
                    row[name + "_fraction"] = (power / off).tolist()
                rows.append(row)
            files[str(path.relative_to(REPO))] = digest(path)
            print(
                f"sigma={sigma:g}, {profile.kind} f={profile.fraction:g}",
                flush=True,
            )
    record = dict(
        fingerprint=key,
        manifest=provenance,
        rows=rows,
        array_sha256=files,
        freqs_mhz=freq.tolist(),
        scope=config["scope"],
        validation_scope=(
            "Analytic/independent moments and quadrature checks; no real "
            "beam, channelization, loading, leakage or noise."
        ),
    )
    write_json(record_path, record)
    return record


def plot(config, record_path, figure_path):
    import matplotlib.pyplot as plt

    _, _, _, key = setup(config)
    record = load_record(record_path, key)
    for name, expected in record["array_sha256"].items():
        if digest(REPO / name) != expected:
            raise ValueError(f"changed cached result: {name}")
    fig, axes = plt.subplots(
        2,
        len(config["sigma_depth_values"]),
        figsize=(7.0, 5.2),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
    for col, sigma in enumerate(config["sigma_depth_values"]):
        rows = [r for r in record["rows"] if r["sigma_depth_rad_m2"] == sigma]
        for row, color in zip(rows, cycle(colors)):
            profile = row["profile"]
            label = profile["kind"]
            if label == "screen":
                label += f" f={profile['fraction']:g}"
            for ax, component in zip(axes[:, col], ("smooth", "speckle")):
                values = np.array(row[component + "_fraction"])
                # Logarithmic plotting needs an explicit display floor for
                # zero, underflow and extremely suppressed components.
                ax.plot(
                    record["freqs_mhz"],
                    np.maximum(values, 1e-12),
                    marker="o",
                    label=label,
                    color=color,
                    lw=1.3,
                )
                zero = values < 1e-12
                ax.scatter(
                    np.array(record["freqs_mhz"])[zero],
                    np.full(zero.sum(), 1e-12),
                    marker="v",
                    color=color,
                )
                ax.set_yscale("log")
                ax.set_ylim(8e-13, 2)
                ax.grid(alpha=0.2)
        axes[0, col].set_title(rf"$\sigma_\phi={sigma:g}$ rad m$^{{-2}}$")
        axes[1, col].set_xlabel("Frequency [MHz]")
    axes[0, 0].set_ylabel("Smooth power fraction")
    axes[1, 0].set_ylabel("Speckle power fraction")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "Depth-profile benchmark: conditional flat-plane moments", fontsize=10
    )
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    fig.savefig(figure_path.with_suffix(".png"), dpi=160)
    plt.close(fig)
    caption = (
        "Smooth |E[P]|² and speckle E[|P-E[P]|²], each divided by the same "
        "unrotated beam-weighted power, at 10/30/50 MHz. The Gaussian is "
        "truncated to f in [0,1]; all profile parameters are in the record. "
        "The frequency points are monochromatic. Downward triangles mark "
        "values below the 1e-12 display floor, including zero and underflow. "
        "Screen/source/beam parameters and provenance are in the result "
        "record. Local emissivity and screen statistics are illustrative; "
        "no real LuSEE beam, channel averaging, leakage or noise is included."
    )
    figure_path.with_suffix(".caption.txt").write_text(caption + "\n")
    write_json(
        figure_path.with_suffix(".manifest.json"),
        dict(
            fingerprint=key,
            record_sha256=digest(record_path),
            figure_sha256=digest(figure_path),
            caption=caption,
        ),
    )
    print(f"wrote {figure_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("compute", "plot"))
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/paper/depth.json"
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPO / "docs/depth-depolarization-results.json",
    )
    parser.add_argument(
        "--figure", type=Path, default=FIG_DIR / "depth_depolarization.pdf"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage == "compute":
        compute(config, args.record)
    else:
        plot(config, args.record, args.figure)


if __name__ == "__main__":
    main()
