"""Compute and validate native-band and measured-bin mixed-depth moments."""

import common  # noqa: F401
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from common import CACHE_DIR, REPO
from faraday_covariance import SOURCES as BASE_SOURCES
from faraday_covariance import projector, spectral_amplitude
from paper_common import digest, load_record, manifest, write_json
from paper_kernels import load as load_kernel
from paper_kernels import setup as kernel_setup
from lusee_faraday import channelization as ch
from lusee_faraday.config import fine_freqs, parent_centers
from lusee_faraday.conventions import lambda_squared
from lusee_faraday.depth_channel_covariance import (
    DepthChannelProjector,
    channel_phase_coefficients,
    depth_phase_coefficients,
    hermite_degree,
)
from lusee_faraday.depth_depolarization import DepthProfile, DepthQuadrature
from lusee_faraday.frequency_covariance import gaussian_visibility_draw
from lusee_faraday.spin_covariance import AngularScreen

SOURCES = sorted(
    set(
        BASE_SOURCES
        + [
            "scripts/mixed_depth_predictions.py",
            "src/lusee_faraday/depth_channel_covariance.py",
            "src/lusee_faraday/depth_depolarization.py",
            "src/lusee_faraday/gaussian_screen.py",
        ]
    )
)


def setup(config):
    base = json.loads((REPO / config["covariance_config"]).read_text())
    parameters = json.loads((REPO / base["kernel_config"]).read_text())
    _, _, _, kernel_key, _ = kernel_setup(parameters)
    kr = load_record(REPO / base["kernel_record"], kernel_key)
    complete = dict(config, covariance=base, kernel_fingerprint=kernel_key)
    previous = REPO / config["foreground_reference"]
    complete["foreground_reference_sha256"] = digest(previous)
    provenance, key = manifest(complete, SOURCES)
    directory = CACHE_DIR / "mixed_depth" / key
    return base, kr, provenance, key, directory


def frequency_map(center, kind, stride=1):
    if kind == "native":
        return np.array([center]), np.ones((1, 1)), np.array([center])
    fine = fine_freqs(center)[::stride]
    parents = parent_centers(center)
    w = np.r_[
        ch.channel_operator(fine, parents),
        ch.channel_operator(fine, parents, "zoom"),
    ]
    nominal = np.r_[
        parents, (parents[:, None] + ch.zoom_bin_offsets_hz() * 1e-6).ravel()
    ]
    return fine, w, nominal


def coefficients(
    center, kind, screen, profile, numerical, degree, base, config, stride=1
):
    frequency, w, nominal = frequency_map(center, kind, stride)
    result = depth_phase_coefficients(
        lambda_squared(frequency),
        screen,
        profile,
        numerical,
        tail=config["mode_tail"],
        max_degree=config["max_degree"],
    )
    value = np.pad(
        result["coefficients"], ((0, 0), (0, degree - result["degree"]))
    )
    amp = spectral_amplitude(frequency, base["intrinsic"])
    channel = channel_phase_coefficients(value, w, amp)
    tail_rms = abs(w) @ (
        abs(amp) * np.full(len(frequency), np.sqrt(result["tail_bound"]))
    )
    off = np.zeros_like(channel)
    off[:, 0] = w @ amp
    return channel, off, tail_rms, result, nominal


def compare(reference, alternative, off, config):
    result = {}
    for name in ("smooth", "speckle", "total"):
        a, b = reference[name], alternative[name]
        n = a.shape[0] * a.shape[1]
        a, b = a.reshape(n, n), b.reshape(n, n)
        off_diag = np.diag(off.reshape(n, n))
        diag = np.diag(a)
        if diag.min() < -1e-10 * max(
            float(off_diag.max()), np.finfo(float).tiny
        ):
            raise ValueError("negative readout variance")
        on_scale = np.sqrt(np.outer(np.maximum(diag, 0), np.maximum(diag, 0)))
        off_scale = np.sqrt(
            np.outer(np.maximum(off_diag, 0), np.maximum(off_diag, 0))
        )
        allowed = (
            config["relative_tolerance"] * on_scale
            + config["absolute_off_tolerance"] * off_scale
        )
        allowed = np.maximum(allowed, np.finfo(float).tiny)
        result[name] = float(np.max(abs(a - b) / allowed))
    return result


def check_previous_foreground(center, screen, moments, off, config):
    previous = json.loads((REPO / config["foreground_reference"]).read_text())
    for row in previous["results"]:
        if row["center_mhz"] == center and row["case"]["parameters"] == screen:
            path = REPO / row["path"]
            if digest(path) != row["sha256"]:
                raise ValueError("corrupt foreground reference")
            indices = np.r_[np.arange(3), np.arange(3 + 64, 3 + 128)]
            with np.load(path) as data:
                current = {
                    name: value[indices][:, :, indices, :]
                    for name, value in moments.items()
                }
                expected = {
                    name: data["covariance"] if name == "total" else data[name]
                    for name in current
                }
                check = compare(
                    current, expected, off[indices][:, :, indices, :], config
                )
            return check
    raise ValueError("no parameter-matched foreground reference")


def compute(config, record_path):
    base, kr, provenance, key, directory = setup(config)
    directory.mkdir(parents=True, exist_ok=True)
    numerical = DepthQuadrature(**config["depth_numerical"])
    rows = []
    for center in config["native_centers_mhz"]:
        item = kr["arrays"][f"{center:g}"]
        if digest(REPO / item["path"]) != item["sha256"]:
            raise ValueError("kernel artifact differs from record")
        kernels = load_kernel(REPO / item["path"], kr["fingerprint"])[
            "stokes_kernels"
        ]
        angular = projector(kernels, base)
        angular_refined = projector(kernels, base, refinement=2)
        angular_cut = projector(kernels, base, lmax=base["check_lmax"])
        for case_index, case in enumerate(base["screens"]):
            screen = AngularScreen(**case["parameters"])
            for kind in (
                ["native", "channels"]
                if center in config["channel_centers_mhz"]
                else ["native"]
            ):
                frequency, _, _ = frequency_map(center, kind)
                degree, _ = hermite_degree(
                    lambda_squared(frequency).max(),
                    screen,
                    DepthProfile(),
                    config["mode_tail"],
                    config["max_degree"],
                )
                solver = DepthChannelProjector(angular, screen, degree)
                refined_solver = DepthChannelProjector(
                    angular_refined, screen, degree
                )
                cut_solver = DepthChannelProjector(angular_cut, screen, degree)
                for profile_index, parameters in enumerate(config["profiles"]):
                    tag = (
                        f"{center:g}mhz_{kind}_s{case_index}_p{profile_index}"
                    )
                    path = directory / f"{tag}.npz"
                    if path.exists():
                        check = load_record(path.with_suffix(".json"), key)
                        if digest(path) != check["sha256"]:
                            raise ValueError("corrupt mixed-depth checkpoint")
                        rows.append(check["result"])
                        print("verified", tag, flush=True)
                        continue
                    print("computing", tag, "degree", degree, flush=True)
                    profile = DepthProfile(**parameters)
                    a, off_a, tail, result, nominal = coefficients(
                        center,
                        kind,
                        screen,
                        profile,
                        numerical,
                        degree,
                        base,
                        config,
                    )
                    moments = solver.moments(a)
                    off = solver.moments(off_a)["total"]
                    high_numerical = replace(
                        numerical,
                        order=max(numerical.order, result["depth_order"]),
                        max_order=max(
                            numerical.max_order, 2 * result["depth_order"]
                        ),
                        epsabs=numerical.epsabs / 10,
                        epsrel=numerical.epsrel / 10,
                    )
                    depth_a, *_ = coefficients(
                        center,
                        kind,
                        screen,
                        profile,
                        high_numerical,
                        degree,
                        base,
                        config,
                    )
                    checks = {
                        "depth": compare(
                            moments, solver.moments(depth_a), off, config
                        ),
                        "angular": compare(
                            moments, refined_solver.moments(a), off, config
                        ),
                        "bandlimit": compare(
                            moments, cut_solver.moments(a), off, config
                        ),
                    }
                    if kind == "channels":
                        coarse_a, *_ = coefficients(
                            center,
                            kind,
                            screen,
                            profile,
                            numerical,
                            degree,
                            base,
                            config,
                            stride=2,
                        )
                        checks["frequency"] = compare(
                            moments, solver.moments(coarse_a), off, config
                        )
                        if profile.kind == "screen" and profile.fraction == 1:
                            checks["foreground_reference"] = (
                                check_previous_foreground(
                                    center,
                                    case["parameters"],
                                    moments,
                                    off,
                                    config,
                                )
                            )
                    bound = solver.omitted_covariance_bound(tail)
                    n = off.shape[0] * off.shape[1]
                    off_diag = np.diag(off.reshape(n, n))
                    off_scale = np.sqrt(
                        np.outer(
                            np.maximum(off_diag, 0), np.maximum(off_diag, 0)
                        )
                    )
                    tail_ratio = float(
                        np.max(
                            bound.reshape(n, n)
                            / np.maximum(
                                config["absolute_off_tolerance"] * off_scale,
                                np.finfo(float).tiny,
                            )
                        )
                    )
                    passed = (
                        all(
                            v <= 1 for c in checks.values() for v in c.values()
                        )
                        and tail_ratio <= 1
                    )
                    draws = {}
                    eigen = {}
                    for component_index, name in enumerate(
                        ("smooth", "speckle")
                    ):
                        draw = gaussian_visibility_draw(
                            moments[name].reshape(n, n),
                            [
                                config["seed"],
                                int(round(center * 1e6)),
                                case_index,
                                profile_index,
                                component_index,
                            ],
                            relative_tolerance=1e-8,
                        )
                        draws["draw_" + name] = draw["sample"].reshape(
                            len(nominal), -1
                        )
                        eigen[name] = dict(
                            minimum=draw["min_eigenvalue"],
                            tolerance=draw["tolerance"],
                            clipped_roundoff=draw[
                                "clipped_roundoff_eigenvalues"
                            ],
                        )
                    np.savez_compressed(
                        path,
                        **moments,
                        off=off,
                        coefficients=a,
                        nominal_mhz=nominal,
                        **draws,
                    )
                    diagonal = np.diag(moments["total"].reshape(n, n)).reshape(
                        len(nominal), -1
                    )
                    row = dict(
                        center_mhz=center,
                        kind=kind,
                        case=case,
                        profile=parameters,
                        degree=degree,
                        depth_order=result["depth_order"],
                        depth_error=result["depth_error"],
                        phase_tail=result["tail_bound"],
                        tail_gate_ratio=tail_ratio,
                        checks=checks,
                        passed=passed,
                        eigen=eigen,
                        rms_k=np.sqrt(np.maximum(diagonal, 0)).tolist(),
                        path=str(path.relative_to(REPO)),
                        sha256=digest(path),
                    )
                    write_json(
                        path.with_suffix(".json"),
                        dict(fingerprint=key, sha256=digest(path), result=row),
                    )
                    rows.append(row)
                    print(
                        "saved",
                        tag,
                        "PASS" if passed else "FAIL",
                        checks,
                        flush=True,
                    )
    write_json(
        record_path,
        dict(
            fingerprint=key,
            provenance=provenance,
            results=rows,
            all_passed=all(row["passed"] for row in rows),
            scope=(
                "Shared-column mixed-depth moments at native 10/30/50 MHz "
                "and all three parent/zoom responses at 30/50 MHz. Gaussian "
                "draws use separate smooth/centered components; full "
                "visibility Gaussianity is not asserted."
            ),
        ),
    )
    if not all(row["passed"] for row in rows):
        raise SystemExit("Retained mixed-depth numerical gate failures.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["compute", "validate"])
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/paper/mixed-depth.json"
    )
    parser.add_argument(
        "--record", type=Path, default=REPO / "docs/mixed-depth-results.json"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage == "compute":
        compute(config, args.record)
    else:
        _, _, _, key, _ = setup(config)
        record = load_record(args.record, key)
        for row in record["results"]:
            if digest(REPO / row["path"]) != row["sha256"]:
                raise ValueError("corrupt mixed-depth output")
        if not record["all_passed"]:
            raise SystemExit("Retained failed mixed-depth gates.")
        print("Verified", len(record["results"]), "mixed-depth cases.")


if __name__ == "__main__":
    main()
