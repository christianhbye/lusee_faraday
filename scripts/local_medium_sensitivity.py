"""Signed local columns and emissivity through the common estimator."""

import common  # noqa: F401
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from common import CACHE_DIR, REPO
from faraday_covariance import projector, spectral_amplitude
from paper_common import digest, load_record, manifest, write_json
from paper_kernels import load as load_kernel
from residual_signal import SOURCES as SIGNAL_SOURCES
from residual_signal import common_fit, setup as signal_setup
from lusee_faraday.conventions import lambda_squared
from lusee_faraday.depth_channel_covariance import (
    DepthChannelProjector,
    depth_phase_coefficients,
    hermite_degree,
)
from lusee_faraday.depth_depolarization import DepthProfile, DepthQuadrature
from lusee_faraday.faraday_horizon import faraday_rate
from lusee_faraday.frequency_covariance import interpolated_channel_operator
from lusee_faraday.moment_diagonal import diagonal_moments, mixture_diagonals
from lusee_faraday.observing import sidereal_bin_window
from lusee_faraday.spin_covariance import AngularScreen


def setup(config):
    sep = json.loads((REPO / config["separability_config"]).read_text())
    horizon = json.loads((REPO / config["horizon_config"]).read_text())
    mixed, base, kr, _, reference_key, _ = signal_setup(sep)
    if horizon["emissivity_ref_mhz"] != base["intrinsic"]["ref_mhz"]:
        raise ValueError(
            "emissivity and intrinsic reference frequencies differ"
        )
    complete = dict(
        config,
        separability=sep,
        horizon=horizon,
        reference_fingerprint=reference_key,
        system_record_sha256=digest(
            REPO / "docs/measured-system-results.json"
        ),
        exposure_record_sha256=digest(REPO / sep["exposure_record"]),
    )
    provenance, key = manifest(
        complete,
        sorted(
            set(
                SIGNAL_SOURCES
                + [
                    "scripts/local_medium_sensitivity.py",
                    "src/lusee_faraday/moment_diagonal.py",
                    "src/lusee_faraday/faraday_horizon.py",
                ]
            )
        ),
    )
    return (
        sep,
        horizon,
        mixed,
        base,
        kr,
        provenance,
        key,
        CACHE_DIR / "local_medium" / key,
    )


def mean_columns(horizon, config):
    values = {}
    for source in horizon["scenarios"]:
        for projection in config["signed_field_projections"]:
            mean = float(
                faraday_rate(source["ne_cm3"], source["field_ug"] * projection)
                * horizon["emitting_path_pc"]
            )
            label = round(mean, 14)
            values.setdefault(label, dict(mean_depth=mean, sources=[]))[
                "sources"
            ].append(
                dict(
                    scenario=source["name"],
                    projection=projection,
                    density_cm3=source["ne_cm3"],
                    field_ug=source["field_ug"],
                    path_pc=horizon["emitting_path_pc"],
                )
            )
    return sorted(values.values(), key=lambda row: row["mean_depth"])


def diagonal_gate(reference, alternative, off, config):
    result = {}
    for name in reference:
        allowed = config["relative_covariance_tolerance"] * abs(
            reference[name]
        ) + (config["absolute_off_covariance_tolerance"] * abs(off))
        result[name] = float(
            np.max(
                abs(reference[name] - alternative[name])
                / np.maximum(allowed, 1e-300)
            )
        )
    return result


def covariance_change_gate(solver, low, high, probabilities, off, qu, config):
    """Cauchy bound on full Q/U cross-frequency covariance refinement."""
    first = mixture_diagonals(solver, low, probabilities)
    second = mixture_diagonals(solver, high, probabilities)
    error = mixture_diagonals(
        solver, [b - a for a, b in zip(low, high)], probabilities
    )
    results = {}
    off_rms = np.sqrt(np.maximum(off[:, qu].ravel(), 0))
    off_scale = np.outer(off_rms, off_rms)
    for name in first:
        on = np.sqrt(
            np.maximum(np.maximum(first[name][:, qu], second[name][:, qu]), 0)
        ).ravel()
        err = np.sqrt(np.maximum(error[name][:, qu], 0)).ravel()
        bound = np.outer(err, on) + np.outer(on, err) + np.outer(err, err)
        allowance = config["relative_covariance_tolerance"] * np.outer(
            on, on
        ) + (config["absolute_off_covariance_tolerance"] * off_scale)
        results[name] = float(np.max(bound / np.maximum(allowance, 1e-300)))
    return results


def compute(config, path):
    sep, horizon, mixed, base, kr, provenance, key, directory = setup(config)
    directory.mkdir(parents=True, exist_ok=True)
    means = mean_columns(horizon, config)
    physical_rms = (
        horizon["emissivity_ref_k_per_pc"] * horizon["emitting_path_pc"]
    )
    amplitude_scale = physical_rms / base["intrinsic"]["p_ref_k"]
    numerical = DepthQuadrature(
        **dict(mixed["depth_numerical"], max_order=config["depth_max_order"])
    )
    qu = np.concatenate(
        [np.arange(1, 3) + 4 * t for t in range(len(base["sidereal_turns"]))]
    )
    rows = []
    weight_files = {}
    for center in mixed["channel_centers_mhz"]:
        fine, w, nominal, r, _ = common_fit(center, sep["smooth_degree"])
        item = kr["arrays"][f"{center:g}"]
        kernels = load_kernel(REPO / item["path"], kr["fingerprint"])[
            "stokes_kernels"
        ]
        kernels = kernels * sidereal_bin_window(kernels.shape[-2] - 1)
        initial = AngularScreen(**base["screens"][-1]["parameters"])
        largest = replace(
            initial,
            sigma_depth=max(
                s["parameters"]["sigma_depth"] for s in base["screens"]
            )
            * np.sqrt(max(sep["variance_mixture_scales"])),
        )
        degree, _ = hermite_degree(
            lambda_squared(fine).max(),
            largest,
            DepthProfile(),
            mixed["mode_tail"],
            mixed["max_degree"],
        )
        solver = DepthChannelProjector(
            projector(kernels, base, refinement=2), largest, degree
        )
        angular = DepthChannelProjector(
            projector(kernels, base), largest, degree
        )
        cut = DepthChannelProjector(
            projector(kernels, base, lmax=base["check_lmax"], refinement=2),
            largest,
            degree,
        )
        modes = []
        for mode in range(degree + 1):
            basis = np.zeros((2, degree + 1), complex)
            basis[:, mode] = [1, 1j]
            covariance = solver.moments(basis)["total"].reshape(
                2 * solver.size, -1
            )
            minimum = float(
                np.linalg.eigvalsh((covariance + covariance.T) / 2)[0]
            )
            scale = max(float(abs(covariance).max()), 1e-300)
            modes.append(
                dict(
                    mode=mode,
                    minimum=minimum,
                    scale=scale,
                    passed=minimum >= -1e-8 * scale,
                )
            )
        if not all(m["passed"] for m in modes):
            raise ValueError("local-grid latent-mode positivity failed")
        weights_path = directory / f"weights_{center:g}.npz"
        np.savez_compressed(
            weights_path,
            weights=solver.weights,
            angular_weights=angular.weights,
            cut_weights=cut.weights,
            tail_norm=solver.tail_norm,
            nominal_mhz=nominal,
            operator=r,
        )
        weight_files[f"{center:g}"] = dict(
            path=str(weights_path.relative_to(REPO)),
            sha256=digest(weights_path),
            mode_psd=modes,
        )
        operators = {}

        def frequency_operator(count):
            if count not in operators:
                nodes = np.linspace(fine[0], fine[-1], count)
                operators[count] = (
                    nodes,
                    interpolated_channel_operator(fine, nodes, w),
                )
            return operators[count]

        def coefficients(screen, profile, count, settings):
            nodes, operator = frequency_operator(count)
            result = depth_phase_coefficients(
                lambda_squared(nodes),
                screen,
                profile,
                settings,
                tail=mixed["mode_tail"],
                max_degree=mixed["max_degree"],
            )
            a = np.pad(
                result["coefficients"],
                ((0, 0), (0, degree - result["degree"])),
            )
            amp = (
                spectral_amplitude(nodes, base["intrinsic"]) * amplitude_scale
            )
            value = operator @ (a * amp[:, None])
            off = np.zeros_like(value)
            off[:, 0] = operator @ amp
            tail = abs(amp) * np.sqrt(result["tail_bound"])
            result["channel_tail_rms"] = abs(operator) @ tail
            result["residual_tail_rms"] = abs(r @ operator) @ tail
            return value, off, result

        for mi, mean in enumerate(means):
            for si, case in enumerate(base["screens"]):
                for pi, parameters in enumerate(mixed["profiles"]):
                    profile = DepthProfile(**parameters)
                    for model in ("gaussian", "variance_mixture"):
                        tag = f"{center:g}_m{mi}_s{si}_p{pi}_{model}"
                        target = directory / (tag + ".npz")
                        if (
                            target.exists()
                            and target.with_suffix(".json").exists()
                        ):
                            saved = load_record(
                                target.with_suffix(".json"), key
                            )
                            if digest(target) != saved["sha256"]:
                                raise ValueError(
                                    "corrupt local-mean checkpoint"
                                )
                            rows.append(saved["result"])
                            continue
                        scales = (
                            [1.0]
                            if model == "gaussian"
                            else sep["variance_mixture_scales"]
                        )
                        probabilities = (
                            [1.0]
                            if model == "gaussian"
                            else sep["variance_mixture_weights"]
                        )
                        screens = [
                            AngularScreen(
                                **dict(
                                    case["parameters"],
                                    mean_depth=mean["mean_depth"],
                                    sigma_depth=case["parameters"][
                                        "sigma_depth"
                                    ]
                                    * np.sqrt(scale),
                                )
                            )
                            for scale in scales
                        ]
                        count = config["initial_frequency_nodes"]
                        low = [
                            coefficients(s, profile, count, numerical)[0]
                            for s in screens
                        ]
                        while True:
                            count = 2 * count - 1
                            if count > config["maximum_frequency_nodes"]:
                                raise ValueError(
                                    "local-mean frequency refinement exhausted"
                                )
                            values = []
                            depth_orders = []
                            tail_bounds = []
                            for s in screens:
                                value, off, result = coefficients(
                                    s, profile, count, numerical
                                )
                                values.append(value)
                                depth_orders.append(result["depth_order"])
                                tail_bounds.append(result)
                            off_diag = diagonal_moments(solver, off)["total"]
                            frequency_gate = {
                                "raw": covariance_change_gate(
                                    solver,
                                    low,
                                    values,
                                    probabilities,
                                    off_diag,
                                    qu,
                                    config,
                                ),
                                "residual": covariance_change_gate(
                                    solver,
                                    [r @ a for a in low],
                                    [r @ a for a in values],
                                    probabilities,
                                    off_diag,
                                    qu,
                                    config,
                                ),
                            }
                            if all(
                                v <= 1
                                for group in frequency_gate.values()
                                for v in group.values()
                            ):
                                break
                            low = values
                        depth = []
                        for s, order in zip(screens, depth_orders):
                            settings = replace(
                                numerical,
                                order=max(numerical.order, order),
                                max_order=max(numerical.max_order, 2 * order),
                                epsabs=numerical.epsabs / 10,
                                epsrel=numerical.epsrel / 10,
                            )
                            depth.append(
                                coefficients(s, profile, count, settings)[0]
                            )
                        raw = mixture_diagonals(solver, values, probabilities)
                        filtered = [r @ a for a in values]
                        residual = mixture_diagonals(
                            solver, filtered, probabilities
                        )
                        gates = dict(frequency=frequency_gate)
                        off_rms = np.sqrt(np.maximum(off_diag.ravel(), 0))
                        off_scale = np.outer(off_rms, off_rms)
                        for name, field in (
                            ("raw", "channel_tail_rms"),
                            ("residual", "residual_tail_rms"),
                        ):
                            tail = np.max(
                                [b[field] for b in tail_bounds], axis=0
                            )
                            bound = solver.omitted_covariance_bound(
                                tail
                            ).reshape(len(off_rms), len(off_rms))
                            gate = float(
                                np.max(
                                    bound
                                    / np.maximum(
                                        config[
                                            "absolute_off_covariance_tolerance"
                                        ]
                                        * off_scale,
                                        1e-300,
                                    )
                                )
                            )
                            gates.setdefault("mode_tail", {})[name] = {
                                "total": gate
                            }
                        for name, other in (("depth", depth),):
                            gates[name] = {
                                "raw": covariance_change_gate(
                                    solver,
                                    values,
                                    other,
                                    probabilities,
                                    off_diag,
                                    qu,
                                    config,
                                ),
                                "residual": covariance_change_gate(
                                    solver,
                                    filtered,
                                    [r @ a for a in other],
                                    probabilities,
                                    off_diag,
                                    qu,
                                    config,
                                ),
                            }
                        for name, alternative in (
                            ("angular", angular),
                            ("bandlimit", cut),
                        ):
                            gates[name] = {
                                "raw": diagonal_gate(
                                    raw,
                                    mixture_diagonals(
                                        alternative, values, probabilities
                                    ),
                                    off_diag,
                                    config,
                                ),
                                "residual": diagonal_gate(
                                    residual,
                                    mixture_diagonals(
                                        alternative, filtered, probabilities
                                    ),
                                    off_diag,
                                    config,
                                ),
                            }
                        passed = all(
                            v <= 1
                            for groups in gates.values()
                            for group in groups.values()
                            for v in group.values()
                        )
                        np.savez_compressed(
                            target,
                            coefficients=values,
                            residual_coefficients=filtered,
                            coarse_coefficients=low,
                            depth_coefficients=depth,
                            probabilities=probabilities,
                            off=off_diag,
                            **{"raw_" + n: v for n, v in raw.items()},
                            **{
                                "residual_" + n: v for n, v in residual.items()
                            },
                        )
                        row = dict(
                            case_id=tag,
                            center_mhz=center,
                            mean_index=mi,
                            mean_depth_rad_m2=mean["mean_depth"],
                            mean_sources=mean["sources"],
                            screen_index=si,
                            sigma_depth_rad_m2=case["parameters"][
                                "sigma_depth"
                            ],
                            profile_index=pi,
                            profile=parameters,
                            model=model,
                            intrinsic_rms_k=physical_rms,
                            frequency_nodes=count,
                            depth_orders=depth_orders,
                            gates=gates,
                            passed=bool(passed),
                            path=str(target.relative_to(REPO)),
                            sha256=digest(target),
                        )
                        write_json(
                            target.with_suffix(".json"),
                            dict(
                                fingerprint=key,
                                sha256=digest(target),
                                result=row,
                            ),
                        )
                        rows.append(row)
                        print(
                            tag,
                            "PASS" if passed else "FAIL",
                            "nodes",
                            count,
                            flush=True,
                        )
    write_json(
        path,
        dict(
            fingerprint=key,
            provenance=provenance,
            results=rows,
            weights=weight_files,
            all_passed=all(r["passed"] for r in rows),
            means=means,
            intrinsic_rms_k=physical_rms,
            scope=(
                "Full signed uniform-mean sensitivity from the adopted "
                "density/field/path pairs. Dispersion and angular statistics "
                "remain conditional assumptions. Marginal variances contract "
                "the complete four-branch operator; phase coefficients and "
                "weights retain the joint C/J. Frequency/depth refinement "
                "bounds the full Q/U covariance change; "
                "angular and band-limit checks cover every reported marginal. "
                "The joint stage reconstructs any requested complete "
                "covariance."
            ),
        ),
    )
    if not all(r["passed"] for r in rows):
        raise SystemExit("Retained local-medium refinement failures")


def joint(config, record_path, case_id):
    from local_medium_joint import compute as compute_joint

    compute_joint(config, record_path, case_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["compute", "joint"])
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/paper/local-medium-sensitivity.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPO / "docs/local-medium-sensitivity-results.json",
    )
    parser.add_argument("--case")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage == "compute":
        compute(config, args.record)
    elif args.case:
        joint(config, args.record, args.case)
    else:
        parser.error("joint requires --case CASE_ID")
