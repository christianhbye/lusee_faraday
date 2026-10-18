"""Decision table, simulated observations and future calibrated-data path."""

import common  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np

from common import CACHE_DIR, REPO
from faraday_covariance import projector
from paper_common import digest, load_record, manifest, write_json
from paper_kernels import load as load_kernel
from residual_signal import SOURCES as SIGNAL_SOURCES
from residual_signal import setup as signal_setup
from lusee_faraday import channelization as ch
from lusee_faraday.depth_channel_covariance import DepthChannelProjector
from lusee_faraday.frequency_covariance import gaussian_visibility_draw
from lusee_faraday.measurement_statistics import (
    gaussian_mean_square,
    repeat_difference,
)
from lusee_faraday.separability import fractional_residual
from lusee_faraday.spin_covariance import AngularScreen
from lusee_faraday.observing import sidereal_bin_window
from lusee_faraday.visibility_draws import draw_depth_coefficients
from lusee_faraday.gaussian_screen import real_covariance
from lusee_faraday.receiver_noise import stokes_quadratic_matrices


def load_array(row):
    path = REPO / row["path"]
    if digest(path) != row["sha256"]:
        raise ValueError("array content differs from completed record")
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def check_record_sources(record):
    for name, expected in record["provenance"]["sources"].items():
        if digest(REPO / name) != expected:
            raise ValueError(f"stale result source: {name}")


def selected_block(covariance, phase, component):
    index = 4 * phase + component
    return covariance[:, index, :, index]


def rms_bins(covariance, indices):
    return float(np.sqrt(max(np.diag(covariance)[indices].mean(), 0)))


def ratio(a, b):
    return float(a / b) if b > 0 else None


def complex_polarization_moments(c):
    """Joint real measured Q/U -> complex C and J for P=Q+iU."""
    qq, uu, qu, uq = c[:, 0, :, 0], c[:, 1, :, 1], c[:, 0, :, 1], c[:, 1, :, 0]
    covariance = qq + uu + 1j * (uq - qu)
    relation = qq - uu + 1j * (qu + uq)
    expected = np.block([[qq, qu], [uq, uu]])
    if not np.allclose(
        real_covariance(covariance, relation),
        expected,
        rtol=1e-10,
        atol=1e-12 * max(abs(expected).max(), 1e-300),
    ):
        raise ValueError("complex/real measured covariance round trip failed")
    return covariance, relation


def analyze_observation(path, nominal, operator, noise_covariance, reference):
    """Read calibrated Stokes data, require coordinates, export K and Q/I.

    The simulator writes exactly this input format. A mission reduction must
    supply its actual common calibration, flags and channel-response operator;
    these illustrative settings do not replace that calibration.
    """
    with np.load(path) as data:
        if not np.array_equal(data["frequency_mhz"], nominal):
            raise ValueError("observation frequency coordinates disagree")
        stokes = data["stokes_k"]
    if stokes.shape != (len(nominal), 4):
        raise ValueError("observation needs calibrated I/Q/U/V in kelvin")
    fraction = fractional_residual(stokes, noise_covariance, operator)
    fixed = fractional_residual(
        stokes, noise_covariance, operator, intensity_reference=reference
    )
    return dict(
        residual_stokes_k=operator @ stokes,
        estimated_i_fraction=fraction["value"],
        estimated_i_fraction_covariance=fraction["covariance"],
        fixed_i_fraction=fixed["value"],
        fixed_i_fraction_covariance=fixed["covariance"],
        intensity_fractional_noise=fraction[
            "maximum_intensity_fractional_rms"
        ],
    )


def draw(covariance, seed):
    n = covariance.shape[0] * covariance.shape[1]
    result = gaussian_visibility_draw(
        covariance.reshape(n, n), seed, relative_tolerance=1e-8
    )
    return result["sample"].reshape(covariance.shape[:2])


def predictive_statistics(covariance, mean, seed, groups, nominal):
    """Correlated model ensembles; ratio quantiles need no inverse moment."""
    count = 512
    dimension = covariance.shape[0] * covariance.shape[1]
    c = covariance.reshape(dimension, dimension)
    eigen, vectors = np.linalg.eigh((c + c.T) / 2)
    scale = max(float(abs(c).max()), 1e-300)
    if eigen[0] < -1e-8 * scale:
        raise ValueError("predictive statistic covariance is not PSD")
    rng = np.random.default_rng(seed)
    values = (
        rng.normal(size=(count, dimension)) * np.sqrt(np.maximum(eigen, 0))
    ) @ vectors.T
    values = values.reshape(count, len(nominal), 4) + mean
    ratios, power_checks = [], []
    for component in (1, 2):
        powers = {}
        for group, indices in groups.items():
            powers[group] = np.mean(values[:, indices, component] ** 2, axis=1)
            block = covariance[:, component, :, component][
                np.ix_(indices, indices)
            ]
            prediction = gaussian_mean_square(block, mean[indices, component])
            sem = prediction["sampling_rms_k2"] / np.sqrt(count)
            error = abs(powers[group].mean() - prediction["mean_k2"])
            gate = error / max(sem, 1e-300)
            power_checks.append(
                dict(
                    component=component,
                    channel_type=group,
                    monte_carlo_mean_k2=float(powers[group].mean()),
                    exact_prediction=prediction,
                    mean_error_over_mc_sem=float(gate),
                )
            )
        sampled_ratio = np.divide(powers["zoom"], powers["parent"])
        ratios.append(
            dict(
                component="Q" if component == 1 else "U",
                percentiles=[16, 84],
                zoom_parent_power_ratio_interval=np.percentile(
                    sampled_ratio, [16, 84]
                ).tolist(),
                ratio_summary=(
                    "Predictive quantiles; no expectation of a ratio is "
                    "assumed. Parent residual has one degree of freedom."
                ),
            )
        )
    if any(row["mean_error_over_mc_sem"] > 5 for row in power_checks):
        raise ValueError(
            "predictive power Monte Carlo fails preset 5-SEM gate"
        )
    return dict(
        ensembles=count,
        ratios=ratios,
        power_checks=power_checks,
        mode_minimum_eigenvalue=float(eigen[0]),
    )


def compute(config, record_path):
    _, base, kernel_record, _, signal_key, _ = signal_setup(config)
    signal_record = load_record(
        REPO / "docs/residual-signal-results.json", signal_key
    )
    system_record = json.loads(
        (REPO / "docs/measured-system-results.json").read_text()
    )
    check_record_sources(system_record)
    exposure = json.loads((REPO / config["exposure_record"]).read_text())
    noise_config = json.loads((REPO / config["noise_config"]).read_text())
    system_config = system_record["provenance"]["configuration"]
    if any(system_config.get(k) != value for k, value in config.items()):
        raise ValueError("measured-system estimator/configuration is stale")
    if system_config["noise"] != noise_config or (
        system_config["exposure_record_sha256"]
        != digest(REPO / config["exposure_record"])
    ):
        raise ValueError("noise or exposure inputs changed")
    leakage_config = json.loads((REPO / config["leakage_config"]).read_text())
    if leakage_config["sidereal_turns"] != base["sidereal_turns"]:
        raise ValueError("signal and I-only phase coordinates disagree")
    if not signal_record["all_passed"] or not system_record["all_passed"]:
        raise ValueError("upstream numerical gates must pass")
    complete = dict(
        config,
        input_records={
            name: digest(REPO / name)
            for name in [
                "docs/residual-signal-results.json",
                "docs/measured-system-results.json",
                config["exposure_record"],
                config["noise_config"],
            ]
        },
    )
    provenance, key = manifest(
        complete,
        SIGNAL_SOURCES
        + [
            "scripts/faraday_separability.py",
            "scripts/paper_common.py",
            "src/lusee_faraday/frequency_covariance.py",
            "src/lusee_faraday/separability.py",
            "src/lusee_faraday/measurement_statistics.py",
            "src/lusee_faraday/visibility_draws.py",
        ],
    )
    directory = CACHE_DIR / "faraday_separability" / key
    directory.mkdir(parents=True, exist_ok=True)
    groups = {"parent": np.arange(3), "zoom": np.arange(3, 195)}
    noise_summary, leakage_summary, leaks = {}, {}, {}
    for row in system_record["leakage"]:
        leaks[row["center_mhz"], row["response"]] = load_array(row)
        if row["response"] == "asbuilt":
            item = kernel_record["arrays"][f"{row['center_mhz']:g}"]
            k = load_kernel(REPO / item["path"], kernel_record["fingerprint"])
            expected = stokes_quadratic_matrices(
                k["x_vec"], k["y_vec"], float(k["normalization"])
            )
            if not np.allclose(
                leaks[row["center_mhz"], "asbuilt"]["readout"],
                expected,
                rtol=1e-10,
                atol=1e-12 * abs(expected).max(),
            ):
                raise ValueError("signal and noise calibration planes differ")
    for center in sorted(
        set(r["center_mhz"] for r in system_record["leakage"])
    ):
        reference = leaks[center, "asbuilt"]
        for phase in range(len(base["sidereal_turns"])):
            for component in (1, 2):
                for group, indices in groups.items():
                    column = phase * 4 + component
                    anchor = reference["linear_residual_k"][indices, column]
                    alternatives = [
                        leaks[center, variant][name + "_residual_k"][
                            indices, column
                        ]
                        for variant in ("asbuilt", "c4sym", "diagza")
                        for name in ("linear", "pchip")
                    ]
                    leakage_summary[center, phase, component, group] = dict(
                        asbuilt_linear_rms_k=float(
                            np.sqrt(np.mean(anchor**2))
                        ),
                        residual_model_rms_range_k=[
                            float(
                                min(
                                    np.sqrt(np.mean(a * a))
                                    for a in alternatives
                                )
                            ),
                            float(
                                max(
                                    np.sqrt(np.mean(a * a))
                                    for a in alternatives
                                )
                            ),
                        ],
                        worst_common_calibration_mismatch_rms_k=float(
                            max(
                                np.sqrt(np.mean((a - anchor) ** 2))
                                for a in alternatives
                            )
                        ),
                        asbuilt_interpolation_difference_rms_k=float(
                            np.sqrt(
                                np.mean(
                                    (
                                        reference["pchip_residual_k"][
                                            indices, column
                                        ]
                                        - anchor
                                    )
                                    ** 2
                                )
                            )
                        ),
                    )
    for row in system_record["noise"]:
        data = load_array(row)
        for phase in range(len(base["sidereal_turns"])):
            for component in (1, 2):
                for group, indices in groups.items():
                    c = data["residual_covariance_per_second"][
                        phase, :, component, :, component
                    ]
                    noise_summary[
                        row["center_mhz"],
                        row["voltage_nv_rt_hz"],
                        row["current_pa_rt_hz"],
                        phase,
                        component,
                        group,
                    ] = rms_bins(c, indices)
    decisions, samples = [], []
    for row in signal_record["results"]:
        data = load_array(row)
        center = row["center_mhz"]
        numerical_error = row["maximum_residual_covariance_error_k2"][
            "speckle"
        ]
        for phase, turn in enumerate(base["sidereal_turns"]):
            for component in (1, 2):
                raw = selected_block(data["raw_speckle"], phase, component)
                residual = selected_block(
                    data["residual_speckle"], phase, component
                )
                smooth = selected_block(
                    data["residual_smooth"], phase, component
                )
                for group, indices in groups.items():
                    raw_rms = rms_bins(raw, indices)
                    res_rms, smooth_rms = rms_bins(
                        residual, indices
                    ), rms_bins(smooth, indices)
                    leakage = leakage_summary[center, phase, component, group]
                    for exp in [
                        e
                        for e in exposure["results"]
                        if e["sidereal_turn"] == turn
                    ]:
                        for e in noise_config["voltage_asd_nv_rt_hz"]:
                            for current in noise_config[
                                "current_asd_pa_rt_hz"
                            ]:
                                noise_rms = (
                                    (
                                        noise_summary[
                                            center,
                                            e,
                                            current,
                                            phase,
                                            component,
                                            group,
                                        ]
                                        / np.sqrt(exp["exposure_s"])
                                    )
                                    if exp["available"]
                                    else None
                                )
                                decisions.append(
                                    dict(
                                        center_mhz=center,
                                        screen_index=row["screen_index"],
                                        profile_index=row["profile_index"],
                                        model=row["model"],
                                        phase_index=phase,
                                        sidereal_turn=turn,
                                        component=(
                                            "Q" if component == 1 else "U"
                                        ),
                                        channel_type=group,
                                        lunations=exp["lunations"],
                                        exposure_s=exp["exposure_s"],
                                        voltage_nv_rt_hz=e,
                                        current_pa_rt_hz=current,
                                        raw_speckle_rms_k=raw_rms,
                                        residual_speckle_rms_k=res_rms,
                                        residual_smooth_rms_k=smooth_rms,
                                        retained_speckle_power=ratio(
                                            res_rms**2, raw_rms**2
                                        ),
                                        residual_radiometer_rms_k=noise_rms,
                                        speckle_to_leakage_amplitude=ratio(
                                            res_rms,
                                            leakage["asbuilt_linear_rms_k"],
                                        ),
                                        speckle_to_noise_amplitude=(
                                            ratio(res_rms, noise_rms)
                                            if noise_rms is not None
                                            else None
                                        ),
                                        numerical_covariance_error_k2=(
                                            numerical_error
                                        ),
                                        **leakage,
                                    )
                                )
        selected = (
            row["model"] == "gaussian"
            and row["profile_index"] == config["figure_profile_index"]
            and row["screen_index"] == config["figure_screen_index"]
        )
        if selected:
            print("simulated observation", center, flush=True)
            samples.append(
                simulate(
                    row,
                    data,
                    config,
                    base,
                    system_record,
                    exposure,
                    noise_config,
                    leaks[center, "asbuilt"],
                    directory,
                    groups,
                    kernel_record,
                )
            )
    finite = [
        d for d in decisions if d["speckle_to_noise_amplitude"] is not None
    ]
    write_json(
        record_path,
        dict(
            fingerprint=key,
            provenance=provenance,
            decisions=decisions,
            observations=samples,
            largest_speckle_to_noise_amplitude=max(
                d["speckle_to_noise_amplitude"] for d in finite
            ),
            unavailable_noise_rows=sum(
                d["residual_radiometer_rms_k"] is None for d in decisions
            ),
            scope=(
                "Q and U RMS are reported separately, with power transmission "
                "distinct from amplitude ratios. All local profiles, "
                "screens, "
                "Gaussian/scale-mixture moments, bands, phases and channel "
                "types "
                "use the same smooth fit and time average. Voltage/current "
                "noise "
                "and lunar temperatures remain assumed scenarios. "
                "Native-beam "
                "interpolants and ablations give model spread, not a "
                "calibrated "
                "uncertainty or detection significance. Fixed sky repeats "
                "only "
                "reduce independent measurement noise."
            ),
        ),
    )


def simulate(
    row,
    signal,
    config,
    base,
    system_record,
    exposure,
    noise_config,
    leakage,
    directory,
    groups,
    kernel_record,
):
    center, phase = row["center_mhz"], config["figure_phase_index"]
    exp = next(
        e
        for e in exposure["results"]
        if e["lunations"] == config["figure_lunations"]
        and e["sidereal_turn"] == base["sidereal_turns"][phase]
    )
    if not exp["available"]:
        raise ValueError("selected figure phase has no nightside exposure")
    noise_row = next(
        n
        for n in system_record["noise"]
        if n["center_mhz"] == center
        and n["voltage_nv_rt_hz"] == noise_config["voltage_reference_nv_rt_hz"]
        and n["current_pa_rt_hz"] == noise_config["current_reference_pa_rt_hz"]
    )
    noise = load_array(noise_row)
    nc = noise["covariance_per_second"][phase] / exp["exposure_s"]
    ncr = noise["residual_covariance_per_second"][phase] / exp["exposure_s"]
    r, nominal = signal["operator"], signal["nominal_mhz"]
    kernel_file = kernel_record["arrays"][f"{center:g}"]
    kernels = load_kernel(
        REPO / kernel_file["path"], kernel_record["fingerprint"]
    )["stokes_kernels"]
    kernels = kernels * sidereal_bin_window(kernels.shape[-2] - 1)
    coefficients = signal["conditional_coefficients"][0]
    solver = DepthChannelProjector(
        projector(kernels, base, refinement=2),
        AngularScreen(**row["screen"]),
        coefficients.shape[1] - 1,
    )
    modes = draw_depth_coefficients(
        solver, coefficients, [config["seed"], int(center), 0]
    )
    smooth, speckle = modes["smooth"][0], modes["speckle"][0]
    start, stop = phase * 4, phase * 4 + 4
    sky_smooth, sky_speckle = smooth[:, start:stop], speckle[:, start:stop]
    leak = leakage["linear_stokes_k"][:, start:stop]
    fixed_sky = leak + sky_smooth + sky_speckle
    measurement_noise = [
        draw(nc, [config["seed"], int(center), k]) for k in (2, 3, 4)
    ]
    observation = fixed_sky + measurement_noise[0]
    repeat = fixed_sky + measurement_noise[1]
    control = leak + measurement_noise[0]
    independent_mock = fixed_sky + measurement_noise[2]
    single_visit_noise = nc * exp["eligible_visits"]
    single_visit_first = fixed_sky + draw(
        single_visit_noise, [config["seed"], int(center), 5]
    )
    single_visit_repeat = fixed_sky + draw(
        single_visit_noise, [config["seed"], int(center), 6]
    )
    observation_path = directory / f"observation_{center:g}.npz"
    np.savez_compressed(
        observation_path, frequency_mhz=nominal, stokes_k=observation
    )
    analyzed = analyze_observation(
        observation_path, nominal, r, nc, leak[:, 0]
    )
    independent_path = directory / f"independent_mock_{center:g}.npz"
    np.savez_compressed(
        independent_path, frequency_mhz=nominal, stokes_k=independent_mock
    )
    independent_result = analyze_observation(
        independent_path, nominal, r, nc, leak[:, 0]
    )
    stats = []
    residual_mean = r @ leak
    block_covariance = (
        signal["residual_total"][:, start:stop, :, start:stop] + ncr
    )
    predictive = predictive_statistics(
        block_covariance,
        residual_mean,
        [config["seed"], int(center), 17],
        groups,
        nominal,
    )
    for component in (1, 2):
        sc = selected_block(signal["residual_total"], phase, component)
        for name, indices in groups.items():
            noise_selected = ncr[:, component, :, component][
                np.ix_(indices, indices)
            ]
            model_cov = sc[np.ix_(indices, indices)] + noise_selected
            mean = (r @ leak)[indices, component]
            prediction = gaussian_mean_square(model_cov, mean)
            difference = repeat_difference(
                (r @ observation)[indices, component],
                (r @ repeat)[indices, component],
                noise_selected,
                noise_selected,
            )
            single_difference = repeat_difference(
                (r @ single_visit_first)[indices, component],
                (r @ single_visit_repeat)[indices, component],
                noise_selected * exp["eligible_visits"],
                noise_selected * exp["eligible_visits"],
            )
            stats.append(
                dict(
                    component="Q" if component == 1 else "U",
                    channel_type=name,
                    observed_mean_square_k2=float(
                        np.mean((r @ observation)[indices, component] ** 2)
                    ),
                    independent_mock_mean_square_k2=float(
                        np.mean(
                            independent_result["residual_stokes_k"][
                                indices, component
                            ]
                            ** 2
                        )
                    ),
                    conditional_gaussian_power_prediction=prediction,
                    repeat_difference_mean_square_k2=difference[
                        "observed_mean_square_k2"
                    ],
                    repeat_difference_prediction=difference["prediction"],
                    single_visit_repeat_difference_mean_square_k2=(
                        single_difference["observed_mean_square_k2"]
                    ),
                    single_visit_repeat_difference_prediction=(
                        single_difference["prediction"]
                    ),
                )
            )
    power_ratios, frequency_pairs = [], []
    for component in (1, 2):

        def powers(value):
            return {
                name: float(np.mean(value[indices, component] ** 2))
                for name, indices in groups.items()
            }

        observed_power = powers(r @ observation)
        mock_power = powers(independent_result["residual_stokes_k"])
        power_ratios.append(
            dict(
                component="Q" if component == 1 else "U",
                observed_zoom_parent_power_ratio=ratio(
                    observed_power["zoom"], observed_power["parent"]
                ),
                independent_mock_zoom_parent_power_ratio=ratio(
                    mock_power["zoom"], mock_power["parent"]
                ),
            )
        )
        residual = (r @ observation)[:, component] - residual_mean[
            :, component
        ]
        cov = block_covariance[:, component, :, component]

        def zoom_at(frequency):
            matches = np.flatnonzero(
                np.isclose(nominal[3:], frequency, rtol=0, atol=1e-10)
            )
            if len(matches) != 1:
                raise ValueError("named zoom frequency is not unique")
            return int(matches[0] + 3)

        reference = zoom_at(center)
        for offset in (1, 16, 32):
            other = zoom_at(center + offset * ch.ZOOM_STEP_HZ * 1e-6)
            expected = float(cov[reference, other])
            variance = float(cov[reference, reference] * cov[other, other])
            frequency_pairs.append(
                dict(
                    component="Q" if component == 1 else "U",
                    frequency_a_mhz=float(nominal[reference]),
                    frequency_b_mhz=float(nominal[other]),
                    observed_pair_product_k2=float(
                        residual[reference] * residual[other]
                    ),
                    predicted_covariance_k2=expected,
                    predicted_pair_product_sampling_rms_k2=float(
                        np.sqrt(max(variance + expected**2, 0))
                    ),
                    predicted_correlation=(
                        ratio(abs(expected), np.sqrt(max(variance, 0)))
                        * np.sign(expected)
                        if variance > 0
                        else None
                    ),
                )
            )
    path = directory / f"spectrum_products_{center:g}.npz"
    complex_moments = {}
    for label, covariance in (
        ("raw_signal", signal["raw_total"][:, start:stop, :, start:stop]),
        (
            "residual_signal",
            signal["residual_total"][:, start:stop, :, start:stop],
        ),
        ("raw_noise", nc),
        ("residual_noise", ncr),
    ):
        c, j = complex_polarization_moments(covariance[:, 1:3, :, 1:3])
        complex_moments[label + "_polarization_covariance"] = c
        complex_moments[label + "_polarization_relation"] = j
    np.savez_compressed(
        path,
        nominal_mhz=nominal,
        operator=r,
        observation=observation,
        repeat=repeat,
        single_visit_first=single_visit_first,
        single_visit_repeat=single_visit_repeat,
        single_visit_residual_noise_covariance=(ncr * exp["eligible_visits"]),
        leakage_noise_control=control,
        leakage=leak,
        smooth=sky_smooth,
        speckle=sky_speckle,
        noise_covariance=nc,
        residual_noise_covariance=ncr,
        residual_signal_covariance=signal["residual_total"][
            :, start:stop, :, start:stop
        ],
        raw_signal_covariance=signal["raw_total"][
            :, start:stop, :, start:stop
        ],
        **{
            k: v
            for k, v in analyzed.items()
            if k != "intensity_fractional_noise"
        },
        **complex_moments,
    )
    return dict(
        center_mhz=center,
        phase_index=phase,
        sidereal_turn=base["sidereal_turns"][phase],
        lunations=exp["lunations"],
        exposure_s=exp["exposure_s"],
        single_visit_exposure_s=(exp["bin_seconds"] * exp["duty_cycle"]),
        screen=row["screen"],
        profile=row["profile"],
        intrinsic=base["intrinsic"],
        electronics_duty_cycle=exp["duty_cycle"],
        voltage_reference_nv_rt_hz=noise_config["voltage_reference_nv_rt_hz"],
        current_reference_pa_rt_hz=noise_config["current_reference_pa_rt_hz"],
        statistics=stats,
        parent_zoom_power_ratios=power_ratios,
        frequency_pair_statistics=frequency_pairs,
        predictive_statistics=predictive,
        intensity_fractional_noise=analyzed["intensity_fractional_noise"],
        signal_mode_diagnostics=modes["mode_diagnostics"],
        native_to_signal_pol_rms_fraction=float(
            max(
                np.sqrt(
                    max(
                        np.diag(
                            selected_block(signal["raw_total"], phase, c)
                        ).max(),
                        0,
                    )
                )
                for c in (1, 2)
            )
            / leak[:, 0].min()
        ),
        observation_path=str(observation_path.relative_to(REPO)),
        observation_sha256=digest(observation_path),
        independent_mock_path=str(independent_path.relative_to(REPO)),
        independent_mock_sha256=digest(independent_path),
        path=str(path.relative_to(REPO)),
        sha256=digest(path),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/paper/separability.json"
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPO / "docs/faraday-separability-results.json",
    )
    args = parser.parse_args()
    compute(json.loads(args.config.read_text()), args.record)
