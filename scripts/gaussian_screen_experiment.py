"""Information assessment and bounded toy inference using validated screens."""

import common  # noqa: F401
import argparse
import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
from scipy.linalg import eigh

from lusee_faraday.conventions import lambda_squared
from lusee_faraday.gaussian_screen import ScreenModel, qu_moments
from lusee_faraday.screen_inference import (
    gaussian_kl,
    profile_expected_kl,
    local_stationary_template,
)
from lusee_faraday.screen_validation import noise_weighted_error
from validate_gaussian_screen import CACHE, OUT, ROOT, provenance, write_json

FREQ = np.linspace(45, 50, 5)
SCALES = np.geomspace(0.05, 5, 41)
NOISE_FRACTION = 0.05


def noise_covariance(freq, model, fraction=NOISE_FRACTION):
    amplitude = (
        model.eta * model.p_ref_k * (freq / model.ref_mhz) ** model.beta
    )
    # Equal total inverse-variance budget when checking denser frequency grids.
    sd = (
        fraction * np.r_[amplitude, amplitude] * np.sqrt(len(freq) / len(FREQ))
    )
    return np.diag(sd**2)


def assess():
    model = ScreenModel()
    noise = noise_covariance(FREQ, model)
    truth = qu_moments(FREQ, model, method="special")
    assessment = dict(
        model=asdict(model),
        freq_mhz=FREQ,
        noise_fraction=NOISE_FRACTION,
        scales_pc=SCALES,
        rms_phase=2 * model.sigma_depth * lambda_squared(FREQ),
        attenuation=np.exp(
            -2 * model.sigma_depth**2 * lambda_squared(FREQ) ** 2
        ),
        recovery_rule="2 profiled KL >= 4 for both factor-two alternatives, "
        "with free amplitude/angle/beta and converged optimization",
        recovery=[],
        grid_checks=[],
        templates=[],
    )
    for ell in (0.05, 0.15, 0.5, 1.5, 5.0):
        cfg = replace(model, correlation_pc=ell)
        true = qu_moments(FREQ, cfg, method="special")
        eig = eigh(true["covariance"], noise, eigvals_only=True)
        row = dict(ell=ell, noise_relative_eigenvalues=eig, comparisons=[])
        for alternative in (ell / 2, ell * 2):
            candidate = qu_moments(
                FREQ,
                replace(model, correlation_pc=alternative),
                method="special",
            )
            result = dict(
                alternative=alternative,
                fixed_2kl=2
                * gaussian_kl(
                    true["mean"],
                    true["covariance"] + noise,
                    candidate["mean"],
                    candidate["covariance"] + noise,
                ),
                stages=[],
            )
            for stage in (1, 2, 3):
                result["stages"].append(
                    profile_expected_kl(
                        true, candidate, noise, FREQ, model, stage
                    )
                )
            row["comparisons"].append(result)
        row["supports_recovery"] = all(
            r["stages"][1]["converged"] and 2 * r["stages"][1]["kl"] >= 4
            for r in row["comparisons"]
        )
        assessment["recovery"].append(row)
    for n in (5, 9, 17):
        freq = np.linspace(45, 50, n)
        true = qu_moments(freq, model, method="special")
        other = qu_moments(
            freq, replace(model, correlation_pc=1.0), method="special"
        )
        value = profile_expected_kl(
            true, other, noise_covariance(freq, model), freq, model, 2
        )
        assessment["grid_checks"].append(dict(n=n, **value))
    # Exact source-size ambiguity at fixed ell/w, with beam dilution absorbed
    # by an allowed polarized-amplitude change.
    changed = replace(model, source_sigma_deg=1.0)
    changed = replace(
        changed,
        correlation_pc=model.correlation_pc
        * changed.weight_sigma_pc
        / model.weight_sigma_pc,
        p_ref_k=model.p_ref_k * model.eta / changed.eta,
    )
    other = qu_moments(FREQ, changed, method="special")
    assessment["source_width_degeneracy"] = dict(
        model=asdict(changed),
        max_mean_error=float(np.max(abs(other["mean"] - truth["mean"]))),
        max_covariance_error=float(
            np.max(abs(other["covariance"] - truth["covariance"]))
        ),
    )
    # Small-area limit: doubling ell multiplies covariance by four at fixed
    # frequency/source. This is a limit check, not a new validated screen grid.
    tiny = qu_moments(
        FREQ, replace(model, correlation_pc=0.001), method="special"
    )
    twice = qu_moments(
        FREQ, replace(model, correlation_pc=0.002), method="special"
    )
    assessment["small_area_scaling_error"] = float(
        np.linalg.norm(twice["covariance"] - 4 * tiny["covariance"])
        / np.linalg.norm(twice["covariance"])
    )
    for lo, hi in (
        (45.0, 50.0),
        (25.0, 35.0),
        (29.9, 30.1),
        (19.9, 20.1),
        (9.9, 10.1),
    ):
        freq = np.linspace(lo, hi, 17)
        exact = qu_moments(freq, model)
        fast = qu_moments(freq, model, method="special")
        template = local_stationary_template(freq, model, (lo + hi) / 2)
        cov = exact["covariance"]
        approx = template["covariance"]
        sd = np.sqrt(np.diag(cov))
        sa = np.sqrt(np.diag(approx))
        correlation = cov / np.outer(sd, sd)
        approximate_correlation = approx / np.outer(sa, sa)
        fitted = float(np.sum(cov * approx) / np.sum(approx * approx))
        wn = noise_covariance(freq, model)
        assessment["templates"].append(
            dict(
                freq_mhz=freq,
                mean=exact["mean"],
                covariance=cov,
                template_covariance=approx,
                normalized_template_variance=template["normalized_variance"],
                max_diagonal_relative_error=float(
                    np.max(abs(np.diag(approx) / np.diag(cov) - 1))
                ),
                max_correlation_error=float(
                    np.max(abs(correlation - approximate_correlation))
                ),
                fitted_amplitude_power=fitted,
                fitted_covariance_relative_error=float(
                    np.linalg.norm(fitted * approx - cov) / np.linalg.norm(cov)
                ),
                whitened_errors=noise_weighted_error(
                    -exact["mean"], approx - cov, cov + wn
                ),
                special_vs_quadrature=noise_weighted_error(
                    fast["mean"] - exact["mean"],
                    fast["covariance"] - cov,
                    cov + wn,
                ),
            )
        )
    # The strong same-frequency asymptote and full integral, without claiming
    # direct-screen validation at frequencies below the saved ensemble window.
    assessment["strong_diagonal"] = []
    for nu in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 45.0, 50.0):
        exact = qu_moments([nu], model, method="special")
        approx = local_stationary_template(np.array([nu]), model, nu)
        assessment["strong_diagonal"].append(
            dict(
                freq_mhz=nu,
                centered_power=float(np.trace(exact["covariance"])),
                mean_power=float(exact["mean"] @ exact["mean"]),
                approximate_power=float(np.trace(approx["covariance"])),
            )
        )
    assessment["recovery_enabled"] = any(
        r["supports_recovery"] for r in assessment["recovery"]
    )
    assessment["provenance"] = provenance(
        dict(
            model=asdict(model),
            freq_mhz=FREQ.tolist(),
            noise_fraction=NOISE_FRACTION,
        )
    )
    assessment["assessment_sources"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "scripts/gaussian_screen_experiment.py",
            "src/lusee_faraday/screen_inference.py",
        )
    }
    write_json(OUT / "assessment.json", assessment)
    print(
        "Recovery criterion:",
        [(r["ell"], r["supports_recovery"]) for r in assessment["recovery"]],
        flush=True,
    )
    print("Grid KL:", assessment["grid_checks"], flush=True)


def check_gates():
    for name in ("covariance-validation", "strong-validation"):
        record = json.loads((OUT / f"{name}.json").read_text())
        if not record["complete"] or not all(
            r["numerical_pass"] and r["moments"]["passed"]
            for r in record["results"]
        ):
            raise RuntimeError(f"{name} gate not passed; inference is blocked")
        required_scales = (
            [0.05, 0.15, 0.5, 1.5, 5.0]
            if name == "covariance-validation"
            else [0.5]
        )
        required_freq = (
            FREQ if name == "covariance-validation" else [20, 25, 30, 35]
        )
        if sorted(r["ell"] for r in record["results"]) != required_scales:
            raise RuntimeError(f"{name}: required scale coverage is missing")
        for row in record["results"]:
            stored = json.loads(
                (CACHE / row["cache"] / "config.json").read_text()
            )
            if stored != provenance(row["config"]):
                raise RuntimeError(f"{name}: stale validation provenance")
            if row["samples"] < 128 or not np.array_equal(
                row["config"]["freq"], required_freq
            ):
                raise RuntimeError(
                    f"{name}: incomplete sample/frequency coverage"
                )
            z = (
                np.array(row["spatial_empirical"])
                - np.array(row["spatial_expected"])
            ) / np.array(row["spatial_se"])
            if np.max(abs(z)) > row["moments"]["threshold"]:
                raise RuntimeError(f"{name} generator covariance gate failed")


def freeze():
    check_gates()
    assessment = json.loads((OUT / "assessment.json").read_text())
    if assessment["provenance"] != provenance(
        dict(
            model=asdict(ScreenModel()),
            freq_mhz=FREQ.tolist(),
            noise_fraction=NOISE_FRACTION,
        )
    ):
        raise RuntimeError("stale information assessment")
    for name, digest in assessment["assessment_sources"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"stale assessment source: {name}")
    if assessment["recovery_enabled"]:
        raise RuntimeError(
            "Information supports recovery: configure interval calibration "
            "before production"
        )
    config = dict(
        freq_mhz=FREQ.tolist(),
        model=asdict(ScreenModel()),
        noise_fraction=NOISE_FRACTION,
        scale_grid_pc=SCALES.tolist(),
        calibration_samples=4096,
        rate_halfwidth_design_target=0.012,
        simultaneous_null_comparisons=18,
        count_design="4096 independent trials per null configuration: "
        "normal-design simultaneous half-width below 0.012 at "
        "p=0.05; exact binomial intervals determine the result",
        null_validation_samples=4096,
        alternative_samples=1024,
        alternative_scales_pc=[0.5, 1.5],
        false_positive_target=0.05,
        false_positive_upper_tolerance=0.015,
        simultaneous_confidence=0.95,
        seed_calibration=200000,
        seed_null_validation=300000,
        seed_alternative_noise=500000,
        seed_alternative_screen=60000,
        seed_mixture_screen=80000,
        mixture_amplitudes=[float(np.sqrt(0.5)), float(np.sqrt(1.5))],
        source_nuisance_stages=["amplitude_angle", "amplitude_angle_beta"],
        beta_bounds=[-4.0, -1.0],
        constant_depth_bounds=[-0.1, 0.1],
        null_depths=[-0.04, 0.0, 0.04],
        numerical_mean_whitened_tolerance=0.01,
        numerical_covariance_whitened_tolerance=0.01,
        numerical_statistic_relative_tolerance=0.001,
        numerical_detection_flip_tolerance=1,
        recovery="disabled by finite-model information assessment",
        null_truth_scope="amplitude=1 K, angle=20 deg, beta=config.BETA_QU; "
        "constant depths -0.04, 0, +0.04 rad/m^2 only",
        discrepancy="fixed fiducial ell=0.5 pc covariance weighting; "
        "no scale scan",
        known_source_statistic="maximum Gaussian log likelihood over "
        "41 scales "
        "versus zero rotation; empirically calibrated",
        scope="conditional toy benchmark; source-known likelihood and "
        "source-profiled discrepancy tests; no radiometer forecast",
    )
    sources = [
        "scripts/gaussian_screen_experiment.py",
        "src/lusee_faraday/screen_inference.py",
        "scripts/calibrate_gaussian_screen.py",
        "src/lusee_faraday/screen_validation.py",
    ]
    config["source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in sources
    }
    config["assessment_sha256"] = hashlib.sha256(
        (OUT / "assessment.json").read_bytes()
    ).hexdigest()
    path = OUT / "frozen-experiment.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError(
            "Existing frozen design differs; preserve it and use "
            "a new run directory"
        )
    write_json(path, config)
    print("Frozen", path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["assess", "freeze"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "assess":
        assess()
    else:
        freeze()


if __name__ == "__main__":
    main()
