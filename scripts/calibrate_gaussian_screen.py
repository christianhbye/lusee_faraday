"""Calibrated toy tests with disjoint null and physical-screen samples.

Requires the frozen experiment and both spatial moment gates. All expensive
null fits are checkpointed in chunks. No receiver sensitivity is inferred.
"""

import common  # noqa: F401
import hashlib
import json
import time

import numpy as np
from scipy.stats import beta as beta_distribution, skew, kurtosis

from lusee_faraday.gaussian_screen import ScreenModel, qu_moments
from lusee_faraday.screen_inference import (
    ablate_frequency_covariance,
    binomial_interval,
    covariance_discrepancy,
    fixed_source_scores,
    mixture_moments,
    profile_null_batch,
)
from lusee_faraday.screen_validation import (
    moment_comparison,
    noise_weighted_error,
)
from gaussian_screen_experiment import check_gates, noise_covariance
from validate_gaussian_screen import CACHE, OUT, ROOT, provenance, write_json
from dataclasses import replace


def load_samples(label, ell):
    ledger = json.loads((OUT / f"{label}.json").read_text())
    if not ledger["complete"]:
        raise RuntimeError(f"incomplete ensemble {label}")
    entry = next(r for r in ledger["results"] if np.isclose(r["ell"], ell))
    if not entry["numerical_pass"]:
        raise RuntimeError(f"numerical gate failed for {label}/{ell}")
    folder = CACHE / entry["cache"]
    stored = json.loads((folder / "config.json").read_text())
    if stored != provenance(entry["config"]):
        raise RuntimeError("screen cache source/configuration mismatch")
    rows = []
    middle = []
    for seed in range(
        entry["seed_start"], entry["seed_start"] + entry["samples"]
    ):
        with np.load(folder / f"{seed}.npz", allow_pickle=False) as saved:
            rows.append(saved["levels"][-1])
            middle.append(saved["levels"][-2])
    return np.array(rows), np.array(middle), entry


def main():
    check_gates()
    frozen = OUT / "frozen-experiment.json"
    config = json.loads(frozen.read_text())
    for name, digest in config["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"frozen source changed: {name}")
    key = hashlib.sha256(frozen.read_bytes()).hexdigest()
    cache = CACHE / f"calibration-{key}"
    cache.mkdir(exist_ok=True)
    freq = np.array(config["freq_mhz"])
    model = ScreenModel(**config["model"])
    noise = noise_covariance(freq, model, config["noise_fraction"])
    sigma = np.sqrt(np.diag(noise))
    fiducial = qu_moments(freq, model, method="special")
    signals = [
        fiducial["covariance"],
        ablate_frequency_covariance(fiducial["covariance"]),
    ]
    moments = [
        qu_moments(freq, replace(model, correlation_pc=ell), method="special")
        for ell in config["scale_grid_pc"]
    ]
    diagonal_moments = [
        dict(m, covariance=ablate_frequency_covariance(m["covariance"]))
        for m in moments
    ]
    zero = qu_moments(
        freq, replace(model, mean_depth=0.0, sigma_depth=0.0), method="special"
    )
    output = dict(
        config=config,
        config_sha256=key,
        null_tests=[],
        power=[],
        likelihood_diagnostics=[],
        mixture={},
        thresholds={},
        statistic_stability=[],
        nuisance_fits=[],
        conditioning=[
            dict(
                ell=ell,
                total_condition_number=float(
                    np.linalg.cond(m["covariance"] + noise)
                ),
                minimum_signal_eigenvalue=float(
                    np.linalg.eigvalsh(m["covariance"])[0]
                ),
            )
            for ell, m in zip(config["scale_grid_pc"], moments)
        ],
    )

    def boundary_fractions(parameters, constant_depth, free_beta):
        amplitudes, _, depths, betas = parameters.T
        limit = np.minimum(
            model.i_ref_k,
            np.min(
                model.i_ref_k
                * (freq[None, :] / model.ref_mhz)
                ** (model.beta - betas[:, None]),
                axis=1,
            ),
        )
        hits = dict(
            amplitude_lower=np.isclose(amplitudes, 0, atol=1e-5),
            amplitude_upper=np.isclose(amplitudes, limit, atol=1e-5),
        )
        if constant_depth:
            hits["depth"] = np.any(
                np.isclose(
                    depths[:, None], config["constant_depth_bounds"], atol=1e-5
                ),
                axis=1,
            )
        if free_beta:
            hits["beta"] = np.any(
                np.isclose(betas[:, None], config["beta_bounds"], atol=1e-5),
                axis=1,
            )
        return dict(
            any=float(np.any(list(hits.values()), axis=0).mean()),
            **{name: float(value.mean()) for name, value in hits.items()},
        )

    ncal = config["calibration_samples"]
    nval = config["null_validation_samples"]
    nulls = {}
    started = time.monotonic()
    for split, count, seedbase in [
        ("calibration", ncal, config["seed_calibration"]),
        ("validation", nval, config["seed_null_validation"]),
    ]:
        for di, depth in enumerate(config["null_depths"]):
            predicted = qu_moments(
                freq,
                replace(model, mean_depth=depth, sigma_depth=0.0),
                method="special",
            )
            chunks = []
            for start in range(0, count, 128):
                path = cache / f"{split}-{di}-{start}.npz"
                if not path.exists():
                    n = min(128, count - start)
                    rng = np.random.default_rng([seedbase, di, start])
                    data = (
                        predicted["mean"]
                        + rng.normal(size=(n, len(sigma))) * sigma
                    )
                    record = dict(data=data)
                    for stage, free_beta in enumerate((False, True)):
                        for null, is_constant in [("H0", False), ("Hc", True)]:
                            if null == "H0" and depth != 0:
                                continue
                            fit = profile_null_batch(
                                data,
                                freq,
                                model,
                                noise,
                                is_constant,
                                free_beta,
                                beta_bounds=tuple(config["beta_bounds"]),
                                depth_bounds=tuple(
                                    config["constant_depth_bounds"]
                                ),
                            )
                            if not fit["converged"].all():
                                raise RuntimeError("unconverged null fit")
                            record[f"{stage}-{null}-parameters"] = fit[
                                "parameters"
                            ]
                            for weight, signal in enumerate(signals):
                                record[f"{stage}-{null}-{weight}"] = (
                                    covariance_discrepancy(
                                        fit["residual"], signal, noise
                                    )
                                )
                    if depth == 0:
                        nullscore = fixed_source_scores(data, [zero], noise)[
                            :, 0
                        ]
                        for weight, alternatives in enumerate(
                            (moments, diagonal_moments)
                        ):
                            record[f"known-{weight}"] = 2 * (
                                fixed_source_scores(
                                    data, alternatives, noise
                                ).max(axis=1)
                                - nullscore
                            )
                    tmp = path.with_suffix(".tmp.npz")
                    np.savez(tmp, **record)
                    tmp.replace(path)
                with np.load(path, allow_pickle=False) as saved:
                    chunks.append({name: saved[name] for name in saved.files})
            nulls[split, depth] = {
                name: np.concatenate([c[name] for c in chunks])
                for name in chunks[0]
            }
            print(
                f"{split} depth={depth}: {count} completed "
                f"({time.monotonic()-started:.1f}s)",
                flush=True,
            )
    alpha = (1 - config["simultaneous_confidence"]) / config[
        "simultaneous_null_comparisons"
    ]
    quantile = 1 - config["false_positive_target"]
    for stage in range(2):
        for null in ("H0", "Hc"):
            depths = [0.0] if null == "H0" else config["null_depths"]
            for weight in range(2):
                name = f"{stage}-{null}-{weight}"
                thresholds = [
                    float(
                        np.quantile(
                            nulls["calibration", d][name],
                            quantile,
                            method="higher",
                        )
                    )
                    for d in depths
                ]
                threshold = max(thresholds)
                output["thresholds"][name] = threshold
                for depth in depths:
                    score = nulls["validation", depth][name]
                    detections = int(np.count_nonzero(score > threshold))
                    interval = binomial_interval(detections, len(score), alpha)
                    parameters = nulls["validation", depth][
                        f"{stage}-{null}-parameters"
                    ]
                    boundaries = boundary_fractions(
                        parameters, null == "Hc", bool(stage)
                    )
                    output["null_tests"].append(
                        dict(
                            name=name,
                            depth=depth,
                            n=len(score),
                            detections=detections,
                            rate=detections / len(score),
                            interval=interval,
                            threshold=threshold,
                            nuisance_boundary_fraction=boundaries["any"],
                            nuisance_boundaries=boundaries,
                            passed=bool(
                                interval[1]
                                <= config["false_positive_target"]
                                + config["false_positive_upper_tolerance"]
                            ),
                        )
                    )
    for weight in range(2):
        name = f"known-{weight}"
        threshold = float(
            np.quantile(
                nulls["calibration", 0.0][name], quantile, method="higher"
            )
        )
        output["thresholds"][name] = threshold
        score = nulls["validation", 0.0][name]
        detections = int(np.count_nonzero(score > threshold))
        interval = binomial_interval(detections, len(score), alpha)
        output["null_tests"].append(
            dict(
                name=name,
                depth=0.0,
                n=len(score),
                detections=detections,
                rate=detections / len(score),
                interval=interval,
                threshold=threshold,
                passed=bool(
                    interval[1]
                    <= config["false_positive_target"]
                    + config["false_positive_upper_tolerance"]
                ),
            )
        )
    # Exact order-statistic uncertainty for each calibration quantile, before
    # taking the conservative max across the declared Hc depths.
    rank = int(np.ceil(quantile * (ncal - 1))) + 1
    output["calibration_tail_probability_interval"] = beta_distribution.ppf(
        [alpha / 2, 1 - alpha / 2], ncal + 1 - rank, rank
    ).tolist()
    output["null_calibration_passed"] = all(
        r["passed"] for r in output["null_tests"]
    )
    for label, ell, noise_seed in [
        ("final-validation", 0.5, config["seed_alternative_noise"]),
        ("final-validation", 1.5, config["seed_alternative_noise"] + 1),
        ("mixture-validation", 0.5, config["seed_alternative_noise"] + 2),
    ]:
        sample, middle, entry = load_samples(label, ell)
        if len(sample) != config["alternative_samples"]:
            raise RuntimeError(
                "alternative sample count differs from frozen design"
            )
        rng = np.random.default_rng(noise_seed)
        data = sample + rng.normal(size=sample.shape) * sigma
        expected = qu_moments(
            freq, replace(model, correlation_pc=ell), method="special"
        )
        total = expected["covariance"] + noise
        numeric = noise_weighted_error(
            (sample - middle).mean(0),
            np.cov(sample, rowvar=False) - np.cov(middle, rowvar=False),
            total,
        )
        if (
            numeric[0] > config["numerical_mean_whitened_tolerance"]
            or numeric[1] > config["numerical_covariance_whitened_tolerance"]
        ):
            raise RuntimeError(
                "noise-weighted numerical error exceeds tolerance"
            )
        white = np.linalg.solve(
            np.linalg.cholesky(total), (data - expected["mean"]).T
        ).T
        output["likelihood_diagnostics"].append(
            dict(
                label=label,
                ell=ell,
                n=len(sample),
                mean_whitened=white.mean(0),
                variance_whitened=white.var(0, ddof=1),
                skew=skew(white, axis=0),
                excess_kurtosis=kurtosis(white, axis=0),
                tail_counts_3sigma=(abs(white) > 3).sum(axis=0),
                squared_radius_quantiles=np.quantile(
                    np.sum(white**2, axis=1), [0.5, 0.9, 0.95, 0.99]
                ),
                numerical_whitened_errors=numeric,
            )
        )
        record = dict(data=data, physical_sample=sample, mean=expected["mean"])
        refined_data = data + middle - sample

        def record_stability(name, fine_score, coarse_score):
            threshold = output["thresholds"][name]
            change = float(np.max(abs(fine_score - coarse_score)))
            flips = int(
                np.count_nonzero(
                    (fine_score > threshold) != (coarse_score > threshold)
                )
            )
            passed = bool(
                change / max(1.0, abs(threshold))
                < config["numerical_statistic_relative_tolerance"]
                and flips <= config["numerical_detection_flip_tolerance"]
            )
            output["statistic_stability"].append(
                dict(
                    label=label,
                    ell=ell,
                    name=name,
                    max_score_change=change,
                    detection_flips=flips,
                    passed=passed,
                )
            )
            if not passed:
                raise RuntimeError(
                    "statistic is unstable under quadrature refinement"
                )

        for stage, free_beta in enumerate((False, True)):
            for null, is_constant in [("H0", False), ("Hc", True)]:
                fit = profile_null_batch(
                    data,
                    freq,
                    model,
                    noise,
                    is_constant,
                    free_beta,
                    beta_bounds=tuple(config["beta_bounds"]),
                    depth_bounds=tuple(config["constant_depth_bounds"]),
                )
                if not fit["converged"].all():
                    raise RuntimeError("unconverged alternative null fit")
                refined_fit = profile_null_batch(
                    refined_data,
                    freq,
                    model,
                    noise,
                    is_constant,
                    free_beta,
                    beta_bounds=tuple(config["beta_bounds"]),
                    depth_bounds=tuple(config["constant_depth_bounds"]),
                )
                if not refined_fit["converged"].all():
                    raise RuntimeError("unconverged quadrature-check null fit")
                record[f"{stage}-{null}-parameters"] = fit["parameters"]
                output["nuisance_fits"].append(
                    dict(
                        label=label,
                        ell=ell,
                        stage=stage,
                        null=null,
                        converged=int(fit["converged"].sum()),
                        n=len(data),
                        boundaries=boundary_fractions(
                            fit["parameters"], is_constant, free_beta
                        ),
                    )
                )
                for weight, signal in enumerate(signals):
                    name = f"{stage}-{null}-{weight}"
                    score = covariance_discrepancy(
                        fit["residual"], signal, noise
                    )
                    record_stability(
                        name,
                        score,
                        covariance_discrepancy(
                            refined_fit["residual"], signal, noise
                        ),
                    )
                    record[name] = score
                    count = int(
                        np.count_nonzero(score > output["thresholds"][name])
                    )
                    output["power"].append(
                        dict(
                            label=label,
                            ell=ell,
                            name=name,
                            n=len(score),
                            detections=count,
                            rate=count / len(score),
                            interval=binomial_interval(
                                count, len(score), 0.05 / 30
                            ),
                        )
                    )
        nullscore = fixed_source_scores(data, [zero], noise)[:, 0]
        for weight, alternatives in enumerate((moments, diagonal_moments)):
            name = f"known-{weight}"
            score = 2 * (
                fixed_source_scores(data, alternatives, noise).max(axis=1)
                - nullscore
            )
            coarse_score = 2 * (
                fixed_source_scores(refined_data, alternatives, noise).max(
                    axis=1
                )
                - fixed_source_scores(refined_data, [zero], noise)[:, 0]
            )
            record_stability(name, score, coarse_score)
            record[name] = score
            count = int(np.count_nonzero(score > output["thresholds"][name]))
            output["power"].append(
                dict(
                    label=label,
                    ell=ell,
                    name=name,
                    n=len(score),
                    detections=count,
                    rate=count / len(score),
                    interval=binomial_interval(count, len(score), 0.05 / 30),
                )
            )
        if label.startswith("mixture"):
            exact = mixture_moments(freq, model)
            output["mixture"] = dict(
                gaussian_moments=moment_comparison(sample, expected),
                conditional_mixture_moments=moment_comparison(sample, exact),
                gaussian_model_whitened_error=noise_weighted_error(
                    exact["mean"] - expected["mean"],
                    exact["covariance"] - expected["covariance"],
                    total,
                ),
            )
        np.savez(OUT / f"observation-{label}-{ell}.npz", **record)
        print(f"{label} ell={ell} scored", flush=True)
        write_json(OUT / "inference-results.json", output)
    output["elapsed_seconds"] = time.monotonic() - started
    output["complete"] = True
    write_json(OUT / "inference-results.json", output)
    print(
        "Null calibration passed:",
        output["null_calibration_passed"],
        flush=True,
    )


if __name__ == "__main__":
    main()
