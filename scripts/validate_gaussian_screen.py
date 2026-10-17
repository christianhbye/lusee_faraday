"""Resumable independent direct-screen convergence and moment validation.

Use --help for explicit parameters. Every saved screen has three angular
quadratures of the same pinned Fourier field and a nested cutoff check.
"""

import common  # noqa: F401
import argparse
import hashlib
import importlib.metadata
import json
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.fft import next_fast_len

from lusee_faraday.conventions import lambda_squared
from lusee_faraday.gaussian_screen import ScreenModel, qu_moments
from lusee_faraday.screen_draws import (
    FourierScreen,
    ScreenGrid,
    direct_readouts,
    spectral_covariance,
)
from lusee_faraday.screen_validation import moment_comparison

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated_data/gaussian_screen_inference"
CACHE = ROOT / "generated_data/cache/gaussian_screen_inference"
SOURCES = [
    "scripts/validate_gaussian_screen.py",
    "src/lusee_faraday/gaussian_screen.py",
    "src/lusee_faraday/screen_draws.py",
    "src/lusee_faraday/screen_validation.py",
    "src/lusee_faraday/conventions.py",
    "src/lusee_faraday/config.py",
]


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, default=plain, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def provenance(config):
    return dict(
        config=config,
        sources={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "lusee", "croissant-sim")
        },
    )


def geometry(model, freq, domain_factor=12.0):
    ell, w = model.correlation_pc, model.weight_sigma_pc
    width = max(domain_factor * w, 12 * ell)
    cutoff = 7.5 / ell
    phase = 2 * model.sigma_depth * lambda_squared(freq).max()
    # Initial spacing resolves modes and phase oscillations. All draws still
    # undergo two doubling checks; this prescription is not itself a gate.
    n = next_fast_len(
        int(
            np.ceil(
                max(
                    cutoff * width / np.pi + 2,
                    width / ell * phase,
                    2 * width / w,
                )
            )
        ),
        real=True,
    )
    n += n % 2
    return width, cutoff, n


def readout(field, freq, model, width):
    grid = ScreenGrid(field.shape[0], np.rad2deg(width / model.distance_pc))
    values = direct_readouts(field, freq, model, grid, names=("Q", "U"))
    return np.r_[values["Q"][1], values["U"][1]]


def one_screen(model, freq, seed, width, cutoff, n, mixture=1.0):
    started = time.monotonic()
    screen = FourierScreen.draw(model, width, cutoff, seed)
    amp = model.eta * model.p_ref_k * (freq / model.ref_mhz) ** model.beta
    scale = np.r_[amp, amp]
    field = screen.evaluate(4 * n)
    field = model.mean_depth + mixture * (field - model.mean_depth)
    levels = np.array(
        [
            readout(field[::step, ::step], freq, model, width)
            for step in (4, 2, 1)
        ]
    )
    changes = abs(np.diff(levels, axis=0)) / scale
    low = screen.truncate(6.5 / model.correlation_pc).evaluate(2 * n)
    low = model.mean_depth + mixture * (low - model.mean_depth)
    cutoff_change = abs(readout(low, freq, model, width) - levels[1]) / scale
    # Parseval spatial covariance of this exact Fourier realization. Keep DC.
    m = screen.coefficients.shape[0] // 2
    k = 2 * np.pi / width * np.arange(-m, m + 1)
    lags = model.correlation_pc * np.array([0.0, 0.5, 1.0, 2.0])
    spatial_cov = np.cos(lags[:, None] * k) @ (
        abs(screen.coefficients) ** 2
    ).sum(axis=1)
    return dict(
        levels=levels,
        changes=changes,
        cutoff_change=cutoff_change,
        spatial_cov=spatial_cov,
        seconds=time.monotonic() - started,
        mixture=mixture,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ell", type=float, nargs="+", default=[0.05, 0.5, 5.0]
    )
    parser.add_argument(
        "--freq", type=float, nargs="+", default=[45.0, 47.5, 50.0]
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--label", default="pilot")
    parser.add_argument("--domain-factor", type=float, default=12.0)
    parser.add_argument("--mixture", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or min(args.ell) <= 0:
        parser.error("positive sample count and scales required")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    all_results = []
    started = time.monotonic()
    for ell in args.ell:
        model = ScreenModel(correlation_pc=ell)
        freq = np.array(args.freq)
        width, cutoff, n = geometry(model, freq, args.domain_factor)
        config = dict(
            model=asdict(model),
            freq=freq.tolist(),
            width_pc=width,
            cutoff=cutoff,
            n=n,
            mixture=args.mixture,
            mixture_amplitudes=[np.sqrt(0.5), np.sqrt(1.5)],
        )
        prov = provenance(config)
        key = hashlib.sha256(
            json.dumps(prov, sort_keys=True).encode()
        ).hexdigest()
        folder = CACHE / key
        folder.mkdir(exist_ok=True)
        write_json(folder / "config.json", prov)
        print(f"ell={ell}, n={n},{2*n},{4*n}, cache={key[:12]}", flush=True)
        records = []
        for seed in range(args.seed_start, args.seed_start + args.samples):
            path = folder / f"{seed}.npz"
            if not path.exists():
                mix = (
                    np.random.default_rng([seed, 937]).choice(
                        [np.sqrt(0.5), np.sqrt(1.5)]
                    )
                    if args.mixture
                    else 1.0
                )
                record = one_screen(model, freq, seed, width, cutoff, n, mix)
                tmp = path.with_suffix(".tmp.npz")
                np.savez(tmp, **record)
                tmp.replace(path)
            with np.load(path, allow_pickle=False) as saved:
                records.append({name: saved[name] for name in saved.files})
            if len(records) % 8 == 0 or len(records) == args.samples:
                print(
                    f"  {len(records)}/{args.samples}, "
                    f'last {float(records[-1]["seconds"]):.2f}s',
                    flush=True,
                )
        samples = np.array([r["levels"][-1] for r in records])
        reference = qu_moments(freq, model)
        max_change = max(float(r["changes"].max()) for r in records)
        max_cutoff = max(float(r["cutoff_change"].max()) for r in records)
        lags = ell * np.array([0.0, 0.5, 1.0, 2.0])
        expected = model.sigma_depth**2 * np.exp(-0.5 * (lags / ell) ** 2)
        spec = spectral_covariance(model, width, cutoff, lags)
        domain = spectral_covariance(model, 1.5 * width, cutoff, lags)
        spatial = np.array([r["spatial_cov"] for r in records])
        tail = np.exp(-(width**2) / (8 * model.weight_sigma_pc**2))
        phase_bound = (
            2
            * model.sigma_depth
            * lambda_squared(freq).max()
            * np.exp(-0.25 * ell**2 * cutoff**2)
        )
        numerical_pass = bool(
            max_change < 1e-4
            and max_cutoff < 1e-4
            and tail < 1e-6
            and phase_bound < 1e-4
            and max(abs(spec - expected)) < 1e-8 * expected[0]
        )
        result = dict(
            ell=ell,
            cache=key,
            config=config,
            seed_start=args.seed_start,
            samples=args.samples,
            max_quadrature_change=max_change,
            max_cutoff_change=max_cutoff,
            spatial_lags_pc=lags,
            spatial_expected=expected,
            spatial_spectral=spec,
            spatial_larger_domain=domain,
            spatial_empirical=spatial.mean(0),
            spatial_se=(
                spatial.std(0, ddof=1) / np.sqrt(len(spatial))
                if len(spatial) > 1
                else None
            ),
            weight_tail_bound=tail,
            missing_phase_rms_bound=phase_bound,
            numerical_pass=numerical_pass,
            mean_seconds_per_screen=np.mean([r["seconds"] for r in records]),
        )
        if len(samples) >= 8:
            result["moments"] = moment_comparison(
                samples, reference, family_size=len(args.ell)
            )
        all_results.append(result)
        write_json(
            OUT / f"{args.label}.json",
            dict(
                results=all_results,
                elapsed_seconds=time.monotonic() - started,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
                provenance=prov,
                complete=len(all_results) == len(args.ell),
            ),
        )
        print(
            f"  numerical_pass={numerical_pass}, max shift={max_change:.3g}",
            flush=True,
        )
    print(f"Wrote {OUT/args.label}.json", flush=True)


if __name__ == "__main__":
    main()
