"""Inject mixed-depth covariance faults into disposable source copies."""

import common  # noqa: F401
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from common import GEN_DIR, REPO
from paper_common import digest, write_json

SOURCE = "src/lusee_faraday/depth_channel_covariance.py"
TEST = "tests/test_depth_channel_covariance.py::"
FAULTS = [
    (
        "lost Hermite phase",
        "np.array([1, 1j, -1, -1j])[k % 4]",
        "np.ones_like(k, dtype=complex)",
        "test_fractional_screen_matches_exact_phase_kernels",
    ),
    (
        "replace slab by mean emitting depth",
        "a = x[i:stop, None] * nodes",
        "a = x[i:stop, None] * np.full_like(nodes, weights @ nodes)",
        "test_slab_phase_moments_match_direct_bivariate_gaussian_averages",
    ),
    (
        "unnormalized Gaussian depth average",
        'result[i:stop] = np.einsum("fdk,d->fk", values, weights)',
        'result[i:stop] = .9 * np.einsum("fdk,d->fk", values, weights)',
        "test_truncated_gaussian_mean_matches_independent_adaptive_integral",
    ),
    (
        "all modes inherit an underflowed mean",
        "values = np.exp(log_magnitude) * phase_k",
        "values = np.exp(-t[..., None]**2/2) * phase_k",
        "test_log_coefficients_recover_power_after_mean_underflows",
    ),
    (
        "silent Hermite truncation",
        "if not np.isfinite(peak) or poisson.sf(max_degree, peak) > tail:",
        "if False:",
        "test_log_coefficients_recover_power_after_mean_underflows",
    ),
    (
        "drop nonconjugated readout term",
        "plus[0].real + plus[1].imag + minus[0].real + minus[1].imag",
        "plus[0].real + plus[1].imag + minus[0].real",
        "test_loaded_angular_channels_match_direct_double_depth_covariance",
    ),
    (
        "drop intrinsic frequency amplitude",
        "return w @ (a * amplitude[..., None])",
        "return w @ a",
        "test_loaded_angular_channels_match_direct_double_depth_covariance",
    ),
    (
        "claim zero omitted-mode bound",
        'return np.einsum("b,c,pq->bpcq", bound, bound, self.tail_norm)',
        'return 0*np.einsum("b,c,pq->bpcq", bound, bound, self.tail_norm)',
        "test_omitted_mode_bound_covers_actual_channel_error",
    ),
    (
        "unresolved depth returns zeros",
        'raise DepthConvergenceError("depth phase scale needs refinement")',
        "return dict(coefficients=np.zeros((len(x),1)))",
        "test_unresolved_and_invalid_depth_channels_are_rejected",
    ),
    (
        "phase relation conjugated",
        'nonconjugated_second=np.einsum("ik,jk,nk->ijn", a, b, powers)',
        'nonconjugated_second=np.einsum("ik,jk,nk->ijn", a, b.conj(), powers)',
        "test_slab_phase_moments_match_direct_bivariate_gaussian_averages",
    ),
]


def main():
    results = []
    for index, (name, before, after, test) in enumerate(FAULTS):
        with tempfile.TemporaryDirectory(prefix="depth_channel_fault_") as tmp:
            work = Path(tmp)
            for folder in ("src", "tests"):
                shutil.copytree(
                    REPO / folder,
                    work / folder,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            path = work / SOURCE
            text = path.read_text()
            if text.count(before) != 1:
                raise ValueError(f"fault target not unique: {name}")
            path.write_text(text.replace(before, after))
            result = subprocess.run(
                [
                    str(REPO / ".venv/bin/python"),
                    "-m",
                    "pytest",
                    TEST + test,
                    "-q",
                    "-o",
                    "addopts=",
                ],
                cwd=work,
                env=dict(
                    os.environ,
                    PYTHONPATH=str(work / "src"),
                    PYTHONDONTWRITEBYTECODE="1",
                ),
                capture_output=True,
                text=True,
                timeout=120,
            )
            log = GEN_DIR / f"depth_channel_fault_{index:02d}.log"
            log.write_text(result.stdout + result.stderr)
            caught = (
                result.returncode == 1
                and "FAILED" in result.stdout
                and "ERROR collecting" not in result.stdout
            )
            results.append(
                dict(
                    fault=name,
                    test=TEST + test,
                    caught=caught,
                    log_sha256=digest(log),
                )
            )
            print(name, "caught" if caught else "FAILED CHECK", flush=True)
    write_json(
        GEN_DIR / "depth_channel_fault_checks.json",
        dict(
            source=SOURCE,
            source_sha256=digest(REPO / SOURCE),
            test_sha256=digest(
                REPO / "tests/test_depth_channel_covariance.py"
            ),
            runner_sha256=digest(Path(__file__)),
            faults=results,
            all_caught=all(row["caught"] for row in results),
        ),
    )
    if not all(row["caught"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
