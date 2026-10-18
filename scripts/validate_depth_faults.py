"""Prove the new tests catch named faults in disposable source copies."""

import common  # noqa: F401
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from common import GEN_DIR, REPO
from paper_common import digest, write_json

DEPTH = "src/lusee_faraday/depth_depolarization.py"
HORIZON = "src/lusee_faraday/faraday_horizon.py"
DEPTH_TEST = "tests/test_depth_depolarization.py::"
HORIZON_TEST = "tests/test_faraday_horizon.py::"
DRIVER_TEST = "tests/test_paper_drivers.py::"

# Each entry pairs the claim with a specific implementation mistake.
FAULTS = [
    (
        "unnormalized slab",
        DEPTH,
        "return (z + 1) / 2, w / 2",
        "return (z + 1) / 2, w / 3",
        DEPTH_TEST + "test_profile_normalization_and_zero_rotation",
    ),
    (
        "ignored screen fraction",
        DEPTH,
        "np.array([profile.fraction])",
        "np.array([1.0])",
        DEPTH_TEST + "test_fraction_zero_and_scaled_foreground_screen",
    ),
    (
        "wrong relation phase",
        DEPTH,
        "(a + b))",
        "(a - b))",
        DEPTH_TEST + "test_phase_moments_against_direct_bivariate_depths",
    ),
    (
        "halved rotation",
        DEPTH,
        "2j * model.mean_depth * x * nodes",
        "1j * model.mean_depth * x * nodes",
        DEPTH_TEST + "test_deterministic_slab_and_truncated_gaussian_response",
    ),
    (
        "halved uniform-screen variance",
        DEPTH,
        "return kernel(1.0, 0.0)[:2], 0.0",
        "return 0.5 * kernel(1.0, 0.0)[:2], 0.0",
        DEPTH_TEST + "test_uniform_random_screen_slab_beam_moments",
    ),
    (
        "doubled speckle power",
        DEPTH,
        "speckle = np.diag(c).real",
        "speckle = 2 * np.diag(c).real",
        DEPTH_TEST + "test_joint_covariance_power_split_and_refinement",
    ),
    (
        "missing raw relation cross term",
        DEPTH,
        "raw_j = np.exp(-base - cross * rho)",
        "raw_j = np.exp(-base)",
        DEPTH_TEST + "test_raw_screen_relation_is_not_the_centered_relation",
    ),
    (
        "skipped depth convergence",
        DEPTH,
        'if profile.kind == "screen":\n        result, _ = evaluate(2)',
        "if True:\n        result, _ = evaluate(2)",
        DEPTH_TEST + "test_unresolved_depth_ridge_raises",
    ),
    (
        "unvalidated profile",
        DEPTH,
        'if self.kind not in ("screen", "slab", "gaussian"):',
        "if False:",
        DEPTH_TEST + "test_invalid_depth_parameters",
    ),
    (
        "wrong density units",
        HORIZON,
        "return ROTATION_PER_PC * ne * b",
        "return 10 * ROTATION_PER_PC * ne * b",
        HORIZON_TEST + "test_rate_units_and_sign",
    ),
    (
        "missing polarization phase factor",
        HORIZON,
        "denominator = 2 * rate * lambda_squared(freq)",
        "denominator = rate * lambda_squared(freq)",
        HORIZON_TEST
        + "test_horizon_reaches_requested_phase_and_scales_with_frequency",
    ),
    (
        "finite zero-field horizon",
        HORIZON,
        "np.full(denominator.shape, np.inf)",
        "np.zeros(denominator.shape)",
        HORIZON_TEST
        + "test_zero_field_has_no_finite_horizon_but_volume_is_finite",
    ),
    (
        "negative density allowed",
        HORIZON,
        "if not np.all(np.isfinite(ne) & (ne >= 0) & np.isfinite(b)):",
        "if False:",
        HORIZON_TEST + "test_invalid_horizon_inputs",
    ),
    (
        "stale manifest allowed",
        "scripts/paper_common.py",
        'if record["fingerprint"] != expected_key:',
        "if False:",
        DRIVER_TEST + "test_manifest_refuses_changed_config_and_source",
    ),
    (
        "corrupt covariance allowed",
        "scripts/depth_depolarization.py",
        'if digest(path) != check["array_sha256"]:',
        "if False:",
        DRIVER_TEST + "test_depth_driver_cache_rejects_corruption",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=GEN_DIR / "depth_fault_checks.json"
    )
    args = parser.parse_args()
    results = []
    for i, (name, source, before, after, test) in enumerate(FAULTS):
        with tempfile.TemporaryDirectory(prefix="depth_fault_") as folder:
            root = Path(folder)
            for subdir in ("src", "tests"):
                shutil.copytree(
                    REPO / subdir,
                    root / subdir,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            (root / "scripts").mkdir()
            for script in (
                "common.py",
                "paper_common.py",
                "depth_depolarization.py",
            ):
                shutil.copy2(
                    REPO / "scripts" / script, root / "scripts" / script
                )
            path = root / source
            original = path.read_text()
            if original.count(before) != 1:
                raise ValueError(f"fault target is not unique: {name}")
            path.write_text(original.replace(before, after))
            env = dict(
                os.environ,
                PYTHONPATH=str(root / "src"),
                PYTHONDONTWRITEBYTECODE="1",
            )
            run = subprocess.run(
                [
                    str(REPO / ".venv/bin/python"),
                    "-m",
                    "pytest",
                    test,
                    "-q",
                    "-o",
                    "addopts=",
                ],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            log = GEN_DIR / f"depth_fault_{i:02d}.log"
            log.write_text(run.stdout + run.stderr)
            caught = (
                run.returncode == 1
                and "FAILED" in run.stdout
                and "ERROR collecting" not in run.stdout
            )
            results.append(
                dict(
                    fault=name,
                    test=test,
                    source=source,
                    caught=caught,
                    returncode=run.returncode,
                    source_sha256=digest(REPO / source),
                )
            )
            print(
                f"{name}: {'caught' if caught else 'FAILED CHECK'}", flush=True
            )
    write_json(
        args.output,
        dict(
            runner_sha256=digest(Path(__file__)),
            faults=results,
            all_caught=all(r["caught"] for r in results),
            scope="Disposable mutated copies; working source was not changed.",
        ),
    )
    if not all(r["caught"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
