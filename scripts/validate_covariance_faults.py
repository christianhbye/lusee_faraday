"""Require test failures for named faults in disposable covariance copies."""

import common  # noqa: F401
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from common import GEN_DIR, REPO
from paper_common import digest, write_json

CHANNEL = "src/lusee_faraday/channelization.py"
SPIN = "src/lusee_faraday/spin_covariance.py"
VIS = "src/lusee_faraday/visibility_covariance.py"
FREQ = "src/lusee_faraday/frequency_covariance.py"
CT = "tests/test_channel_covariance.py::"
ST = "tests/test_spin_covariance.py::"
VT = "tests/test_visibility_covariance.py::"
FT = "tests/test_frequency_covariance.py::"

FAULTS = [
    (
        "interpolation weights halved",
        FREQ,
        "values = np.column_stack([1 - fraction, fraction]).ravel()",
        "values = 0.5 * np.column_stack([1 - fraction, fraction]).ravel()",
        FT
        + "test_interpolation_is_exact_for_affine_process_and_preserves_gain",
    ),
    (
        "compressed cross-frequency tiles omitted",
        FREQ,
        "value = np.asarray(kernel(nodes[si], nodes[sj]))",
        "if i != j:\n                continue\n            "
        "value = np.asarray(kernel(nodes[si], nodes[sj]))",
        FT
        + "test_compressed_covariance_refines_to_direct_nonstationary_modes",
    ),
    (
        "Gaussian draw variance halved",
        FREQ,
        "np.sqrt(np.maximum(eigen, 0)) * noise",
        "np.sqrt(0.5 * np.maximum(eigen, 0)) * noise",
        FT + "test_gaussian_draw_reproducibility_covariance_and_psd_rejection",
    ),
    (
        "complex power map silently accepted",
        FREQ,
        "or not np.isrealobj(w)",
        "or False",
        FT + "test_invalid_compression_coordinates_and_blocks",
    ),
    (
        "unnormalized channel rows",
        CHANNEL,
        "operator[rows, np.flatnonzero(sel)] = weights",
        "operator[rows, np.flatnonzero(sel)] = 2 * weights",
        CT + "test_channel_operator_matches_direct_bin_responses",
    ),
    (
        "zoom fold replaced by ideal response",
        CHANNEL,
        "weights = zoom_weights(offsets).T",
        "weights = ideal_zoom_weights(offsets).T",
        CT + "test_zoom_operator_retains_nyquist_fold",
    ),
    (
        "covariance loses conjugation",
        CHANNEL,
        "operator @ covariance @ operator.conj().T",
        "operator @ covariance @ operator.T",
        CT + "test_complex_moments_match_independent_finite_sample_ensemble",
    ),
    (
        "off-diagonal frequency blocks omitted",
        CHANNEL,
        "values = kernel(fine[si], fine[sj])",
        "if i != j:\n                continue\n            "
        "values = kernel(fine[si], fine[sj])",
        CT + "test_block_projection_includes_broad_nonstationary_covariance",
    ),
    (
        "noise bandwidth units lost",
        CHANNEL,
        "operator @ operator.T / fine_step_hz",
        "operator @ operator.T",
        CT + "test_white_noise_overlap_and_effective_bandwidth",
    ),
    (
        "unknown channel accepted",
        CHANNEL,
        'if kind not in ("parent", "zoom", "ideal_zoom"):',
        "if False:",
        CT + "test_invalid_channel_covariance_coordinates_and_shapes",
    ),
    (
        "spin Jacobi indices swapped",
        SPIN,
        "eval_jacobi(ell - 2, 0, 4, x)",
        "eval_jacobi(ell - 2, 4, 0, x)",
        ST + "test_wigner_rows_match_independent_factorial_sum",
    ),
    (
        "spin inverse missing two pi",
        SPIN,
        "plus=2 * np.pi * (plus * self.weights)",
        "plus=np.pi * (plus * self.weights)",
        ST + "test_spin_roundtrip_including_eb_relation",
    ),
    (
        "relation deterministic phase halved",
        SPIN,
        "phase_j = np.exp(2j * screen.mean_depth * (x + y))",
        "phase_j = np.exp(1j * screen.mean_depth * (x + y))",
        ST
        + "test_uniform_deterministic_rotation_mixes_eb_without_losing_power",
    ),
    (
        "Gaussian phase cross term halved",
        SPIN,
        "cross = 4 * screen.sigma_depth**2 * x * y",
        "cross = 2 * screen.sigma_depth**2 * x * y",
        ST + "test_screen_phase_kernels_match_direct_gaussian_ensemble",
    ),
    (
        "speckle relation omitted",
        SPIN,
        'xi["minus"] * phase["relation"]',
        '0 * xi["minus"] * phase["relation"]',
        ST + "test_rotated_power_split_conservation_and_spectral_positivity",
    ),
    (
        "coherence phase factor halved",
        SPIN,
        "strength = 4 * (self.sigma_depth * fraction * x) ** 2",
        "strength = 2 * (self.sigma_depth * fraction * x) ** 2",
        ST + "test_strong_screen_recovers_scalar_low_ell_floor_and_coherence",
    ),
    (
        "inadmissible angular exponent accepted",
        SPIN,
        "or not 0 < self.alpha <= 2",
        "or not 0 < self.alpha <= 3",
        ST + "test_invalid_spin_inputs_are_rejected",
    ),
    (
        "harmonic relation parity omitted",
        VIS,
        "relation *= (-1.0) ** m[:, None, None]",
        "relation *= 1.0",
        VT + "test_pair_covariance_and_relation_match_harmonic_ensemble",
    ),
    (
        "cross-frequency helicity conjugation lost",
        VIS,
        '"pclm,...lcd,qdlm->...mpq", b, h.conj(), d.conj(), optimize=True',
        '"pclm,...lcd,qdlm->...mpq", b, h, d.conj(), optimize=True',
        VT + "test_cross_frequency_rotation_and_different_pair_kernels",
    ),
    (
        "real readout drops relation",
        VIS,
        "a @ real_covariance(c, j) @ b.T",
        "a @ real_covariance(c, 0 * j) @ b.T",
        VT + "test_real_readout_keeps_relation_and_batch_dimensions",
    ),
    (
        "effective kernel conjugate parity omitted",
        VIS,
        "beam[:, flip, :, ::-1].conj() * parity",
        "beam[:, flip, :, ::-1].conj() + 0 * parity",
        VT + "test_real_kernels_match_pair_readout_on_"
        "independent_harmonic_samples",
    ),
    (
        "precontracted angular normalization halved",
        VIS,
        'np.einsum("lpq,ln,n->npq", response[k], d, transform.weights)',
        'np.einsum("lpq,ln,n->npq", 0.5 * response[k], d, transform.weights)',
        VT
        + "test_precontracted_angular_weights_match_full_spin_pair_time_chain",
    ),
    (
        "blackbody normalization doubled",
        VIS,
        "scale = normalization if kelvin else 1.0",
        "scale = 2 * normalization if kelvin else 1.0",
        VT + "test_loaded_operator_matches_direct_instrument_and_blackbody",
    ),
    (
        "unknown readout accepted",
        VIS,
        'if kind not in ("stokes", "science"):',
        "if False:",
        VT + "test_invalid_covariance_readout_requests",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=GEN_DIR / "covariance_fault_checks.json"
    )
    args = parser.parse_args()
    results = []
    for i, (name, source, before, after, test) in enumerate(FAULTS):
        with tempfile.TemporaryDirectory(prefix="covariance_fault_") as tmp:
            root = Path(tmp)
            for folder in ("src", "tests"):
                shutil.copytree(
                    REPO / folder,
                    root / folder,
                    ignore=shutil.ignore_patterns("__pycache__"),
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
            result = subprocess.run(
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
            log = GEN_DIR / f"covariance_fault_{i:02d}.log"
            log.write_text(result.stdout + result.stderr)
            caught = (
                result.returncode == 1
                and "FAILED" in result.stdout
                and "ERROR collecting" not in result.stdout
            )
            results.append(
                dict(
                    fault=name,
                    test=test,
                    source=source,
                    caught=caught,
                    source_sha256=digest(REPO / source),
                    log_sha256=digest(log),
                    returncode=result.returncode,
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
            all_caught=all(row["caught"] for row in results),
            scope="Disposable mutated copies; working source was not changed.",
        ),
    )
    if not all(row["caught"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
