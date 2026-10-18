"""Require receiver/radiometer test failures for named injected mistakes."""

import common  # noqa: F401
from fault_audit import run_faults

SOURCE = "src/lusee_faraday/receiver_noise.py"
TEST = "tests/test_receiver_noise.py::"
FAULTS = [
    (
        "current noise ignores parallel loading",
        "parallel = np.asarray(loading_matrix(za, zl)) @ za",
        "parallel = za",
        "test_loaded_voltage_current_noise_matches_direct_kirchhoff_samples",
    ),
    (
        "omit voltage/current correlation",
        "+ ei @ parallel.swapaxes(-1, -2).conj()",
        "+ 0 * ei @ parallel.swapaxes(-1, -2).conj()",
        "test_loaded_voltage_current_noise_matches_direct_kirchhoff_samples",
    ),
    (
        "voltage ASD treated as PSD",
        "e[..., :, None] ** 2 * np.eye(4)",
        "e[..., :, None] * np.eye(4)",
        "test_explicit_asds_and_missing_receiver_data",
    ),
    (
        "science imaginary product sign flipped",
        "imag[a, b], imag[b, a] = 0.5j, -0.5j",
        "imag[a, b], imag[b, a] = -0.5j, 0.5j",
        "test_quadratic_readouts_match_production_science_and_stokes",
    ),
    (
        "drop cross-product radiometer covariance",
        "return result.real / n[..., None, None]",
        "return result.real * np.eye(result.shape[-1]) / n[..., None, None]",
        "test_full_product_covariance_matches_exact_gaussian_quadrature",
    ),
    (
        "two-sided sample-count factor inserted",
        "return result.real / n[..., None, None]",
        "return result.real / (2 * n[..., None, None])",
        "test_diagonal_variances_match_luseepy_and_scale_with_integration",
    ),
    (
        "drop noise covariance between overlapping bins",
        "return value / (fine_step_hz * integration_s)",
        "return value * np.eye(w.shape[0])[:,None,:,None] "
        "/ (fine_step_hz * integration_s)",
        "test_channel_overlap_and_variable_system_power_"
        "match_voltage_monte_carlo",
    ),
    (
        "ignore frequency variation of system power",
        "fine = quadratic_radiometer_covariance(system_psd, readout)",
        "fine = quadratic_radiometer_covariance(np.asarray(system_psd)[0] "
        "if np.asarray(system_psd).ndim == 3 else system_psd, readout)",
        "test_channel_overlap_and_variable_system_power_"
        "match_voltage_monte_carlo",
    ),
    (
        "negative spectral power accepted",
        "if np.any(np.linalg.eigvalsh(s)[..., 0] < -1e-11 * scale):",
        "if False:",
        "test_invalid_system_readout_and_exposure_are_rejected",
    ),
]


if __name__ == "__main__":
    run_faults(
        [
            (name, SOURCE, before, after, TEST + test)
            for name, before, after, test in FAULTS
        ],
        "receiver_noise",
        __file__,
    )
