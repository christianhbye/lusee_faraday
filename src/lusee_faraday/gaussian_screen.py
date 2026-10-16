"""Conditional flat-plane Gaussian-screen visibility moments.

See docs/specs/2026-10-15-gaussian-screen-step1.md. No angular pixel grid,
real-beam response, channel averaging, or stochastic sky is evaluated here.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.special import hyp1f1

from .config import BETA_QU
from .conventions import lambda_squared

# (k_I, k_Q, k_U) of V = int B [k_I I + k_Q Q + k_U U]. Ideal x/y dipoles
# give <E E^dagger> = [[I+Q, U], [U, I-Q]]/2 for V_Stokes = 0; Q and U are the
# beam-weighted Stokes parameters themselves.
READOUTS = {
    "xx": (0.5, 0.5, 0.0),
    "yy": (0.5, -0.5, 0.0),
    "xy": (0.0, 0.0, 0.5),
    "Q": (0.0, 1.0, 0.0),
    "U": (0.0, 0.0, 1.0),
}


@dataclass(frozen=True)
class ScreenModel:
    """Physical parameters; angular widths are Gaussian standard deviations."""

    distance_pc: float = 100.0
    correlation_pc: float = 0.5
    mean_depth: float = 0.01  # rad m^-2
    sigma_depth: float = 0.02  # rad m^-2
    beam_sigma_deg: float = 1.0
    source_sigma_deg: float = 1.5
    i_ref_k: float = 2.0
    p_ref_k: float = 1.0
    angle_deg: float = 20.0
    ref_mhz: float = 30.0
    beta: float = BETA_QU
    jones_angle_deg: float = 45.0
    jones_phase_deg: float = 60.0
    # (k_I, k_Q, k_U) overriding the Jones pair, e.g. READOUTS["xx"].
    readout: "tuple | None" = None

    def __post_init__(self):
        positive = (
            self.distance_pc,
            self.beam_sigma_deg,
            self.source_sigma_deg,
            self.ref_mhz,
        )
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError(
                "Distances, widths and reference must be positive"
            )
        if not self.correlation_pc > 0:
            raise ValueError("Correlation length must be positive (or inf)")
        if not np.isfinite(self.sigma_depth) or self.sigma_depth < 0:
            raise ValueError("Screen sigma must be finite and nonnegative")
        if not 0 <= self.p_ref_k <= self.i_ref_k:
            raise ValueError("Require I >= P >= 0")
        if self.readout is not None and (
            len(self.readout) != 3
            or not np.all(np.isfinite(np.asarray(self.readout, complex)))
        ):
            raise ValueError("readout must be three finite (k_I, k_Q, k_U)")
        for name in (
            "mean_depth",
            "i_ref_k",
            "p_ref_k",
            "angle_deg",
            "beta",
            "jones_angle_deg",
            "jones_phase_deg",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

    @property
    def eta(self):
        """Beam dilution of the source peak."""
        return self.source_sigma_deg**2 / (
            self.source_sigma_deg**2 + self.beam_sigma_deg**2
        )

    @property
    def weight_sigma_pc(self):
        """Width of the normalized beam times source on the screen."""
        angular = (self.source_sigma_deg**-2 + self.beam_sigma_deg**-2) ** -0.5
        return self.distance_pc * np.deg2rad(angular)

    @property
    def kappa(self):
        return 2 * (self.weight_sigma_pc / self.correlation_pc) ** 2

    @property
    def stokes_coefficients(self):
        """k_I, k_Q, k_U for p=(1,0), q=(cos psi, exp(i delta)sin psi).

        An explicit ``readout`` replaces the Jones pair.
        """
        if self.readout is not None:
            return np.asarray(self.readout, complex)
        psi = np.deg2rad(self.jones_angle_deg)
        delta = np.deg2rad(self.jones_phase_deg)
        return np.array(
            [
                0.5 * np.cos(psi),
                0.5 * np.cos(psi),
                0.5 * np.sin(psi) * np.exp(-1j * delta),
            ]
        )


@dataclass(frozen=True)
class SeparationQuadrature:
    """Numerical settings; none defines a physical correlation length."""

    epsabs: float = 2e-11
    epsrel: float = 2e-10
    upper_u: float = 40.0

    def __post_init__(self):
        if not all(
            np.isfinite(v) and v > 0
            for v in (self.epsabs, self.epsrel, self.upper_u)
        ):
            raise ValueError("Quadrature settings must be positive and finite")

    @property
    def centered_tail_bound(self):
        return 2 * np.exp(-self.upper_u)


def _centered_amplitude(u, x, y, sign_product, model):
    """Stable exp(-B) [exp(A rho)-1], with all large exponents <= 0."""
    sig2 = model.sigma_depth**2
    a = 4 * sig2 * sign_product * x * y
    b = 2 * sig2 * (x * x + y * y)
    one_minus_rho = -np.expm1(-model.kappa * u)
    rho = np.exp(-model.kappa * u)
    if a >= 0:
        return np.exp(-2 * sig2 * (x - y) ** 2 - a * one_minus_rho) * (
            -np.expm1(-a * rho)
        )
    return np.exp(-b) * np.expm1(a * rho)


def centered_branch(x, y, s, t, model, numerical=SeparationQuadrature()):
    """K_st = E[(Z_s-a_s)(Z_t-a_t)*], scalar x,y in m^2.

    Split at the coherence scale and rescale each interval before adaptive
    integration. Returns complex centered moment; absolute tail <= 2e^-umax.
    """
    if s not in (-1, 1) or t not in (-1, 1) or min(x, y) < 0:
        raise ValueError("Use branch signs +/-1 and nonnegative lambda^2")
    if model.sigma_depth == 0 or x == 0 or y == 0:
        return 0j
    phase = np.exp(2j * model.mean_depth * (s * x - t * y))
    if model.kappa == 0:
        return phase * _centered_amplitude(0.0, x, y, s * t, model)
    a_abs = 4 * model.sigma_depth**2 * x * y
    core = 1 / (1 + a_abs * model.kappa)
    scales = [core, 1 / model.kappa, 1.0]
    points = {0.0, numerical.upper_u}
    for scale in scales:
        for factor in (0.1, 1.0, 10.0, 100.0, 1000.0):
            point = scale * factor
            if 0 < point < numerical.upper_u:
                points.add(point)
    points = sorted(points)
    total = 0.0
    for left, right in zip(points[:-1], points[1:]):
        width = right - left

        def integrand(z):
            u = left + width * z
            return (
                width * np.exp(-u) * _centered_amplitude(u, x, y, s * t, model)
            )

        value, _ = quad(
            integrand,
            0.0,
            1.0,
            epsabs=numerical.epsabs / (len(points) - 1),
            epsrel=numerical.epsrel,
        )
        total += value
    return phase * total


def branch_second_special(x, y, s, t, model):
    """Uncentered branch second moment from the exact special-function form."""
    sig2 = model.sigma_depth**2
    a = 4 * sig2 * s * t * x * y
    b = 2 * sig2 * (x * x + y * y)
    phase = np.exp(2j * model.mean_depth * (s * x - t * y))
    if model.kappa == 0:
        amplitude = np.exp(-2 * sig2 * (s * x - t * y) ** 2)
    elif a >= 0:
        amplitude = np.exp(-2 * sig2 * (x - y) ** 2) * hyp1f1(
            1, 1 + 1 / model.kappa, -a
        )
    else:
        amplitude = np.exp(-b) * hyp1f1(
            1 / model.kappa, 1 + 1 / model.kappa, a
        )
    return phase * amplitude


def source_terms(freq_mhz, model):
    """Return lambda^2, v_I, b_(+,-), and branch means."""
    freq = np.atleast_1d(np.asarray(freq_mhz, dtype=float))
    if freq.ndim != 1 or not np.all(np.isfinite(freq) & (freq > 0)):
        raise ValueError("Frequencies must be a positive finite vector in MHz")
    x = lambda_squared(freq)
    f = (freq / model.ref_mhz) ** model.beta
    ki, kq, ku = model.stokes_coefficients
    signs = np.array([1, -1])
    b = np.array(
        [
            model.eta
            * model.p_ref_k
            * f
            * np.exp(2j * s * np.deg2rad(model.angle_deg))
            * (kq - 1j * s * ku)
            / 2
            for s in signs
        ]
    )
    means = np.exp(
        2j * signs[:, None] * model.mean_depth * x
        - 2 * model.sigma_depth**2 * x**2
    )
    return x, model.eta * model.i_ref_k * f * ki, b, means


def visibility_moments(
    freq_mhz, model, numerical=SeparationQuadrature(), diagonal_only=False
):
    """Paired off/on moments in K and K^2, including both branch statistics.

    diagonal_only returns vector diagonals for inexpensive parameter sweeps.
    Full matrices use O(Nfreq^2) storage; no pixel-pair arrays are allocated.
    """
    x, vi, b, a = source_terms(freq_mhz, model)
    off = vi + b.sum(axis=0)
    mean = vi + np.sum(b * a, axis=0)
    shape = (len(x),) if diagonal_only else (len(x), len(x))
    cov = np.zeros(shape, complex)
    relation = np.zeros(shape, complex)
    pairs = (
        ((i, i) for i in range(len(x)))
        if diagonal_only
        else ((i, j) for i in range(len(x)) for j in range(len(x)))
    )
    signs = [1, -1]
    for i, j in pairs:
        # K depends on signs through st and the separable mean phase.
        branch = np.array(
            [
                [
                    centered_branch(x[i], x[j], s, t, model, numerical)
                    for t in signs
                ]
                for s in signs
            ]
        )
        index = i if diagonal_only else (i, j)
        cov[index] = b[:, i] @ branch @ b[:, j].conj()
        relation[index] = b[:, i] @ branch[:, ::-1] @ b[:, j]
    delta_mean = mean - off
    if diagonal_only:
        on_second = cov + mean * mean.conj()
        delta_second = cov + delta_mean * delta_mean.conj()
        off_second = off * off.conj()
        on_off_second = mean * off.conj()
        on_nonconjugated = relation + mean**2
    else:
        on_second = cov + np.outer(mean, mean.conj())
        delta_second = cov + np.outer(delta_mean, delta_mean.conj())
        off_second = np.outer(off, off.conj())
        on_off_second = np.outer(mean, off.conj())
        on_nonconjugated = relation + np.outer(mean, mean)
    diag_delta = delta_second if diagonal_only else np.diag(delta_second)
    return dict(
        off=off,
        mean_on=mean,
        mean_delta=delta_mean,
        covariance=cov,
        relation=relation,
        second_on=on_second,
        second_off=off_second,
        second_on_off=on_off_second,
        second_on_nonconjugated=on_nonconjugated,
        second_delta=delta_second,
        expected_abs_delta2=diag_delta.real,
    )
