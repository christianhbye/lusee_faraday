"""Isotropic spin-2 correlations, Gaussian rotation and helicity spectra.

The COSMO field P=Q+iU carries the P_PLUS harmonic coefficients a_+=E+iB;
its conjugate carries a_-=E-iB in P_MINUS. E/B signs are defined here by
these identities. Transform normalization follows the correlation method
of Challinor & Lewis (2005), Eqs. 64–65, with this explicit basis choice.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import eval_jacobi, roots_legendre

from .conventions import lambda_squared


def wigner_spin2(theta_rad, lmax):
    """Return d^ell_(2,2) and d^ell_(2,-2), with rows ell=0..lmax.

    The Jacobi expressions avoid building complete Wigner matrices. Rows
    ell<2 vanish; the ell=2 limits are cos(theta/2)^4 and sin(theta/2)^4.
    """
    theta = np.atleast_1d(np.asarray(theta_rad, float))
    if (
        not isinstance(lmax, int)
        or lmax < 2
        or theta.ndim != 1
        or not np.all(np.isfinite(theta) & (theta >= 0) & (theta <= np.pi))
    ):
        raise ValueError("need lmax >= 2 and finite angles in [0, pi]")
    plus = np.zeros((lmax + 1, theta.size))
    minus = np.zeros_like(plus)
    x = np.cos(theta)
    for ell in range(2, lmax + 1):
        plus[ell] = np.cos(theta / 2) ** 4 * eval_jacobi(ell - 2, 0, 4, x)
        minus[ell] = np.sin(theta / 2) ** 4 * eval_jacobi(ell - 2, 4, 0, x)
    return plus, minus


class SpinTransform:
    """Reusable spin transform on an explicitly weighted angular grid.

    Weights integrate d(cos theta) with positive orientation, including
    sin(theta) for a theta-grid quadrature. Reuse this object across
    frequency pairs; it stores two (lmax+1)-by-ntheta matrices only.
    """

    def __init__(self, theta_rad, weights, lmax):
        self.theta = np.array(theta_rad, dtype=float, copy=True)
        self.weights = np.array(weights, dtype=float, copy=True)
        if (
            self.theta.ndim != 1
            or self.theta.size < 2
            or self.weights.shape != self.theta.shape
            or not np.all(np.diff(self.theta) > 0)
            or not np.all(np.isfinite(self.weights) & (self.weights >= 0))
        ):
            raise ValueError("invalid angular quadrature")
        self.plus, self.minus = wigner_spin2(self.theta, lmax)
        self.lmax = lmax

    @classmethod
    def gauss_legendre(cls, lmax, nodes=None):
        """Exact for band-limited correlations; resolve Faraday cores apart."""
        nodes = lmax + 1 if nodes is None else nodes
        if not isinstance(nodes, int) or nodes < lmax + 1:
            raise ValueError("Gauss quadrature needs at least lmax+1 nodes")
        x, w = roots_legendre(nodes)
        return cls(np.arccos(x[::-1]), w[::-1], lmax)

    @classmethod
    def theta_grid(cls, theta_rad, lmax):
        """Trapezoidal angular rule; callers must refine coherence cores."""
        theta = np.asarray(theta_rad, float)
        if (
            theta.ndim != 1
            or theta.size < 2
            or theta[0] != 0
            or abs(theta[-1] - np.pi) > 1e-14
        ):
            raise ValueError("theta grid must cover [0, pi]")
        dt = np.diff(theta)
        weight = np.zeros_like(theta)
        weight[:-1] += dt / 2
        weight[1:] += dt / 2
        return cls(theta, weight * np.sin(theta), lmax)

    def correlations(self, plus, minus):
        """Synthesize xi_+ and xi_- from their complex angular spectra."""
        plus, minus = np.asarray(plus), np.asarray(minus)
        if (
            plus.shape != minus.shape
            or plus.ndim < 1
            or plus.shape[-1] > self.lmax + 1
            or plus.shape[-1] < 3
            or np.any(plus[..., :2] != 0)
            or np.any(minus[..., :2] != 0)
            or not np.all(np.isfinite(plus))
            or not np.all(np.isfinite(minus))
        ):
            raise ValueError("invalid spin spectra")
        count = plus.shape[-1]
        normalization = (2 * np.arange(count) + 1) / (4 * np.pi)
        return dict(
            plus=(plus * normalization) @ self.plus[:count],
            minus=(minus * normalization) @ self.minus[:count],
        )

    def spectra(self, plus, minus):
        """Transform geodesic-basis xi_+/xi_- into angular spectra."""
        plus, minus = np.asarray(plus), np.asarray(minus)
        if (
            plus.shape != minus.shape
            or plus.ndim < 1
            or plus.shape[-1] != self.theta.size
            or not np.all(np.isfinite(plus))
            or not np.all(np.isfinite(minus))
        ):
            raise ValueError("correlation dimensions disagree with quadrature")
        return dict(
            plus=2 * np.pi * (plus * self.weights) @ self.plus.T,
            minus=2 * np.pi * (minus * self.weights) @ self.minus.T,
        )


def eb_spectra(ee, bb, eb=0.0):
    """Physical same-frequency E/B spectra -> plus/minus spectra.

    plus=EE+BB and minus=EE-BB+2i EB with a_+=E+iB. EB is real;
    its sign follows this declared COSMO coefficient convention.
    """
    if not all(np.isrealobj(v) for v in (ee, bb, eb)):
        raise ValueError("same-frequency E/B spectra must be real")
    ee, bb, eb = np.broadcast_arrays(ee, bb, eb)
    if (
        ee.ndim < 1
        or ee.shape[-1] < 3
        or not np.all(np.isfinite(ee) & np.isfinite(bb) & np.isfinite(eb))
        or np.any(ee < 0)
        or np.any(bb < 0)
        or np.any(eb**2 > ee * bb + 1e-14 * np.maximum(ee, bb) ** 2)
        or np.any(ee[..., :2] != 0)
        or np.any(bb[..., :2] != 0)
        or np.any(eb[..., :2] != 0)
    ):
        raise ValueError(
            "E/B spectra are not positive semidefinite spin-2 data"
        )
    return dict(plus=ee + bb, minus=ee - bb + 2j * eb)


def helicity_covariance(plus, minus):
    """H_cd=E[a_c(nu) a_d(nu')*], ordered (P_MINUS, P_PLUS).

    Input spectra may describe a frequency pair. The other pair ordering
    must obey plus(y,x)=plus(x,y)* and minus(y,x)=minus(x,y).
    """
    plus, minus = np.broadcast_arrays(plus, minus)
    out = np.empty(plus.shape + (2, 2), complex)
    out[..., 0, 0] = plus.conj()
    out[..., 1, 1] = plus
    out[..., 0, 1] = minus.conj()
    out[..., 1, 0] = minus
    return out


@dataclass(frozen=True)
class AngularScreen:
    """Admissible Gaussian column covariance exp[-(chord/theta0)^alpha]."""

    sigma_depth: float = 0.02
    mean_depth: float = 0.01
    theta0_rad: float = 0.01
    alpha: float = 2.0

    def __post_init__(self):
        if (
            not all(
                np.isfinite(v)
                for v in (
                    self.sigma_depth,
                    self.mean_depth,
                    self.theta0_rad,
                    self.alpha,
                )
            )
            or self.sigma_depth < 0
            or self.theta0_rad <= 0
            or not 0 < self.alpha <= 2
        ):
            raise ValueError("invalid Gaussian angular screen")

    def exponent(self, theta_rad):
        theta = np.asarray(theta_rad, float)
        if not np.all(np.isfinite(theta) & (theta >= 0) & (theta <= np.pi)):
            raise ValueError("invalid angular separation")
        return (2 * np.sin(theta / 2) / self.theta0_rad) ** self.alpha

    def coherence_angle(self, freq_mhz, fraction=1.0):
        """Angle satisfying 2 lambda^4 f² D(theta)=1, or inf if absent."""
        freq = np.asarray(freq_mhz, float)
        if (
            not np.all(np.isfinite(freq) & (freq > 0))
            or not np.isfinite(fraction)
            or not 0 <= fraction <= 1
        ):
            raise ValueError("invalid frequency or screen fraction")
        x = lambda_squared(freq_mhz)
        strength = 4 * (self.sigma_depth * fraction * x) ** 2
        valid = strength > 1
        chord = np.full(x.shape, np.inf)
        chord[valid] = self.theta0_rad * (-np.log1p(-1 / strength[valid])) ** (
            1 / self.alpha
        )
        result = np.full(x.shape, np.inf)
        valid = chord <= 2
        result[valid] = 2 * np.arcsin(chord[valid] / 2)
        return result


def screen_phase_moments(x, y, theta_rad, screen, fraction=1.0):
    """Exact raw/centered phase kernels for one fractional Gaussian screen.

    Uses lambda² inputs and the COSMO positive rotation. A depth-mixed
    emission profile requires its own double-depth average, not rescaling
    these moments by a scalar depolarization factor.
    """
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if (
        not np.all(np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0))
        or not np.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("invalid wavelength or screen fraction")
    x, y = x[..., None] * fraction, y[..., None] * fraction
    exponent = screen.exponent(np.atleast_1d(theta_rad))
    if exponent.ndim != 1:
        raise ValueError("angular separations must be one dimensional")
    rho, one_minus_rho = np.exp(-exponent), -np.expm1(-exponent)
    cross = 4 * screen.sigma_depth**2 * x * y
    base = 2 * screen.sigma_depth**2 * (x * x + y * y)
    phase_c = np.exp(2j * screen.mean_depth * (x - y))
    phase_j = np.exp(2j * screen.mean_depth * (x + y))
    raw_c = phase_c * np.exp(
        -2 * screen.sigma_depth**2 * (x - y) ** 2 - cross * one_minus_rho
    )
    return dict(
        mean_x=np.exp(
            2j * screen.mean_depth * x - 2 * screen.sigma_depth**2 * x * x
        )[..., 0],
        mean_y=np.exp(
            2j * screen.mean_depth * y - 2 * screen.sigma_depth**2 * y * y
        )[..., 0],
        covariance=raw_c * (-np.expm1(-cross * rho)),
        relation=phase_j * np.exp(-base) * np.expm1(-cross * rho),
        second=raw_c,
        nonconjugated_second=phase_j * np.exp(-base - cross * rho),
    )


def rotated_spin_spectra(transform, intrinsic, x, y, screen, fraction=1.0):
    """Split rotated angular spectra into smooth and screen-induced terms."""
    xi = transform.correlations(intrinsic["plus"], intrinsic["minus"])
    phase = screen_phase_moments(x, y, transform.theta, screen, fraction)
    smooth_plus = phase["mean_x"] * phase["mean_y"].conjugate()
    smooth_minus = phase["mean_x"] * phase["mean_y"]
    return dict(
        smooth=transform.spectra(
            xi["plus"] * smooth_plus[..., None],
            xi["minus"] * smooth_minus[..., None],
        ),
        speckle=transform.spectra(
            xi["plus"] * phase["covariance"], xi["minus"] * phase["relation"]
        ),
        total=transform.spectra(
            xi["plus"] * phase["second"],
            xi["minus"] * phase["nonconjugated_second"],
        ),
    )
