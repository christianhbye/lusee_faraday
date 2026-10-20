"""Depth-mixed spin-2 moments with an explicitly bounded Hermite expansion.

For a shared Gaussian column Phi=mu+sigma G, expand the depth-averaged
phase in He_k(G)/sqrt(k!). Correlated unit Gaussians obey E[h_k h_j']
= delta_kj rho^k. Averaging emitting depths therefore precedes channel
projection without treating the depths as independent emitting layers.
"""

import numpy as np
from scipy.special import gammaln, xlogy
from scipy.stats import poisson

from .depth_depolarization import (
    DepthConvergenceError,
    DepthProfile,
    DepthQuadrature,
    depth_nodes,
)


def hermite_degree(
    max_lambda_squared, screen, profile, tail=1e-12, max_degree=4096
):
    """Degree whose omitted pointwise phase power is at most tail.

    The single-depth coefficient power is Poisson with mean
    (2 sigma f lambda²)^2. Jensen's inequality bounds a normalized mixed
    profile by its worst-depth tail, including the f=0 boundary.
    """
    if (
        not np.isfinite(max_lambda_squared)
        or max_lambda_squared < 0
        or not np.isfinite(tail)
        or not 0 < tail < 1
        or not isinstance(max_degree, int)
        or max_degree < 0
    ):
        raise ValueError("invalid Hermite bound settings")
    fraction = profile.fraction if profile.kind == "screen" else 1.0
    peak = (2 * screen.sigma_depth * max_lambda_squared * fraction) ** 2
    if not np.isfinite(peak) or poisson.sf(max_degree, peak) > tail:
        raise DepthConvergenceError("Hermite tail needs more than max_degree")
    low, high = -1, max_degree
    while high - low > 1:
        middle = (high + low) // 2
        if poisson.sf(middle, peak) <= tail:
            high = middle
        else:
            low = middle
    return high, float(poisson.sf(high, peak))


def _log_poisson_power(mean, k):
    """Avoid subtraction of O(k log k) terms near a high-order peak.

    For k>=16 the Stirling correction through k^-9 has next term below
    1.1e-16. The nearby-mean deviance uses log1p; lower orders use gammaln.
    """
    result = -mean + xlogy(k, mean) - gammaln(k + 1)
    high = k >= 16
    if np.any(high):
        n = k[high].astype(float)
        ratio = (mean - n) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            deviance = np.where(
                abs(ratio) < 0.5,
                n * (np.log1p(ratio) - ratio),
                n * (np.log(mean / n) - ratio),
            )
        correction = (
            1 / (12 * n)
            - 1 / (360 * n**3)
            + 1 / (1260 * n**5)
            - 1 / (1680 * n**7)
            + 1 / (1188 * n**9)
        )
        result[..., high] = deviance - np.log(2 * np.pi * n) / 2 - correction
    return result


def _coefficients(x, screen, profile, degree, order):
    nodes, weights = depth_nodes(profile, order)
    k = np.arange(degree + 1)
    phase_k = np.array([1, 1j, -1, -1j])[k % 4]
    result = np.empty((len(x), degree + 1), complex)
    # Bound temporary arrays, even when 10 MHz needs >1000 modes.
    chunk = max(1, 1_000_000 // (len(nodes) * (degree + 1)))
    for i in range(0, len(x), chunk):
        stop = min(i + chunk, len(x))
        a = x[i:stop, None] * nodes
        t = 2 * screen.sigma_depth * a
        log_magnitude = _log_poisson_power(t[..., None] ** 2, k) / 2
        values = np.exp(log_magnitude) * phase_k
        values *= np.exp(2j * screen.mean_depth * a)[..., None]
        result[i:stop] = np.einsum("fdk,d->fk", values, weights)
    return result


def depth_phase_coefficients(
    lambda_squared,
    screen,
    profile=DepthProfile(),
    numerical=DepthQuadrature(),
    tail=1e-12,
    max_degree=4096,
):
    """Converged normalized Hermite coefficients of the mixed phase.

    Output coefficients have shape (frequency, mode), including the mean
    at mode zero. Depth convergence is measured in coefficient L2 norm;
    no angular or frequency-coherence grid sets a physical amplitude.
    """
    x = np.atleast_1d(np.asarray(lambda_squared, float))
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x) & (x >= 0)):
        raise ValueError("lambda squared must be a nonnegative finite vector")
    degree, bound = hermite_degree(x.max(), screen, profile, tail, max_degree)
    if profile.kind == "screen":
        value = _coefficients(x, screen, profile, degree, 2)
        order, error = 1, 0.0
    else:
        scale = x.max() * max(screen.sigma_depth, abs(screen.mean_depth))
        order = max(numerical.order, int(np.ceil(2 * scale)))
        if 2 * order > numerical.max_order:
            raise DepthConvergenceError("depth phase scale needs refinement")
        previous = _coefficients(x, screen, profile, degree, order)
        error = np.inf
        while 2 * order <= numerical.max_order:
            order *= 2
            value = _coefficients(x, screen, profile, degree, order)
            change = np.sqrt(np.sum(abs(value - previous) ** 2, axis=1))
            size = np.sqrt(np.sum(abs(value) ** 2, axis=1))
            error = float(change.max())
            if np.all(change <= numerical.epsabs + numerical.epsrel * size):
                break
            previous = value
        else:
            raise DepthConvergenceError(
                "depth coefficient order did not converge"
            )
    if np.max(np.sum(abs(value) ** 2, axis=1)) > 1 + 1e-12:
        raise DepthConvergenceError("phase coefficient power exceeds unity")
    return dict(
        coefficients=value,
        degree=degree,
        tail_bound=bound,
        depth_order=order,
        depth_error=error,
    )


def phase_moments_from_coefficients(left, right, rho):
    """Raw/centered C and J from coefficient rows, all frequency pairs."""
    a, b, rho = np.asarray(left), np.asarray(right), np.atleast_1d(rho)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != b.shape[1]
        or rho.ndim != 1
        or not np.isrealobj(rho)
        or not np.all(np.isfinite(rho) & (abs(rho) <= 1))
    ):
        raise ValueError("invalid Hermite coefficient or correlation shape")
    powers = rho[:, None] ** np.arange(a.shape[1])
    return dict(
        mean_x=a[:, 0],
        mean_y=b[:, 0],
        second=np.einsum("ik,jk,nk->ijn", a, b.conj(), powers),
        nonconjugated_second=np.einsum("ik,jk,nk->ijn", a, b, powers),
        covariance=np.einsum(
            "ik,jk,nk->ijn", a[:, 1:], b[:, 1:].conj(), powers[:, 1:]
        ),
        relation=np.einsum("ik,jk,nk->ijn", a[:, 1:], b[:, 1:], powers[:, 1:]),
    )


def channel_phase_coefficients(fine_coefficients, operator, amplitude=1.0):
    """Apply the same power-response estimator to each phase mode."""
    a, w = np.asarray(fine_coefficients), np.asarray(operator)
    amplitude = np.asarray(amplitude)
    if (
        a.ndim != 2
        or w.ndim != 2
        or w.shape[1] != a.shape[0]
        or not np.isrealobj(w)
        or not np.all(np.isfinite(w))
        or amplitude.ndim > 1
        or (amplitude.ndim == 1 and amplitude.shape != (a.shape[0],))
        or not np.all(np.isfinite(amplitude))
    ):
        raise ValueError("channel coefficients and coordinates disagree")
    return w @ (a * amplitude[..., None])


class DepthChannelProjector:
    """Reuse the spin/instrument angular projector on each Hermite mode.

    Its mode contraction includes both phase moment types and every
    helicity/pair term. It allocates only modes-by-selected-readout weights
    and the requested channel covariance, never the fine-frequency square.
    """

    def __init__(self, angular_projector, screen, degree):
        if not isinstance(degree, int) or degree < 0:
            raise ValueError("Hermite degree must be nonnegative")
        self.degree = degree
        z = screen.exponent(angular_projector.theta)
        powers = np.exp(-np.arange(degree + 1)[:, None] * z)
        xi = np.stack(
            [angular_projector.xi["plus"]] * 2
            + [angular_projector.xi["minus"]] * 2
        )
        self.weights = np.einsum(
            "rn,knpq,kn->krpq",
            powers,
            angular_projector.weights,
            xi,
            optimize=True,
        )
        self.size = angular_projector.size
        self.tail_norm = np.einsum(
            "knpq,kn->pq", abs(angular_projector.weights), abs(xi)
        )

    def moments(self, coefficients):
        a = np.asarray(coefficients)
        if (
            a.ndim != 2
            or a.shape[1] != self.degree + 1
            or not np.all(np.isfinite(a))
        ):
            raise ValueError("channel coefficient dimensions disagree")

        def contract(selection):
            b, weights = a[:, selection], self.weights[:, selection]
            plus = [
                np.einsum(
                    "br,cr,rpq->bpcq", b, b.conj(), weights[k], optimize=True
                )
                for k in (0, 1)
            ]
            minus = [
                np.einsum("br,cr,rpq->bpcq", b, b, weights[k], optimize=True)
                for k in (2, 3)
            ]
            return plus[0].real + plus[1].imag + minus[0].real + minus[1].imag

        smooth, speckle = contract(slice(0, 1)), contract(slice(1, None))
        return dict(smooth=smooth, speckle=speckle, total=smooth + speckle)

    def omitted_covariance_bound(self, channel_tail_rms):
        """Absolute elementwise bound from Minkowski/Cauchy-Schwarz."""
        bound = np.atleast_1d(np.asarray(channel_tail_rms, float))
        if not np.all(np.isfinite(bound) & (bound >= 0)):
            raise ValueError("tail RMS bounds must be nonnegative")
        return np.einsum("b,c,pq->bpcq", bound, bound, self.tail_norm)
