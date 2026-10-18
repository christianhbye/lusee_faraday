"""Conditional moments for emission at fractions of one Gaussian screen.

All depths share the same field: P = P0 int w(f) exp(2i f Phi lambda²) df.
This is a depth-profile bracket, not independent turbulent layers. Separation
is in pc, using ScreenModel's flat-plane covariance. No sky pixels are used.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import roots_legendre

from .conventions import lambda_squared
from .gaussian_screen import SeparationQuadrature, real_covariance


@dataclass(frozen=True)
class DepthProfile:
    """Unit-integral emission profile on the column fraction 0 <= f <= 1.

    ``gaussian`` is truncated to this interval. Its ``center`` and ``width``
    describe the underlying normal, not the truncated distribution's moments.
    """

    kind: str = "screen"
    fraction: float = 1.0
    center: float = 0.5
    width: float = 0.15

    def __post_init__(self):
        if self.kind not in ("screen", "slab", "gaussian"):
            raise ValueError("profile kind must be screen, slab or gaussian")
        if not all(
            np.isfinite(v) for v in (self.fraction, self.center, self.width)
        ):
            raise ValueError("profile parameters must be finite")
        if not 0 <= self.fraction <= 1 or not 0 <= self.center <= 1:
            raise ValueError("fraction and center must lie in [0, 1]")
        if self.width <= 0:
            raise ValueError("Gaussian profile width must be positive")


@dataclass(frozen=True)
class DepthQuadrature:
    """Depth-order doubling and an independently controlled separation rule.

    Orders describe integration accuracy, never a physical coherence scale.
    Unresolved depth ridges raise DepthConvergenceError rather than returning
    a grid-dependent amplitude. epsabs is in unit unrotated phase power.
    """

    order: int = 16
    max_order: int = 256
    epsabs: float = 2e-10
    epsrel: float = 2e-6
    separation: SeparationQuadrature = field(
        default_factory=SeparationQuadrature
    )

    def __post_init__(self):
        if (
            not isinstance(self.order, int)
            or not isinstance(self.max_order, int)
            or self.order < 2
            or self.max_order < 2 * self.order
        ):
            raise ValueError("require integer max_order >= 2*order >= 4")
        if not all(
            np.isfinite(v) and v > 0 for v in (self.epsabs, self.epsrel)
        ):
            raise ValueError("depth tolerances must be positive and finite")


class DepthConvergenceError(RuntimeError):
    """The requested depth or separation accuracy was not established."""


def depth_nodes(profile, order=32):
    """Return normalized quadrature nodes/weights for the named profile.

    This primitive does not certify convergence; phase_moments and
    depth_qu_moments compare successive orders before returning results.
    """
    if not isinstance(order, int) or order < 2:
        raise ValueError("depth order must be an integer >= 2")
    if profile.kind == "screen":
        return np.array([profile.fraction]), np.ones(1)
    z, w = roots_legendre(order)
    if profile.kind == "gaussian":
        # Integrate in Gaussian standard deviations. The finite support is
        # exact; splitting at the center resolves even very narrow profiles.
        edges = sorted({0.0, profile.center, 1.0})
        nodes, weights = [], []
        for left, right in zip(edges[:-1], edges[1:]):
            # Beyond 12 sigma the omitted probability is < 4e-33.
            left = max(left, profile.center - 12 * profile.width)
            right = min(right, profile.center + 12 * profile.width)
            if right <= left:
                continue
            f = left + (z + 1) * (right - left) / 2
            nodes.append(f)
            weights.append(
                w
                * (right - left)
                / 2
                * np.exp(-0.5 * ((f - profile.center) / profile.width) ** 2)
            )
        nodes, weights = np.concatenate(nodes), np.concatenate(weights)
        return nodes, weights / weights.sum()
    return (z + 1) / 2, w / 2


def _phase_mean(x, model, nodes, weights):
    return (
        np.exp(
            2j * model.mean_depth * x * nodes
            - 2 * (model.sigma_depth * x * nodes) ** 2
        )
        @ weights
    )


def _node_kernels(x, y, model, nodes, weights):
    a, b = x * nodes[:, None], y * nodes[None, :]
    cross = 4 * model.sigma_depth**2 * a * b
    base = 2 * model.sigma_depth**2 * (a**2 + b**2)
    difference = 2 * model.sigma_depth**2 * (a - b) ** 2
    phase_c = np.exp(2j * model.mean_depth * (a - b))
    phase_j = np.exp(2j * model.mean_depth * (a + b))
    weight = weights[:, None] * weights[None, :]

    def evaluate(rho, one_minus_rho):
        raw_c = np.exp(-difference - cross * one_minus_rho)
        raw_j = np.exp(-base - cross * rho)
        # Both differences are stable with no exp(+large) intermediate.
        cov = raw_c * (-np.expm1(-cross * rho))
        relation = np.exp(-base) * np.expm1(-cross * rho)
        return np.array(
            [
                np.sum(weight * phase_c * cov),
                np.sum(weight * phase_j * relation),
                np.sum(weight * phase_c * raw_c),
                np.sum(weight * phase_j * raw_j),
            ]
        )

    return evaluate


def _converged(evaluate, model, profile, max_x, numerical):
    if profile.kind == "screen":
        result, _ = evaluate(2)
        result.update(depth_order=1, depth_error=0.0)
        return result
    # Resolve the full-depth phase scale before trusting an order comparison;
    # otherwise both rules could miss an off-diagonal narrow ridge.
    scale = max_x * max(model.sigma_depth, abs(model.mean_depth))
    order = max(numerical.order, int(np.ceil(2 * scale)))
    if 2 * order > numerical.max_order:
        raise DepthConvergenceError(
            f"depth phase scale {scale:.3g} needs initial order {order} "
            f"and a refinement; max_order={numerical.max_order}"
        )
    _, old = evaluate(order)
    while 2 * order <= numerical.max_order:
        order *= 2
        result, new = evaluate(order)
        error = np.abs(new - old)
        if np.all(error <= numerical.epsabs + numerical.epsrel * abs(new)):
            result.update(depth_order=order, depth_error=float(error.max()))
            return result
        old = new
    raise DepthConvergenceError(
        f"depth integral did not converge by order {order}; "
        f"last maximum change={error.max():.6g} in off-power units"
    )


def phase_moments(
    x,
    y,
    separation_pc,
    model,
    profile=DepthProfile(),
    numerical=DepthQuadrature(),
):
    """Phase means, C=E[delta A_x delta A_y*], J=E[delta A_x delta A_y].

    x and y are scalar lambda² in m². Raw moments are integrated directly,
    including the nonconjugated relation; they are not a circular-Gaussian
    approximation. Separation is a finite nonnegative scalar or 1D pc array.
    """
    if not all(np.isfinite(v) and v >= 0 for v in (x, y)):
        raise ValueError("lambda squared must be finite and nonnegative")
    h = np.atleast_1d(np.asarray(separation_pc, float))
    if h.ndim != 1 or not np.all(np.isfinite(h) & (h >= 0)):
        raise ValueError("separation must be a finite nonnegative pc vector")
    z = 0.5 * (h / model.correlation_pc) ** 2

    def evaluate(order):
        nodes, weights = depth_nodes(profile, order)
        kernel = _node_kernels(x, y, model, nodes, weights)
        values = np.array([kernel(np.exp(-v), -np.expm1(-v)) for v in z]).T
        mx = _phase_mean(x, model, nodes, weights)
        my = _phase_mean(y, model, nodes, weights)
        result = dict(
            mean_x=mx,
            mean_y=my,
            covariance=values[0],
            relation=values[1],
            second=values[2],
            nonconjugated_second=values[3],
        )
        return result, np.r_[mx, my, values.ravel()]

    return _converged(evaluate, model, profile, max(x, y), numerical)


def _beam_kernels(x, y, model, nodes, weights, numerical):
    if model.sigma_depth == 0 or x == 0 or y == 0:
        return np.zeros(2, complex), 0.0
    kernel = _node_kernels(x, y, model, nodes, weights)
    if model.kappa == 0:
        return kernel(1.0, 0.0)[:2], 0.0
    # Gaussian source times beam gives radial separation weight exp(-u).
    # Subdivide at both the narrow phase core and the screen-correlation scale.
    core = 1 / (1 + 4 * model.sigma_depth**2 * x * y * model.kappa)
    points = {0.0, numerical.upper_u}
    for scale in (core, 1 / model.kappa, 1.0):
        for factor in (0.1, 1.0, 10.0, 100.0, 1000.0):
            point = scale * factor
            if 0 < point < numerical.upper_u:
                points.add(point)
    points = sorted(points)
    total = np.zeros(2, complex)
    error = 0.0
    for left, right in zip(points[:-1], points[1:]):
        width = right - left

        def integrand(z):
            u = left + width * z
            v = model.kappa * u
            return width * np.exp(-u) * kernel(np.exp(-v), -np.expm1(-v))[:2]

        value, err, info = quad_vec(
            integrand,
            0.0,
            1.0,
            epsabs=numerical.epsabs / (len(points) - 1),
            epsrel=numerical.epsrel,
            full_output=True,
        )
        if not info.success:
            raise DepthConvergenceError(info.message)
        total += value
        error += err
    return total, error + numerical.centered_tail_bound


def depth_qu_moments(
    freq_mhz,
    model,
    profile=DepthProfile(),
    numerical=DepthQuadrature(),
    diagonal_only=False,
):
    """Flat-plane beam-averaged Q/U moments in K and K², depth mixed.

    Matches gaussian_screen.qu_moments' deterministic Gaussian source/beam.
    With diagonal_only, complex_covariance/relation are frequency diagonals;
    full real covariance/second matrices are then omitted, never zero-filled.
    Depth convergence is assessed on dimensionless phase moments before
    applying brightness, so the error is independent of source normalization.
    """
    freq = np.atleast_1d(np.asarray(freq_mhz, float))
    if freq.ndim != 1 or not np.all(np.isfinite(freq) & (freq > 0)):
        raise ValueError("frequencies must be a finite positive MHz vector")
    x = lambda_squared(freq)

    def evaluate(order):
        nodes, weights = depth_nodes(profile, order)
        mean = np.array([_phase_mean(v, model, nodes, weights) for v in x])
        shape = (len(x),) if diagonal_only else (len(x), len(x))
        c, j = np.zeros(shape, complex), np.zeros(shape, complex)
        errors = np.zeros(shape)
        for i in range(len(x)):
            for k in ([i] if diagonal_only else range(i, len(x))):
                values, error = _beam_kernels(
                    x[i], x[k], model, nodes, weights, numerical.separation
                )
                index = i if diagonal_only else (i, k)
                c[index], j[index], errors[index] = *values, error
                if not diagonal_only:
                    c[k, i], j[k, i], errors[k, i] = (
                        values[0].conjugate(),
                        values[1],
                        error,
                    )
        result = dict(
            phase_mean=mean,
            phase_covariance=c,
            phase_relation=j,
            separation_error_estimate=float(errors.max()),
        )
        return result, np.r_[mean, c.ravel(), j.ravel()]

    result = _converged(evaluate, model, profile, x.max(), numerical)
    amp = model.eta * model.p_ref_k * (freq / model.ref_mhz) ** model.beta
    intrinsic = amp * np.exp(2j * np.deg2rad(model.angle_deg))
    mean_p = intrinsic * result.pop("phase_mean")
    c, j = result.pop("phase_covariance"), result.pop("phase_relation")
    if diagonal_only:
        c, j = abs(intrinsic) ** 2 * c, intrinsic**2 * j
        speckle = c.real
    else:
        c = np.outer(intrinsic, intrinsic.conj()) * c
        j = np.outer(intrinsic, intrinsic) * j
        speckle = np.diag(c).real
    smooth = abs(mean_p) ** 2
    result.update(
        mean=np.r_[mean_p.real, mean_p.imag],
        off=np.r_[intrinsic.real, intrinsic.imag],
        complex_covariance=c,
        complex_relation=j,
        smooth_power=smooth,
        speckle_power=speckle,
        total_power=smooth + speckle,
        off_power=amp**2,
    )
    if not diagonal_only:
        covariance = real_covariance(c, j)
        mean, off = result["mean"], result["off"]
        result.update(
            covariance=covariance,
            second=covariance + np.outer(mean, mean),
            delta_second=covariance + np.outer(mean - off, mean - off),
        )
    return result
