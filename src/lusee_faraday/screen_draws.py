"""Drawn Gaussian screens and a direct angular sum of the off/on readouts.

Illustration and cross-check of ``gaussian_screen``; it shares none of that
module's separation-integral code. Each readout is summed pixel by pixel from
the Stokes brightness, ``P_on = P_0 exp(+2 i lambda^2 phi)`` (COSMO), so the
result does not use the Jones-branch algebra of the analytic derivation.
"""

from dataclasses import dataclass

import numpy as np

from .conventions import lambda_squared
from .gaussian_screen import READOUTS


@dataclass(frozen=True)
class ScreenGrid:
    """Square periodic tangent-plane grid, centred on zenith."""

    n_pix: int = 4096
    width_deg: float = 8.0

    def __post_init__(self):
        if self.n_pix < 2 or self.n_pix % 2:
            raise ValueError("n_pix must be an even integer >= 2")
        if not self.width_deg > 0:
            raise ValueError("width_deg must be positive")

    @property
    def pixel_rad(self):
        return np.deg2rad(self.width_deg) / self.n_pix

    @property
    def axis_rad(self):
        """Pixel-centre coordinates in radians, zero on a pixel centre."""
        return (np.arange(self.n_pix) - self.n_pix // 2) * self.pixel_rad


def draw_depth(model, grid, seed):
    """One stationary Gaussian depth map phi(theta) in rad m^-2.

    Spectral construction on the periodic grid, with the angular spectrum
    S(k) = 2 pi sigma^2 ell^2 exp(-ell^2 k^2 / 2) for the covariance
    sigma^2 exp(-h^2 / 2 ell^2), ell = correlation_pc / distance_pc in radians.
    """
    if not np.isfinite(model.correlation_pc):
        # one random constant over the whole plane
        offset = np.random.default_rng(seed).standard_normal()
        return np.full(
            (grid.n_pix,) * 2, model.mean_depth + model.sigma_depth * offset
        )
    ell = model.correlation_pc / model.distance_pc
    k = 2 * np.pi * np.fft.fftfreq(grid.n_pix, d=grid.pixel_rad)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    spectrum = (
        2 * np.pi * model.sigma_depth**2 * ell**2 * np.exp(-0.5 * ell**2 * k2)
    )
    white = np.random.default_rng(seed).standard_normal((grid.n_pix,) * 2)
    field = np.fft.ifft2(np.sqrt(spectrum) * np.fft.fft2(white))
    return model.mean_depth + field.real / grid.pixel_rad


def _weights(model, grid):
    """B dOmega on the grid (normalized beam) and the source profile."""
    axis = grid.axis_rad
    r2 = axis[:, None] ** 2 + axis[None, :] ** 2
    s_beam = np.deg2rad(model.beam_sigma_deg)
    s_src = np.deg2rad(model.source_sigma_deg)
    beam = np.exp(-0.5 * r2 / s_beam**2) / (2 * np.pi * s_beam**2)
    return beam * grid.pixel_rad**2, np.exp(-0.5 * r2 / s_src**2)


def direct_readouts(
    depth, freq_mhz, model, grid, names=("Q", "U", "xx", "yy", "xy")
):
    """Faraday-off and -on readouts of one depth map, by direct pixel sum.

    Returns ``{name: (off, on)}`` with each entry a vector over ``freq_mhz``,
    in K. ``depth`` is an (n_pix, n_pix) map aligned with ``grid``.
    """
    freq = np.atleast_1d(np.asarray(freq_mhz, float))
    beam, source = _weights(model, grid)
    f = (freq / model.ref_mhz) ** model.beta
    chi = np.exp(2j * np.deg2rad(model.angle_deg))
    i_sum = model.i_ref_k * beam.ravel() @ source.ravel()
    weight = (beam * source).ravel()
    flat = depth.ravel()
    x = lambda_squared(freq)
    off = np.empty(len(freq), complex)  # beam-weighted Q + iU, unrotated
    on = np.empty(len(freq), complex)
    for n, xn in enumerate(x):
        off[n] = model.p_ref_k * f[n] * chi * weight.sum()
        on[n] = model.p_ref_k * f[n] * chi * (weight @ np.exp(2j * xn * flat))
    intensity = i_sum * f
    out = {}
    for name in names:
        ki, kq, ku = READOUTS[name]
        out[name] = tuple(
            ki * intensity + kq * p.real + ku * p.imag for p in (off, on)
        )
    return out


@dataclass(frozen=True)
class FourierScreen:
    """Pinned real periodic Fourier field on a physical pc square.

    Modes use exp(2 pi i (nx*x+ny*y)/L); coefficients have units rad/m^2.
    ``coefficients`` is ordered from -m to m on each axis. Its Hermitian
    symmetry defines one real continuous field, independent of quadrature.
    """

    width_pc: float
    coefficients: np.ndarray
    mean_depth: float
    cutoff_per_pc: float

    def __post_init__(self):
        c = np.array(self.coefficients, dtype=complex, copy=True)
        if (
            c.ndim != 2
            or c.shape[0] != c.shape[1]
            or c.shape[0] % 2 != 1
            or not np.all(np.isfinite(c))
        ):
            raise ValueError("coefficients must be a finite odd square")
        if not np.isfinite(self.width_pc) or self.width_pc <= 0:
            raise ValueError("width_pc must be positive and finite")
        if not np.allclose(c, c[::-1, ::-1].conj(), atol=1e-15, rtol=1e-13):
            raise ValueError("coefficients must be Hermitian")
        c.flags.writeable = False
        object.__setattr__(self, "coefficients", c)

    @classmethod
    def draw(cls, model, width_pc, cutoff_per_pc, seed):
        """Gaussian Fourier coefficients with E|c_k|^2 = S(k)/L^2.

        Finite periodic covariance is sum S(k) cos(k.h)/L^2. It is
        explicitly checked against the continuum covariance by the driver.
        """
        rng = np.random.default_rng(seed)
        ell = model.correlation_pc
        if not np.isfinite(ell) or model.sigma_depth == 0:
            c = np.array([[model.sigma_depth * rng.standard_normal()]])
            return cls(width_pc, c, model.mean_depth, cutoff_per_pc)
        m = int(np.ceil(cutoff_per_pc * width_pc / (2 * np.pi)))
        mode = np.arange(-m, m + 1)
        k2 = (2 * np.pi / width_pc) ** 2 * (
            mode[:, None] ** 2 + mode[None, :] ** 2
        )
        spectrum = (
            2
            * np.pi
            * model.sigma_depth**2
            * ell**2
            * np.exp(-0.5 * ell**2 * k2)
        )
        z = (
            rng.standard_normal(k2.shape) + 1j * rng.standard_normal(k2.shape)
        ) / np.sqrt(2)
        z = (z + z[::-1, ::-1].conj()) / np.sqrt(2)
        c = z * np.sqrt(spectrum) / width_pc
        c[k2 > cutoff_per_pc**2] = 0
        return cls(width_pc, c, model.mean_depth, cutoff_per_pc)

    def truncate(self, cutoff_per_pc):
        """Lower the physical cutoff without redrawing retained modes."""
        if not 0 < cutoff_per_pc <= self.cutoff_per_pc:
            raise ValueError("cutoff must lie within the stored spectrum")
        m = self.coefficients.shape[0] // 2
        mode = np.arange(-m, m + 1)
        k2 = (2 * np.pi / self.width_pc) ** 2 * (
            mode[:, None] ** 2 + mode[None, :] ** 2
        )
        c = self.coefficients.copy()
        c[k2 > cutoff_per_pc**2] = 0
        return FourierScreen(self.width_pc, c, self.mean_depth, cutoff_per_pc)

    def evaluate(self, n_pix):
        """Evaluate the same field on a centered, periodic Cartesian grid."""
        m = self.coefficients.shape[0] // 2
        if n_pix % 2 or n_pix <= 2 * m:
            raise ValueError("even quadrature size must resolve stored modes")
        mode = np.arange(-m, m + 1)
        spectrum = np.zeros((n_pix, n_pix), complex)
        spectrum[np.ix_(mode % n_pix, mode % n_pix)] = self.coefficients
        field = np.fft.fftshift(np.fft.ifft2(spectrum)).real * n_pix**2
        return self.mean_depth + field

    def save(self, path):
        """Persist physical modes separately from readout quadrature."""
        np.savez(
            path,
            width_pc=self.width_pc,
            coefficients=self.coefficients,
            mean_depth=self.mean_depth,
            cutoff_per_pc=self.cutoff_per_pc,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                float(data["width_pc"]),
                data["coefficients"],
                float(data["mean_depth"]),
                float(data["cutoff_per_pc"]),
            )


def spectral_covariance(model, width_pc, cutoff_per_pc, separation_pc):
    """Exact covariance of the discretized generator, along the x axis."""
    ell = model.correlation_pc
    h = np.atleast_1d(separation_pc)
    if not np.isfinite(ell):
        return np.full(h.shape, model.sigma_depth**2)
    m = int(np.ceil(cutoff_per_pc * width_pc / (2 * np.pi)))
    k = 2 * np.pi / width_pc * np.arange(-m, m + 1)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    power = (
        2
        * np.pi
        * model.sigma_depth**2
        * ell**2
        * np.exp(-0.5 * ell**2 * k2)
        / width_pc**2
    )
    power[k2 > cutoff_per_pc**2] = 0
    return np.cos(h[:, None] * k) @ power.sum(axis=1)
