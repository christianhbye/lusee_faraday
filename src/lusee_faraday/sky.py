"""The Faraday sky, decomposed into spectrally separable components.

Faraday rotation is diagonal in croissant's harmonic dual, so a region of
constant Faraday depth contributes one frequency-independent component
alm plus a per-frequency, per-block coefficient.  A sky is therefore

    alm(nu) = sum_k coeff[k, nu, c] * component_alms[k, c]

which is exact -- not an approximation -- whenever ``phi_FD`` is
piecewise constant, and which makes the 16,384-channel fine grid cost
one einsum rather than 16,384 spherical transforms.

Input Stokes Q/U are healpy/COSMO.  They are handed to
``croissant.PolarizedSky`` with ``convention="COSMO"`` so croissant does
the IAU conversion; the stored component alms are therefore IAU duals,
which is what :func:`lusee_faraday.conventions.dual_block_phase` was
derived against.
"""

import numpy as np

from .conventions import dual_block_phase

STOKES_IQUV = ("I", "Q", "U", "V")


def _component_alm(I, Q, U, lmax, coord):  # noqa: E741
    """One region's IAU dual alms from COSMO Stokes maps."""
    import croissant as cro

    data = np.stack(
        [
            np.asarray(I, dtype=float),
            np.asarray(Q, dtype=float),
            np.asarray(U, dtype=float),
            np.zeros_like(np.asarray(I, dtype=float)),
        ]
    )[None]
    sky = cro.PolarizedSky(
        data,
        np.array([1.0]),  # placeholder: the spectrum lives in coeffs
        sampling="healpix",
        coord=coord,
        convention="COSMO",
    )
    return np.asarray(sky.compute_alm(lmax=int(lmax)))[0]


class FaradaySky:
    """A sky whose frequency dependence separates into components."""

    units = "K"
    convention = "IAU"
    stokes = STOKES_IQUV
    tangent_basis = "theta-phi"
    frequency_units = "MHz"

    def __init__(
        self,
        component_alms,
        phi_fd,
        beta=None,
        ref_freq_mhz=None,
        coord="galactic",
    ):
        # A copy, not a view: croissant's compute_alm returns a
        # jax-backed array that numpy exposes as read-only, and
        # i_only() below needs to zero out blocks in place.
        self.component_alms = np.array(component_alms)
        if self.component_alms.ndim != 4:
            raise ValueError(
                "component_alms must have shape (K, 4, L, 2L-1); got "
                f"{self.component_alms.shape}"
            )
        n_components = self.component_alms.shape[0]
        self.phi_fd = np.atleast_1d(np.asarray(phi_fd, dtype=float))
        if self.phi_fd.size != n_components:
            raise ValueError(
                f"phi_fd has {self.phi_fd.size} entries for "
                f"{n_components} components"
            )
        self.beta = (
            np.zeros((n_components, 4))
            if beta is None
            else np.asarray(beta, dtype=float)
        )
        self.ref_freq_mhz = (
            np.ones((n_components, 4))
            if ref_freq_mhz is None
            else np.asarray(ref_freq_mhz, dtype=float)
        )
        self.lmax = self.component_alms.shape[2] - 1
        self.coord = coord
        self.frame = coord

    @property
    def n_components(self):
        return self.component_alms.shape[0]

    def coeffs(self, freqs_mhz):
        """Per-frequency, per-block coefficients; shape ``(K, nfreq, 4)``."""
        freqs = np.atleast_1d(np.asarray(freqs_mhz, dtype=float))
        scale = (
            freqs[None, :, None] / self.ref_freq_mhz[:, None, :]
        ) ** self.beta[:, None, :]
        return scale * dual_block_phase(self.phi_fd, freqs)

    def polarized_alm_at_freq(self, target_freqs, lmax=None):
        """Sky alms at each target frequency; the luseepy sky protocol."""
        target_lmax = self.lmax if lmax is None else int(lmax)
        if target_lmax > self.lmax:
            raise ValueError(
                f"requested lmax={target_lmax} exceeds the sky's "
                f"{self.lmax}"
            )
        alms = self.component_alms
        if target_lmax < self.lmax:
            # m = 0 sits at index lmax, so trimming ell also trims m
            # symmetrically from both ends.
            drop = self.lmax - target_lmax
            hi = alms.shape[3] - drop
            alms = alms[:, :, : target_lmax + 1, drop:hi]
        return np.einsum(
            "kclm,kfc->fclm",
            alms,
            self.coeffs(target_freqs),
            optimize=True,
        )

    @classmethod
    def from_maps(
        cls,
        I,
        Q,
        U,
        phi_fd,
        lmax,
        beta=None,
        ref_freq_mhz=None,
        coord="galactic",
    ):
        """One region of constant Faraday depth."""
        alm = _component_alm(I, Q, U, lmax, coord)[None]
        return cls(alm, [float(phi_fd)], beta, ref_freq_mhz, coord)

    @classmethod
    def uniform_screen(
        cls,
        I,
        Q,
        U,
        phi_fd,
        lmax,
        beta_i=0.0,
        ref_freq_i=1.0,
        beta_qu=0.0,
        ref_freq_qu=1.0,
        coord="galactic",
    ):
        """A constant Faraday depth across the whole sky."""
        beta = np.array([[beta_i, beta_i, beta_qu, beta_qu]])
        ref = np.array([[ref_freq_i, ref_freq_i, ref_freq_qu, ref_freq_qu]])
        return cls.from_maps(I, Q, U, phi_fd, lmax, beta, ref, coord)

    @classmethod
    def i_only(
        cls,
        I,  # noqa: E741
        lmax,
        beta_i=0.0,
        ref_freq_i=1.0,
        coord="galactic",
    ):
        """Perfect depolarization: Stokes I only, no polarized blocks."""
        zeros = np.zeros_like(np.asarray(I, dtype=float))
        beta = np.array([[beta_i, beta_i, 0.0, 0.0]])
        ref = np.array([[ref_freq_i, ref_freq_i, 1.0, 1.0]])
        sky = cls.from_maps(I, zeros, zeros, 0.0, lmax, beta, ref, coord)
        sky.component_alms[:, 2:] = 0.0
        return sky

    @classmethod
    def point_source(
        cls,
        theta,
        phi,
        stokes,
        phi_fd,
        nside,
        lmax,
        beta=None,
        ref_freq_mhz=None,
        coord="galactic",
    ):
        """Discrete sources, each with its own Faraday depth.

        Parameters
        ----------
        theta, phi : (n_sources,) float
            Source directions in ``coord``, radians.
        stokes : (n_sources, 3) float
            Per-source I, Q, U in the healpy/COSMO convention.  The
            values land in a single HEALPix pixel each, so they carry
            the pixel's solid angle.
        phi_fd : (n_sources,) float
            Faraday depth per source, rad/m^2.
        """
        import healpy as hp

        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        stokes = np.atleast_2d(np.asarray(stokes, dtype=float))
        phi_fd = np.atleast_1d(np.asarray(phi_fd, dtype=float))
        n = theta.size
        if not (phi.size == n and stokes.shape == (n, 3) and phi_fd.size == n):
            raise ValueError(
                "theta, phi, stokes and phi_fd must describe the same "
                "number of sources"
            )
        npix = hp.nside2npix(int(nside))
        pix = hp.ang2pix(int(nside), theta, phi)
        alms = []
        for k in range(n):
            maps = np.zeros((3, npix))
            maps[:, pix[k]] = stokes[k]
            alms.append(_component_alm(*maps, lmax, coord))
        return cls(np.stack(alms), phi_fd, beta, ref_freq_mhz, coord)

    @classmethod
    def binned_screen(
        cls,
        I,  # noqa: E741
        Q,
        U,
        rm_map,
        dphi,
        lmax,
        beta=None,
        ref_freq_mhz=None,
        coord="galactic",
    ):
        """Partition a Faraday screen into bins of constant depth.

        Each component holds I, Q and U masked to its own bin, so the
        components partition the sky rather than overlapping it:
        summing them reproduces the input maps exactly.
        """
        dphi = float(dphi)
        if dphi <= 0:
            raise ValueError("dphi must be positive")
        I = np.asarray(I, dtype=float)  # noqa: E741
        Q = np.asarray(Q, dtype=float)
        U = np.asarray(U, dtype=float)
        rm = np.asarray(rm_map, dtype=float)
        index = np.floor(rm / dphi).astype(int)
        alms, depths = [], []
        for value in np.unique(index):
            mask = index == value
            alms.append(
                _component_alm(I * mask, Q * mask, U * mask, lmax, coord)
            )
            depths.append(float(rm[mask].mean()))
        return cls(np.stack(alms), depths, beta, ref_freq_mhz, coord)
