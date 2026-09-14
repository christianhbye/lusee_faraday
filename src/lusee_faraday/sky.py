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

import logging

import numpy as np

from .conventions import dual_block_phase

logger = logging.getLogger(__name__)

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
    logger.info(
        "component transform engines: %s (%s)",
        sky.engine,
        sky.engine_reason,
    )
    return np.asarray(sky.compute_alm(lmax=int(lmax)))[0]


AUDIT_REFERENCE = (
    "see the 2026-08-18 audit (commit 4b401c5): the diffuse-sky Faraday "
    "signature is HEALPix shot noise when the screen is unresolved"
)


def spectral_component_count(phi_min, phi_max, freqs_mhz):
    """Constant-depth components needed to resolve the phase in-band.

    The Faraday phase turns by ``2 * phi * lambda^2``, so across a band
    spanning ``d(lambda^2)`` two depths differing by more than
    ``pi / (2 d(lambda^2))`` are no longer coherent and need separate
    components.  This governs cost, not validity.
    """
    from .conventions import lambda_squared

    lam2 = lambda_squared(freqs_mhz)
    span = float(lam2.max() - lam2.min())
    width = float(phi_max) - float(phi_min)
    if span <= 0 or width <= 0:
        return 1
    return int(np.ceil(width / (np.pi / (2 * span))))


def nyquist_nside(rm_map, freq_mhz, percentile=99.9):
    """The nside at which the screen is resolved between adjacent pixels.

    This is the audit's criterion.  At 30 MHz the real Hutschenreuter map
    returns of order 3e5, i.e. ~1e12 pixels: the input does not determine
    the answer at any computable resolution, and no engine choice
    rescues it.
    """
    import healpy as hp

    from .conventions import lambda_squared

    rm = np.asarray(rm_map, dtype=float)
    nside0 = hp.npix2nside(rm.size)
    neighbours = hp.get_all_neighbours(nside0, np.arange(rm.size))
    valid = neighbours >= 0
    diffs = np.abs(rm[np.where(valid, neighbours, 0)] - rm[None, :])
    step = float(np.percentile(diffs[valid], percentile))
    lam2 = float(lambda_squared(freq_mhz).max())
    phase_step = 2.0 * step * lam2
    return nside0 * max(1.0, phase_step / np.pi)


def effective_pixel_count(weights):
    """The number of pixels a weighted coherent sum actually averages.

    ``N_eff = (sum w)^2 / sum w^2``.  Under random phases the
    normalised coherent pixel sum has

        sqrt(<|P|^2>) / sum_n |w_n|
            = sqrt(sum_n |w_n|^2) / sum_n |w_n|
            = N_eff^{-1/2},

    which is the random walk's floor with no free parameter in it.
    ``N_eff = N_pix`` only when the weights are equal, and that is the
    case the textbook ``N_pix^{-1/2}`` assumes.  The beam- and
    emissivity-weighted sky is not that case -- it is concentrated
    enough that at nside 512 the two differ by a factor 10 -- so a
    convergence plot drawn against ``N_pix^{-1/2}`` shows a shortfall
    that is an artefact of the guide rather than of the estimator.

    Takes ``|w|`` or ``|w|^2``: whichever quantity enters the sum
    linearly is the one to pass.
    """
    w = np.abs(np.asarray(weights, dtype=float))
    total = w.sum()
    if total <= 0.0:
        raise ValueError("effective_pixel_count needs positive weight")
    return float(total**2 / np.square(w).sum())


def _both_criteria(
    used_nside, needed_nside, freq_mhz, n_needed, max_components
):
    """The shared "here are both numbers" clause for both refusals.

    Whichever criterion trips the refusal, the message reports both,
    since bypassing the one that fired can still leave the other
    unmet -- the message is how the audit finding reaches a user who
    never read the audit.
    """
    return (
        f"nside={used_nside} used, nside~{needed_nside:.3g} needed at "
        f"{freq_mhz:g} MHz; {n_needed} spectral components needed "
        f"across the band (cap {max_components})"
    )


def audit_screen(
    rm_map, freqs_mhz, allow_pixelwise=False, max_components=4096
):
    """Compute, report and enforce both audit criteria for one screen.

    Every screen built from an RM map goes through here -- it is called
    by :meth:`FaradaySky.binned_screen`, which is the only constructor
    that turns a map of Faraday depths into components, and which
    :meth:`FaradaySky.from_rm_map` delegates to.  Putting the check on
    the *inner* constructor is deliberate: an earlier arrangement had it
    on ``from_rm_map`` alone, so calling ``binned_screen`` directly
    silently built a screen the map could not resolve, which is exactly
    the regime the 2026-08-18 audit exists to refuse.

    Both numbers are logged at INFO on every build, whether or not
    either criterion trips, so a successful build reports them too and
    not only a raising one.  Returns ``(needed_nside, n_needed)``.
    """
    import healpy as hp

    rm = np.asarray(rm_map, dtype=float)
    used_nside = hp.npix2nside(rm.size)
    freq_mhz = float(np.min(freqs_mhz))
    needed_nside = nyquist_nside(rm, freq_mhz)
    n_needed = spectral_component_count(
        float(rm.min()), float(rm.max()), freqs_mhz
    )
    both = _both_criteria(
        used_nside, needed_nside, freq_mhz, n_needed, max_components
    )
    logger.info("Faraday screen audit: %s", both)
    unresolved = needed_nside > used_nside
    too_many = n_needed > max_components
    if allow_pixelwise:
        if unresolved or too_many:
            logger.warning(
                "allow_pixelwise=True: building a screen that fails the "
                "audit (%s; %s)",
                both,
                AUDIT_REFERENCE,
            )
        return needed_nside, n_needed
    if unresolved:
        raise ValueError(
            f"Faraday screen is not resolved: {both}. The pixel "
            f"sum is a random walk, not a quadrature "
            f"({AUDIT_REFERENCE}). Pass allow_pixelwise=True to "
            "build it anyway."
        )
    if too_many:
        raise ValueError(
            f"Faraday screen needs too many spectral components: "
            f"{both} ({AUDIT_REFERENCE}). Pass allow_pixelwise=True "
            "to build it anyway or narrow the band."
        )
    return needed_nside, n_needed


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
        """Perfect depolarization: Stokes I only, no polarized blocks.

        The explicit zeroing of ``P_MINUS``/``P_PLUS`` is **defensive,
        not load-bearing**: croissant returns exactly ``0.0`` for the
        polarized alms of a ``Q = U = 0`` map, so deleting the line
        fails no test and changes no number.  It is kept because this
        constructor's entire contract is "no polarized blocks", and it
        should not silently depend on an external library's exact-zero
        behaviour to deliver that.
        """
        zeros = np.zeros_like(np.asarray(I, dtype=float))
        beta = np.array([[beta_i, beta_i, 0.0, 0.0]])
        ref = np.array([[ref_freq_i, ref_freq_i, 1.0, 1.0]])
        sky = cls.from_maps(I, zeros, zeros, 0.0, lmax, beta, ref, coord)
        sky.component_alms[:, 2:] = 0.0  # defensive; see the docstring
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
        freqs_mhz,
        allow_pixelwise=False,
        max_components=4096,
        beta=None,
        ref_freq_mhz=None,
        coord="galactic",
    ):
        """Partition a Faraday screen into bins of constant depth.

        Each component holds I, Q and U masked to its own bin, so the
        components partition the sky rather than overlapping it:
        summing them reproduces the input maps exactly.

        ``freqs_mhz`` is the observing band.  It is required rather than
        optional because both audit criteria are band-dependent and a
        screen cannot be judged without one: this is the constructor
        that turns an RM map into components, so it is where
        :func:`audit_screen` runs.  ``allow_pixelwise`` and
        ``max_components`` mean what they mean in :meth:`from_rm_map`,
        which is now a thin wrapper that only chooses ``dphi``.
        """
        dphi = float(dphi)
        if dphi <= 0:
            raise ValueError("dphi must be positive")
        audit_screen(rm_map, freqs_mhz, allow_pixelwise, max_components)
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

    @classmethod
    def from_rm_map(
        cls,
        I,  # noqa: E741
        Q,
        U,
        rm_map,
        freqs_mhz,
        lmax,
        allow_pixelwise=False,
        max_components=4096,
        beta=None,
        ref_freq_mhz=None,
        coord="galactic",
    ):
        """Build a binned screen at the band-matched bin width.

        A thin wrapper: it chooses ``dphi`` from the band's
        ``lambda^2`` span and hands everything to
        :meth:`binned_screen`, which is where :func:`audit_screen`
        reports both criteria and refuses an unresolved screen unless
        the caller has explicitly opted in.
        """
        from .conventions import lambda_squared

        rm = np.asarray(rm_map, dtype=float)
        span = float(np.ptp(lambda_squared(freqs_mhz)))
        dphi = np.pi / (2 * span) if span > 0 else (np.ptp(rm) or 1.0)
        return cls.binned_screen(
            I,
            Q,
            U,
            rm,
            dphi,
            lmax,
            freqs_mhz,
            allow_pixelwise=allow_pixelwise,
            max_components=max_components,
            beta=beta,
            ref_freq_mhz=ref_freq_mhz,
            coord=coord,
        )
