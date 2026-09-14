"""The reproduction arm: pixel-space four-port engine.

Not deprecated, and not transitional.  This module is a deliberately
*independent* quadrature of the same integral the production path
computes harmonically (``response`` + ``engine`` + ``instrument``), and
it exists for two reasons that do not expire:

1. **It is what makes the 2026-08-18 audit checkable.**  Claiming that a
   published diffuse-Faraday result is HEALPix shot noise is only
   meaningful if the result can still be produced.  The analyses it
   refutes live at the ``audit-2026-08-18`` tag and run on this module.

2. **It is the correctness evidence for the production path.**  The
   cross-implementation tests pin the two arms against each other at
   2.5e-16 (covariance assembly), 0.0e+00 (direction-space kernel),
   3.4e-14 (the ported pixel arm vs its pre-port artifact) and 3.054e-4
   (whole-sky engine agreement on the real sky).  Those numbers are the
   reason to believe the harmonic path is right, and they cease to exist
   the moment this module is "cleaned up" into calling the new one.

So do not shrink this module toward the production modules, however much
duplication it looks like: the duplication *is* the independence.  The one
exception already taken is ``load_response_fast``, a FITS-schema reader
whose second copy could only drift, never disagree informatively.

New analyses belong on the production path.  Import this module only from
the reproduction scripts, the cross-check scripts, and tests that compare
the two arms.

One caveat on that last rule: ``topo_rotation_matrix`` used to live here
and is now in :mod:`lusee_faraday.conventions`, because it defines the
response frame rather than either engine's quadrature.  It is re-exported
above so the reproduction scripts need no edit.

----

Four-port (luseepy new-engine) Faraday waterfall machinery.

This module reimplements the lusee_faraday analysis on top of the
luseepy four-port instrument response (`lusee.InstrumentResponse`),
producing all sixteen correlation products (4 autos + Re/Im of the 6
crosses) of the N, E, S, W monopole ports.

Conventions (pinned against the BGL_v16 response artifact and the
old_vs_new diagnostics):

- Response frame: theta = colatitude from zenith, phi measured from
  local East towards North (phi = 90 deg - astronomical azimuth),
  right-handed about +z.  We therefore build the topocentric frame with
  cartesian basis x = East, y = North, z = zenith, which is a *proper*
  rotation of the galactic frame (det = +1).
- Port order 0, 1, 2, 3 = N, E, S, W (header PORTNAMES).
- Pair-Stokes kernels P^I, P^Q, P^U, P^V follow the formalism in
  (e_theta, e_phi) basis under e^{+j omega t}; the sky coherency is
  B_E = (k_B eta0/lambda^2) [[I+Q, U-jV], [U+jV, I-Q]].
- Sky Q, U maps are taken in the healpy/COSMO convention, which in the
  (e_theta, e_phi) basis of the map graticule reads
  Q = <E_th^2> - <E_ph^2>, U = 2<E_th E_ph>.  This identification is
  validated against the croissant harmonic engine (which consumes
  IAU-convention maps, U_IAU = -U_COSMO) in scripts/validate_engine.py.
- Faraday rotation follows the paper: in the galactic map basis
  (Q + iU)_rot = (Q + iU) exp(+2 i phi_FD lambda^2).
"""

import numpy as np
from scipy.constants import Boltzmann, c, physical_constants

from .conventions import topo_rotation_matrix  # noqa: F401  (re-export)

ETA0 = physical_constants["characteristic impedance of vacuum"][0]
KB = Boltzmann

# Port pairs a <= b and the 16 packed real channels, matching
# lusee.Covariance.default_product_labels().
PORT_PAIRS = tuple((a, b) for a in range(4) for b in range(a, 4))
PRODUCT_LABELS = []
for _a, _b in PORT_PAIRS:
    if _a == _b:
        PRODUCT_LABELS.append(f"{_a}{_b}R")
    else:
        PRODUCT_LABELS.extend((f"{_a}{_b}R", f"{_a}{_b}I"))
PRODUCT_LABELS = tuple(PRODUCT_LABELS)

# Pseudo-dipoles from the port voltages (ports N, E, S, W = 0, 1, 2, 3)
Y_VEC = np.array([1.0, 0.0, -1.0, 0.0])  # Y = N - S
X_VEC = np.array([0.0, 1.0, 0.0, -1.0])  # X = E - W


def load_response_fast(path):
    """Load a v3 response artifact without re-running the slow validation.

    Kept as a name, not as an implementation: it is an alias for
    :func:`lusee_faraday.response.load_response`.  The two were separate
    near-verbatim copies while the harmonic path was being built beside
    this one, so that a change to either could not perturb the other's
    validation; with this module demoted to the validation arm that
    reason is gone, and two copies of a FITS-schema reader is exactly
    the kind of duplication that drifts silently.
    """
    from .response import load_response

    return load_response(path)


def sample_periodic_maps(values, theta_deg, phi_deg, theta_rad, phi_rad):
    """Vectorized bilinear sampling of maps with axes (..., theta, phi).

    The phi grid must include the 0/360 wraparound bin.  This is the
    array version of lusee.InstrumentResponse._sample_periodic_maps: the
    already-formed kernel is interpolated (never the fields).
    """
    theta_rad = np.asarray(theta_rad, dtype=np.float64)
    phi_rad = np.asarray(phi_rad, dtype=np.float64) % (2 * np.pi)
    tg = np.radians(np.asarray(theta_deg, dtype=np.float64))
    pg = np.radians(np.asarray(phi_deg, dtype=np.float64)[:-1])
    if theta_rad.min() < tg[0] - 1e-12 or theta_rad.max() > tg[-1] + 1e-12:
        raise ValueError("theta outside the stored response region")
    theta_rad = np.clip(theta_rad, tg[0], tg[-1])

    ti = np.searchsorted(tg, theta_rad, side="right")
    thi = np.clip(ti, 1, tg.size - 1)
    tlo = thi - 1
    ta = (theta_rad - tg[tlo]) / (tg[thi] - tg[tlo])

    pi_ = np.searchsorted(pg, phi_rad, side="right")
    plo = (pi_ - 1) % pg.size
    phid = pi_ % pg.size
    phi_lo = pg[plo]
    phi_hi = np.where(phid == 0, pg[phid] + 2 * np.pi, pg[phid])
    phi_eval = np.where(phi_rad >= phi_lo, phi_rad, phi_rad + 2 * np.pi)
    pa = (phi_eval - phi_lo) / (phi_hi - phi_lo)

    flat = values.reshape(values.shape[:-2] + (-1,))
    nphi = len(phi_deg)

    def gather(it, ip):
        return flat[..., it * nphi + ip]

    low = (1 - pa) * gather(tlo, plo) + pa * gather(tlo, phid)
    high = (1 - pa) * gather(thi, plo) + pa * gather(thi, phid)
    return (1 - ta) * low + ta * high


class FixedFreqKernel:
    """Pair-Stokes kernel of the four-port response at one native channel.

    Holds the ten (a <= b) complex pair-Stokes maps (10, 4=IQUV, Ntheta,
    Nphi), the antenna impedance, the receiver loading matrix M and the
    blackbody normalization at that frequency.  The beam is deliberately
    achromatic: everything is evaluated at this single native channel.
    """

    def __init__(self, resp, freq_mhz, receiver):
        freq = np.asarray(resp.freq)
        idx = int(np.argmin(np.abs(freq - freq_mhz)))
        if abs(freq[idx] - freq_mhz) > 1e-9:
            raise ValueError(
                f"{freq_mhz} MHz is not a native response channel."
            )
        self.freq_mhz = float(freq[idx])
        self.native_index = idx
        self.K = np.asarray(resp.all_pair_stokes_maps(freq_ndx=[idx]))[
            :, 0
        ]  # (10, 4, Ntheta, Nphi) complex, bare m^2 units
        self.theta_deg = np.asarray(resp.theta_deg)
        self.phi_deg = np.asarray(resp.phi_deg)
        self.lam = c / (self.freq_mhz * 1e6)
        self.prefac = ETA0 / self.lam**2
        self.ZA = np.asarray(resp.ZA)[idx]
        self.Rmoon = np.asarray(resp.Rmoon_native)[idx]
        self.Rloss = np.asarray(resp.Rloss_native)[idx]
        ZL = np.asarray(receiver.Z(np.array([self.freq_mhz])))[0]
        self.ZL = ZL
        # M = ZL (ZA + ZL)^{-1}, computed with a solve (right division)
        self.M = np.linalg.solve((self.ZA + ZL).T, ZL.T).T
        dissipative = 0.5 * (self.ZA + self.ZA.conj().T)
        self.blackbody = (
            4 * KB * (self.M @ dissipative @ self.M.conj().T)
        )  # V^2/Hz per K of blackbody enclosure

    def sample(self, theta_rad, phi_rad):
        """Kernel at arbitrary directions -> (10, 4, N) complex (m^2)."""
        return sample_periodic_maps(
            self.K, self.theta_deg, self.phi_deg, theta_rad, phi_rad
        )


class GalacticGrid:
    """Galactic HEALPix pixel directions and tangent bases."""

    def __init__(self, nside):
        import healpy as hp

        self.nside = nside
        self.npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(self.npix))
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        self.vec = np.stack([st * cp, st * sp, ct], axis=-1)
        self.e_theta = np.stack([ct * cp, ct * sp, -st], axis=-1)
        self.e_phi = np.stack([-sp, cp, np.zeros_like(sp)], axis=-1)
        self.pix_area = 4 * np.pi / self.npix


def transport(R, grid):
    """Transport galactic pixels into the response frame at one time.

    Returns (theta, phi, alpha, up): response-frame polar angles of every
    galactic pixel, the local polarization-basis rotation angle alpha
    (transported galactic e_theta = cos(a) e_theta + sin(a) e_phi in the
    response basis), and the above-horizon mask.
    """
    n_t = grid.vec @ R.T
    z = np.clip(n_t[:, 2], -1.0, 1.0)
    theta = np.arccos(z)
    phi = np.arctan2(n_t[:, 1], n_t[:, 0]) % (2 * np.pi)
    up = theta <= np.pi / 2 + 1e-12
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    e_theta_t = np.stack([ct * cp, ct * sp, -st], axis=-1)
    e_phi_t = np.stack([-sp, cp, np.zeros_like(sp)], axis=-1)
    tvec = grid.e_theta @ R.T
    alpha = np.arctan2(
        np.sum(tvec * e_phi_t, axis=-1), np.sum(tvec * e_theta_t, axis=-1)
    )
    return theta, phi, alpha, up


class SkyWaterfallSim:
    """Beam-weighted Faraday synthesis of the ten pair integrals.

    The sky is held constant across the (narrow) band at its value at
    the kernel frequency; only the Faraday screen phase
    exp(+/- 2 i phi_FD lambda^2) varies with frequency.  The pair
    integral for each frequency is evaluated exactly (per pixel) with a
    type-3 NUFFT over the Faraday-depth distribution.
    """

    def __init__(self, kernel, grid, I_map, Q_map, U_map, RM_map):
        self.kernel = kernel
        self.grid = grid
        self.I_map = np.asarray(I_map, dtype=np.float64)
        self.P_map = np.asarray(Q_map, dtype=np.float64) + 1j * np.asarray(
            U_map, dtype=np.float64
        )
        self.RM_map = np.asarray(RM_map, dtype=np.float64)

    def pair_integrals(self, R, lam2, faraday=True, eps=1e-10):
        """Ten complex pair integrals at one time -> (10, Nfreq).

        lam2 : array of lambda^2 values (m^2) at the target frequencies.
        With faraday=False the Faraday screen is switched off and the
        (frequency-independent) integrals are broadcast over lam2.
        """
        import finufft

        theta, phi, alpha, up = transport(R, self.grid)
        K = self.kernel.sample(theta[up], phi[up])  # (10, 4, Nup)
        scale = self.kernel.prefac * self.grid.pix_area
        S_I = K[:, 0] @ self.I_map[up]  # (10,)
        P_t = self.P_map[up] * np.exp(2j * alpha[up])
        a = 0.5 * (K[:, 1] - 1j * K[:, 2]) * P_t[None, :]
        b = 0.5 * (K[:, 1] + 1j * K[:, 2]) * np.conj(P_t)[None, :]
        lam2 = np.atleast_1d(np.asarray(lam2, dtype=np.float64))
        if faraday:
            x = np.concatenate([2 * self.RM_map[up], -2 * self.RM_map[up]])
            cvec = np.concatenate([a, b], axis=1)  # (10, 2 Nup)
            pol = finufft.nufft1d3(
                x, cvec, lam2, eps=eps, isign=1
            )  # (10, Nfreq)
        else:
            pol = np.broadcast_to(
                (a + b).sum(axis=1)[:, None], (10, lam2.size)
            ).copy()
        return scale * (S_I[:, None] + pol)


def assemble_covariance(pair_integrals, M):
    """(..., 10) open pair integrals -> loaded covariance (..., 4, 4).

    Applies K_sky = k_B * pairmatrix and C = M K M^H (T_moon = 0).
    """
    pv = np.asarray(pair_integrals)
    Kmat = np.zeros(pv.shape[:-1] + (4, 4), dtype=complex)
    for k, (a, b) in enumerate(PORT_PAIRS):
        Kmat[..., a, b] = pv[..., k]
        if a != b:
            Kmat[..., b, a] = np.conj(pv[..., k])
    Kmat *= KB
    C = np.einsum("ab,...bc,dc->...ad", M, Kmat, np.conj(M))
    return 0.5 * (C + np.conj(np.swapaxes(C, -1, -2)))


def pack_products(C):
    """Hermitian covariance (..., 4, 4) -> 16 real channels (..., 16)."""
    out = np.empty(C.shape[:-2] + (16,), dtype=np.float64)
    k = 0
    for a, b in PORT_PAIRS:
        if a == b:
            out[..., k] = C[..., a, b].real
            k += 1
        else:
            out[..., k] = C[..., a, b].real
            out[..., k + 1] = C[..., a, b].imag
            k += 2
    return out


def unpack_products(channels):
    """16 real channels (..., 16) -> Hermitian covariance (..., 4, 4)."""
    ch = np.asarray(channels)
    C = np.zeros(ch.shape[:-1] + (4, 4), dtype=complex)
    k = 0
    for a, b in PORT_PAIRS:
        if a == b:
            C[..., a, b] = ch[..., k]
            k += 1
        else:
            C[..., a, b] = ch[..., k] + 1j * ch[..., k + 1]
            C[..., b, a] = np.conj(C[..., a, b])
            k += 2
    return C


def polarimeter(C, x_vec=None, y_vec=None):
    """Ideal polarimeter from the port covariance (..., 4, 4).

    X = x_vec . v, Y = y_vec . v (defaults X = E - W, Y = N - S);
    XX = <|X|^2>, YY = <|Y|^2>, XY = <Y X*>;
    I = (XX+YY)/2, Q = (XX-YY)/2, U = Re XY, V = Im XY.
    Returns (..., 4) array with the pseudo-Stokes I, Q, U, V.
    Pass the vectors from `zenith_port_weights` for a polarimeter
    calibrated on an unpolarized zenith source.
    """
    xv = X_VEC if x_vec is None else np.asarray(x_vec)
    yv = Y_VEC if y_vec is None else np.asarray(y_vec)
    # X = sum_a x_a v_a with complex weights allowed:
    # <X X*> = sum_ab x_a conj(x_b) C_ab, <Y X*> = sum_ab y_a conj(x_b) C_ab
    XX = np.einsum("a,b,...ab->...", xv, np.conj(xv), C).real
    YY = np.einsum("a,b,...ab->...", yv, np.conj(yv), C).real
    XY = np.einsum("a,b,...ab->...", yv, np.conj(xv), C)
    return np.stack(
        [0.5 * (XX + YY), 0.5 * (XX - YY), XY.real, XY.imag], axis=-1
    )


def polarimeter_from_channels(channels, x_vec=None, y_vec=None):
    return polarimeter(unpack_products(channels), x_vec, y_vec)


def zenith_port_weights(kern, null_q=True):
    """Port weights that calibrate the polarimeter on a zenith source.

    Computes the covariance C0 of an unpolarized (Stokes I) source at
    exact zenith and returns (x_vec, y_vec, C0) with
    X = wE E - wW W, Y = wN N - wS S,  w_p = sqrt(g / C0_pp)
    (g = geometric mean of the four autos), so all four ports have
    equal calibrated auto response.  XX and YY still differ through
    the Re(EW) / Re(NS) cross-coupling terms; with null_q=True the X
    and Y pairs are rescaled by a common factor (s, 1/s) so that
    pseudo-Q of the zenith source vanishes exactly.
    """
    K0 = kern.sample(np.array([0.0]), np.array([0.0]))[:, :, 0]
    pair = kern.prefac * K0[:, 0]  # unpolarized: I kernel only
    C0 = assemble_covariance(pair, kern.M)
    autos = np.diagonal(C0).real
    g = np.exp(np.mean(np.log(autos)))
    w = np.sqrt(g / autos)
    x_vec = np.array([0.0, w[1], 0.0, -w[3]])
    y_vec = np.array([w[0], 0.0, -w[2], 0.0])
    if null_q:
        XX = np.einsum("a,b,ab->", x_vec, x_vec, C0).real
        YY = np.einsum("a,b,ab->", y_vec, y_vec, C0).real
        s = (YY / XX) ** 0.25
        x_vec = x_vec * s
        y_vec = y_vec / s
    return x_vec, y_vec, C0


def orthonormalize_xy(C0, x_vec, y_vec):
    """Loewdin-orthonormalize (X, Y) in the metric of a covariance C0.

    With X = sum_a x_a v_a, the pseudo-Stokes of a source with
    covariance C0 vanish in Q, U and V exactly iff conj(x) and
    conj(y) are C0-orthogonal with equal C0-norms (the polarimeter
    forms are XX = p^H C0 p and XY = q^H C0 p with p = conj(x),
    q = conj(y)).  The symmetric G^{-1/2} (Loewdin) transform
    achieves this while perturbing the input dipole vectors as little
    as possible; the pair is rescaled to preserve the geometric-mean
    zenith power.  Returns complex (x_vec, y_vec).
    """
    C0 = np.asarray(C0)
    P = np.stack(
        [
            np.conj(np.asarray(x_vec, dtype=complex)),
            np.conj(np.asarray(y_vec, dtype=complex)),
        ],
        axis=1,
    )  # (4, 2)
    G = P.conj().T @ C0 @ P  # (2, 2) Hermitian positive
    scale = np.sqrt(np.real(G[0, 0] * G[1, 1]))
    evals, evecs = np.linalg.eigh(G)
    if evals.min() <= 0:
        raise ValueError("X/Y are degenerate in the C0 metric")
    G_isqrt = (evecs / np.sqrt(evals)) @ evecs.conj().T
    P_new = P @ G_isqrt * np.sqrt(scale)
    return np.conj(P_new[:, 0]), np.conj(P_new[:, 1])


def zenith_orthonormal_vectors(kern):
    """Fully nulled zenith polarimeter: Q = U = V = 0 for pure zenith I.

    Starts from the gain-weighted `zenith_port_weights` vectors and
    Loewdin-orthonormalizes them in the metric of the unpolarized
    zenith covariance; X and Y become complex combinations of all
    four ports.
    """
    x0, y0, C0 = zenith_port_weights(kern, null_q=True)
    x_vec, y_vec = orthonormalize_xy(C0, x0, y0)
    return x_vec, y_vec, C0


# --------------------------------------------------------------------
# Spectrometer integration
# --------------------------------------------------------------------

ZOOM_STEP_HZ = 25000.0 / 64  # 390.625 Hz
FINE_STEP_HZ = 25000.0 / 2048  # 12.20703125 Hz


def zoom_bin_offsets_hz():
    """Nominal center offsets of the 64 zoom bins (FFT ordering)."""
    k = np.arange(64)
    return np.where(k < 32, k, k - 64) * ZOOM_STEP_HZ


def parent_weights(offsets_hz, notch=0):
    """Normalized parent-bin weights on a fine offset grid."""
    from lusee.SpectrometerResponse import spectrometer_response

    w = spectrometer_response(np.asarray(offsets_hz, dtype=float), notch)
    return w / w.sum()


def zoom_weights(offsets_hz):
    """Normalized zoom-bin weights -> (Noffsets, 64)."""
    from lusee.SpectrometerResponse import spectrometer_response_zoom

    off = np.asarray(offsets_hz, dtype=float)
    W = np.stack(
        [spectrometer_response_zoom(off, k) for k in range(64)], axis=-1
    )
    return W / W.sum(axis=0, keepdims=True)


def ideal_zoom_weights(offsets_hz, fwhm_hz=ZOOM_STEP_HZ):
    """Gaussian 'ideal' zoom bins at the nominal centers -> (Noff, 64)."""
    off = np.asarray(offsets_hz, dtype=float)
    centers = zoom_bin_offsets_hz()
    sigma = fwhm_hz / (2 * np.sqrt(2 * np.log(2)))
    W = np.exp(-0.5 * ((off[:, None] - centers[None, :]) / sigma) ** 2)
    return W / W.sum(axis=0, keepdims=True)


def integrate_spectrometer(
    waterfall, fine_freqs_mhz, parent_centers_mhz, notch=0
):
    """Convolve a fine-frequency waterfall with the spectrometer response.

    waterfall : (..., Nfreq, Nchan) array on the fine grid.
    Returns dict with:
        'parent'     : (..., Nparent, Nchan)
        'zoom'       : (..., Nparent, 64, Nchan)
        'ideal_zoom' : (..., Nparent, 64, Nchan)
    Parent bins whose +-50 kHz response support is not fully covered by
    the fine grid raise an error.
    """
    fine = np.asarray(fine_freqs_mhz, dtype=float)
    df = np.diff(fine)
    if not np.allclose(df, df[0], rtol=1e-9):
        raise ValueError("fine frequency grid must be uniform")
    parents, zooms, ideals = [], [], []
    for fc in parent_centers_mhz:
        off = (fine - fc) * 1e6
        sel = np.abs(off) <= 50000.0 + 1e-6
        if off[sel].min() > -50000.0 + 1e-3 or off[sel].max() < 50000.0 - 1e-3:
            raise ValueError(
                f"fine grid does not cover the response of bin {fc} MHz"
            )
        chunk = waterfall[..., sel, :]
        o = off[sel]
        wp = parent_weights(o, notch=notch)
        wz = zoom_weights(o)
        wi = ideal_zoom_weights(o)
        parents.append(np.einsum("...fc,f->...c", chunk, wp))
        zooms.append(np.einsum("...fc,fz->...zc", chunk, wz))
        ideals.append(np.einsum("...fc,fz->...zc", chunk, wi))
    return {
        "parent": np.stack(parents, axis=-2),
        "zoom": np.stack(zooms, axis=-3),
        "ideal_zoom": np.stack(ideals, axis=-3),
    }


def zoom_frequency_grid(parent_centers_mhz):
    """Sorted absolute frequencies (MHz) and index maps for zoom bins.

    Returns (freqs_sorted, order) where order is a list of (parent_index,
    zoom_bin) tuples such that freqs_sorted[i] is the nominal center of
    zoom bin order[i][1] of parent order[i][0].  The grid is contiguous
    at the 390.625 Hz spacing across adjacent parents.
    """
    offs = zoom_bin_offsets_hz()
    entries = []
    for p, fc in enumerate(parent_centers_mhz):
        for k in range(64):
            entries.append((fc + offs[k] * 1e-6, p, k))
    entries.sort()
    freqs = np.array([e[0] for e in entries])
    order = [(e[1], e[2]) for e in entries]
    return freqs, order
