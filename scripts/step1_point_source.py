"""Step 1: single linearly polarized transiting source, 30 MHz band.

A fully polarized point source (unit brightness, 1 K sr) with its EVPA
fixed on the sky transits the LuSEE-Night site.  The source shines
through a Faraday screen of depth phi_FD = 250 rad/m^2 (paper value).
The beam is frozen at the 30 MHz native response channel; across the
fine grid (16384 x 25 kHz/2048, +-0.1 MHz) only the Faraday phase
chi(nu) = psi(t) + phi_FD lambda^2(nu) varies.

Outputs (generated_data/):
    step1_fine_waterfall.npy   (1024, 16384, 16) float64 memmap,
                               V^2/Hz, product order = PRODUCT_LABELS
    step1_meta.npz             times, fine freqs, track, psi, no-Faraday
                               waterfall (1024, 16) and config scalars
    step1_binned.npz           parent (1024, 3, 16), zoom and ideal
                               zoom (1024, 3, 64, 16)

The source is placed at ecliptic longitude 120 deg with its ecliptic
latitude equal to the site latitude, so it culminates near zenith
(the lunar equator is within ~1.5 deg of the ecliptic).
"""

import argparse
import time as _time

import numpy as np

from common import (
    CACHE_DIR,
    FINE_STEP_MHZ,
    GEN_DIR,
    N_FINE,
    N_TIMES,
    PHI_FD_POINT,
    RESPONSE_PATH,
    fine_freqs,
    lam2,
    moon_location,
    parent_centers,
    times,
)
from lusee_faraday import channelization as chan
from lusee_faraday import instrument
from lusee_faraday import polarimeter as pol
from lusee_faraday import response as rsp
from lusee_faraday.conventions import PRODUCT_LABELS

ECL_LON_DEG = 120.0
CENTER_MHZ = 30.0
TIME_CHUNK = 32


def source_track(tt, loc, ecl_lat_deg):
    """theta, phi (response convention) and parallactic angle psi.

    psi is the position angle of ecliptic north in the tangent basis,
    measured from e_theta towards e_phi; a source with its electric
    vector along ecliptic north has instrument-frame
    (Q + iU) = exp(2 i psi).  Cached, as lunarsky transforms take
    minutes for 1024 times.
    """
    import astropy.units as u
    from astropy.coordinates import BarycentricTrueEcliptic
    from astropy.coordinates import SkyCoord as ASkyCoord
    from lunarsky import LunarTopo

    cache = CACHE_DIR / "step1_track.npz"
    if cache.exists():
        d = np.load(cache)
        if (
            d["theta"].shape == (N_TIMES,)
            and float(d["ecl_lat_deg"]) == ecl_lat_deg
        ):
            return d["theta"], d["phi"], d["psi"]

    def to_topo(lat_deg):
        src = ASkyCoord(
            lon=ECL_LON_DEG * u.deg,
            lat=lat_deg * u.deg,
            frame=BarycentricTrueEcliptic,
            distance=1e6 * u.pc,
        )
        topo = src.transform_to(LunarTopo(location=loc, obstime=tt))
        alt = np.asarray(topo.alt.rad, dtype=float)
        az = np.asarray(topo.az.rad, dtype=float)
        return np.pi / 2 - alt, (np.pi / 2 - az) % (2 * np.pi)

    print("computing source track (lunarsky)...", flush=True)
    theta, phi = to_topo(ecl_lat_deg)
    delta = 1e-3  # deg, towards ecliptic north
    theta_n, phi_n = to_topo(ecl_lat_deg + delta)
    d_theta = theta_n - theta
    d_phi = (phi_n - phi + np.pi) % (2 * np.pi) - np.pi
    psi = np.arctan2(d_phi * np.sin(theta), d_theta)
    np.savez(cache, theta=theta, phi=phi, psi=psi, ecl_lat_deg=ecl_lat_deg)
    return theta, phi, psi


def fine_grid(n_fine=N_FINE):
    """``common.fine_freqs`` with the grid size made explicit.

    ``fine_freqs`` reads ``N_FINE`` off the config module, so nothing can
    ask it for a smaller grid.  Same formula, same centering, same
    values at the default -- pinned by
    ``test_fine_grid_default_matches_common_fine_freqs``.
    """
    k = np.arange(n_fine) - n_fine // 2
    return CENTER_MHZ + k * FINE_STEP_MHZ


def make_waterfall(
    kern,
    resp,
    receiver,
    theta,
    phi,
    psi,
    fd,
    out_path,
    n_times=N_TIMES,
    n_fine=N_FINE,
):
    """Stream the (T, F, 16) fine waterfall to a memmapped .npy.

    ``n_times`` and ``n_fine`` exist so a test can run the real physics
    at a size that fits in a test suite.  At the defaults this is the
    production run: 1024 x 16384 x 16 float64, a 2 GB memmap, which is
    why nothing could reach this function before they were added.
    """
    ff = fine_grid(n_fine)
    l2 = lam2(ff)
    wf = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float64,
        shape=(n_times, n_fine, 16),
    )
    nofar = np.zeros((n_times, 16))
    up = theta <= np.pi / 2
    # exp(2i chi) with chi = psi + fd * lam2; the frequency factor is
    # shared by all times in the chunk.
    e_freq = np.exp(2j * fd * l2)  # (F,)
    one = np.array([CENTER_MHZ])
    # The beam AND the impedances are frozen at the native channel: Z_A
    # is steep here, so a chromatic Z_A would put an 11% ramp into the
    # band.  T_moon = T_ant = 0 keeps this the sky-only covariance the
    # legacy assembler computed.
    frozen = dict(impedance_freq_mhz=CENTER_MHZ, T_moon=0.0, T_ant=0.0)
    t0 = _time.time()
    for s in range(0, n_times, TIME_CHUNK):
        sl = slice(s, min(s + TIME_CHUNK, n_times))
        mask = up[sl]
        # Every time in the chunk is carried, with the below-horizon
        # ones zeroed afterwards, rather than slicing the array down to
        # the above-horizon count: instrument.covariance runs through
        # @jax.jit, so a shape that tracked that count would recompile
        # once per distinct count (three on the paper track, so 6 jit
        # compilations rather than 2).  The price is that the 13 chunks
        # that are entirely below the horizon now do full-shape work the
        # sliced version skipped.  With T_moon = T_ant = 0 a zero pair
        # integral gives an exactly zero covariance and an exactly zero
        # packed block, so the answer is unchanged.
        th = np.where(mask, theta[sl], 0.0)
        K = kern.sample(th, phi[sl])  # (10, 4, Nt), physical W
        e_t = np.exp(2j * psi[sl])  # (Nt,)
        # pair integral: K @ (1, cos 2chi, sin 2chi, 0)
        #   = K_I + 0.5 (K_Q - i K_U) e^{2i chi}
        #         + 0.5 (K_Q + i K_U) e^{-2i chi}.
        # K_Q, K_U are complex for cross pairs, so the second
        # coefficient is NOT conj(first): conjugating it breaks
        # Hermiticity/PSD and gives pseudo-p > 1.
        base = K[:, 0]  # (10, Nt)
        cpol_p = 0.5 * (K[:, 1] - 1j * K[:, 2])
        cpol_m = 0.5 * (K[:, 1] + 1j * K[:, 2])
        e_tf = e_t[None, :, None] * e_freq[None, None, :]
        pair = (
            base[:, :, None]
            + cpol_p[:, :, None] * e_tf
            + cpol_m[:, :, None] * np.conj(e_tf)
        )
        pair = np.moveaxis(pair, 0, -1)  # (Nt, F, 10)
        pair[~mask] = 0.0
        block, _ = instrument.channels(
            instrument.covariance(pair, resp, receiver, ff, **frozen)
        )
        # Runtime invariant, not just a test: sqrt(Q^2+U^2+V^2) <= I for
        # any physical covariance.  It caught a real sign bug in the
        # complex cross-pair decomposition above.
        pol.check_psd(pol.pseudo_stokes_from_channels(block))
        wf[sl] = block
        # no-Faraday reference: chi = psi only
        pair0 = (
            base + cpol_p * e_t[None, :] + cpol_m * np.conj(e_t)[None, :]
        ).T  # (Nt, 10)
        pair0[~mask] = 0.0
        nf_block, _ = instrument.channels(
            instrument.covariance(
                pair0[:, None, :], resp, receiver, one, **frozen
            )
        )
        nofar[sl] = nf_block[:, 0]
        if (sl.stop % 128) < TIME_CHUNK:
            print(
                f"  waterfall {sl.stop}/{n_times}"
                f"  ({_time.time()-t0:.0f} s)",
                flush=True,
            )
    wf.flush()
    return nofar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", type=float, default=PHI_FD_POINT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.ReceiverImpedance import JFETReceiver

    loc = moon_location()
    tt = times()
    theta, phi, psi = source_track(tt, loc, ecl_lat_deg=float(loc.lat.deg))
    print(
        f"track: min zenith angle {np.degrees(theta.min()):.2f} deg, "
        f"up {100 * np.mean(theta <= np.pi / 2):.0f}% of the day",
        flush=True,
    )

    wf_path = GEN_DIR / "step1_fine_waterfall.npy"
    meta_path = GEN_DIR / "step1_meta.npz"
    if wf_path.exists() and meta_path.exists() and not args.force:
        print("fine waterfall exists; use --force to redo", flush=True)
    else:
        t0 = _time.time()
        resp = rsp.load_response(RESPONSE_PATH)
        receiver = JFETReceiver()
        kern = rsp.FixedChannelKernel(resp, CENTER_MHZ)
        print(f"kernel ready {_time.time()-t0:.1f} s", flush=True)
        nofar = make_waterfall(
            kern, resp, receiver, theta, phi, psi, args.fd, wf_path
        )
        # PSD / rank-1 sanity: a single fully polarized source gives a
        # rank-1 covariance, so sqrt(Q^2+U^2+V^2)/I must be <= 1
        # (equality up to the tiny mixing of the bilinear kernel
        # interpolation) in every fine channel.
        wf_check = np.load(wf_path, mmap_mode="r")
        it = int(np.argmin(theta))
        S = pol.pseudo_stokes_from_channels(np.asarray(wf_check[it]))
        ptot = np.sqrt((S[:, 1:] ** 2).sum(-1)) / S[:, 0]
        print(
            f"rank-1 check at transit: sqrt(Q^2+U^2+V^2)/I in "
            f"[{ptot.min():.6f}, {ptot.max():.6f}]", flush=True,
        )
        assert (ptot <= 1 + 1e-9).all() and ptot.min() > 0.98
        np.savez(
            meta_path,
            t_unix=tt.unix,
            fine_freqs_mhz=fine_freqs(CENTER_MHZ),
            theta=theta,
            phi=phi,
            psi=psi,
            nofaraday=nofar,
            fd=args.fd,
            center_mhz=CENTER_MHZ,
            labels=np.array(PRODUCT_LABELS),
        )
        print(f"saved {wf_path.name}, {meta_path.name}", flush=True)

    binned_path = GEN_DIR / "step1_binned.npz"
    if binned_path.exists() and not args.force:
        print("binned products exist; use --force to redo", flush=True)
        return
    print("integrating spectrometer response...", flush=True)
    wf = np.load(wf_path, mmap_mode="r")
    centers = parent_centers(CENTER_MHZ)
    parents = np.empty((N_TIMES, 3, 16))
    zooms = np.empty((N_TIMES, 3, 64, 16))
    ideals = np.empty((N_TIMES, 3, 64, 16))
    ff = fine_freqs(CENTER_MHZ)
    for s in range(0, N_TIMES, TIME_CHUNK):
        sl = slice(s, min(s + TIME_CHUNK, N_TIMES))
        out = chan.integrate(np.asarray(wf[sl]), ff, centers)
        parents[sl] = out["parent"]
        zooms[sl] = out["zoom"]
        ideals[sl] = out["ideal_zoom"]
    np.savez(
        binned_path,
        parent=parents,
        zoom=zooms,
        ideal_zoom=ideals,
        parent_centers_mhz=centers,
        zoom_offsets_hz=chan.zoom_bin_offsets_hz(),
    )
    print(f"saved {binned_path.name}", flush=True)


if __name__ == "__main__":
    main()
