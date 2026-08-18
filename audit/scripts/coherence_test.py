"""Is there COHERENT sky structure under the shot noise?

The delay transform refocuses pixels of equal RM, so unlike |V| at a
single lambda^2 it CAN retain sky coherence.  If real degree-scale
polarized structure contributes, the measured delay power must exceed
the incoherent prediction sum_p|w_p|^2, and the excess must GROW as we
smooth the polarized map to scales where WMAP K is signal-dominated.
If the ripple is WMAP noise + pixelization, the ratio stays at 1.
"""
import os as _os, pathlib as _p
_ROOT = _p.Path(__file__).resolve().parents[2]
_OUT = _p.Path(__file__).resolve().parent.parent / "generated"
_OUT.mkdir(exist_ok=True)
_os.chdir(_ROOT)          # data/ paths are repo-relative
SP = str(_OUT) + "/"      # generated artefacts land here

import numpy as np, healpy as hp, h5py, finufft, gc
from astropy.io import fits

C, LMAX = 299792458.0, 1023
with h5py.File("data/faraday2020v2.hdf5", "r") as f:
    RM = f["faraday_sky_mean"][:]
with fits.open("data/wmap_band_iqumap_r9_9yr_K_v5.fits") as h:
    d = h["Stokes Maps"].data
    Q = hp.reorder(d["Q_POLARISATION"].astype(np.float64), n2r=True)
    U = hp.reorder(d["U_POLARISATION"].astype(np.float64), n2r=True)
    NOBS = hp.reorder(d["N_OBS"].astype(np.float64), n2r=True)
alm_RM = hp.map2alm(RM, lmax=LMAX, iter=0)
axis = np.array([0.3, -0.5, 0.81]); axis /= np.linalg.norm(axis)

# WMAP K sigma0 for Q/U = 1.435 mK per observation (LAMBDA); map is in mK
sig = 1.435 / np.sqrt(np.maximum(NOBS, 1e-9))
print("WMAP K nside=512 per-pixel Q noise: median %.4f mK" % np.median(sig))
print("map Q rms %.4f mK  ->  per-pixel S/N ~ %.2f\n"
      % (Q.std(), Q.std() / np.median(sig)))

for f_mhz in (30.0, 50.0):
    l2g = (C / ((f_mhz + (np.arange(16384) - 8192) * (25e-3 / 2048)) * 1e6)) ** 2
    han = np.hanning(l2g.size)
    print("=== %.0f MHz" % f_mhz)
    print("  smooth   nside |  delay power   incoherent pred | ratio")
    for fwhm_deg in (0.0, 1.0, 3.0, 10.0):
        aq = hp.map2alm(Q, lmax=LMAX, iter=0)
        au = hp.map2alm(U, lmax=LMAX, iter=0)
        if fwhm_deg:
            bl = hp.gauss_beam(np.radians(fwhm_deg), lmax=LMAX)
            aq = hp.almxfl(aq, bl); au = hp.almxfl(au, bl)
        for ns in (512, 1024):
            rm = hp.alm2map(alm_RM, ns, lmax=LMAX)
            q = hp.alm2map(aq, ns, lmax=LMAX)
            u = hp.alm2map(au, ns, lmax=LMAX)
            mu = axis @ np.array(hp.pix2vec(ns, np.arange(hp.nside2npix(ns))))
            B = np.where(mu > 0, mu ** 2, 0.0)
            w = B * (q + 1j * u) / B.sum()
            V = finufft.nufft1d3(np.ascontiguousarray(2.0 * rm), w,
                                 np.ascontiguousarray(l2g), eps=1e-9, isign=1)
            tot = (np.abs(np.fft.fft(V * han)) ** 2).sum()
            pred = l2g.size * (np.abs(w) ** 2).sum() * (han ** 2).sum()
            print("  %4.1f deg  %5d | %.5e  %.5e  | %6.2f"
                  % (fwhm_deg, ns, tot, pred, tot / pred))
            del rm, q, u, w, V, mu, B; gc.collect()
    print()
