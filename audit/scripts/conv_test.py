"""Is the beam-integrated Faraday-rotated sky-pol term converged in nside?

Take the SAME band-limited fields (RM, Q, U as represented by their
nside=512 alm, lmax=1023), evaluate them on progressively finer HEALPix
quadrature grids, and see whether the oscillatory beam integral
    V = sum_p B_p (Q+iU)_p exp(2i RM_p lam^2) dOmega / sum_p B_p dOmega
settles down.  Only the quadrature changes; the underlying function does not.
"""
import os as _os, pathlib as _p
_ROOT = _p.Path(__file__).resolve().parents[2]
_OUT = _p.Path(__file__).resolve().parent.parent / "generated"
_OUT.mkdir(exist_ok=True)
_os.chdir(_ROOT)          # data/ paths are repo-relative
SP = str(_OUT) + "/"      # generated artefacts land here

import numpy as np, healpy as hp, h5py, time
from astropy.io import fits

C = 299792458.0
LMAX = 1023
NS_IN = 512

t0 = time.time()
with h5py.File("data/faraday2020v2.hdf5", "r") as f:
    RM = f["faraday_sky_mean"][:]
with fits.open("data/wmap_band_iqumap_r9_9yr_K_v5.fits") as h:
    d = h["Stokes Maps"].data
    Q = hp.reorder(d["Q_POLARISATION"].astype(np.float64), n2r=True)
    U = hp.reorder(d["U_POLARISATION"].astype(np.float64), n2r=True)
print("loaded %.1fs" % (time.time() - t0), flush=True)

alm_RM = hp.map2alm(RM, lmax=LMAX, iter=0)
alm_Q = hp.map2alm(Q, lmax=LMAX, iter=0)
alm_U = hp.map2alm(U, lmax=LMAX, iter=0)
print("alms %.1fs" % (time.time() - t0), flush=True)

# smooth "beam": cos^2-ish lobe about a fixed galactic axis, lmax ~ 2
axis = np.array([0.3, -0.5, 0.81])
axis /= np.linalg.norm(axis)

for f_mhz in (30.0, 50.0):
    lam2 = (C / (f_mhz * 1e6)) ** 2
    print("\n=== %.0f MHz  (lambda^2 = %.1f m^2)" % (f_mhz, lam2), flush=True)
    prev = None
    for ns in (256, 512, 1024, 2048):
        rm = hp.alm2map(alm_RM, ns, lmax=LMAX, verbose=False)
        q = hp.alm2map(alm_Q, ns, lmax=LMAX, verbose=False)
        u = hp.alm2map(alm_U, ns, lmax=LMAX, verbose=False)
        vec = np.array(hp.pix2vec(ns, np.arange(hp.nside2npix(ns))))
        mu = axis @ vec
        B = np.where(mu > 0, mu**2, 0.0)
        norm = B.sum()
        P = q + 1j * u
        V_far = (B * P * np.exp(2j * rm * lam2)).sum() / norm
        V_off = (B * P).sum() / norm
        I_like = (B * np.abs(P)).sum() / norm
        msg = ("  nside=%5d  |V_faraday|=%.4e   |V_nofaraday|=%.4e"
               "   supp=%.1e" % (ns, abs(V_far), abs(V_off), abs(V_far) / abs(V_off)))
        if prev is not None:
            msg += "   ratio_to_coarser=%.3f" % (abs(V_far) / prev)
        prev = abs(V_far)
        print(msg, flush=True)
        del rm, q, u, vec, mu, B, P
print("\ndone %.1fs" % (time.time() - t0))
