"""Is the delay power a physical signal, or the shot noise of the pixel grid?

If the per-pixel Faraday phases are effectively random across the fine
frequency grid, then E|V(l2)|^2 = sum_p |w_p|^2 exactly -- a pure
pixelization shot floor that falls as 1/npix.  Compare measured against
that prediction, and watch the nside trend.
"""
import os as _os, pathlib as _p
_ROOT = _p.Path(__file__).resolve().parents[2]
_OUT = _p.Path(__file__).resolve().parent.parent / "generated"
_OUT.mkdir(exist_ok=True)
_os.chdir(_ROOT)          # data/ paths are repo-relative
SP = str(_OUT) + "/"      # generated artefacts land here

import numpy as np, healpy as hp, h5py, finufft, gc
from astropy.io import fits

C = 299792458.0
LMAX = 1023
with h5py.File("data/faraday2020v2.hdf5", "r") as f:
    RM = f["faraday_sky_mean"][:]
with fits.open("data/wmap_band_iqumap_r9_9yr_K_v5.fits") as h:
    d = h["Stokes Maps"].data
    Q = hp.reorder(d["Q_POLARISATION"].astype(np.float64), n2r=True)
    U = hp.reorder(d["U_POLARISATION"].astype(np.float64), n2r=True)
alm = {k: hp.map2alm(m, lmax=LMAX, iter=0) for k, m in (("RM", RM), ("Q", Q), ("U", U))}
axis = np.array([0.3, -0.5, 0.81]); axis /= np.linalg.norm(axis)

k = np.arange(16384) - 8192
l2 = (C / ((30.0 + k * (25e-3 / 2048)) * 1e6) ** 1) ** 2
han = np.hanning(l2.size)

print(" nside | measured tot.pow | shot-noise pred |  meas/pred | ratio to coarser")
prev = None
for ns in (256, 512, 1024, 2048):
    rm = hp.alm2map(alm["RM"], ns, lmax=LMAX)
    q = hp.alm2map(alm["Q"], ns, lmax=LMAX)
    u = hp.alm2map(alm["U"], ns, lmax=LMAX)
    npix = hp.nside2npix(ns)
    Bsum = 0.0
    w = np.empty(npix, dtype=complex)
    CH = 1 << 22
    for s in range(0, npix, CH):
        e = min(s + CH, npix)
        vec = np.array(hp.pix2vec(ns, np.arange(s, e)))
        mu = axis @ vec
        B = np.where(mu > 0, mu ** 2, 0.0)
        Bsum += B.sum()
        w[s:e] = B * (q[s:e] + 1j * u[s:e])
        del vec, mu, B
    w /= Bsum
    V = finufft.nufft1d3(np.ascontiguousarray(2.0 * rm), w,
                         np.ascontiguousarray(l2), eps=1e-9, isign=1)
    tot = (np.abs(np.fft.fft(V * han)) ** 2).sum()
    pred = l2.size * (np.abs(w) ** 2).sum() * (han ** 2).sum()
    line = " %5d | %.6e   | %.6e  |   %6.3f  " % (ns, tot, pred, tot / pred)
    if prev is not None:
        line += "|   %.2f" % (prev / tot)
    prev = tot
    print(line, flush=True)
    del rm, q, u, w, V
    gc.collect()
