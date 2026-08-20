"""Where does the diffuse-sky Faraday integral stop being physical?

Sweep lambda^2 from short wavelengths (where the phase IS resolved by the
grid) down to the LuSEE band, at three resolutions.  While the answer is
converged, all three nsides agree and |V| follows the true depolarization
law.  Once the grid loses the phase, the curves separate and each one
flattens onto its own 1/sqrt(npix) shot floor.
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

freqs = np.array([3000., 1400., 800., 400., 200., 100., 50., 30., 10.])
l2 = (C / (freqs * 1e6)) ** 2

out = {}
for ns in (512, 1024, 2048):
    rm = hp.alm2map(alm["RM"], ns, lmax=LMAX)
    q = hp.alm2map(alm["Q"], ns, lmax=LMAX)
    u = hp.alm2map(alm["U"], ns, lmax=LMAX)
    npix = hp.nside2npix(ns)
    w = np.empty(npix, dtype=complex); Bsum = 0.0
    for s in range(0, npix, 1 << 22):
        e = min(s + (1 << 22), npix)
        mu = axis @ np.array(hp.pix2vec(ns, np.arange(s, e)))
        B = np.where(mu > 0, mu ** 2, 0.0)
        Bsum += B.sum(); w[s:e] = B * (q[s:e] + 1j * u[s:e])
    w /= Bsum
    V = finufft.nufft1d3(np.ascontiguousarray(2.0 * rm), w,
                         np.ascontiguousarray(l2), eps=1e-9, isign=1)
    out[ns] = (np.abs(V), np.sqrt((np.abs(w) ** 2).sum()))
    del rm, q, u, w, V; gc.collect()

V0 = 9.1491e-03  # unrotated |V|, identical at every nside
print("  nu[MHz]  lam^2   |  suppression |V|/|V_nofar| at nside      | spread |  shot floor / |V| ")
print("                    |   512       1024       2048            |        |  (512, 1024, 2048)")
for i, f in enumerate(freqs):
    a, b, c = out[512][0][i], out[1024][0][i], out[2048][0][i]
    sf = [out[n][1] / out[n][0][i] for n in (512, 1024, 2048)]
    print("  %7.0f %7.2f | %.3e %.3e %.3e | %5.1fx | %.2f %.2f %.2f"
          % (f, l2[i], a / V0, b / V0, c / V0,
             max(a, b, c) / min(a, b, c), *sf))
print("\nshot floor sqrt(sum|w|^2)/|V_nofar| :  512 %.2e   1024 %.2e   2048 %.2e"
      % tuple(out[n][1] / V0 for n in (512, 1024, 2048)))
