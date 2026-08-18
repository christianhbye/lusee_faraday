"""Why the Step-4 delay spectrum LOOKED right: it is the RM histogram.

If the per-pixel phases decorrelate, E|F(phi)|^2 = sum_p |w_p|^2 W(phi-2phi_p)
-- i.e. exactly the |w|^2-weighted histogram of the RM map, with no
information about the polarized sky's spatial coherence in it at all.
Compare the measured delay-power profile to that histogram.
"""
import os as _os, pathlib as _p
_ROOT = _p.Path(__file__).resolve().parents[2]
_OUT = _p.Path(__file__).resolve().parent.parent / "generated"
_OUT.mkdir(exist_ok=True)
_os.chdir(_ROOT)          # data/ paths are repo-relative
SP = str(_OUT) + "/"      # generated artefacts land here

import numpy as np, healpy as hp, h5py, finufft
from astropy.io import fits

C, LMAX, NS = 299792458.0, 1023, 1024
with h5py.File("data/faraday2020v2.hdf5", "r") as f:
    RM = f["faraday_sky_mean"][:]
with fits.open("data/wmap_band_iqumap_r9_9yr_K_v5.fits") as h:
    d = h["Stokes Maps"].data
    Q = hp.reorder(d["Q_POLARISATION"].astype(np.float64), n2r=True)
    U = hp.reorder(d["U_POLARISATION"].astype(np.float64), n2r=True)
alm = {k: hp.map2alm(m, lmax=LMAX, iter=0) for k, m in (("RM", RM), ("Q", Q), ("U", U))}
rm = hp.alm2map(alm["RM"], NS, lmax=LMAX)
q = hp.alm2map(alm["Q"], NS, lmax=LMAX)
u = hp.alm2map(alm["U"], NS, lmax=LMAX)
axis = np.array([0.3, -0.5, 0.81]); axis /= np.linalg.norm(axis)
mu = axis @ np.array(hp.pix2vec(NS, np.arange(hp.nside2npix(NS))))
B = np.where(mu > 0, mu ** 2, 0.0)
w = B * (q + 1j * u) / B.sum()

k = np.arange(16384) - 8192
l2 = (C / ((30.0 + k * (25e-3 / 2048)) * 1e6) ** 1) ** 2
V = finufft.nufft1d3(np.ascontiguousarray(2.0 * rm), w, np.ascontiguousarray(l2), eps=1e-9, isign=1)
han = np.hanning(V.size)
P = np.abs(np.fft.fftshift(np.fft.fft(V * han))) ** 2
phi = np.fft.fftshift(np.fft.fftfreq(V.size, d=abs(np.diff(l2).mean()))) * np.pi

# the shot-noise prediction: |w|^2-weighted histogram of RM on the same phi grid
edges = np.concatenate([phi - 0.5 * (phi[1] - phi[0]), [phi[-1] + 0.5 * (phi[1] - phi[0])]])
hist, _ = np.histogram(rm, bins=edges, weights=np.abs(w) ** 2)
hist = hist * V.size * (han ** 2).sum()

m = P > 0
print("measured delay power vs |w|^2-weighted RM histogram")
print("  log-log correlation over all %d delay bins : %.4f"
      % (m.sum(), np.corrcoef(np.log10(P[m] + 1e-40), np.log10(hist[m] + 1e-40))[0, 1]))
print("  total power ratio measured/histogram        : %.3f" % (P.sum() / hist.sum()))
print("\n  |phi| band       measured frac   histogram frac")
for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 1e9)):
    s = (np.abs(phi) >= lo) & (np.abs(phi) < hi)
    print("  %5.0f - %-8.0f   %.4f          %.4f"
          % (lo, min(hi, 999), P[s].sum() / P.sum(), hist[s].sum() / hist.sum()))
