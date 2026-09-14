"""Both spectra should BE histograms -- just with different weightings.

Real map: |w|^2 and RM are correlated (bright polarized plane = high |RM|,
and half of it sits under the horizon mask).  Shuffled: w and RM are
independent, so the weighting is flat.  Overlay each spectrum on its own
predicted histogram sum_p |w_p|^2 delta(phi - 2 RM_p).
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
rng = np.random.default_rng(20260818)
with h5py.File("data/faraday2020v2.hdf5", "r") as f:
    RM0 = f["faraday_sky_mean"][:]
with fits.open("data/wmap_band_iqumap_r9_9yr_K_v5.fits") as h:
    d = h["Stokes Maps"].data
    Q = hp.reorder(d["Q_POLARISATION"].astype(np.float64), n2r=True)
    U = hp.reorder(d["U_POLARISATION"].astype(np.float64), n2r=True)
rm = hp.alm2map(hp.map2alm(RM0, lmax=LMAX, iter=0), NS, lmax=LMAX)
q = hp.alm2map(hp.map2alm(Q, lmax=LMAX, iter=0), NS, lmax=LMAX)
u = hp.alm2map(hp.map2alm(U, lmax=LMAX, iter=0), NS, lmax=LMAX)
axis = np.array([0.3, -0.5, 0.81]); axis /= np.linalg.norm(axis)
mu = axis @ np.array(hp.pix2vec(NS, np.arange(hp.nside2npix(NS))))
B = np.where(mu > 0, mu ** 2, 0.0)
w = np.ascontiguousarray(B * (q + 1j * u) / B.sum())
perm = rng.permutation(rm.size)
rms = np.ascontiguousarray(rm[perm])

l2 = (C / ((30.0 + (np.arange(16384) - 8192) * (25e-3 / 2048)) * 1e6)) ** 2
han = np.hanning(l2.size); dl2 = abs(np.diff(l2).mean())
phi = np.fft.fftshift(np.fft.fftfreq(l2.size, d=dl2)) * np.pi
step = phi[1] - phi[0]
edges = np.concatenate([phi - step / 2, [phi[-1] + step / 2]])
norm = l2.size * (han ** 2).sum()

out = {"phi": phi}
for src, tag in ((rm, "real"), (rms, "shuf")):
    V = finufft.nufft1d3(np.ascontiguousarray(2.0 * src), w,
                         np.ascontiguousarray(l2), eps=1e-9, isign=1)
    P = np.abs(np.fft.fftshift(np.fft.fft(V * han))) ** 2
    hist, _ = np.histogram(src, bins=edges, weights=np.abs(w) ** 2)
    hist = hist * norm
    out["P_" + tag] = P; out["H_" + tag] = hist
    m = (np.abs(phi) < 2600)
    good = m & (P > 0) & (hist > 0)
    print("%-5s  total P/H = %.3f   log-log corr over support = %.3f   "
          "support(P) %.0f..%.0f   support(H) %.0f..%.0f"
          % (tag, P.sum() / hist.sum(),
             np.corrcoef(np.log10(P[good]), np.log10(hist[good]))[0, 1],
             phi[m][P[m] > 1e-8].min(), phi[m][P[m] > 1e-8].max(),
             phi[m][hist[m] > 0].min(), phi[m][hist[m] > 0].max()))
np.savez(SP + "hist_overlay.npz", **out)

# why the real support is narrower: where do the extreme RM pixels sit?
vis = B > 0
print("\nextreme-RM pixels vs the horizon mask:")
for lab, sel in (("|RM|>1200", np.abs(rm) > 1200), ("|RM|>2000", np.abs(rm) > 2000)):
    print("   %-10s  %7d pixels,  %5.1f%% above horizon,  mean B = %.4f"
          % (lab, sel.sum(), 100 * vis[sel].mean(), B[sel].mean()))
print("   %-10s  %7d pixels,  %5.1f%% above horizon,  mean B = %.4f"
      % ("all sky", rm.size, 100 * vis.mean(), B.mean()))
