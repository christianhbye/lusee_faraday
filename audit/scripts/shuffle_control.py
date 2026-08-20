"""Random-phase control for the diffuse Faraday delay spectrum.

Shuffling the RM values across pixels preserves the RM histogram EXACTLY
and destroys every trace of spatial structure.  If the delay spectrum is
unchanged by that, it carries no information about the sky -- only the
histogram.

Positive control: rescale RM by a factor f.  At small f the level-set
bands are wider than a pixel, the calculation is resolved, and shuffling
MUST change the answer.  If it does, the test has power and the null
result at f=1 means something.
"""
import os as _os, pathlib as _p
_ROOT = _p.Path(__file__).resolve().parents[2]
_OUT = _p.Path(__file__).resolve().parent.parent / "generated"
_OUT.mkdir(exist_ok=True)
_os.chdir(_ROOT)          # data/ paths are repo-relative
SP = str(_OUT) + "/"      # generated artefacts land here

import numpy as np, healpy as hp, h5py, finufft, gc, json
from astropy.io import fits

C, LMAX = 299792458.0, 1023
rng = np.random.default_rng(20260818)

with h5py.File("data/faraday2020v2.hdf5", "r") as f:
    RM0 = f["faraday_sky_mean"][:]
with fits.open("data/wmap_band_iqumap_r9_9yr_K_v5.fits") as h:
    d = h["Stokes Maps"].data
    Q = hp.reorder(d["Q_POLARISATION"].astype(np.float64), n2r=True)
    U = hp.reorder(d["U_POLARISATION"].astype(np.float64), n2r=True)
alm_RM = hp.map2alm(RM0, lmax=LMAX, iter=0)
alm_Q = hp.map2alm(Q, lmax=LMAX, iter=0)
alm_U = hp.map2alm(U, lmax=LMAX, iter=0)
axis = np.array([0.3, -0.5, 0.81]); axis /= np.linalg.norm(axis)

# the paper's fine grid at 30 MHz
l2 = (C / ((30.0 + (np.arange(16384) - 8192) * (25e-3 / 2048)) * 1e6)) ** 2
han = np.hanning(l2.size)
dl2 = abs(np.diff(l2).mean())
phi_ax = np.fft.fftshift(np.fft.fftfreq(l2.size, d=dl2)) * np.pi
dRM_cell = 2 * np.sqrt(3) / (l2.max() - l2.min())   # RMSF FWHM

def build(ns):
    rm = hp.alm2map(alm_RM, ns, lmax=LMAX)
    q = hp.alm2map(alm_Q, ns, lmax=LMAX)
    u = hp.alm2map(alm_U, ns, lmax=LMAX)
    mu = axis @ np.array(hp.pix2vec(ns, np.arange(hp.nside2npix(ns))))
    B = np.where(mu > 0, mu ** 2, 0.0)
    w = B * (q + 1j * u) / B.sum()
    return rm, w

def spectrum(rm, w, f):
    V = finufft.nufft1d3(np.ascontiguousarray(2.0 * f * rm),
                         np.ascontiguousarray(w),
                         np.ascontiguousarray(l2), eps=1e-9, isign=1)
    F = np.fft.fftshift(np.fft.fft(V * han))
    return np.abs(F) ** 2

results = {}
print("RMSF FWHM (delay cell) = %.2f rad/m^2\n" % dRM_cell)
print("%-6s %-6s %-14s %-13s %-13s %-8s %-8s" %
      ("nside", "f", "band/pixel", "real", "RM-shuffled", "real/sh", "real/incoh"))
for ns in (512, 1024, 2048):
    rm, w = build(ns)
    pix = np.sqrt(4 * np.pi / hp.nside2npix(ns))
    perm = rng.permutation(rm.size)
    incoh = l2.size * (np.abs(w) ** 2).sum() * (han ** 2).sum()
    for f in (1.0, 0.1, 0.02):
        gradf = 4327.8 * f
        band_over_pix = (dRM_cell / gradf) / pix
        Pr = spectrum(rm, w, f)
        Ps = spectrum(rm[perm], w, f)
        tr, ts = Pr.sum(), Ps.sum()
        print("%-6d %-6.2f %-14.3f %-13.5e %-13.5e %-8.2f %-8.2f" %
              (ns, f, band_over_pix, tr, ts, tr / ts, tr / incoh))
        results["%d_%g" % (ns, f)] = dict(
            nside=ns, f=f, band_over_pix=band_over_pix, real=tr,
            shuffled=ts, incoh=incoh)
        if ns == 1024 and f in (1.0, 0.02):
            np.savez(SP + "spec_f%g.npz" % f,
                     phi=phi_ax, real=Pr, shuf=Ps,
                     rm=rm[::37], w2=np.abs(w[::37]) ** 2, f=f)
    del rm, w; gc.collect()
    print()

json.dump(results, open(SP + "shuffle_results.json", "w"), indent=1)
