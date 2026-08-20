"""How well do the two input maps actually determine the answer?"""
import os as _os, pathlib as _p
_ROOT = _p.Path(__file__).resolve().parents[2]
_OUT = _p.Path(__file__).resolve().parent.parent / "generated"
_OUT.mkdir(exist_ok=True)
_os.chdir(_ROOT)          # data/ paths are repo-relative
SP = str(_OUT) + "/"      # generated artefacts land here

import numpy as np, healpy as hp, h5py
from astropy.io import fits
C = 299792458.0

# ---------------------------------------------- 1. Hutschenreuter RM map
with h5py.File("data/faraday2020v2.hdf5", "r") as f:
    RM, SIG = f["faraday_sky_mean"][:], f["faraday_sky_std"][:]
print("=== faraday2020v2 reconstruction uncertainty  (rad/m^2)")
for q in (10, 50, 90, 99):
    print("   sigma p%-2d = %7.2f" % (q, np.percentile(SIG, q)))
print("   sigma/|RM| median = %.2f" % np.median(SIG / np.maximum(np.abs(RM), 1e-9)))
print("   fraction of sky with sigma > |RM| : %.1f%%" % (100 * (SIG > np.abs(RM)).mean()))

sig = np.median(SIG)
lam2_pi = np.pi / (2 * sig)                      # 2 sigma lam^2 = pi
print("\n   phase uncertainty 2*sigma*lam^2 reaches pi at lam^2 = %.3f m^2"
      % lam2_pi, " -> nu = %.0f MHz" % (C / np.sqrt(lam2_pi) / 1e6))
print("   band     median turns   frac of sky > 1 turn")
for f0 in (10., 30., 50., 100., 400., 800.):
    l2 = (C / (f0 * 1e6)) ** 2
    turns = 2 * SIG * l2 / (2 * np.pi)
    print("   %4.0f MHz  %12.1f   %10.1f%%" % (f0, np.median(turns), 100 * (turns > 1).mean()))

# ---------------------------------------------- 2. resolution threshold
GRAD = 4327.8
print("\n=== resolution: 2|grad phi| lam^2 * pixel < pi")
for ns in (512, 1024, 2048):
    pix = np.sqrt(4 * np.pi / hp.nside2npix(ns))
    l2 = np.pi / (2 * GRAD * pix)
    print("   nside %4d -> converged only for lam^2 < %6.3f m^2  (nu > %4.0f MHz)"
          % (ns, l2, C / np.sqrt(l2) / 1e6))

# ---------------------------------------------- 3. WMAP K polarization
with fits.open("data/wmap_band_iqumap_r9_9yr_K_v5.fits") as h:
    d = h["Stokes Maps"].data
    Q = hp.reorder(d["Q_POLARISATION"].astype(np.float64), n2r=True)
    U = hp.reorder(d["U_POLARISATION"].astype(np.float64), n2r=True)
    N = hp.reorder(d["N_OBS"].astype(np.float64), n2r=True)
sQ = 1.435 / np.sqrt(np.maximum(N, 1e-9))        # sigma0 for K-band Q/U, mK
nvar = np.mean(sQ ** 2)
tot = 0.5 * (Q.var() + U.var())
print("\n=== WMAP K polarization, nside 512")
print("   per-pixel noise sigma  median %.4f mK" % np.median(sQ))
print("   map rms (Q,U avg)             %.4f mK" % np.sqrt(tot))
print("   noise-subtracted signal rms   %.4f mK" % np.sqrt(max(tot - nvar, 0)))
print("   per-pixel S/N (signal/noise)  %.2f" % np.sqrt(max(tot - nvar, 0) / nvar))
print("   noise share of per-pixel variance: %.0f%%" % (100 * nvar / tot))

# where does signal power cross noise power?
ee, bb = hp.anafast([np.zeros_like(Q), Q, U], lmax=700, pol=True)[1:3]
Npix = hp.nside2npix(512); Nl = 4 * np.pi / Npix * nvar
ell = np.arange(ee.size)
sig_l = 0.5 * (ee + bb)
cross = ell[(ell > 10) & (sig_l < Nl)]
print("   noise C_l = %.3e mK^2 ;  signal falls below it at ell ~ %d  (~%.1f deg)"
      % (Nl, cross[0] if cross.size else -1, 180.0 / (cross[0] if cross.size else 1)))

f_ext = (30.0 / 23e3) ** -2.8
print("\n=== extrapolation 23 GHz -> 30 MHz at beta=-2.8 : x %.2e" % f_ext)
