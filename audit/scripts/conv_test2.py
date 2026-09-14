"""Two decisive follow-ups.

(A) Quadrature-arbitrariness: the same physical integral evaluated on
    grids that differ only by an irrelevant rigid rotation.  If the
    result is a real number, all rotations agree.
(B) Is the RM map itself even determinate at these wavelengths?  The
    Hutschenreuter reconstruction ships a per-pixel std.
"""
import os as _os, pathlib as _p
_ROOT = _p.Path(__file__).resolve().parents[2]
_OUT = _p.Path(__file__).resolve().parent.parent / "generated"
_OUT.mkdir(exist_ok=True)
_os.chdir(_ROOT)          # data/ paths are repo-relative
SP = str(_OUT) + "/"      # generated artefacts land here

import numpy as np, healpy as hp, h5py
from astropy.io import fits

C = 299792458.0
LMAX, NS = 1023, 1024

with h5py.File("data/faraday2020v2.hdf5", "r") as f:
    RM = f["faraday_sky_mean"][:]
    SIG = f["faraday_sky_std"][:]
with fits.open("data/wmap_band_iqumap_r9_9yr_K_v5.fits") as h:
    d = h["Stokes Maps"].data
    Q = hp.reorder(d["Q_POLARISATION"].astype(np.float64), n2r=True)
    U = hp.reorder(d["U_POLARISATION"].astype(np.float64), n2r=True)

alm = {k: hp.map2alm(m, lmax=LMAX, iter=0) for k, m in
       (("RM", RM), ("Q", Q), ("U", U))}

axis0 = np.array([0.3, -0.5, 0.81]); axis0 /= np.linalg.norm(axis0)
lam2 = (C / 30e6) ** 2

print("(A) same integral, quadrature grid rotated by an arbitrary angle")
print("    (rotate the FIELDS and the beam axis together => identical physics)")
vals = []
for k, ang in enumerate([0.0, 0.017, 0.11, 0.7]):
    r = hp.Rotator(rot=[ang, 0.31 * ang, 0.7 * ang], deg=False)
    a = {kk: r.rotate_alm(v.copy()) for kk, v in alm.items()}
    rm = hp.alm2map(a["RM"], NS, lmax=LMAX)
    q = hp.alm2map(a["Q"], NS, lmax=LMAX)
    u = hp.alm2map(a["U"], NS, lmax=LMAX)
    ax = np.asarray(r(axis0))
    vec = np.array(hp.pix2vec(NS, np.arange(hp.nside2npix(NS))))
    mu = ax @ vec
    B = np.where(mu > 0, mu ** 2, 0.0)
    V = (B * (q + 1j * u) * np.exp(2j * rm * lam2)).sum() / B.sum()
    V0 = (B * (q + 1j * u)).sum() / B.sum()
    vals.append(V)
    print("    rot=%.3f rad: |V_far|=%.4e  arg=%+.2f rad   (|V_nofar|=%.4e)"
          % (ang, abs(V), np.angle(V), abs(V0)))
    del rm, q, u, vec, mu, B
v = np.array(vals)
print("    -> spread of |V| across grids: min %.3e max %.3e  (factor %.1f)"
      % (abs(v).min(), abs(v).max(), abs(v).max() / abs(v).min()))
print("    -> scatter/mean of complex V: %.2f" % (np.std(v) / np.abs(np.mean(v))))

print("\n(B) is the RM map determinate?  phase uncertainty 2*sigma_RM*lambda^2")
print("    sigma_RM: med %.1f  p90 %.1f  rad/m^2" % (np.median(SIG), np.percentile(SIG, 90)))
for f_mhz in (10., 30., 50.):
    l2 = (C / (f_mhz * 1e6)) ** 2
    print("    %2.0f MHz: median phase uncertainty = %.1f rad  (= %.1f full turns)"
          % (f_mhz, 2 * np.median(SIG) * l2, 2 * np.median(SIG) * l2 / (2 * np.pi)))
