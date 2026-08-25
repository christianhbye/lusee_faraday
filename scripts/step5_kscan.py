"""Scan the emissivity geometry k continuously (spec S4.2, S4.2.1).

``step5_template.py`` builds the template at three geometries -- the
bracket k -> inf, k = 0, k -> -1 -- because those are the ones the
report quotes.  Three points do not show the SHAPE of the dependence,
and the shape is the robustness claim: the retained power fraction
varies by only ~1.7x over the whole defensible half of the family,
k in [0, inf), against ~5.4x over k in [-0.9, 0].  It is SHALLOW above
k = 0, not flat -- f still climbs 0.27 -> 0.46 at 30 MHz -- but 1.7x
in power is 1.3x in amplitude, which is why the geometry is a ~30%
systematic on detection and the depolarisation floor is a factor 818.
This script fills in between.

It is a RE-ANALYSIS of the committed products, not a re-run.  The
k -> inf column of ``step5_template.npz`` IS the folded, |w|^2-weighted
histogram of |RM| per band, which is exactly the input a pushforward to
any other k needs -- so ``dispersion.pushforward_histogram`` re-casts
it in milliseconds instead of re-running the 20-40 minute template job.
``--validate`` (default on) pins that reconstruction against the stored
k = 0 column and against ``step5_detection.npz``'s retained fraction,
so a silent drift between the scan and the detection table cannot
survive.

Two limits on what the scan answers, both real:

1. It re-casts the geometry ONLY.  The weight |w|^2 and the RM map are
   frozen at whatever built the npz, which is correct -- k does not
   touch either -- but it means this cannot pick up an interaction
   between geometry and beam.  ``step5_template.py --arm two-port`` is
   the leg that varies the beam.
2. The reconstruction inherits the 1 rad/m^2 display binning of
   ``d["phi"]``, not the fine ``phi_edges`` grid the stored templates
   were accumulated on.  Measured cost: KS ~ 1e-3 against the stored
   k = 0 column, against the repo's 1e-2 shape gate.

Cheap: seconds, no data/ artifacts, reads the step5 products only.

Usage:
  uv run python step5_kscan.py [--arm four-port|two-port] [--no-validate]
"""

import argparse

import common  # noqa: F401
import numpy as np

from common import GEN_DIR
from lusee_faraday import dispersion as dsp
from lusee_faraday import noise
from lusee_faraday.config import SIDEREAL_DAY_S
from lusee_faraday.conventions import lambda_squared

# The cuts the report discusses; 27.5 is the window-budget cut that
# every headline number uses.
CUTS = np.array([5.0, 10.0, 27.5, 100.0])
SAFE_CUT = 27.5

# Kept in step with scripts/step5_detection.py: the threshold must be
# recomputed on the TRUNCATED template at every k, never read off the
# full one.  Comparing a cut signal against an uncut threshold is the
# specific error the report records an earlier version making.
TAU_S = SIDEREAL_DAY_S / 1024
ZOOM_ENBW_HZ = 563.4
NIGHT_FRACTION = 0.55
DRIFT_FACTOR = 0.54


def schedule(lunations):
    n_lst = int(round(1024 * min(1.0, NIGHT_FRACTION * lunations)))
    return max(1.0, DRIFT_FACTOR * lunations), max(n_lst, 1)


def detection_ratio(
    phi, H, cut, amplitude, lam2_bins, N, n_coh, n_lst,
    allow_one_sided=False,
):
    """A_bracket * sqrt(f) / A_5sigma for one geometry's template.

    ``phi``/``H`` are the SIGNED distribution: the covariance is the
    complex transform of it (the observable is P = Q + iU), while the
    cut and the retained fraction ``f`` are |phi| quantities.  Folding
    before this point understates A_5sigma by ~18%.
    """
    # |phi| alone: an amplitude floor can amputate one sign of the
    # signed grid and rebuild the folded-template bug (step5_detection).
    sel = np.abs(phi) >= cut
    if sel.sum() < 2:
        return np.nan, np.nan
    f = H[sel].sum() / H.sum()
    S = noise.faraday_signal_covariance(
        phi[sel], H[sel], lam2_bins, allow_one_sided=allow_one_sided
    )
    A = noise.matched_filter_threshold(S, N, n_coh, n_lst)
    return float(amplitude * np.sqrt(f) / A), float(A)


def best_squeeze(phi, H_far, H_fid):
    """Smallest KS between F_0 and F_inf after rescaling the phi axis.

    Answers "is the pushforward just a squeeze of the sightline
    mixture?".  If it were, some scale factor would drive this to zero.
    """
    c_far = np.cumsum(H_far) / H_far.sum()
    c_fid = np.cumsum(H_fid) / H_fid.sum()
    scales = np.linspace(1.0, 8.0, 701)
    ks = [
        np.abs(np.interp(phi, phi / sc, c_far) - c_fid).max() for sc in scales
    ]
    j = int(np.argmin(ks))
    return float(scales[j]), float(ks[j]), float(np.abs(c_far - c_fid).max())


# k grid.  Dense near -1 where f falls off a cliff, coarser above 0
# where it only creeps -- and 0.0 exactly, because it is the fiducial
# and the figure marks it.  The upper end is 20 rather than something
# larger because f approaches k -> inf slowly (still 1.5e-2 short at
# k = 40): the asymptote is drawn as a line, not chased with samples.
K_GRID = np.unique(
    np.concatenate(
        [
            np.linspace(-0.99, -0.5, 60),
            np.linspace(-0.5, 0.0, 40),
            np.linspace(0.0, 2.0, 40),
            np.linspace(2.0, 20.0, 60),
        ]
    )
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm", choices=("four-port", "two-port"), default="four-port"
    )
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    ap.add_argument("--lunations", type=int, default=24)
    args = ap.parse_args()

    suffix = "" if args.arm == "four-port" else "_two_port"
    d = np.load(GEN_DIR / f"step5_template{suffix}.npz")
    phi, bands = d["phi"], d["bands"]
    kf = int(d["k_fiducial_index"]) if "k_fiducial_index" in d.files else 1
    # The k -> inf column is the folded |w|^2-weighted |RM| histogram.
    # It is stored FIRST because KS = (np.inf, 0.0, -1.0); the npz's
    # own "ks" array labels it 100.0, which is a stand-in for inf and
    # not a finite geometry (see step5_template.py).
    ki = int(np.argmax(np.where(np.isfinite(d["ks"]), d["ks"], np.inf)))

    nb, nk, nc = len(bands), K_GRID.size, len(CUTS)
    frac = np.zeros((nb, nc, nk))
    frac_inf = np.zeros((nb, nc))
    knee = np.zeros((nb, nk))
    knee_inf = np.zeros(nb)
    valid = np.full((nb, 3), np.nan)  # KS(recon, stored), f_recon, f_stored
    ratio = np.zeros((nb, nk))  # detection ratio at the slab floor
    ratio_inf = np.zeros(nb)
    squeeze = np.zeros((nb, 3))  # best scale, residual KS, raw KS
    # coherence-tilted: f, detection ratio (FOLDED), knee, and the
    # convention-cancelling factor vs the folded baseline
    tilt = np.zeros((nb, 4))
    n_coh, n_lst = schedule(args.lunations)

    det_path = GEN_DIR / f"step5_detection{suffix}.npz"
    det = np.load(det_path) if det_path.exists() else None

    for ib, band in enumerate(bands):
        far = d["H"][ib, ki]
        # signed twin of ``far``, for every covariance below
        phis = d["phi_signed"]
        far_s = d["H_signed"][ib, ki]
        for ic, cut in enumerate(CUTS):
            frac_inf[ib, ic] = dsp.retained_fraction(phi, far, cut, np.inf)
            for ik, k in enumerate(K_GRID):
                frac[ib, ic, ik] = dsp.retained_fraction(phi, far, cut, k)
        # detection ratio across the family, threshold recomputed per k
        _, bins, W = dsp.zoom_bin_matrix(band)
        N = noise.zoom_noise_covariance(
            W, noise.radiometer_sigma(1.0, ZOOM_ENBW_HZ, TAU_S)
        )
        lam2b = np.asarray(lambda_squared(bins), dtype=float)
        slab = float(d["bracket"][ib, 1])
        ratio_inf[ib], _ = detection_ratio(
            phis, far_s, SAFE_CUT, slab, lam2b, N, n_coh, n_lst
        )
        for ik, k in enumerate(K_GRID):
            ratio[ib, ik], _ = detection_ratio(
                phis,
                dsp.pushforward_signed(phis, far_s, k),
                SAFE_CUT,
                slab,
                lam2b,
                N,
                n_coh,
                n_lst,
            )
        squeeze[ib] = best_squeeze(phi, far, d["H"][ib, kf])
        # The COHERENT-limit shape, for the report's "what this map
        # cannot decide" section.  Quantified so the limitation can be
        # stated with a number instead of drawn as a peer curve: the
        # tilt needs theta_c, which clamps (d["theta_c_clamped"]).
        hc = d["H_coh"][ib, kf]
        tilt[ib, 0] = hc[phi >= SAFE_CUT].sum() / hc.sum()
        # The tilt is a FOLDED shape statement and has no signed twin
        # stored; its threshold is therefore indicative only, which is
        # all the report uses it for (it enters no verdict).
        tilt[ib, 1], _ = detection_ratio(
            phi, hc, SAFE_CUT, slab, lam2b, N, n_coh, n_lst,
            allow_one_sided=True,
        )
        tilt[ib, 2] = phi[np.searchsorted(np.cumsum(hc) / hc.sum(), 0.9)]
        # The tilt ratio above is FOLDED, so it must not be compared
        # against the signed detection table directly -- that would
        # conflate the coherence tilt with the folded/signed error.
        # Quote it as a FACTOR against the folded baseline instead,
        # where the convention cancels.
        hf = d["H"][ib, kf]
        base_folded, _ = detection_ratio(
            phi, hf, SAFE_CUT, slab, lam2b, N, n_coh, n_lst,
            allow_one_sided=True,
        )
        tilt[ib, 3] = base_folded / max(tilt[ib, 1], 1e-300)
        knee_inf[ib] = dsp.mass_quantile_knee(phi, far)
        for ik, k in enumerate(K_GRID):
            knee[ib, ik] = dsp.mass_quantile_knee(
                phi, dsp.pushforward_histogram(phi, far, k)
            )

        print(f"\n=== {band:.0f} MHz")
        print(
            f"  knee: k->inf {knee_inf[ib]:8.2f}   k=0 "
            f"{knee[ib, int(np.argmin(np.abs(K_GRID)))]:8.2f} rad/m^2"
        )
        ics = int(np.argmin(np.abs(CUTS - SAFE_CUT)))
        for k in (-0.9, -0.5, 0.0, 1.0, 4.0, 12.0):
            ik = int(np.argmin(np.abs(K_GRID - k)))
            print(
                f"  f(k={K_GRID[ik]:+5.2f}, cut {SAFE_CUT}) = "
                f"{frac[ib, ics, ik]:.4f}"
            )
        print(f"  f(k-> inf, cut {SAFE_CUT}) = {frac_inf[ib, ics]:.4f}")
        print(
            f"  detection ratio at the slab floor: k=0 "
            f"{ratio[ib, int(np.argmin(np.abs(K_GRID)))]:.2f}  "
            f"k->inf {ratio_inf[ib]:.2f}"
        )
        print(
            f"  coherence-tilted (theta_c clamped="
            f"{bool(d['theta_c_clamped'][ib])}): f {tilt[ib, 0]:.4f} "
            f"ratio {tilt[ib, 1]:.2f} knee {tilt[ib, 2]:.1f}"
        )
        print(
            f"  best squeeze of F_inf onto F_0: scale {squeeze[ib, 0]:.2f}, "
            f"residual KS {squeeze[ib, 1]:.3f} (raw {squeeze[ib, 2]:.3f})"
        )

        if args.validate:
            recon = dsp.pushforward_histogram(phi, far, 0.0)
            stored = d["H"][ib, kf]
            c = lambda h: np.cumsum(h) / h.sum()  # noqa: E731
            ks = float(np.abs(c(recon) - c(stored)).max())
            f_recon = dsp.retained_fraction(phi, far, SAFE_CUT, 0.0)
            f_stored = (
                float(
                    det["power_fraction"][
                        ib, int(np.argmin(np.abs(det["cuts"] - SAFE_CUT)))
                    ]
                )
                if det is not None
                else float(stored[phi >= SAFE_CUT].sum() / stored.sum())
            )
            valid[ib] = [ks, f_recon, f_stored]
            src = "step5_detection.npz" if det is not None else "stored H"
            print(
                f"  VALIDATE k=0: KS(recon, stored) = {ks:.2e} "
                f"(gate 1e-2);  f {f_recon:.4f} vs {f_stored:.4f} "
                f"({src}), diff {abs(f_recon - f_stored):.1e}"
            )
            if ks > 1e-2:
                raise SystemExit(
                    f"band {band}: reconstruction KS {ks:.2e} exceeds the "
                    "1e-2 shape gate -- the scan does not reproduce the "
                    "stored template and must not be published"
                )

    out = GEN_DIR / f"step5_kscan{suffix}.npz"
    np.savez(
        out,
        bands=bands,
        ks=K_GRID,
        cuts=CUTS,
        safe_cut=SAFE_CUT,
        power_fraction=frac,
        power_fraction_kinf=frac_inf,
        knee=knee,
        knee_kinf=knee_inf,
        validation=valid,
        ratio_slab=ratio,
        ratio_slab_kinf=ratio_inf,
        squeeze=squeeze,
        tilt=tilt,
        theta_c=d["theta_c"],
        theta_c_clamped=d["theta_c_clamped"],
        lunations=args.lunations,
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
