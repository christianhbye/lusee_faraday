"""Detection SNR against the low-depth systematics cut (spec S4.10).

The question this answers is DETECTION -- is there any evidence of
Faraday rotation -- not localisation.  Those need different statistics
and they give very different answers, and an earlier version of the
write-up quoted a localisation verdict as if it were the detection one.

For each band and each cut ``phi >= phi_min``:

  * ``f``    -- the fraction of template POWER retained by the cut;
  * ``A_mf`` -- the 5-sigma whitened matched-filter threshold computed
    on the TRUNCATED template, not the full one.  This is the part the
    tail gate got wrong: if a statistic only looks above the cut, the
    matched filter only has the truncated shape to work with, and its
    threshold is higher (28-35% higher at the p99 cut).
  * the ratio ``A_bracket * sqrt(f) / A_mf`` against both usable
    bracket floors.  ``sqrt(f)`` and not ``f`` because ``f`` is a
    fraction of power and the bracket is an amplitude.

The cut exists because instrumental ``I -> Q,U`` leakage is spectrally
smooth and therefore sits at ``phi ~ 0``.  Cutting it away is what
breaks the degeneracy between a Faraday detection and a systematic.

BASIS INDEPENDENCE.  ``tau_FD = 2 phi c^2 / (pi nu^3)`` is monotonic in
``phi``, so every cut here is exactly a cut in delay and the retained
power fraction is identical.  Nothing in this script prefers one basis;
the ``tau_us`` column is the same cut expressed in delay.

The LST-averaged numbers are the headline, because a full-mission
detection coadds every LST bin -- ``schedule`` already carries that as
``n_lst``.  The LST-resolved leg exists for the LOCALISATION statistic
(the tail gate), where Galactic Centre transit matters because the tail
is 2-4x its LST-mean there; it needs ``H_lst`` in the template npz.

Cheap: seconds, no data/ artifacts, reads the step5 products only.

Usage:
  uv run python step5_detection.py [--lunations 24] [--arm four-port]
"""

import argparse

import common  # noqa: F401
import numpy as np

from common import GEN_DIR
from lusee_faraday import dispersion as dsp
from lusee_faraday import noise
from lusee_faraday.config import SIDEREAL_DAY_S
from lusee_faraday.conventions import lambda_squared

C_LIGHT = 299792458.0
TAU_S = SIDEREAL_DAY_S / 1024
ZOOM_ENBW_HZ = 563.4
NIGHT_FRACTION = 0.55
DRIFT_FACTOR = 0.54

# The window-budget cut (docs section "The window budget on the
# Faraday-depth axis"): beyond phi = 27.5 a phi ~ 0 foreground at
# |P|/I = 0.15 leaks <= 1e-6 through BH4.  The others bracket it.
CUTS = np.array([0.0, 5.0, 10.0, 27.5, 50.0, 100.0, 200.0, 400.0])


def schedule(lunations):
    """(coherent nights per LST bin, LST bins covered); see S4.10."""
    n_lst = int(round(1024 * min(1.0, NIGHT_FRACTION * lunations)))
    n_coh = max(1.0, DRIFT_FACTOR * lunations)
    return n_coh, max(n_lst, 1)


def delay_of_depth(phi, band_mhz):
    """tau_FD in seconds: the delay a depth phi sits at, at band_mhz.

    d(phase)/d(nu) of exp(2i phi lambda^2) against exp(-2 pi i nu tau).
    Verified against the refuted Step 4's own table, which quotes
    5 ms at 30 MHz and 1.1 ms at 50 MHz for phi_max ~ 2400.
    """
    return 2.0 * np.asarray(phi) * C_LIGHT**2 / (np.pi * (band_mhz * 1e6) ** 3)


def threshold_for(H, phi, sel, lam2_bins, N, n_coh, n_lst):
    """5-sigma threshold for the template restricted to ``sel``.

    ``sel`` must select on |phi| ALONE.  A relative-amplitude floor
    (``H > H.max()*1e-6``) is fine for the folded template that sets
    ``f``, but on the signed grid it can drop every bin of one sign --
    measured at two-port 50 MHz above the p99 cut -- which silently
    rebuilds the folded-template bug the signed grid exists to fix.
    Zero-weight bins contribute exactly nothing to S, so there is
    nothing to gain by excluding them.

    ``phi``/``H`` must be the SIGNED depth distribution, not the folded
    one.  The observable is the complex P = Q + iU, so the frequency
    covariance is the complex transform of the signed distribution;
    feeding it the folded template models a sky whose every column has
    one sign of RM and understates this threshold by ~18%.  The cut is
    still applied on |phi|, and the retained power fraction ``f`` is
    still a folded quantity -- only the covariance reads the sign.
    """
    S = noise.faraday_signal_covariance(phi[sel], H[sel], lam2_bins)
    return noise.matched_filter_threshold(S, N, n_coh, n_lst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lunations", type=int, default=24)
    ap.add_argument(
        "--arm", choices=("four-port", "two-port"), default="four-port"
    )
    args = ap.parse_args()

    suffix = "" if args.arm == "four-port" else "_two_port"
    d = np.load(GEN_DIR / f"step5_template{suffix}.npz")
    phi, bands = d["phi"], d["bands"]
    kf = int(d["k_fiducial_index"]) if "k_fiducial_index" in d.files else 1
    has_lst = "H_lst" in d.files
    n_coh, n_lst = schedule(args.lunations)
    print(
        f"arm {args.arm}, {args.lunations} lunations: "
        f"n_coh={n_coh:.2f}, n_lst={n_lst}"
    )
    if not has_lst:
        print(
            "NOTE: no H_lst in the npz -- the LST-resolved (transit) "
            "leg is skipped. Re-run step5_template.py to add it."
        )

    nb, nc = len(bands), len(CUTS)
    frac = np.zeros((nb, nc))
    a_mf = np.zeros((nb, nc))
    tau_us = np.zeros((nb, nc))
    ratio_slab = np.zeros((nb, nc))
    ratio_disp = np.zeros((nb, nc))
    tail_corrected = np.zeros((nb, 3))  # f, A_mf(tail), ratio at slab

    for ib, band in enumerate(bands):
        _, bins, W = dsp.zoom_bin_matrix(band)
        N = noise.zoom_noise_covariance(
            W, noise.radiometer_sigma(1.0, ZOOM_ENBW_HZ, TAU_S)
        )
        lam2b = np.asarray(lambda_squared(bins), dtype=float)
        H = d["H"][ib, kf]
        # folded for f and the cut; signed for the covariance (S4.10)
        phis, Hs = d["phi_signed"], d["H_signed"][ib, kf]
        total = H.sum()
        slab, disp = d["bracket"][ib, 1], d["bracket"][ib, 2]
        floor_rel = 1e-6  # numerical, matches step5_sensitivity
        floor = H.max() * floor_rel
        print(f"\n=== {band:.0f} MHz  A_slab={slab:.3e}  A_disp={disp:.3e}")
        print("  phi cut      tau      keeps      A_mf      slab     disp")
        for ic, cut in enumerate(CUTS):
            sel = (phi >= cut) & (H > floor)
            if sel.sum() < 2:
                continue
            f = H[sel].sum() / total
            sels = np.abs(phis) >= cut
            A = threshold_for(Hs, phis, sels, lam2b, N, n_coh, n_lst)
            frac[ib, ic], a_mf[ib, ic] = f, A
            tau_us[ib, ic] = delay_of_depth(cut, band) * 1e6
            ratio_slab[ib, ic] = slab * np.sqrt(f) / A
            ratio_disp[ib, ic] = disp * np.sqrt(f) / A
            print(
                f"  {cut:7.1f} {tau_us[ib, ic]:8.1f}us {f:9.4f} "
                f"{A:10.3e} {ratio_slab[ib, ic]:9.2f} "
                f"{ratio_disp[ib, ic]:8.4f}"
            )

        # The localisation (tail) statistic, done consistently: the
        # GC-transit power fraction against a threshold computed on the
        # transit template restricted to the same p99 cut.
        p99 = d["weighted_percentiles"][2]
        if has_lst:
            il = int(np.argmax(d["tail_frac_lst"][ib]))
            Ht = d["H_lst"][ib, il]
            selt = (phi >= p99) & (Ht > Ht.max() * 1e-6)
            ft = Ht[selt].sum() / Ht.sum()
            Hts = d["H_lst_signed"][ib, il]
            selts = np.abs(phis) >= p99
            At = threshold_for(Hts, phis, selts, lam2b, N, n_coh, n_lst)
            src = f"LST bin {il} (transit)"
        else:
            selts = np.abs(phis) >= p99
            ft = float(d["tail_frac_lst"][ib].max())
            At = threshold_for(Hs, phis, selts, lam2b, N, n_coh, n_lst)
            src = "LST-averaged shape, transit f (approximate)"
        tail_corrected[ib] = [ft, At, slab * np.sqrt(ft) / At]
        print(
            f"  tail gate, {src}: f={ft:.4f} A_mf={At:.3e} "
            f"ratio at slab floor = {tail_corrected[ib, 2]:.2f}"
        )

    out = GEN_DIR / f"step5_detection{suffix}.npz"
    np.savez(
        out,
        bands=bands,
        cuts=CUTS,
        tau_us=tau_us,
        power_fraction=frac,
        a_mf=a_mf,
        ratio_slab=ratio_slab,
        ratio_dispersion=ratio_disp,
        tail_corrected=tail_corrected,
        lunations=args.lunations,
        lst_resolved_tail=has_lst,
        bracket=d["bracket"],
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
