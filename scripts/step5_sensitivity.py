"""The detectability threshold (spec S4.10): matched filter + schedule.

The whitened matched filter on the zoom-bin covariance is the
deliverable; the corrected closed form is printed as the sanity check.
The schedule model carries the night-fraction and sidereal-drift
corrections (S4.10): one lunation covers ~55% of LST bins, and a given
LST bin is dark in ~0.54 of lunations.

Units (spec S4.10, R24): ``A_mf`` and ``A_closed`` are dimensionless
fractional amplitudes relative to T_sys -- ``noise.radiometer_sigma``
is called with ``T_sys=1.0``, so ``sigma_bin``, and everything derived
from it, is measured in units of T_sys.  ``step5_template.npz``'s
``bracket`` column is a fractional polarized amplitude referred to the
SKY (P/I).  These are two different reference temperatures: comparing
``A_mf``/``A_closed`` against ``bracket`` on one axis is only valid
after multiplying ``A_mf``/``A_closed`` by T_sys/T_sky (the
``tsys_over_tsky`` ratio this script also computes).  This script does
NOT apply that rescaling -- it reports the ratio and leaves the
correction to whichever task plots the two together.

T_sys/T_sky (Ruling R28): T_sys = T_sky + T_loading + T_amp for a
sky-noise-dominated radiometer, so the ratio is ``1 + r + t_amp/t_sky``,
where ``r`` is the luseepy loading model's Moon+antenna-loss
contribution (250 K each) referenced to the sky via a 1 K blackbody,
and ``t_sky`` is the mean Haslam-scaled sky temperature
(``common.sky_at_freq``, which includes the T_CMB offset).

TWO LIMITS ON WHAT THAT RATIO ANSWERS, and both must travel with it:

1. Amplifier noise is NOT in this chain, and ``--t-amp`` defaults to
   0, so what is printed is ``1 + T_loading/T_sky`` and nothing else.
   Spec S4.10's risk item -- "sky domination must be computed, not
   asserted; for a short mismatched dipole on regolith it is not
   automatic at 50 MHz" -- is precisely about the receiver term this
   sets to zero.  1.0044 / 1.0170 / 1.0006 is therefore a genuine
   LOWER BOUND on T_sys/T_sky, not a computed sky-domination result
   (``r`` alone was a vacuous bound: it omits the sky's own
   contribution to its own system temperature).  Re-run with
   ``--t-amp <T_rx>`` once the collaboration supplies one.
2. ``t_sky = mean(I_sky)`` is an ALL-SKY mean, not a beam-weighted
   antenna temperature.  The beam is not uniform, so the antenna
   temperature the receiver actually sees differs from it; nothing
   here weights the sky by ``|w(n)|^2``.
"""

import argparse
from pathlib import Path

import common  # noqa: F401
import numpy as np

from common import GEN_DIR, RESPONSE_PATH, load_sky_maps, sky_at_freq
from lusee_faraday import dispersion as dsp
from lusee_faraday import noise
from lusee_faraday.config import SIDEREAL_DAY_S
from lusee_faraday.conventions import lambda_squared

TAU_S = SIDEREAL_DAY_S / 1024
COH_BW_HZ = {50.0: 50176.0, 30.0: 10838.0, 10.0: 401.0}
NIGHT_FRACTION = 0.55
DRIFT_FACTOR = 0.54


def schedule(lunations):
    """(coherent nights per LST bin, LST bins covered)."""
    n_lst = int(round(1024 * min(1.0, NIGHT_FRACTION * lunations)))
    n_coh = max(1.0, DRIFT_FACTOR * lunations)
    return n_coh, max(n_lst, 1)


def template_for(band):
    """(phi, normalized H, bracket row, theta_c_clamped).

    ``theta_c_clamped`` is ``None`` for the standalone fallback (it
    carries no coherence-angle statement at all); otherwise it is the
    bool read straight from ``step5_template.npz``.  When it is True,
    the bracket's UPPER entry -- and only that one -- is a clamped
    quantity, and the clamp OVERSTATES it: see the docstrings of
    ``dispersion.coherence_angle`` and ``dispersion.amplitude_bracket``.
    ``lower_slab`` and ``lower_dispersion`` are theta_c-free closed
    forms and are unaffected.
    """
    f = GEN_DIR / "step5_template.npz"
    if f.exists():
        d = np.load(f)
        ib = int(np.argmin(np.abs(d["bands"] - band)))
        clamped = bool(d["theta_c_clamped"][ib])
        # Folded AND signed: |phi| statistics use the first, the
        # matched-filter covariance the second (S4.10).
        return (
            d["phi"],
            d["H"][ib, 1],
            d["phi_signed"],
            d["H_signed"][ib, 1],
            d["bracket"][ib],
            clamped,
        )
    # standalone fallback: uniform slab to the map median depth
    phi = np.arange(0.0, 2500.0, 1.0)
    H = np.where(phi < 18.4, 1.0, 0.0)
    phis = np.arange(-2500.0, 2500.0, 1.0)
    Hs = np.where(np.abs(phis) < 18.4, 1.0, 0.0)
    return (
        phi,
        H / H.sum(),
        phis,
        Hs / Hs.sum(),
        np.array([5.4e-4, 1.0e-4, 5.2e-7]),
        None,
    )


def tsys_over_tsky(band, t_amp_k):
    try:
        from lusee.ReceiverImpedance import JFETReceiver

        from lusee_faraday import instrument, response as rsp

        resp = rsp.load_response(RESPONSE_PATH)
        receiver = JFETReceiver()
        bb = instrument.blackbody_normalization(
            resp, receiver, np.array([band]), impedance_freq_mhz=band
        )
        load = instrument.covariance(
            np.zeros((1, 1, 10)),
            resp,
            receiver,
            np.array([band]),
            T_moon=250.0,
            T_ant=250.0,
            impedance_freq_mhz=band,
        )
        I_sky, _, _ = sky_at_freq(load_sky_maps(), band)
        t_sky = float(np.mean(I_sky))
        r = np.nanmean(
            np.abs(np.diagonal(load[0, 0]))
            / (np.abs(np.diagonal(bb[0])) * t_sky)
        )
        return float(1.0 + r + t_amp_k / t_sky)
    except Exception as e:  # artifact or receiver model unavailable
        if Path(RESPONSE_PATH).exists():
            # The artifact IS present -- a failure here is a bug, not
            # a legitimate fallback (R23): do not paper over it with
            # nan.
            raise
        print(
            f"T_sys/T_sky at {band} MHz unavailable "
            f"({type(e).__name__}): {e}"
        )
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bands", type=float, nargs="+", default=[30.0, 50.0, 10.0]
    )
    ap.add_argument("--lunations", type=int, default=24)
    ap.add_argument("--t-amp", type=float, default=0.0)
    args = ap.parse_args()

    lun = np.arange(1, args.lunations + 1)
    A_mf = np.zeros((len(args.bands), lun.size))
    A_cf = np.zeros_like(A_mf)
    bracket = np.zeros((len(args.bands), 3))
    ratios = np.zeros(len(args.bands))

    for ib, band in enumerate(args.bands):
        fine, bins, W = dsp.zoom_bin_matrix(band)
        sigma_bin = noise.radiometer_sigma(1.0, 563.4, TAU_S)
        N = noise.zoom_noise_covariance(W, sigma_bin)
        phi, H, phis, Hs, bracket[ib], clamped = template_for(band)
        if clamped:
            print(
                f"{band:.0f} MHz: theta_c is CLAMPED at the grid's low "
                f"edge, which OVERSTATES it -- bracket['upper'] "
                f"({bracket[ib, 0]:.2e}) is an upper bound not "
                f"computable from this map, not a measurement; the two "
                f"lower ends are theta_c-free and stand"
            )
        lam2b = np.asarray(lambda_squared(bins), dtype=float)
        # SIGNED template: the observable is the complex P = Q + iU, so
        # S is the complex transform of the signed depth distribution.
        # The folded template understates A_5sigma by ~18% (S4.10).
        # no amplitude floor: it can amputate one sign of the signed
        # grid (see step5_detection.threshold_for)
        keep = np.ones(Hs.size, dtype=bool)
        S = noise.faraday_signal_covariance(phis[keep], Hs[keep], lam2b)
        n_modes_cf = 75000.0 / COH_BW_HZ[band]
        for j, L in enumerate(lun):
            n_coh, n_lst = schedule(int(L))
            A_mf[ib, j] = noise.matched_filter_threshold(S, N, n_coh, n_lst)
            A_cf[ib, j] = noise.closed_form_threshold(
                COH_BW_HZ[band], TAU_S, n_modes_cf * n_lst, n_coh
            )
        ratios[ib] = tsys_over_tsky(band, args.t_amp)
        print(
            f"{band:.0f} MHz: A(1 lun) mf {A_mf[ib, 0]:.2e} "
            f"closed {A_cf[ib, 0]:.2e}; A({args.lunations} lun) mf "
            f"{A_mf[ib, -1]:.2e} closed {A_cf[ib, -1]:.2e}; "
            f"T_sys/T_sky >= {ratios[ib]:.5f}"
        )

    np.savez(
        GEN_DIR / "step5_sensitivity.npz",
        lunations=lun,
        A_mf=A_mf,
        A_closed=A_cf,
        bands=np.array(args.bands),
        bracket=bracket,
        tsys_over_tsky=ratios,
    )
    print(f"wrote {GEN_DIR / 'step5_sensitivity.npz'}")


if __name__ == "__main__":
    main()
