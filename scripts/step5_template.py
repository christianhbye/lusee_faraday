"""Build the diffuse Faraday delay templates (spec S4.2--S4.3, S4.9).

Per band and per geometry k, the |w|^2-weighted depth distribution of
the RM map, with w = pair beam x polarised emissivity, LST-resolved.
Outputs the normalised template family, the coherence-tilted variant
(S4.4.1), the roll-off knees (plain and plane-tapered, S4.2.1, S4.2.2),
the LST-resolved tail fraction that decides the S4.2.2 gate, and the
amplitude bracket inputs.

Heavy: run in the background under ulimit -v 16000000 with a log in
generated_data/.  ~20-40 min at --lst 128 on the as-built kernel.

Usage:
  uv run python step5_template.py [--arm four-port|two-port]
      [--lst 128] [--bands 30 50 10] [--sigma-eff 9.8]
"""

import argparse

import common  # noqa: F401
import numpy as np

import healpy as hp
from common import DATA_DIR, GEN_DIR, RESPONSE_PATH, load_sky_maps
from lusee_faraday import dispersion as dsp
from lusee_faraday import response as rsp
from lusee_faraday.config import (
    BETA_QU,
    FREQ_REF_QU,
    MAP_NSIDE,
    moon_location,
    times,
)
from lusee_faraday.conventions import lambda_squared

KS = (np.inf, 0.0, -1.0)
COARSE_DPHI = 1.0  # rad/m^2, the display/npz grid


class _TwoPortKernel:
    """Duck-typed kernel for the symmetric pseudo-dipole arm (S4.9)."""

    def __init__(self, path, freq_mhz):
        h_theta, h_phi = rsp.two_port_jones_from_fits(path, freq_mhz)
        maps = rsp.pair_stokes_from_jones(
            h_theta[:, None], h_phi[:, None], pairs=rsp.TWO_PORT_PAIRS
        )[:, 0]
        # sample_periodic_maps requires the duplicated 0/360 column,
        # which two_port_jones_from_fits strips.  Wrap it back on.
        self.K = np.concatenate([maps, maps[..., :1]], axis=-1)
        self.theta_deg = np.arange(self.K.shape[-2], dtype=float)
        self.phi_deg = np.arange(self.K.shape[-1], dtype=float)

    def sample(self, theta_rad, phi_rad):
        return rsp.sample_periodic_maps(
            self.K, self.theta_deg, self.phi_deg, theta_rad, phi_rad
        )


def build_kernel(arm, freq_mhz):
    if arm == "two-port":
        return _TwoPortKernel(
            DATA_DIR / "hfss_lbl_3m_75deg.2port.fits", freq_mhz
        )
    resp = rsp.load_response(RESPONSE_PATH)
    return rsp.FixedChannelKernel(resp, freq_mhz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm", default="four-port", choices=["four-port", "two-port"]
    )
    ap.add_argument("--lst", type=int, default=128)
    ap.add_argument(
        "--bands", type=float, nargs="+", default=[30.0, 50.0, 10.0]
    )
    ap.add_argument("--sigma-eff", type=float, default=9.8)
    args = ap.parse_args()

    maps = load_sky_maps()
    rm = np.asarray(maps["RM"], dtype=float)
    rm_abs = np.abs(rm)
    # Fixed |rm| binning for the tail gate (S6.14, Ruling R19): |rm|
    # never changes, so bin it ONCE and accumulate w2 into those bins
    # per LST with bincount, instead of storing a full depth histogram
    # per LST (320k x 128 x 8 bytes = 328 MB/band of pure addition).
    # For k=inf the template mass above a depth T is exactly the w2
    # weight of pixels with |rm| > T -- no depth histogram needed.
    rm_bin_edges = np.linspace(0.0, rm_abs.max(), 2001)
    rm_idx = np.clip(
        np.searchsorted(rm_bin_edges, rm_abs, side="right") - 1, 0, 1999
    )
    loc = moon_location()
    t_all = times()
    lst_idx = np.linspace(0, len(t_all) - 1, args.lst, dtype=int)
    pix_area = hp.nside2pixarea(MAP_NSIDE)
    _, b_gal = hp.pix2ang(MAP_NSIDE, np.arange(rm.size), lonlat=True)
    taper = np.sin(np.radians(b_gal)) ** 2  # |b| taper, S4.2.1

    theta_grid = np.geomspace(0.2, 30.0, 24)
    D = dsp.structure_function(rm, theta_grid)

    coarse = np.arange(0.0, 2500.0 + COARSE_DPHI, COARSE_DPHI)
    ccent = 0.5 * (coarse[1:] + coarse[:-1])
    nb, nk = len(args.bands), len(KS)
    H_out = np.zeros((nb, nk, ccent.size))
    Hc_out = np.zeros_like(H_out)
    knee = np.zeros((nb, nk))
    knee_taper = np.zeros((nb, nk))
    tail = np.zeros((nb, args.lst))
    theta_cs = np.zeros(nb)
    clamped = np.zeros(nb, dtype=bool)
    bracket = np.zeros((nb, 3))
    w2_accum = np.zeros(rm.size)  # all-band, for w2_mean only

    for ib, band in enumerate(args.bands):
        lam2 = float(lambda_squared(band)[0])
        kernel = build_kernel(args.arm, band)
        p_emis = (
            np.hypot(maps["Q23"], maps["U23"])
            * (band / FREQ_REF_QU) ** BETA_QU
        )
        edges = dsp.phi_edges(band)
        cent = dsp.phi_centers(edges)

        theta_cs[ib] = dsp.coherence_angle(theta_grid, D, lam2)
        lo, hi = np.radians(theta_grid[0]), np.radians(theta_grid[-1])
        clamped[ib] = bool(
            np.isclose(theta_cs[ib], lo) or np.isclose(theta_cs[ib], hi)
        )
        if clamped[ib]:
            print(
                f"WARNING band {band}: theta_c CLAMPED to the sampled "
                f"range at {np.degrees(theta_cs[ib]):.3f} deg -- the true "
                f"coherence angle lies outside [{theta_grid[0]}, "
                f"{theta_grid[-1]}] deg, so N_patch and the amplitude "
                f"bracket are grid-limited, not measured.",
                flush=True,
            )

        Hsum = np.zeros((nk, cent.size))
        Hsum_taper = np.zeros_like(Hsum)
        w2_band = np.zeros(rm.size)  # reset per band
        tail_hist = np.zeros((args.lst, 2000))  # w2 binned by |rm|
        for il, ti in enumerate(lst_idx):
            wb = rsp.pair_weight_maps(kernel, t_all[ti], loc, MAP_NSIDE)
            # pair beam weight (both Faraday branches) x polarised
            # emissivity, squared and pair-summed
            w2 = ((wb * p_emis[None, :]) ** 2).sum(axis=0)
            w2_band += w2
            for ik, k in enumerate(KS):
                H = dsp.depth_distribution(rm, w2, edges, k=k)
                Hsum[ik] += H
                Hsum_taper[ik] += dsp.depth_distribution(
                    rm, w2 * taper, edges, k=k
                )
            tail_hist[il] = np.bincount(rm_idx, weights=w2, minlength=2000)
            print(f"band {band} LST {il + 1}/{args.lst}", flush=True)

        w2_accum += w2_band
        # tail gate (S6.14): the threshold is FIXED per band (the
        # beam-weighted p99 of |RM| over the band's LST-summed weight)
        # while the numerator -- the k=inf template mass beyond it --
        # varies with LST.  Recomputing the percentile per LST from the
        # same w2 that built that LST's template would make the
        # fraction identically 1% by definition of the percentile and
        # measure nothing (Ruling R19).
        p99_band = dsp.weighted_percentiles(rm_abs, w2_band, [99.0])[0]
        above = rm_bin_edges[:-1] > p99_band
        for il in range(args.lst):
            tail[ib, il] = tail_hist[il][above].sum() / tail_hist[il].sum()
        print(
            f"band {band}: p99_band = {p99_band:.3f} rad/m^2, "
            f"tail min {tail[ib].min():.2e} max {tail[ib].max():.2e}",
            flush=True,
        )
        tail_hist = None  # free before the next band

        npatch = dsp.patch_counts(rm, w2_band, edges, theta_cs[ib], pix_area)
        for ik in range(nk):
            pa, Hf = dsp.fold_template(cent, Hsum[ik])
            _, Hft = dsp.fold_template(cent, Hsum_taper[ik])
            _, Hfc = dsp.fold_template(
                cent, dsp.coherence_tilt(Hsum[ik], npatch)
            )
            knee[ib, ik] = dsp.mass_quantile_knee(pa, Hf)
            knee_taper[ib, ik] = dsp.mass_quantile_knee(pa, Hft)
            for target, src in ((H_out, Hf), (Hc_out, Hfc)):
                rb, _ = np.histogram(pa, bins=coarse, weights=src)
                target[ib, ik] = rb / max(rb.sum(), 1e-300)

        wpct_band = dsp.weighted_percentiles(
            rm_abs, w2_band, [50.0, 90.0, 99.0, 99.9]
        )
        omega_beam = w2_band.sum() ** 2 / (w2_band**2).sum() * pix_area
        br = dsp.amplitude_bracket(
            lam2, theta_cs[ib], omega_beam, wpct_band[0], args.sigma_eff
        )
        bracket[ib] = [
            br["upper"],
            br["lower_slab"],
            br["lower_dispersion"],
        ]

    # AFTER the band loop -- matches the w2_mean the envelope script
    # re-weights with
    wpct = dsp.weighted_percentiles(rm_abs, w2_accum, [50.0, 90.0, 99.0, 99.9])

    suffix = "" if args.arm == "four-port" else "_two_port"
    out = GEN_DIR / f"step5_template{suffix}.npz"
    np.savez(
        out,
        phi=ccent,
        H=H_out,
        H_coh=Hc_out,
        ks=np.array([100.0, 0.0, -1.0]),
        bands=np.array(args.bands),
        knee=knee,
        knee_taper=knee_taper,
        tail_frac_lst=tail,
        lst_hours=lst_idx * (27.321661 * 24.0 / 1024.0),
        w2_mean=w2_accum / w2_accum.sum(),
        weighted_percentiles=wpct,
        sigma_eff=args.sigma_eff,
        theta_c=theta_cs,
        theta_c_clamped=clamped,
        bracket=bracket,
    )
    print(f"knees:\n{knee}\nplane-tapered:\n{knee_taper}")
    print(f"tail fraction: min {tail.min():.2e} max {tail.max():.2e}")
    print(
        "theta_c clamp status per band: "
        f"{dict(zip(args.bands, clamped.tolist()))}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
