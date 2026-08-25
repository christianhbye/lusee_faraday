"""Precompute the intuition figures' inputs (spec S5).

The report's build-up needs three things the committed products do not
carry: toy skies to show what the depth distribution does to the
channel covariance, an RM-scale scan to show where the refuted coherent
sum is and is not valid, and the input-sensitivity variants.  All of
them need ``data/`` -- the RM map, its uncertainty, the WMAP
polarization.  ``scripts/step5_plots.py`` is deliberately data-free so
that every figure regenerates from the npz products alone, so the
data-dependent part lives here and is written to
``generated_data/step5_intuition.npz``.

Everything here is for FIGURES.  Nothing downstream computes a
published number from it, and the toy skies are deliberately
unphysical: an alternating-pixel map and per-pixel iid Gaussian draws
are absurd as skies but exactly right as depth DISTRIBUTIONS, because
F depends on the map only through the joint distribution of (weight,
depth) over pixels and never on the spatial arrangement.

Two conventions carried through, both of which cost real numbers when
they were got wrong:

* ``S`` is built from the SIGNED depth distribution, never the folded
  one (``noise.faraday_signal_covariance`` now refuses the folded form).
  The toy single screens are genuinely one-sided physical cases and use
  the documented opt-out.
* The panels that show a template use the FOLDED form, because that is
  how the report presents every |phi| statistic and it is what a log
  axis can hold.  Fold for shape, keep the sign for the covariance.

Needs ``data/`` but not the 631 MB response artifact.  ~1 min.

Usage:
  uv run python step5_intuition.py [--band 30]
"""

import argparse

import common  # noqa: F401
import numpy as np

import h5py
import healpy as hp
from common import DATA_DIR, GEN_DIR, load_sky_maps
from lusee_faraday import dispersion as dsp
from lusee_faraday import noise
from lusee_faraday.config import FREQ_REF_QU
from lusee_faraday.conventions import lambda_squared
from lusee_faraday.config import SIDEREAL_DAY_S

NSIDES = np.array([64, 128, 256, 512])
SCALES = np.array([1e-5, 1e-3, 1e-2, 1e-1, 1.0])
COARSE = np.arange(0.0, 2500.0 + 1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=float, default=30.0)
    args = ap.parse_args()
    band = args.band

    maps = load_sky_maps()
    rm = np.asarray(maps["RM"], dtype=float)
    with h5py.File(DATA_DIR / "faraday2020v2.hdf5", "r") as f:
        rm_std = np.asarray(f["faraday_sky_std"][:], dtype=float)
    pol = np.hypot(maps["Q23"], maps["U23"])

    d = np.load(GEN_DIR / "step5_template.npz")
    w2 = d["w2_mean"]
    w2_two = np.load(GEN_DIR / "step5_template_two_port.npz")["w2_mean"]
    edges = dsp.phi_edges(band)
    cent = dsp.phi_centers(edges)
    _, bins, W = dsp.zoom_bin_matrix(band)
    lam2 = np.asarray(lambda_squared(bins), dtype=float)
    lam2_0 = float(lambda_squared(band)[0])
    ccent = 0.5 * (COARSE[1:] + COARSE[:-1])
    rng = np.random.default_rng(0)

    def folded(H):
        pa, Hf = dsp.fold_template(cent, H)
        rb, _ = np.histogram(pa, bins=COARSE, weights=Hf)
        return rb / max(rb.sum(), 1e-300)

    def srow(phis, H, one_sided=False):
        S = noise.faraday_signal_covariance(
            phis, H, lam2, allow_one_sided=one_sided
        )
        return np.abs(S[0])

    # ---- toy skies: same beam, four depth distributions, two geometries
    skies = [
        ("constant RM map", np.full_like(rm, 50.0)),
        ("two-valued map",
         np.where(np.arange(rm.size) % 2 == 0, 30.0, 70.0)),
        ("Gaussian RM map", rng.normal(60.0, 15.0, rm.size)),
        ("Hutschenreuter", rm),
    ]
    toy_H = np.zeros((len(skies), 2, ccent.size))
    toy_S = np.zeros((len(skies), 2, len(bins)))
    toy_cv = np.zeros(len(skies))
    toy_var = np.zeros((len(skies), 2))
    for i, (_, x) in enumerate(skies):
        for j, k in enumerate((np.inf, 0.0)):
            H = dsp.depth_distribution(x, w2, edges, k=k)
            toy_H[i, j] = folded(H)
            toy_S[i, j] = srow(cent, H)
        mu = np.average(x, weights=w2)
        ex2 = np.average(x**2, weights=w2)
        toy_var[i] = [ex2 - mu**2, ex2 / 3.0 - mu**2 / 4.0]
        toy_cv[i] = np.sqrt(max(toy_var[i, 0], 0.0)) / abs(mu)

    # ---- how far apart neighbouring pixels actually are in depth
    # The random-walk argument of report sec:randomwalk rests on this
    # number, and it was quoted as "order 1e3 rad/m^2" -- wrong by ~3
    # decades.  Hutschenreuter is a SMOOTH reconstruction: neighbours
    # differ by ~3% of the median |RM|.  The conclusion survives (a
    # median pair is still many turns apart at LuSEE wavelengths) but
    # the number must be measured, not asserted.
    # healpy returns -1 for absent neighbours; indexing with -1 wraps
    # silently, so those are filtered rather than trusted.
    pix = rng.integers(0, rm.size, 300_000)
    nbr = hp.get_all_neighbours(hp.npix2nside(rm.size), pix)
    dd = np.concatenate([
        np.abs(rm[pix[nbr[j] >= 0]] - rm[nbr[j][nbr[j] >= 0]])
        for j in range(8)
    ])
    adj_med = float(np.median(dd))
    adj_p90 = float(np.percentile(dd, 90.0))
    # turns of Faraday phase between a median neighbour pair, per band
    adj_turns = np.array([
        2.0 * adj_med * float(lambda_squared(bb)[0]) / (2.0 * np.pi)
        for bb in (10.0, 30.0, 50.0)
    ])
    # where the coherent pixel sum WOULD be valid: 2*dRM*lam2 < 1
    coh_freq_mhz = 299.792458 / np.sqrt(1.0 / (2.0 * adj_med))

    # ---- the refuted coherent sum, against pixelisation and RM scale
    coh = np.zeros((SCALES.size, NSIDES.size))
    scan_H = np.zeros((SCALES.size, ccent.size))
    scan_S = np.zeros((SCALES.size, len(bins)))
    for i, s in enumerate(SCALES):
        x = rm * s
        for j, n in enumerate(NSIDES):
            r = x if n == 512 else hp.ud_grade(x, n)
            w = w2 if n == 512 else hp.ud_grade(w2, n)
            # alpha = 0: MAXIMALLY coherent, so any decay is the Faraday
            # phase decorrelating and not the intrinsic angles.
            coh[i, j] = abs((w * np.exp(2j * r * lam2_0)).sum()) / w.sum()
        H = dsp.depth_distribution(x, w2, edges, k=0.0)
        scan_H[i] = folded(H)
        scan_S[i] = srow(cent, H)

    # ---- input sensitivity of S
    Hfid = dsp.depth_distribution(rm, w2, edges, k=0.0)
    S_fid = srow(cent, Hfid)
    var_names = ["k -> inf", "RM map +1 sigma", "two-port beam"]
    S_var = np.stack([
        srow(cent, dsp.depth_distribution(rm, w2, edges, k=np.inf)),
        srow(cent, dsp.depth_distribution(
            rm + rm_std * rng.normal(size=rm.size), w2, edges, k=0.0)),
        srow(cent, dsp.depth_distribution(rm, w2_two, edges, k=0.0)),
    ])
    # The WMAP spectral index cannot appear: (nu/nu_ref)^beta is a
    # SPATIALLY UNIFORM rescaling of the emissivity, so it multiplies w2
    # by a constant and cancels in the normalised template.  Recorded as
    # an exact zero rather than sampled.
    beta_shift = float(
        np.abs(srow(cent, dsp.depth_distribution(
            rm, w2 * (30.0 / FREQ_REF_QU) ** (2 * 0.2), edges, k=0.0))
            - S_fid).max()
    )

    # ---- the Crab: one compact source dominating the weighting
    # The ~20 rad/m^2 feature in the k -> inf curve is NOT instrumental
    # and NOT an extended region: it is Taurus A, whose pixel holds the
    # all-sky maximum of the WMAP K polarized intensity and sits at
    # |RM| ~ 20 in this map.  Quantified so the report can state the
    # contamination and its bound instead of masking, which belongs
    # with the sky model.
    nside = hp.npix2nside(w2.size)
    vcrab = hp.ang2vec(184.557, -5.784, lonlat=True)
    disc1 = hp.query_disc(nside, vcrab, np.radians(1.0))
    crab = np.array([
        100.0 * disc1.size / w2.size,          # % of sky
        100.0 * w2[disc1].sum() / w2.sum(),    # % of all weight
        pol[disc1].mean() / pol.mean(),        # brightness contrast
    ])
    srt = np.sort(w2)[::-1]
    n01 = max(1, int(1e-4 * srt.size))
    conc = 100.0 * srt[:n01].sum() / srt.sum()

    def excess_at(weight, phi0=20.5, nsm=21):
        h, _ = np.histogram(np.abs(rm), bins=COARSE, weights=weight)
        h = h / h.sum()
        sm = np.convolve(h, np.ones(nsm) / nsm, mode="same")
        i = int(np.argmin(np.abs(ccent - phi0)))
        return h[i] / max(sm[i], 1e-300)

    disc5 = hp.query_disc(nside, vcrab, np.radians(5.0))
    w_nocrab = w2.copy()
    w_nocrab[disc5] = 0.0
    crab_excess = np.array([excess_at(w2), excess_at(w_nocrab)])

    # its bounded effect on the deliverable, fiducial geometry
    N = noise.zoom_noise_covariance(
        W, noise.radiometer_sigma(1.0, 563.4, SIDEREAL_DAY_S / 1024)
    )
    n_coh, n_lst = max(1.0, 0.54 * 24), int(round(1024 * min(1.0, 0.55 * 24)))
    slab = float(d["bracket"][0, 1])

    def ratio_and_knee(weight):
        H = dsp.depth_distribution(rm, weight, edges, k=0.0)
        sel = np.abs(cent) >= 27.5
        frac = H[sel].sum() / H.sum()
        S = noise.faraday_signal_covariance(cent[sel], H[sel], lam2)
        A = noise.matched_filter_threshold(S, N, n_coh, n_lst)
        Hf = folded(H)
        return (slab * np.sqrt(frac) / A,
                ccent[np.searchsorted(np.cumsum(Hf), 0.90)])

    crab_deliv = np.array([ratio_and_knee(w2), ratio_and_knee(w_nocrab)])

    # ---- sigma_eff broadens F too, not just the amplitude
    # The dispersion floor asserts turbulence of dispersion sigma_eff
    # co-spatial with the emission, but F never sees it.  Convolving the
    # template with that dispersion measures the size of the
    # inconsistency.  A uniform kernel is an APPROXIMATION: for
    # co-spatial turbulence the scatter accumulates along the path and
    # should grow with f rather than apply equally at every depth.
    sig_eff = float(d["sigma_eff"])
    dphi_c = cent[1] - cent[0]
    sel0 = np.abs(cent) >= 27.5

    def ratio_of(Hx):
        frac = Hx[sel0].sum() / Hx.sum()
        S = noise.faraday_signal_covariance(cent[sel0], Hx[sel0], lam2)
        A = noise.matched_filter_threshold(S, N, n_coh, n_lst)
        return slab * np.sqrt(frac) / A

    base_ratio = ratio_of(Hfid)
    gg = np.arange(-6 * sig_eff, 6 * sig_eff + dphi_c, dphi_c)
    sigma_broaden = []
    for mult in (1.0, 2.0):
        ker = np.exp(-0.5 * (gg / (mult * sig_eff)) ** 2)
        ker /= ker.sum()
        sigma_broaden.append(
            100.0 * (ratio_of(np.convolve(Hfid, ker, mode="same"))
                     / base_ratio - 1.0)
        )
    sigma_broaden = np.array(sigma_broaden)

    # ---- how many independent modes the matched filter really has
    sel = np.abs(cent) >= 27.5
    Sfull = noise.faraday_signal_covariance(cent[sel], Hfid[sel], lam2)
    F = np.linalg.solve(N, Sfull)
    ev = np.linalg.eigvals(F).real
    n_eff = float(ev.sum() ** 2 / np.sum(ev**2))
    Fd = np.linalg.solve(np.diag(np.diag(N)), Sfull)
    diag_penalty = float(
        (np.einsum("ij,ji->", Fd, Fd).real
         / np.einsum("ij,ji->", F, F).real) ** 0.25
    )

    # ---- what LuSEE-Night actually measures
    # Draws from the report's OWN signal model: P is a Gaussian field of
    # covariance A^2 S, so a realization is exactly what the analysis
    # claims the instrument sees.  Seeded, so the figure is reproducible.
    sim_rng = np.random.default_rng(4)
    tau = SIDEREAL_DAY_S / 1024
    sigb = noise.radiometer_sigma(1.0, 563.4, tau)

    def realize(Smat, m, amp, r):
        L = np.linalg.cholesky(Smat + 1e-12 * np.eye(Smat.shape[0]))
        g = (r.normal(size=(Smat.shape[0], m))
             + 1j * r.normal(size=(Smat.shape[0], m))) / np.sqrt(2)
        return amp * (L @ g)

    S_full = noise.faraday_signal_covariance(cent, Hfid, lam2)
    S_dl0 = noise.faraday_signal_covariance(
        np.array([0.0]), np.array([1.0]), lam2, allow_one_sided=True)
    S_d20 = noise.faraday_signal_covariance(
        np.array([20.0]), np.array([1.0]), lam2, allow_one_sided=True)
    demo = np.stack([realize(x, 1, 1.0, sim_rng)[:, 0]
                     for x in (S_dl0, S_d20, S_full)])
    one = (realize(S_full, 1, slab, sim_rng)[:, 0]
           + realize(N / sigb**2, 1, sigb, sim_rng)[:, 0])
    one_sig = realize(S_full, 1, slab, np.random.default_rng(4))[:, 0]

    luns = np.array([1, 6, 24])
    coadd = np.zeros((luns.size, len(bins)))
    for i, lun in enumerate(luns):
        nl = int(round(1024 * min(1.0, 0.55 * lun)))
        M = int(max(1.0, 0.54 * lun) * nl)
        acc = np.zeros((len(bins), len(bins)), complex)
        done = 0
        while done < M:
            mm = min(1024, M - done)
            x = (realize(S_full, mm, slab, sim_rng)
                 + realize(N / sigb**2, mm, sigb, sim_rng))
            acc += x @ x.conj().T
            done += mm
        coadd[i] = np.abs((acc / M - N)[0]) / slab**2

    # ---- the same measurement at all three bands
    # The report leads with 30 MHz; showing 10 and 50 beside it is what
    # lets a reader check that claim rather than take it.  Only the
    # covariance panel is repeated per band -- the angle/spectra panels
    # are illustrative and stay at the lead band.
    all_bands = [10.0, 30.0, 50.0]
    nb = len(all_bands)
    band_S = np.zeros((nb, len(bins)))
    band_S0 = np.zeros((nb, len(bins)))
    band_coadd = np.zeros((nb, len(bins)))
    band_dl2 = np.zeros((nb, len(bins)))
    band_dnu = np.zeros((nb, len(bins)))
    band_snr = np.zeros(nb)
    # Use the STORED signed templates, on the 5000-bin coarse grid the
    # production covariance uses.  Rebuilding on the FINE phi_edges grid
    # would hand faraday_signal_covariance a (192, nphi) dense matrix --
    # 1.0 GB at 30 MHz and 9.0 GB at 10 MHz, which OOM-kills the job.
    kf_i = int(d["k_fiducial_index"])
    phi_sg = d["phi_signed"]
    for ib, bb in enumerate(all_bands):
        _, bins_b, W_b = dsp.zoom_bin_matrix(bb)
        l2_b = np.asarray(lambda_squared(bins_b), dtype=float)
        N_b = noise.zoom_noise_covariance(
            W_b, noise.radiometer_sigma(1.0, 563.4, tau))
        sg_b = noise.radiometer_sigma(1.0, 563.4, tau)
        jb = int(np.argmin(np.abs(np.asarray(d["bands"]) - bb)))
        H_b = d["H_signed"][jb, kf_i]
        S_b = noise.faraday_signal_covariance(phi_sg, H_b, l2_b)
        A_b = float(d["bracket"][jb, 1])
        band_S[ib] = np.abs(S_b[0])
        band_S0[ib] = np.abs(noise.faraday_signal_covariance(
            np.array([0.0]), np.array([1.0]), l2_b, allow_one_sided=True)[0])
        band_dl2[ib] = np.abs(l2_b - l2_b[0])
        band_dnu[ib] = np.abs(bins_b - bins_b[0]) * 1e6      # Hz
        band_snr[ib] = A_b / sg_b
        nl = int(round(1024 * min(1.0, 0.55 * 24)))
        M = int(max(1.0, 0.54 * 24) * nl)
        acc = np.zeros((len(bins_b), len(bins_b)), complex)
        done = 0
        while done < M:
            mm = min(1024, M - done)
            x = (realize(S_b, mm, A_b, sim_rng)
                 + realize(N_b / sg_b**2, mm, sg_b, sim_rng))
            acc += x @ x.conj().T
            done += mm
        band_coadd[ib] = np.abs((acc / M - N_b)[0]) / A_b**2

    # ---- the single-screen and no-Faraday reference covariances
    S_none = srow(np.array([0.0]), np.array([1.0]), one_sided=True)
    S_one = srow(np.array([20.0]), np.array([1.0]), one_sided=True)

    out = GEN_DIR / "step5_intuition.npz"
    np.savez(
        out,
        band=band,
        phi=ccent,
        lam2_bins=lam2,
        sky_names=np.array([s[0] for s in skies]),
        toy_H=toy_H,
        toy_S=toy_S,
        toy_cv=toy_cv,
        toy_var=toy_var,
        adj_median=adj_med,
        adj_p90=adj_p90,
        adj_turns=adj_turns,
        coherent_valid_above_mhz=float(coh_freq_mhz),
        nsides=NSIDES,
        scales=SCALES,
        coherent_norm=coh,
        scan_H=scan_H,
        scan_S=scan_S,
        S_fiducial=S_fid,
        S_variants=S_var,
        variant_names=np.array(var_names),
        beta_shift=beta_shift,
        S_no_faraday=S_none,
        S_one_screen=S_one,
        all_bands=np.array(all_bands),
        band_S=band_S,
        band_S_no_faraday=band_S0,
        band_coadd=band_coadd,
        band_dlam2=band_dl2,
        band_dnu_hz=band_dnu,
        band_sample_snr=band_snr,
        demo_spectra=demo,
        one_sample=one,
        one_signal=one_sig,
        sample_snr=float(slab / sigb),
        tau_s=float(tau),
        coadd_lunations=luns,
        coadd_S=coadd,
        crab=crab,
        crab_excess=crab_excess,
        crab_deliverable=crab_deliv,
        weight_concentration=conc,
        sigma_broaden_pct=sigma_broaden,
        n_eff_modes=n_eff,
        diag_noise_penalty=diag_penalty,
        rm_std_median=float(np.median(rm_std)),
        rm_abs_median=float(np.median(np.abs(rm))),
        pol_max_sep_deg=float(
            np.degrees(hp.rotator.angdist(
                hp.ang2vec(*hp.pix2ang(
                    hp.npix2nside(pol.size), int(np.argmax(pol)),
                    lonlat=True), lonlat=True),
                hp.ang2vec(184.557, -5.784, lonlat=True))[0])
        ),
    )
    print(f"CV per sky: {dict(zip([s[0] for s in skies], toy_cv.round(3)))}")
    print(f"beta shift in |S| (must be 0): {beta_shift:.3e}")
    print(f"Crab: {crab[0]:.4f}% of sky holds {crab[1]:.2f}% of the weight; "
          f"excess {crab_excess[0]:.2f}x -> {crab_excess[1]:.2f}x excised; "
          f"ratio {crab_deliv[0][0]:.2f} -> {crab_deliv[1][0]:.2f}")
    print(f"sigma_eff broadening raises the ratio by "
          f"{sigma_broaden[0]:+.1f}% (1x) / {sigma_broaden[1]:+.1f}% (2x)")
    print(f"adjacent-pixel |dRM|: median {adj_med:.2f}, p90 {adj_p90:.2f} "
          f"rad/m^2; turns 10/30/50 MHz = "
          + "/".join(f"{t:.1f}" for t in adj_turns))
    print(f"coherent pixel sum valid above ~{coh_freq_mhz:.0f} MHz")
    print("per-band single-sample SNR: "
          + ", ".join(f"{b:.0f} MHz {v:.2f}"
                      for b, v in zip(all_bands, band_snr)))
    print(f"effective modes {n_eff:.1f}, diagonal-N penalty "
          f"{diag_penalty:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
