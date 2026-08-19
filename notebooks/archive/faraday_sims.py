#!/usr/bin/env python
# coding: utf-8

# # Faraday Rotation Simulations for LuSEE
# 
# Simulate polarized visibilities with and without Faraday rotation at 10, 30, and 50 MHz.
# 
# **Strategy:**
# - **No-FR case**: Rotate reference maps once per time step, compute beam-weighted pixel sums, then scale analytically to all frequencies. Full spectrometer resolution is free.
# - **FR case**: Apply Faraday rotation + power-law scaling in Galactic frame, then rotate each frequency channel to topocentric. Decimated frequency grids to manage runtime.
# 
# **Output modes:**
# - **Narrow (zoom) bins**: ~1000 frequencies within one 25 kHz parent bin → convolve with 64 narrow sub-bin responses
# - **Wide bins**: 64 parent bin centers spanning 1.6 MHz (bin-center values; no-FR case uses proper convolution at full resolution)

# In[ ]:


from pathlib import Path
import time as pytime

import astropy.units as u
import healpy as hp
from lunarsky import Time, MoonLocation, LunarTopo
import numpy as np

import lusee_faraday as ld
from lusee_faraday.sim import beam_sky_multiply, Simulator, SimConfig
from lusee_faraday.fast_sim import precompute_rotated_maps, compute_vis_fast
from lusee_faraday.rotations import gal2topo, get_rot_mat, rotmat_to_eulerZYX
from lusee_faraday.sky import LUSEE_LOC


# In[ ]:


# --- Configuration ---
DATA_DIR = Path("/home/christian/Documents/research/lusee/lusee_faraday/data/")
OUT_DIR = Path("/home/christian/Documents/research/lusee/lusee_faraday/notebooks/results")
OUT_DIR.mkdir(exist_ok=True)

NSIDE = 128
N_TIMES = 100
CENTER_FREQS = [10, 30, 50]  # MHz
BEAM_FILE = DATA_DIR / "hfss_lbl_3m_75deg.2port.fits"

# Spectral indices
BETA_I = -2.55   # Haslam (from 50 MHz reference)
BETA_QU = -2.8   # WMAP polarization (from 23 GHz reference)
FREQ_REF_I = 50  # MHz
FREQ_REF_QU = 23e3  # MHz
CMB = 2.725  # K

# Time setup: one full lunar sidereal day (~27.3 Earth days)
lusee_loc = MoonLocation(lat=-23.813, lon=182.258)
time_start = Time("2027-01-01T09:00:00", location=lusee_loc)
time_end = time_start + 655.720 * 3600 * u.s
times = np.linspace(time_start, time_end, num=N_TIMES, endpoint=False)
print(f"Time range: {time_start.iso} to {time_end.iso}")
print(f"  {N_TIMES} steps, dt = {655.72/N_TIMES:.1f} hours")


# In[ ]:


# --- Load sky data ---
haslam_ref = np.load(DATA_DIR / "haslam_galactic.npz")["m"]  # (npix,) at 50 MHz
wmap_23ghz = ld.sky.load_wmap(
    DATA_DIR / "wmap_band_iqumap_r9_9yr_K_v5.fits", nside=NSIDE
)  # (3, npix) at 23 GHz
rm_gal = ld.sky.load_rm(DATA_DIR / "faraday2020v2.hdf5")  # (npix,) rad/m^2

I_ref = haslam_ref      # reference I at 50 MHz
Q_ref = wmap_23ghz[1]   # reference Q at 23 GHz
U_ref = wmap_23ghz[2]   # reference U at 23 GHz

# Spectrometer response (full resolution)
spec_full = ld.SpectrometerResponse.from_file(
    DATA_DIR / "spectrometer_bin_response.txt"
)
print(f"Spectrometer response: {len(spec_full.freq_offset_hz)} points, "
      f"{spec_full.freq_offset_hz[0]/1e3:.0f} to "
      f"{spec_full.freq_offset_hz[-1]/1e3:.0f} kHz, "
      f"step {spec_full.df_hz:.0f} Hz")

# Decimation factors for FR narrow mode (more points at low freq
# where Faraday oscillations are faster)
NARROW_DEC = {10: 10, 30: 50, 50: 100}

# If True, compute wide-bin convolution only for the single time step
# where the galaxy contributes most above the horizon.  Reduces the
# wide-bin compute from N_TIMES steps to 1 step.
WIDE_GALAXY_UP_ONLY = True
for cf, dec in NARROW_DEC.items():
    sd = spec_full.decimate(dec)
    print(f"  {cf} MHz narrow: dec={dec}, "
          f"{len(sd.freq_offset_hz)} pts, "
          f"step={sd.df_hz:.0f} Hz")


# ## No-FR Simulations (fast path)
# 
# Rotate reference sky maps once per time step, compute beam-weighted pixel sums,
# then scale analytically to any frequency. Full spectrometer resolution for free.

# In[ ]:


def compute_pixel_sums(I_topo, Q_topo, U_topo, beam, mask):
    """Compute the 12 beam-weighted pixel sums needed for analytical scaling.

    Returns dict with keys like 'sII_x', 'sIc_x', 'sQQ_x', 'sUU_x' etc.
    for xx, yy, xy visibility types.
    """
    w = beam.weights
    m = mask.astype(float)
    I_m = (I_topo - CMB) * m
    Q_m = Q_topo * m
    U_m = U_topo * m
    cmb_m = CMB * m
    sums = {}
    for pol in ("x", "y", "xy"):
        sums[f"sII_{pol}"] = np.sum(w[f"wI_{pol}"] * I_m)
        sums[f"sIc_{pol}"] = np.sum(w[f"wI_{pol}"] * cmb_m)
        sums[f"sQQ_{pol}"] = np.sum(w[f"wQ_{pol}"] * Q_m)
        sums[f"sUU_{pol}"] = np.sum(w[f"wU_{pol}"] * U_m)
    return sums


def vis_from_sums(sums, freqs, norm):
    """Evaluate Rxx, Ryy, Rxy at arbitrary frequencies from pixel sums."""
    scale_I = (freqs / FREQ_REF_I) ** BETA_I
    scale_QU = (freqs / FREQ_REF_QU) ** BETA_QU
    nf = len(freqs)
    vis = np.empty((3, nf))
    for k, pol in enumerate(("x", "y", "xy")):
        vis[k] = (
            scale_I * sums[f"sII_{pol}"]
            + sums[f"sIc_{pol}"]
            + scale_QU * (sums[f"sQQ_{pol}"] + sums[f"sUU_{pol}"])
        ) / norm
    return vis  # (3, nfreq)


def simulate_no_fr(beam, times, freqs, nside=NSIDE):
    """Full no-FR simulation: rotate once per time step, scale analytically."""
    grid = ld.HealpixGrid(nside, horizon=True)
    mask = grid.mask
    w = beam.weights
    norm = np.sum(w["wI_x"] * mask) + np.sum(w["wI_y"] * mask)

    ntimes = len(times)
    vis = np.zeros((ntimes, 3, len(freqs)))
    for i, t in enumerate(times):
        if (i + 1) % 20 == 0:
            print(f"  no-FR time step {i+1}/{ntimes}")
        topo = LunarTopo(location=LUSEE_LOC, obstime=t)
        It, Qt, Ut = gal2topo(I_ref, Q_ref, U_ref, topo_frame=topo)
        sums = compute_pixel_sums(
            It.squeeze(), Qt.squeeze(), Ut.squeeze(), beam, mask
        )
        vis[i] = vis_from_sums(sums, freqs, norm)
    return vis  # (ntimes, 3, nfreq)


# In[ ]:


def run_no_fr(center_mhz):
    """Run no-FR simulation for one center frequency. Returns dict of results."""
    print(f"\n{'='*60}")
    print(f"No-FR @ {center_mhz} MHz")
    print(f"{'='*60}")
    beam = ld.Beam.from_file(BEAM_FILE, frequency=center_mhz, nside=NSIDE)
    beam.precompute_weights()
    print("  Beam loaded and weights precomputed")

    grid = ld.HealpixGrid(NSIDE, horizon=True)
    mask = grid.mask
    w = beam.weights
    norm = np.sum(w["wI_x"] * mask) + np.sum(w["wI_y"] * mask)

    # --- Frequency grids ---
    freqs_narrow = spec_full.freqs(center_mhz)
    f_lusee = ld.utils.freqs_lusee()
    idx_c = np.argmin(np.abs(f_lusee - center_mhz))
    bin_centers = f_lusee[idx_c - 32: idx_c + 32]
    freqs_zoom = ld.utils.freqs_zoom(center=center_mhz, num=64)
    print(f"  Narrow: {len(freqs_narrow)} freqs, "
          f"[{freqs_narrow[0]:.6f}, {freqs_narrow[-1]:.6f}] MHz")
    print(f"  Wide: 64 bins, [{bin_centers[0]:.4f}, {bin_centers[-1]:.4f}] MHz")

    # --- Precompute pixel sums (one rotation per time step) ---
    t0 = pytime.time()
    all_sums = []
    for i, t in enumerate(times):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Rotating time step {i+1}/{N_TIMES} "
                  f"({pytime.time()-t0:.0f}s elapsed)")
        topo = LunarTopo(location=LUSEE_LOC, obstime=t)
        It, Qt, Ut = gal2topo(I_ref, Q_ref, U_ref, topo_frame=topo)
        sums = compute_pixel_sums(
            It.squeeze(), Qt.squeeze(), Ut.squeeze(), beam, mask
        )
        all_sums.append(sums)
    print(f"  Rotations done in {pytime.time() - t0:.1f} sec")

    # --- Narrow mode ---
    print("  Computing narrow-mode visibilities...")
    vis_narrow = np.zeros((N_TIMES, 3, len(freqs_narrow)))
    for i, sums in enumerate(all_sums):
        vis_narrow[i] = vis_from_sums(sums, freqs_narrow, norm)
    pI_raw, pQ_raw, pU_raw = Simulator.compute_stokes(vis_narrow)
    pI_zoom = spec_full.apply_narrow(pI_raw)
    pQ_zoom = spec_full.apply_narrow(pQ_raw)
    pU_zoom = spec_full.apply_narrow(pU_raw)
    print("  Narrow done")

    # --- Wide mode: bin-center values ---
    print("  Computing wide-mode bin-center visibilities...")
    vis_wide = np.zeros((N_TIMES, 3, 64))
    for i, sums in enumerate(all_sums):
        vis_wide[i] = vis_from_sums(sums, bin_centers, norm)
    pI_wide, pQ_wide, pU_wide = Simulator.compute_stokes(vis_wide)

    # --- Wide mode: proper convolution ---
    # Find galaxy-up time step: argmax of beam-weighted I above horizon
    i_gal = int(np.argmax([s["sII_x"] + s["sII_y"] for s in all_sums]))
    if WIDE_GALAXY_UP_ONLY:
        sel_sums = [all_sums[i_gal]]
        print(f"  Wide convolution (galaxy-up only, step {i_gal})")
    else:
        i_gal = None
        sel_sums = all_sums
        print(f"  Wide convolution ({N_TIMES} time steps)")
    n_sel = len(sel_sums)
    pI_wconv = np.zeros((n_sel, 64))
    pQ_wconv = np.zeros((n_sel, 64))
    pU_wconv = np.zeros((n_sel, 64))
    # Concatenate all 64 bin freq grids; split after vis_from_sums
    all_bin_freqs = np.concatenate(
        [spec_full.freqs(fc) for fc in bin_centers]
    )
    n_pb = len(spec_full.freq_offset_hz)
    t1 = pytime.time()
    for ii, sums in enumerate(sel_sums):
        v = vis_from_sums(sums, all_bin_freqs, norm)
        pI_v = v[0] + v[1]
        pQ_v = v[0] - v[1]
        pU_v = 2 * np.real(v[2])
        for k in range(64):
            sl = slice(k * n_pb, (k + 1) * n_pb)
            pI_wconv[ii, k] = spec_full.apply_wide(pI_v[sl])
            pQ_wconv[ii, k] = spec_full.apply_wide(pQ_v[sl])
            pU_wconv[ii, k] = spec_full.apply_wide(pU_v[sl])
    print(f"  Wide convolution done in {pytime.time()-t1:.1f} sec")

    total = pytime.time() - t0
    print(f"  TOTAL no-FR @ {center_mhz} MHz: {total:.0f} sec ({total/60:.1f} min)")
    return {
        "freqs_narrow": freqs_narrow, "freqs_zoom": freqs_zoom,
        "freqs_wide": bin_centers,
        "i_gal": i_gal,
        "pI_raw": pI_raw, "pQ_raw": pQ_raw, "pU_raw": pU_raw,
        "pI_zoom": pI_zoom, "pQ_zoom": pQ_zoom, "pU_zoom": pU_zoom,
        "pI_wide": pI_wide, "pQ_wide": pQ_wide, "pU_wide": pU_wide,
        "pI_wconv": pI_wconv, "pQ_wconv": pQ_wconv, "pU_wconv": pU_wconv,
    }


# In[ ]:


# Run no-FR for all center frequencies
no_fr_results = {}
for cf in CENTER_FREQS:
    no_fr_results[cf] = run_no_fr(cf)


# ## FR Simulations (optimized fast path)
# 
# Exploits SO(2) commutativity: rotate reference maps + RM to topocentric once per
# time step, then apply power-law scaling and Faraday rotation in topocentric frame.
# The beam-weighted sum factorizes as `A·cos(2·RM·λ²) + B·sin(...)` evaluated via
# BLAS matrix-vector products.
# 
# This avoids per-frequency SHT rotations entirely, making proper wide-bin
# convolution feasible (thousands of frequencies, ~minutes instead of hours).

# In[ ]:


def run_fr(center_mhz):
    """Run FR simulation using the optimized fast_sim approach."""
    print(f"\n{'='*60}")
    print(f"FR @ {center_mhz} MHz")
    print(f"{'='*60}")
    beam = ld.Beam.from_file(BEAM_FILE, frequency=center_mhz, nside=NSIDE)
    beam.precompute_weights()
    print("  Beam loaded and weights precomputed")
    grid = ld.HealpixGrid(NSIDE, horizon=True)
    mask = grid.mask

    # --- Precompute rotated reference maps ---
    t0 = pytime.time()
    I_topo, Q_topo, U_topo, rm_topo = precompute_rotated_maps(
        I_ref, Q_ref, U_ref, rm_gal, times, NSIDE, LUSEE_LOC,
    )
    print(f"  Rotations done in {pytime.time() - t0:.1f} sec")

    # --- Narrow mode ---
    dec = NARROW_DEC[center_mhz]
    spec_dec = spec_full.decimate(dec)
    freqs_narrow = spec_dec.freqs(center_mhz)
    print(f"  Narrow: {len(freqs_narrow)} freqs (dec={dec})")

    t1 = pytime.time()
    vis_narrow = compute_vis_fast(
        I_topo, Q_topo, U_topo, rm_topo, beam, freqs_narrow, mask,
    )
    print(f"  Narrow done in {pytime.time() - t1:.1f} sec")
    pI_raw, pQ_raw, pU_raw = Simulator.compute_stokes(vis_narrow)
    pI_zoom = spec_dec.apply_narrow(pI_raw)
    pQ_zoom = spec_dec.apply_narrow(pQ_raw)
    pU_zoom = spec_dec.apply_narrow(pU_raw)

    # --- Wide mode: bin centers ---
    f_lusee = ld.utils.freqs_lusee()
    idx_c = np.argmin(np.abs(f_lusee - center_mhz))
    bin_centers = f_lusee[idx_c - 32: idx_c + 32]
    print(f"  Wide: 64 bins, [{bin_centers[0]:.4f}, {bin_centers[-1]:.4f}] MHz")

    t2 = pytime.time()
    vis_wide_raw = compute_vis_fast(
        I_topo, Q_topo, U_topo, rm_topo, beam, bin_centers, mask,
    )
    print(f"  Wide (bin-center) done in {pytime.time() - t2:.1f} sec")
    pI_wide, pQ_wide, pU_wide = Simulator.compute_stokes(vis_wide_raw)

    # --- Wide mode: proper convolution (union grid + galaxy-up option) ---
    # Adjacent bins are 25 kHz apart but the response spans ±50 kHz,
    # so 64 concat grids have ~3.8x redundancy.  Build the union once.
    wide_dec = NARROW_DEC[center_mhz]
    spec_wide = spec_full.decimate(wide_dec)
    n_per_bin = len(spec_wide.freq_offset_hz)
    bin_step_pts = round(25000.0 / spec_wide.df_hz)  # exact integer
    n_union = (64 - 1) * bin_step_pts + n_per_bin
    f_union_start = bin_centers[0] + spec_wide.freq_offset_mhz[0]
    union_freqs = (
        f_union_start + np.arange(n_union) * spec_wide.df_hz * 1e-6
    )

    # Find galaxy-up time step: argmax of beam-weighted I above horizon
    w = beam.weights
    m = mask.astype(float)
    I_above = np.sum(
        (I_topo - CMB) * m[None, :]
        * (w["wI_x"] + w["wI_y"])[None, :],
        axis=1,
    )
    i_gal = int(np.argmax(I_above))

    if WIDE_GALAXY_UP_ONLY:
        sel = [i_gal]
        print(
            f"  Wide convolution (galaxy-up only, step {i_gal}): "
            f"union {n_union} freqs "
            f"(concat would be {64 * n_per_bin}, "
            f"{64 * n_per_bin / n_union:.1f}x reduction)"
        )
    else:
        i_gal = None
        sel = list(range(N_TIMES))
        print(
            f"  Wide convolution ({N_TIMES} steps): "
            f"union {n_union} freqs "
            f"(concat would be {64 * n_per_bin}, "
            f"{64 * n_per_bin / n_union:.1f}x reduction)"
        )

    t3 = pytime.time()
    vis_union = compute_vis_fast(
        I_topo[sel], Q_topo[sel], U_topo[sel], rm_topo[sel],
        beam, union_freqs, mask,
    )
    print(f"  compute_vis_fast done in {pytime.time() - t3:.1f} sec")
    sI_all, sQ_all, sU_all = Simulator.compute_stokes(vis_union)

    n_sel = len(sel)
    pI_wconv = np.zeros((n_sel, 64))
    pQ_wconv = np.zeros((n_sel, 64))
    pU_wconv = np.zeros((n_sel, 64))
    for k in range(64):
        sl = slice(k * bin_step_pts, k * bin_step_pts + n_per_bin)
        pI_wconv[:, k] = spec_wide.apply_wide(sI_all[:, sl])
        pQ_wconv[:, k] = spec_wide.apply_wide(sQ_all[:, sl])
        pU_wconv[:, k] = spec_wide.apply_wide(sU_all[:, sl])
    dt = pytime.time() - t3
    print(f"  Wide convolution done in {dt:.0f} sec ({dt/60:.1f} min)")

    freqs_zoom = ld.utils.freqs_zoom(center=center_mhz, num=64)
    total = pytime.time() - t0
    print(f"  TOTAL FR @ {center_mhz} MHz: {total:.0f} sec ({total/60:.1f} min)")

    return {
        "freqs_narrow": freqs_narrow,
        "freqs_zoom": freqs_zoom,
        "freqs_wide": bin_centers,
        "i_gal": i_gal,
        "pI_raw": pI_raw, "pQ_raw": pQ_raw, "pU_raw": pU_raw,
        "pI_zoom": pI_zoom, "pQ_zoom": pQ_zoom, "pU_zoom": pU_zoom,
        "pI_wide": pI_wide, "pQ_wide": pQ_wide, "pU_wide": pU_wide,
        "pI_wconv": pI_wconv, "pQ_wconv": pQ_wconv, "pU_wconv": pU_wconv,
    }


# In[ ]:


# Run FR for all center frequencies (this is the slow part)
fr_results = {}
for cf in CENTER_FREQS:
    fr_results[cf] = run_fr(cf)


# ## Save Results

# In[ ]:


times_jd = np.array([t.jd for t in times])

for cf in CENTER_FREQS:
    nf = no_fr_results[cf]
    fr = fr_results[cf]
    outfile = OUT_DIR / f"faraday_sim_{cf}mhz.npz"
    # i_gal is None when WIDE_GALAXY_UP_ONLY=False (all times used)
    i_gal_val = nf["i_gal"] if nf["i_gal"] is not None else -1
    np.savez(
        outfile,
        center_mhz=cf,
        times_jd=times_jd,
        i_gal=i_gal_val,
        # Frequency grids
        freqs_narrow_noFR=nf["freqs_narrow"],
        freqs_narrow_FR=fr["freqs_narrow"],
        freqs_zoom=nf["freqs_zoom"],
        freqs_wide=nf["freqs_wide"],
        # --- No-FR ---
        pI_noFR_raw=nf["pI_raw"], pQ_noFR_raw=nf["pQ_raw"],
        pU_noFR_raw=nf["pU_raw"],
        pI_noFR_zoom=nf["pI_zoom"], pQ_noFR_zoom=nf["pQ_zoom"],
        pU_noFR_zoom=nf["pU_zoom"],
        pI_noFR_wide=nf["pI_wide"], pQ_noFR_wide=nf["pQ_wide"],
        pU_noFR_wide=nf["pU_wide"],
        pI_noFR_wconv=nf["pI_wconv"], pQ_noFR_wconv=nf["pQ_wconv"],
        pU_noFR_wconv=nf["pU_wconv"],
        # --- FR ---
        pI_FR_raw=fr["pI_raw"], pQ_FR_raw=fr["pQ_raw"],
        pU_FR_raw=fr["pU_raw"],
        pI_FR_zoom=fr["pI_zoom"], pQ_FR_zoom=fr["pQ_zoom"],
        pU_FR_zoom=fr["pU_zoom"],
        pI_FR_wide=fr["pI_wide"], pQ_FR_wide=fr["pQ_wide"],
        pU_FR_wide=fr["pU_wide"],
        pI_FR_wconv=fr["pI_wconv"], pQ_FR_wconv=fr["pQ_wconv"],
        pU_FR_wconv=fr["pU_wconv"],
    )
    print(f"Saved {outfile}")

print("\nAll simulations complete!")

