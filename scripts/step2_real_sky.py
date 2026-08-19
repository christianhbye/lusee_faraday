"""Steps 2-3: real sky maps through the Faraday screen.

Sky: Haslam 408 MHz (dsds) scaled with beta = -2.55 for I, WMAP K-band
Q/U scaled with beta = -2.8, Hutschenreuter faraday2020v2 mean RM map.
All maps are used at their NATIVE nside=512 - no degradation (per-pixel
Faraday phases do not commute with ud_grade averaging).

For each of the 1024 times over one lunar sidereal day the ten pair
integrals are evaluated on the 16384-point fine grid via the type-3
NUFFT; the "Faraday off" reference is the same (frequency-independent)
sky evaluated at the band center.  Waterfalls stream to a memmap and
are then integrated against the spectrometer response.

Usage:  step2_real_sky.py --center 30    (Step 2)
        step2_real_sky.py --center 10    (Step 3i)
        step2_real_sky.py --center 50    (Step 3ii)

Quadrature note: the fine grid (12.2 Hz) supports Faraday oscillations
of |RM| < ~700 rad/m^2 at 10 MHz (higher elsewhere); only ~1e-5 of the
faraday2020v2 pixels exceed that, so spectrometer-bin quadrature error
is negligible.

Outputs (generated_data/): real{C}_fine_waterfall.npy (1024, 16384, 16)
float64, real{C}_meta.npz, real{C}_binned.npz with C the center in MHz.
"""

import argparse
import time as _time

import numpy as np

from common import (
    GEN_DIR,
    MAP_NSIDE,
    N_FINE,
    N_TIMES,
    RESPONSE_PATH,
    fine_freqs,
    lam2,
    load_sky_maps,
    parent_centers,
    rotation_matrices,
    sky_at_freq,
    times,
)
from lusee_faraday import _legacy_pixel as fp

TIME_CHUNK = 16  # times per spectrometer-integration chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", type=float, default=30.0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    center = args.center
    tag = f"real{center:g}"

    wf_path = GEN_DIR / f"{tag}_fine_waterfall.npy"
    meta_path = GEN_DIR / f"{tag}_meta.npz"
    binned_path = GEN_DIR / f"{tag}_binned.npz"

    from lusee.ReceiverImpedance import JFETReceiver

    R_all = rotation_matrices()
    ff = fine_freqs(center)
    l2 = lam2(ff)

    if wf_path.exists() and meta_path.exists() and not args.force:
        print("fine waterfall exists; use --force to redo", flush=True)
    else:
        t0 = _time.time()
        maps = load_sky_maps()
        I_map, Q_map, U_map = sky_at_freq(maps, center)
        RM = maps["RM"]
        resp = fp.load_response_fast(RESPONSE_PATH)
        kern = fp.FixedFreqKernel(resp, center, JFETReceiver())
        del resp
        grid = fp.GalacticGrid(MAP_NSIDE)
        sim = fp.SkyWaterfallSim(kern, grid, I_map, Q_map, U_map, RM)
        print(f"setup done {_time.time()-t0:.1f} s", flush=True)

        wf = np.lib.format.open_memmap(
            wf_path,
            mode="w+",
            dtype=np.float64,
            shape=(N_TIMES, N_FINE, 16),
        )
        nofar = np.zeros((N_TIMES, 16))
        t0 = _time.time()
        for it in range(N_TIMES):
            pair = sim.pair_integrals(
                R_all[it], l2, faraday=True, eps=args.eps
            )  # (10, F)
            C = fp.assemble_covariance(pair.T, kern.M)  # (F, 4, 4)
            wf[it] = fp.pack_products(C)
            pair0 = sim.pair_integrals(
                R_all[it], np.array([l2[N_FINE // 2]]), faraday=False
            )[:, 0]
            nofar[it] = fp.pack_products(fp.assemble_covariance(pair0, kern.M))
            if (it + 1) % 32 == 0:
                dt = _time.time() - t0
                eta = dt / (it + 1) * (N_TIMES - it - 1)
                print(
                    f"  t {it + 1}/{N_TIMES}  {dt:.0f} s"
                    f"  eta {eta/60:.0f} min",
                    flush=True,
                )
        wf.flush()
        np.savez(
            meta_path,
            t_unix=times().unix,
            fine_freqs_mhz=ff,
            nofaraday=nofar,
            center_mhz=center,
            labels=np.array(fp.PRODUCT_LABELS),
        )
        print(f"saved {wf_path.name}, {meta_path.name}", flush=True)

    if binned_path.exists() and not args.force:
        print("binned products exist; use --force to redo", flush=True)
        return
    print("integrating spectrometer response...", flush=True)
    wf = np.load(wf_path, mmap_mode="r")
    centers = parent_centers(center)
    parents = np.empty((N_TIMES, 3, 16))
    zooms = np.empty((N_TIMES, 3, 64, 16))
    ideals = np.empty((N_TIMES, 3, 64, 16))
    for s in range(0, N_TIMES, TIME_CHUNK):
        sl = slice(s, min(s + TIME_CHUNK, N_TIMES))
        out = fp.integrate_spectrometer(np.asarray(wf[sl]), ff, centers)
        parents[sl] = out["parent"]
        zooms[sl] = out["zoom"]
        ideals[sl] = out["ideal_zoom"]
    np.savez(
        binned_path,
        parent=parents,
        zoom=zooms,
        ideal_zoom=ideals,
        parent_centers_mhz=centers,
        zoom_offsets_hz=fp.zoom_bin_offsets_hz(),
    )
    print(f"saved {binned_path.name}", flush=True)


if __name__ == "__main__":
    main()
