"""Decision table for the column-screen speckle against LuSEE noise.

For each column-anchored screen scenario (alpha 2, 5/3 at anchors 0.3, 1,
3 rad/m^2 and alpha 1 at anchor 1), the zoom-bin and parent-channel speckle
rms at the WMAP-K polarized brightness, against the loaded radiometer noise
of the measured-system reference receiver case (3 nV/rtHz, 0.1 pA/rtHz) for
the actual-site nightside exposure at one sidereal phase, for 1, 12, 24 and
48 lunations (the 12 and 48 values scale the 24-lunation exposure). Also
the significance of a variance-excess test over the simulated band (192
zoom bins x 10 independent sidereal phases). Writes
docs/column-decision-results.json and prints a table.
"""

import common  # noqa: F401
import json

import numpy as np

from common import REPO
from paper_common import manifest, write_json

WMAP_PIXEL_RMS_30 = 1.257e4  # K, all scales incl. WMAP noise
BETA_QU = -2.8
RECEIVER = (3.0, 0.1)
LUNATIONS = (1, 12, 24, 48)
PHASES_INDEPENDENT = 10  # beam-width-scale decorrelation over a lunar day
SOURCES = ["scripts/column_decision.py"]


def wmap_rms(center):
    return WMAP_PIXEL_RMS_30 * (center / 30.0) ** BETA_QU


def column_rows():
    rows = []
    for name in (
        "faraday-covariance-column-results.json",
        "faraday-covariance-column-alpha1-results.json",
    ):
        rows.extend(json.loads((REPO / "docs" / name).read_text())["results"])
    return rows


def noise_rms_per_second(center):
    """Per-bin noise variance x 1 s (K^2 s) for Q at phase 0: parent, zoom."""
    rec = json.loads((REPO / "docs/measured-system-results.json").read_text())
    row = [
        n
        for n in rec["noise"]
        if n["center_mhz"] == center
        and abs(n["voltage_nv_rt_hz"] - RECEIVER[0]) < 1e-9
        and abs(n["current_pa_rt_hz"] - RECEIVER[1]) < 1e-9
    ][0]
    z = np.load(REPO / row["path"])
    c = z["covariance_per_second"][0]  # phase 0, (195, 4, 195, 4)
    var = np.array([c[b, 1, b, 1] for b in range(c.shape[0])])
    return var[:3], var[3:], z["nominal_mhz"]


def exposure_s(lunations):
    rec = json.loads((REPO / "docs/observing-exposure.json").read_text())
    by = {
        r["lunations"]: r["exposure_s"]
        for r in rec["results"]
        if r["sidereal_turn"] == 0.0
    }
    if lunations in by:
        return by[lunations]
    return by[24] * lunations / 24


def main():
    results = []
    lines = []
    for center in (30.0, 50.0):
        var_parent, var_zoom, nominal = noise_rms_per_second(center)
        for row in column_rows():
            if row["center_mhz"] != center:
                continue
            z = np.load(REPO / row["path"])
            spk = z["speckle"]
            nb = spk.shape[0]
            # phase 0, Q: parent (bin 1 of 3) and the 64 zoom bins
            s_parent = np.sqrt(spk[1, 1, 1, 1].real) * wmap_rms(center)
            s_zoom = np.sqrt(
                np.mean([spk[b, 1, b, 1].real for b in range(3, nb)])
            ) * wmap_rms(center)
            entry = dict(
                center_mhz=center,
                label=row["case"]["label"],
                speckle_rms_parent_k=float(s_parent),
                speckle_rms_zoom_k=float(s_zoom),
                exposures={},
            )
            for lun in LUNATIONS:
                t = exposure_s(lun)
                n_parent = float(np.sqrt(var_parent[1] / t))
                n_zoom = float(np.sqrt(np.mean(var_zoom) / t))
                r_zoom = s_zoom / n_zoom
                n_samples = 192 * PHASES_INDEPENDENT
                var_excess_sigma = r_zoom**2 * np.sqrt(n_samples / 2)
                entry["exposures"][str(lun)] = dict(
                    exposure_s=float(t),
                    noise_rms_parent_k=n_parent,
                    noise_rms_zoom_k=n_zoom,
                    speckle_over_noise_parent=float(s_parent / n_parent),
                    speckle_over_noise_zoom=float(r_zoom),
                    variance_excess_sigma=float(var_excess_sigma),
                )
                lines.append(
                    f"{center:4g} MHz {row['case']['label']:<34} "
                    f"{lun:3d} lun ({t / 3600:5.1f} h): speckle zoom "
                    f"{s_zoom:.3f} K / noise {n_zoom:6.2f} K = "
                    f"{r_zoom:.1e}; parent {s_parent:.3f} / {n_parent:.2f} "
                    f"= {s_parent / n_parent:.1e}; variance-excess "
                    f"{var_excess_sigma:.1e} sigma"
                )
            results.append(entry)
    config = dict(
        receiver=RECEIVER,
        lunations=LUNATIONS,
        wmap_rms_30=WMAP_PIXEL_RMS_30,
        phases_independent=PHASES_INDEPENDENT,
        scope=(
            "Column-screen speckle at the WMAP-K polarized brightness "
            "(all scales, WMAP noise included) against the loaded radiometer "
            "noise of the measured-system reference receiver at one sidereal "
            "phase; 12 and 48 lunations scale the 24-lunation actual-site "
            "exposure. Conditional on the column screen, the structure-"
            "function family and the intrinsic sky."
        ),
    )
    provenance, fingerprint = manifest(config, SOURCES)
    write_json(
        REPO / "docs/column-decision-results.json",
        dict(fingerprint=fingerprint, provenance=provenance, results=results),
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
