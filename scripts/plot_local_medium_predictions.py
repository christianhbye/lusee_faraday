"""Plot the complete conditional signed-column separability grid."""

import common  # noqa: F401
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import REPO
from paper_common import digest
from plot_measurement_predictions import COLORS, save


def plot(path):
    record = json.loads(path.read_text())
    if not record["all_passed"]:
        raise ValueError("local-medium decisions have not passed")
    item = record["arrays"]
    target = REPO / item["path"]
    if digest(target) != item["sha256"]:
        raise ValueError("corrupt local-medium decision grid")
    with np.load(target) as data:
        speckle = data["speckle_to_noise_amplitude"]
        total = data["total_to_noise_amplitude"]
    sensitivity_path = REPO / "docs/local-medium-sensitivity-results.json"
    sensitivity = json.loads(sensitivity_path.read_text())
    horizon = sensitivity["provenance"]["configuration"]["horizon"]
    axes = record["axes"]
    phase_index = axes["phase"].index(0.0)
    lunation_index = axes["lunations"].index(max(axes["lunations"]))
    coordinates = record["coordinates"]
    means = sorted({r["mean_depth_rad_m2"] for r in coordinates})
    fig, panels = plt.subplots(
        2,
        2,
        figsize=(6.4, 5.0),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    all_lows = []
    for row, center in enumerate(
        sorted({r["center_mhz"] for r in coordinates})
    ):
        for column, kind in enumerate(axes["channel_type"]):
            ax = panels[row, column]
            for component, color in enumerate(COLORS[:2]):
                lows, highs, total_highs = [], [], []
                for mean in means:
                    selected = [
                        i
                        for i, case in enumerate(coordinates)
                        if case["center_mhz"] == center
                        and case["mean_depth_rad_m2"] == mean
                    ]
                    values = speckle[
                        selected,
                        phase_index,
                        column,
                        component,
                        lunation_index,
                        :,
                    ].ravel()
                    positive = values[np.isfinite(values) & (values > 0)]
                    if not len(positive):
                        raise ValueError("missing positive rotating profile")
                    lows.append(float(positive.min()))
                    highs.append(float(positive.max()))
                    total_highs.append(
                        float(
                            np.nanmax(
                                total[
                                    selected,
                                    phase_index,
                                    column,
                                    component,
                                    lunation_index,
                                    :,
                                ]
                            )
                        )
                    )
                all_lows.extend(lows)
                label = axes["component"][component]
                ax.fill_between(means, lows, highs, color=color, alpha=0.14)
                ax.plot(
                    means,
                    highs,
                    "o-",
                    color=color,
                    lw=0.8,
                    ms=3,
                    label=label + " speckle maximum",
                )
                ax.plot(
                    means,
                    total_highs,
                    "s--",
                    color=color,
                    lw=0.8,
                    ms=3,
                    mfc="none",
                    label=label + " total maximum",
                )
            ax.axhline(1, color="black", ls=":", lw=0.8)
            ax.set(
                title=f"{center:g} MHz; {kind}",
                yscale="log",
                ylim=(1e-12, 2),
                ylabel="Residual RMS / radiometer RMS",
                xlabel="Mean Faraday depth (rad/m²)",
            )
            ax.grid(alpha=0.15)
    lower = 10 ** np.floor(np.log10(min(all_lows)))
    panels[0, 0].set_ylim(lower, 2)
    handles, labels = panels[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="outside lower center", fontsize=7)
    save(
        fig,
        "faraday_separability",
        "Conditional residual Q/U RMS divided by the same fitted radiometer "
        "RMS over all signed uniform mean columns from the adopted "
        "density/field/path pairs. Every plotted mean includes both assumed "
        "dispersions, Gaussian and global-variance-mixture statistics, all "
        "five depth profiles and nine receiver-noise combinations. Solid "
        "points are maximum speckle ratios; shading spans positive speckle "
        "ratios. The unrotated f=0 case has exactly zero speckle. Dashed "
        "open squares give maximum total (smooth plus speckle) ratios, "
        "including f=0. These are conditional model ranges, not confidence "
        "intervals. The horizontal dotted line is equal RMS, not a detection "
        "threshold. The full arrays retain both phases and 1/24-lunation "
        "availability; this figure uses phase "
        f"{axes['phase'][phase_index]:g} and "
        f"{axes['lunations'][lunation_index]} illustrative lunations. "
        f"Intrinsic RMS is {record['intrinsic_rms_k']:g} K at "
        f"{horizon['emissivity_ref_mhz']:g} MHz, "
        "the assumed volume-averaged polarized emissivity times the full "
        f"{horizon['emitting_path_pc']:g} pc path. "
        "The depth integral supplies "
        "depolarization; no second "
        "horizon cut is made. Alternative emissivity normalizations are "
        "tabulated separately and scale amplitudes linearly.",
        [path, sensitivity_path],
        generator=__file__,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        type=Path,
        default=REPO / "docs/local-medium-separability-results.json",
    )
    args = parser.parse_args()
    plot(args.record)
