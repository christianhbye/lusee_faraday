"""Render the preserved flat-plane depth result at the report's print width."""

import common  # noqa: F401
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from common import REPO
from paper_common import digest
from plot_measurement_predictions import save


def plot():
    path = REPO / "docs/depth-depolarization-results.json"
    record = json.loads(path.read_text())
    for name, expected in record["array_sha256"].items():
        if digest(REPO / name) != expected:
            raise ValueError("corrupt flat depth benchmark")
    config = record["manifest"]["configuration"]
    sigmas = config["sigma_depth_values"]
    fig, axes = plt.subplots(
        2,
        len(sigmas),
        figsize=(6.4, 4.5),
        layout="constrained",
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
    for column, sigma in enumerate(sigmas):
        rows = [r for r in record["rows"] if r["sigma_depth_rad_m2"] == sigma]
        for row, color in zip(rows, colors):
            profile = row["profile"]
            label = profile["kind"]
            if label == "screen":
                label += f" f={profile['fraction']:g}"
            for ax, component in zip(axes[:, column], ("smooth", "speckle")):
                values = np.array(row[component + "_fraction"])
                freq = np.array(record["freqs_mhz"])
                ax.plot(
                    freq,
                    np.maximum(values, 1e-12),
                    "o-",
                    ms=3,
                    lw=1,
                    color=color,
                    label=label,
                )
                hidden = values < 1e-12
                ax.plot(
                    freq[hidden],
                    np.full(hidden.sum(), 1e-12),
                    "v",
                    color=color,
                    ms=4,
                )
                ax.set(yscale="log", ylim=(8e-13, 2))
                ax.grid(alpha=0.15)
        axes[0, column].set_title(f"sigma_phi={sigma:g} rad/m²")
        axes[1, column].set_xlabel("Frequency (MHz)")
    axes[0, 0].set_ylabel("Smooth power / unrotated power")
    axes[1, 0].set_ylabel("Speckle power / unrotated power")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7)
    save(
        fig,
        "paper_depth_depolarization",
        "Preserved flat-plane depth-profile benchmark, rendered at the "
        "report's print width. Smooth |E[P]| squared and centered speckle "
        "power are divided by the same unrotated beam-weighted reference. "
        "The Gaussian profile is supported on f in [0,1]; fractions 0/0.6/1 "
        "retain the original screen limits. Monochromatic points use "
        "10/30/50 MHz; lines guide the eye. Triangles mark values below "
        "the 1e-12 display floor, including exact zeros and underflow. "
        "The source/beam geometry and local screen parameters remain "
        "conditional. This independent benchmark includes no real "
        "four-port beam, channel averaging, leakage or receiver noise; "
        "those enter the subsequent instrument predictions.",
        [path],
        generator=__file__,
    )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    plot()
