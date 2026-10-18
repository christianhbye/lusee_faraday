"""Plot inspected native-band mixed-depth power and loaded RMS predictions."""

import common  # noqa: F401
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np

from common import FIG_DIR, REPO
from lusee_faraday.config import BETA_QU
from mixed_depth_predictions import setup
from paper_common import digest, load_record, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/paper/mixed-depth.json"
    )
    parser.add_argument(
        "--record", type=Path, default=REPO / "docs/mixed-depth-results.json"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    base, _, _, key, _ = setup(config)
    record = load_record(args.record, key)
    if not record["all_passed"]:
        raise ValueError("mixed-depth figure requires passing numerical gates")
    colors = ["#000000", "#CC79A7", "#0072B2", "#009E73", "#D55E00"]
    labels = [
        "Screen f=0",
        "Screen f=0.6",
        "Screen f=1",
        "Uniform slab",
        "Truncated Gaussian",
    ]
    floor = 1e-16
    series = []
    for column, case in enumerate(base["screens"]):
        for index, profile in enumerate(config["profiles"]):
            rows = sorted(
                [
                    r
                    for r in record["results"]
                    if r["kind"] == "native"
                    and r["case"] == case
                    and r["profile"] == profile
                ],
                key=lambda r: r["center_mhz"],
            )
            values = []
            for row in rows:
                path = REPO / row["path"]
                if digest(path) != row["sha256"]:
                    raise ValueError("corrupt mixed-depth figure input")
                with np.load(path) as data:
                    # Predetermined reference phase, Q+U power.
                    off = float(
                        data["off"][0, 1, 0, 1] + data["off"][0, 2, 0, 2]
                    )
                    powers = {
                        name: float(
                            data[name][0, 1, 0, 1] + data[name][0, 2, 0, 2]
                        )
                        for name in ("smooth", "speckle", "total")
                    }
                    values.append(
                        [
                            row["center_mhz"],
                            powers["smooth"] / off,
                            powers["speckle"] / off,
                            np.sqrt(max(powers["total"], 0)),
                        ]
                    )
            series.append((column, index, np.array(values)))
    with plt.rc_context(
        {
            "font.size": 8,
            "axes.titlesize": 10,
            "legend.fontsize": 7,
            "axes.labelsize": 8,
        }
    ):
        fig, axes = plt.subplots(
            3,
            len(base["screens"]),
            figsize=(6.4, 6.8),
            sharex=True,
            sharey="row",
            layout="constrained",
        )
        for column, index, values in series:
            for row in range(3):
                y = values[:, row + 1]
                axes[row, column].plot(
                    values[:, 0],
                    np.maximum(y, floor),
                    "o-",
                    color=colors[index],
                    ms=3,
                    lw=1,
                    label=labels[index],
                )
                clipped = y < floor
                if np.any(clipped):
                    axes[row, column].plot(
                        values[clipped, 0],
                        np.full(clipped.sum(), floor),
                        "v",
                        color=colors[index],
                        ms=4,
                    )
                axes[row, column].set_yscale("log")
        for column, case in enumerate(base["screens"]):
            sigma = case["parameters"]["sigma_depth"]
            axes[0, column].set_title(
                rf"$\sigma_\phi={sigma:g}$ rad m$^{{-2}}$"
            )
            axes[2, column].set_xlabel("Frequency (MHz)")
            axes[2, column].set_xticks(config["native_centers_mhz"])
        axes[0, 0].set_ylabel("Smooth Q+U power / off")
        axes[1, 0].set_ylabel("Speckle Q+U power / off")
        axes[2, 0].set_ylabel("Total Q/U RMS (K)")
        axes[0, 0].set_ylim(floor / 2, 2)
        axes[1, 0].set_ylim(floor / 2, 2)
        handles, legend = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend, loc="outside lower center", ncol=3)
        name = FIG_DIR / "faraday_redistribution_spin2"

        def log_tick(value, position):
            return f"{value:.0e}".replace("e+0", "e+").replace("e-0", "e-")

        for ax in axes.ravel():
            ax.yaxis.set_major_formatter(FuncFormatter(log_tick))
            ax.yaxis.set_minor_formatter(NullFormatter())
        fig.savefig(name.with_suffix(".pdf"))
        fig.savefig(name.with_suffix(".png"), dpi=180)
        plt.close(fig)
    caption = (
        "Conditional as-built loaded four-port predictions at the reference "
        "MEPA sidereal phase, using the two illustrative local Gaussian "
        "columns and the five shared-column emitting-depth profiles. The "
        "first two rows show Var(Q)+Var(U) for the phase-mean and centered "
        "terms, divided by the same unrotated-beam reference power. The "
        "bottom row is sqrt[Var(Q)+Var(U)] in K. Points use their own native "
        "10/30/50 MHz beams; connecting segments guide the eye. Triangles "
        "mark zero/underflow values at the 1e-16 display floor. The intrinsic "
        f"spin-2 RMS is {base['intrinsic']['p_ref_k']:g} K at "
        f"{base['intrinsic']['ref_mhz']:g} MHz with assumed E/B spectra, "
        f"beta={BETA_QU:g} and a {base['intrinsic']['turnover_mhz']:g} MHz "
        "turnover. This is an illustrative brightness law, "
        "especially below 20 MHz. Slab/Gaussian emission depths share one "
        "column rather than independent rotating layers. These are model "
        "second moments, without I-only leakage or measurement noise."
    )
    name.with_suffix(".caption.txt").write_text(caption + "\n")
    write_json(
        name.with_suffix(".manifest.json"),
        dict(
            fingerprint=key,
            record_sha256=digest(args.record),
            plot_source_sha256=digest(Path(__file__)),
            pdf_sha256=digest(name.with_suffix(".pdf")),
            caption=caption,
            display_floor=floor,
        ),
    )
    print("Saved", name.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
