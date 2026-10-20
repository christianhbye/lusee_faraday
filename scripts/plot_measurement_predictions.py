"""Figures from the common-estimator spectra, covariance and decision table."""

import common  # noqa: F401
import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

from common import FIG_DIR, REPO
from faraday_separability import load_array
from paper_common import digest, write_json

plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
COLORS = ["#0072B2", "#D55E00", "#009E73"]


def save(fig, name, caption, records, generator=None):
    def log_tick(value, position):
        return f"{value:.0e}".replace("e+0", "e+").replace("e-0", "e-")

    for ax in fig.axes:
        if ax.get_yscale() == "log":
            if hasattr(ax, "_colorbar"):
                lo, hi = ax.get_ylim()
                if hi / lo < 100:
                    ax.yaxis.set_major_locator(
                        LogLocator(subs=(1, 2, 5), numticks=8)
                    )
            ax.yaxis.set_major_formatter(FuncFormatter(log_tick))
            ax.yaxis.set_minor_formatter(NullFormatter())
    path = FIG_DIR / name
    generator = Path(generator or __file__).resolve()
    fig.savefig(
        path.with_suffix(".pdf"),
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(path.with_suffix(".png"), dpi=170)
    plt.close(fig)
    path.with_suffix(".caption.txt").write_text(caption + "\n")
    write_json(
        path.with_suffix(".manifest.json"),
        dict(
            generator=str(generator.relative_to(REPO)),
            generator_sha256=digest(generator),
            render_helper=str(Path(__file__).relative_to(REPO)),
            render_helper_sha256=digest(__file__),
            input_records={
                str(p.relative_to(REPO)): digest(p) for p in records
            },
            pdf_sha256=digest(path.with_suffix(".pdf")),
            caption=caption,
        ),
    )


def limits(axes, padding=0.05):
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    delta = max(hi - lo, 1e-30)
    for ax in axes:
        ax.set_ylim(lo - padding * delta, hi + padding * delta)


def correlation(c):
    rms = np.sqrt(np.maximum(np.diag(c), 0))
    return np.divide(
        c,
        np.outer(rms, rms),
        out=np.full(c.shape, np.nan),
        where=np.outer(rms, rms) > 0,
    )


def spectra(record, records):
    example = record["observations"][0]
    profile = example["profile"]
    profile_name = {
        "slab": "uniform emitting slab",
        "gaussian": "Gaussian depth profile supported on [0,1]",
        "screen": f"screen at fraction {profile.get('fraction', 1):g}",
    }[profile["kind"]]
    intrinsic = example["intrinsic"]
    voltage = example["voltage_reference_nv_rt_hz"]
    current = example["current_reference_pa_rt_hz"]
    fig, axes = plt.subplots(5, 2, figsize=(6.4, 7.0), layout="constrained")
    for column, row in enumerate(record["observations"]):
        data = load_array(row)
        nominal = data["nominal_mhz"]
        center = row["center_mhz"]
        sort = np.argsort(nominal[3:]) + 3
        x = (nominal[sort] - center) * 1e3
        parent_x = (nominal[:3] - center) * 1e3
        intensity = data["leakage"][:, 0]
        r = data["operator"]
        for component in (1, 2):
            ax = axes[component - 1, column]
            for name, label, color in (
                ("observation", "Simulated data", "black"),
                ("leakage_noise_control", "I-only + same noise", "#888888"),
            ):
                value = data[name][:, component] / intensity
                ax.plot(x, value[sort], color=color, lw=0.6, label=label)
                ax.plot(
                    parent_x, value[:3], "o", mfc="none", color=color, ms=4
                )
            fitted = (
                (np.eye(len(r)) - r) @ data["observation"][:, component]
            ) / intensity
            ax.plot(
                x,
                fitted[sort],
                "--",
                color=COLORS[2],
                label="Smooth fit",
                lw=1,
            )
            ax.set_ylabel(("Q" if component == 1 else "U") + " / I")
            if component == 1:
                ax.set_title(f"{center:g} MHz; {row['exposure_s']:.0f} s")
            if column == 0 and component == 1:
                ax.legend(fontsize=7, loc="best")
            ax = axes[component + 1, column]
            observed = (r @ data["observation"][:, component]) / intensity
            control = (
                r @ data["leakage_noise_control"][:, component]
            ) / intensity
            noise = (
                np.sqrt(
                    np.maximum(
                        np.diag(
                            data["residual_noise_covariance"][
                                :, component, :, component
                            ]
                        ),
                        0,
                    )
                )
                / intensity
            )
            model = (
                np.sqrt(
                    np.maximum(
                        np.diag(
                            (
                                data["residual_noise_covariance"]
                                + data["residual_signal_covariance"]
                            )[:, component, :, component]
                        ),
                        0,
                    )
                )
                / intensity
            )
            model_mean = (r @ data["leakage"][:, component]) / intensity
            ax.fill_between(
                x,
                model_mean[sort] - model[sort],
                model_mean[sort] + model[sort],
                color=COLORS[0],
                alpha=0.15,
                label="Conditional sky + noise RMS",
            )
            ax.plot(
                x,
                model_mean[sort] + noise[sort],
                ":",
                color="black",
                lw=0.8,
                label="Radiometer RMS",
            )
            ax.plot(
                x, model_mean[sort] - noise[sort], ":", color="black", lw=0.8
            )
            ax.plot(x, model_mean[sort], "--", color=COLORS[2], lw=0.8)
            ax.plot(x, observed[sort], color="black", lw=0.6)
            ax.plot(x, control[sort], color="#888888", lw=0.6)
            ax.errorbar(
                parent_x,
                observed[:3],
                yerr=noise[:3],
                fmt="o",
                ms=4,
                color=COLORS[1],
                label="Parents ± noise RMS",
            )
            ax.set_ylabel(
                "Residual " + ("Q" if component == 1 else "U") + " / I"
            )
            if column == 0 and component == 1:
                ax.legend(
                    fontsize=7,
                    loc="lower left",
                    bbox_to_anchor=(0, 1),
                    borderaxespad=0,
                )
        ax = axes[4, column]
        for component in (1, 2):
            for name, style in (("smooth", "--"), ("speckle", "-")):
                ax.plot(
                    x,
                    data[name][sort, component] / intensity[sort],
                    style,
                    color=COLORS[component - 1],
                    lw=0.8,
                    label=("Q" if component == 1 else "U") + " " + name,
                )
        ax.set(
            ylabel="Polarized component / I", xlabel="Frequency offset (kHz)"
        )
        if column == 0:
            ax.legend(ncol=2, fontsize=7)
        for ax in axes[:, column]:
            ax.set_xlim(-40, 40)
            ax.grid(alpha=0.15)
    for i in range(5):
        limits(axes[i])
    save(
        fig,
        "faraday_measured_spectra",
        "Simulated observations, not mission data: as-built calibrated Q/I "
        f"and U/I at preselected sidereal phase {example['sidereal_turn']:g}, "
        f"{profile_name} and sigma_phi={example['screen']['sigma_depth']:g} "
        "rad/m^2. Zoom curves retain "
        "all folded images; open circles are three parent measurements. "
        "A constant and physical-frequency slope pass through each channel "
        "response before fitting. Residuals are subtracted in K, then divided "
        "by the same fixed channelized sky-I reference as the components. "
        "Estimated-I covariances, including I-Q/U correlations, are exported. "
        "The paired I-only control shares the noise draw. The last row shows "
        "the underlying smooth and speckle polarization draws; only noise "
        "changes between repeated visits. Intrinsic RMS is "
        f"{intrinsic['p_ref_k']:g} K at {intrinsic['ref_mhz']:g} MHz; "
        "local structure, turnover, current noise and temperatures "
        "are assumed. "
        f"Voltage/current references are {voltage:g} nV/sqrt(Hz), "
        f"{current:g} pA/sqrt(Hz). "
        f"Exposure spans {example['lunations']} illustrative lunations "
        f"at {100*example['electronics_duty_cycle']:g}% duty. "
        "The colored band shows marginal conditional sky-plus-noise RMS "
        "around the residual I-only model (dashed). Native interpolation "
        "limits "
        "fine-channel leakage interpretation.",
        records,
    )


def covariances(record, records):
    old = FIG_DIR / "faraday_channel_covariance.pdf"
    if (
        old.exists()
        and not (
            FIG_DIR / "faraday_channel_covariance_signal_only.pdf"
        ).exists()
    ):
        for suffix in (".pdf", ".png", ".caption.txt", ".manifest.json"):
            source = old.with_suffix(suffix)
            if source.exists():
                shutil.copy2(
                    source,
                    FIG_DIR
                    / ("faraday_channel_covariance_signal_only" + suffix),
                )
    fig, axes = plt.subplots(3, 2, figsize=(6.4, 7.4), layout="constrained")
    repeat_fig, repeat_axes = plt.subplots(
        2, 2, figsize=(6.4, 4.8), layout="constrained"
    )
    before_images, after_images, maximum_loss = [], [], 0.0
    for column, row in enumerate(record["observations"]):
        data = load_array(row)
        nominal, center, r = (
            data["nominal_mhz"],
            row["center_mhz"],
            data["operator"],
        )
        order = np.argsort(nominal[3:]) + 3
        x = (nominal[order] - center) * 1e3
        extent = [x[0] - (x[1] - x[0]) / 2, x[-1] + (x[1] - x[0]) / 2] * 2
        for i, name in enumerate(
            ("raw_signal_covariance", "residual_signal_covariance")
        ):
            c = correlation(data[name][:, 1, :, 1])
            matrix = c[np.ix_(order, order)]
            if i == 0:
                loss = np.maximum(1 - matrix, 1e-12)
                maximum_loss = max(maximum_loss, float(loss.max()))
                image = axes[i, column].imshow(
                    loss,
                    origin="lower",
                    extent=extent,
                    cmap="viridis",
                    norm=LogNorm(vmin=1e-12, vmax=1),
                    aspect="auto",
                )
                before_images.append(image)
            else:
                image = axes[i, column].imshow(
                    matrix,
                    origin="lower",
                    extent=extent,
                    cmap="RdBu_r",
                    vmin=-1,
                    vmax=1,
                    aspect="auto",
                )
                after_images.append(image)
            axes[i, column].set(
                ylabel="Offset (kHz)",
                xlabel="Offset (kHz)",
                title=f"{center:g} MHz; "
                + ("before: 1-r" if i == 0 else "after: r"),
            )
        ax = axes[2, column]
        reference = int(order[np.argmin(abs(nominal[order] - center))])
        for component in (1, 2):
            for name, style, label in (
                ("raw_signal_covariance", ":", "before"),
                ("residual_signal_covariance", "-", "after"),
            ):
                c = correlation(data[name][:, component, :, component])
                ax.plot(
                    (nominal[order] - nominal[reference]) * 1e3,
                    c[reference, order],
                    style,
                    color=COLORS[component - 1],
                    lw=0.9,
                    label=("Q" if component == 1 else "U") + " " + label,
                )
        noise = correlation(
            data["single_visit_residual_noise_covariance"][:, 1, :, 1]
        )
        ax.plot(
            (nominal[order] - nominal[reference]) * 1e3,
            noise[reference, order],
            "--",
            color="#777777",
            lw=0.8,
            label="Noise control",
        )
        ax.set(
            xlabel="Separation from center bin (kHz)",
            ylabel="Correlation",
            ylim=(-1.05, 1.05),
        )
        ax = repeat_axes[0, column]
        noise_rms = np.sqrt(
            np.maximum(
                np.diag(
                    data["single_visit_residual_noise_covariance"][:, 1, :, 1]
                ),
                0,
            )
        )
        for name, label, color in (
            ("single_visit_first", "First cycle", "black"),
            ("single_visit_repeat", "Second cycle", COLORS[1]),
        ):
            y = r @ data[name][:, 1]
            ax.plot(x, y[order], lw=0.7, color=color, label=label)
            ax.plot(
                (nominal[:3] - center) * 1e3, y[:3], "o", color=color, ms=3
            )
        ax.set(title=f"{center:g} MHz; fixed sky", ylabel="Residual Q (K)")
        if column == 0:
            ax.legend(fontsize=7)
        ax = repeat_axes[1, column]
        difference = r @ (
            data["single_visit_first"][:, 1]
            - data["single_visit_repeat"][:, 1]
        )
        y = difference / (np.sqrt(2) * noise_rms)
        ax.fill_between([-40, 40], -1, 1, color="#cccccc", alpha=0.5)
        ax.plot(x, y[order], color="black", lw=0.8)
        ax.plot(
            (nominal[:3] - center) * 1e3, y[:3], "o", color=COLORS[1], ms=3
        )
        ax.set(
            xlabel="Frequency offset (kHz)",
            ylabel="Difference / noise RMS",
            xlim=(-40, 40),
        )
    handles, labels = axes[2, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=7)
    loss_limit = 10 ** np.ceil(np.log10(max(maximum_loss, 1e-11)))
    for image in before_images:
        image.set_clim(1e-12, loss_limit)
    fig.colorbar(
        before_images[0],
        ax=axes[0].tolist(),
        label="1 - Q correlation",
        shrink=0.8,
    )
    fig.colorbar(
        after_images[0], ax=axes[1].tolist(), label="Q correlation", shrink=0.8
    )
    limits(repeat_axes[0])
    limits(repeat_axes[1])
    save(
        fig,
        "faraday_channel_covariance",
        "Conditional slab-signal Q correlations before and after the common "
        "smooth fit, on all 192 nominal zoom frequencies, for the same model "
        "and time bin as the measured-spectra figure. "
        "Before-fit images show 1-r on a common log scale with floor 1e-12; "
        "after-fit images show signed r. The last row retains "
        "individual Q and U correlations to the named center-frequency bin; "
        "unlike frequency pairs are not averaged into a stationary estimate. "
        "The noise-control line includes channel folds, overlap and the same "
        "fit. Full Q-U cross covariance and both parent/zoom blocks "
        "are exported. "
        "Repeated-cycle differences and their radiometer prediction are shown "
        "in faraday_repeatability.pdf. These conditional correlations are not "
        "a Faraday identification or an observed covariance estimate.",
        records,
    )
    save(
        repeat_fig,
        "faraday_repeatability",
        "Two simulated repeat measurements of the same sky and instrument at "
        f"sidereal phase {record['observations'][0]['sidereal_turn']:g}. "
        "Top: Q residuals, with parent points and "
        "zoom curves. "
        "Bottom: their difference divided by the predicted independent-repeat "
        "radiometer RMS, sqrt(2) times the individual RMS; gray is ±1. "
        "The sky, smooth polarization, speckle and leakage are held fixed. "
        "Noise draws are independent; each cycle uses one eligible "
        "phase-bin exposure at "
        f"{100*record['observations'][0]['electronics_duty_cycle']:g}% "
        "duty, exported in seconds. Correlated "
        "power sampling uncertainties and independent mock observations are "
        "recorded, without counting channels as independent skies.",
        records,
    )


def covariance_amplitudes(record, records):
    """Keep Q and U covariance units beside the normalized correlations."""
    fig, axes = plt.subplots(4, 2, figsize=(6.4, 7.0), layout="constrained")
    raw_images, residual_images, raw_values, residual_values = [], [], [], []
    for column, row in enumerate(record["observations"]):
        data = load_array(row)
        nominal, center = data["nominal_mhz"], row["center_mhz"]
        order = np.argsort(nominal[3:]) + 3
        x = (nominal[order] - center) * 1e3
        step = np.diff(x).mean()
        extent = [x[0] - step / 2, x[-1] + step / 2] * 2
        for filtered, name in enumerate(
            ("raw_signal_covariance", "residual_signal_covariance")
        ):
            for component in (1, 2):
                covariance = data[name][:, component, :, component]
                matrix = covariance[np.ix_(order, order)]
                ax = axes[2 * filtered + component - 1, column]
                stokes = "Q" if component == 1 else "U"
                if filtered:
                    image = ax.imshow(
                        matrix,
                        origin="lower",
                        extent=extent,
                        aspect="auto",
                        cmap="RdBu_r",
                        vmin=-1,
                        vmax=1,
                    )
                    residual_images.append(image)
                    residual_values.append(matrix)
                else:
                    if np.any(matrix <= 0):
                        raise ValueError(
                            "raw illustrated covariance not positive"
                        )
                    image = ax.imshow(
                        matrix,
                        origin="lower",
                        extent=extent,
                        aspect="auto",
                        cmap="viridis",
                        norm=LogNorm(vmin=1e-30, vmax=1),
                    )
                    raw_images.append(image)
                    raw_values.append(matrix)
                ax.set(
                    title=f"{center:g} MHz; {stokes} "
                    + ("after fit" if filtered else "before fit"),
                    xlabel="Offset (kHz)",
                    ylabel="Offset (kHz)",
                )
    raw_limits = min(x.min() for x in raw_values), max(
        x.max() for x in raw_values
    )
    for image in raw_images:
        image.set_clim(*raw_limits)
    residual_limit = max(abs(x).max() for x in residual_values)
    for image in residual_images:
        image.set_clim(-residual_limit, residual_limit)
    fig.colorbar(
        raw_images[0],
        ax=axes[:2].ravel().tolist(),
        label="Before-fit signal covariance (K²)",
        shrink=0.7,
    )
    fig.colorbar(
        residual_images[0],
        ax=axes[2:].ravel().tolist(),
        label="Residual signal covariance (K²)",
        shrink=0.7,
    )
    save(
        fig,
        "faraday_channel_covariance_amplitude",
        "Conditional total polarized-signal Q and U covariances on all "
        "192 nominal zoom frequencies, before and after the common fit, for "
        "the same reference slab and phase as the measured spectra. The "
        "upper four panels share one positive logarithmic scale; the lower "
        "four share one signed linear scale. Each is in K squared at the "
        "common loaded calibration plane. The before/after groups use "
        "different scales because the fit removes most of this coherent "
        "reference signal. These are ensemble signal covariances, not "
        "covariances estimated from a single spectrum. The separate noise "
        "control and full Q-U, parent/zoom and complex C/J blocks are "
        "exported; normalized correlations are shown in the companion.",
        records,
    )


def separability(record, records):
    example = record["observations"][0]
    phase, lunations = example["sidereal_turn"], example["lunations"]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.2), layout="constrained")
    labels = ["f=0", "f=0.6", "f=1", "slab", "Gaussian"]
    for i, center in enumerate((30, 50)):
        for j, kind in enumerate(("parent", "zoom")):
            ax = axes[i, j]
            rows = [
                d
                for d in record["decisions"]
                if d["center_mhz"] == center
                and d["channel_type"] == kind
                and d["lunations"] == example["lunations"]
                and d["phase_index"] == example["phase_index"]
            ]
            for component, color, offset in (
                ("Q", COLORS[0], -0.13),
                ("U", COLORS[1], 0.13),
            ):
                for profile in range(5):
                    values = [
                        d["speckle_to_noise_amplitude"]
                        for d in rows
                        if d["component"] == component
                        and d["profile_index"] == profile
                    ]
                    low, high = min(values), max(values)
                    floor = 1e-14
                    ax.plot(
                        [profile + offset] * 2,
                        [max(low, floor), max(high, floor)],
                        color=color,
                        lw=2,
                        label=component if profile == 0 else None,
                    )
                    ax.plot(
                        profile + offset,
                        max(high, floor),
                        "v" if high == 0 else "o",
                        color=color,
                        ms=4,
                    )
            ax.axhline(1, color="black", ls="--", lw=0.8, label="Equal RMS")
            ax.set(
                yscale="log",
                xticks=np.arange(5),
                xticklabels=labels,
                ylabel="Speckle RMS / radiometer RMS",
                title=f"{center} MHz; {kind}",
            )
            ax.tick_params(axis="x", labelrotation=20)
            ax.grid(axis="y", alpha=0.2)
    all_axes = axes.ravel()
    limits(all_axes, 0)
    all_axes[0].legend(fontsize=7)
    save(
        fig,
        "faraday_separability_reference",
        "Residual Q/U speckle RMS divided by residual radiometer RMS, after "
        "the same constant+slope fit, at preselected phase "
        f"{phase:g} and the {lunations}-lunation "
        "illustrative exposure. Vertical ranges span both assumed local RM "
        "dispersions, Gaussian/global-variance-mixture phase statistics and "
        "all nine adopted voltage/current noise combinations. They are model "
        "ranges, not confidence intervals. Triangles at 1e-14 denote exact "
        "zero phase-induced speckle for f=0. The dashed line is equal RMS, "
        "not a detection threshold. The decision record separately gives "
        "retained signal power, residual leakage, interpolation and ablation "
        "mismatch, numerical errors, both phases and 1/24-lunation "
        "availability. "
        "Intrinsic polarization RMS is "
        f"{example['intrinsic']['p_ref_k']:g} K at "
        f"{example['intrinsic']['ref_mhz']:g} MHz; signal "
        "amplitudes scale linearly and no local emissivity is inferred.",
        records,
    )


def leakage_figure(system, records):
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.7), layout="constrained")
    for row in system["leakage"]:
        data = load_array(row)
        column = 0 if row["center_mhz"] == 30 else 1
        variant = ("asbuilt", "c4sym", "diagza").index(row["response"])
        a = data["readout"]
        s = np.einsum("pab,ftba->ftp", a, data["native_covariance"]).real[:, 0]
        for component in (1, 2):
            axes[component - 1, column].plot(
                row["native_mhz"],
                s[:, component] / s[:, 0],
                "o-",
                color=COLORS[variant],
                ms=3,
                lw=0.9,
                label=row["response"],
            )
            axes[component - 1, column].set(
                xlabel="Native response frequency (MHz)",
                ylabel=("Q" if component == 1 else "U") + " / I",
                title=f"{row['center_mhz']:g} MHz calibration",
            )
    axes[0, 0].legend(fontsize=7)
    for line in axes:
        limits(line)
    save(
        fig,
        "faraday_instrument_leakage",
        "I-only native-frequency Haslam+CMB harmonic predictions at sidereal "
        "phase 0 through the as-built, C4-symmetrized and diagonal-ZA "
        "responses. Every variant uses the as-built band's fixed zenith "
        "calibration and enclosure normalization, plus identical MEPA time "
        "averaging. Curves show actual native frequencies; fine-channel "
        "linear/PCHIP leakage and their common-fit residuals are model "
        "interpolants exported in the result record. Variant differences "
        "are a model spread, not a beam-error probability distribution. "
        "These diffuse-sky leakages need not vanish after zenith calibration.",
        records,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        type=Path,
        default=REPO / "docs/faraday-separability-results.json",
    )
    args = parser.parse_args()
    system_path = REPO / "docs/measured-system-results.json"
    record = json.loads(args.record.read_text())
    records = [args.record, system_path]
    spectra(record, records)
    covariances(record, records)
    covariance_amplitudes(record, records)
    separability(record, records)
    leakage_figure(json.loads(system_path.read_text()), records)
