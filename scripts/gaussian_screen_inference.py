"""Build the executed benchmark notebook and figures from cached results.

This reads the completed validation and calibrated-inference products. It
never launches an ensemble. Regenerate with the project Python interpreter.
"""

import common  # noqa: F401
import hashlib
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import nbformat
import numpy as np
from nbclient import NotebookClient

from calibrate_gaussian_screen import load_samples
from gaussian_screen_experiment import check_gates, noise_covariance
from lusee_faraday.gaussian_screen import ScreenModel, qu_moments
from lusee_faraday.screen_inference import (
    gaussian_kl,
    local_stationary_template,
    transform_source,
)
from validate_gaussian_screen import OUT, ROOT, write_json

matplotlib.use("Agg")

FIG = ROOT / "report/figures"


def save(fig, name):
    fig.savefig(
        FIG / f"{name}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return name


def mode_comparison(assessment):
    """Display cached profiled alternatives in one fixed reference basis."""
    model = ScreenModel(**assessment["model"])
    freq = np.asarray(assessment["freq_mhz"])
    row = next(r for r in assessment["recovery"] if r["ell"] == 0.5)
    reference = qu_moments(freq, model, method="special")
    noise = noise_covariance(freq, model)
    sd = np.sqrt(np.diag(noise))
    eigenvalues, vectors = np.linalg.eigh(
        reference["covariance"] / np.outer(sd, sd)
    )
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    # Fix otherwise arbitrary eigenvector signs for reproducible mean offsets.
    for column in range(len(sd)):
        pivot = np.argmax(abs(vectors[:, column]))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    projection = vectors.T / sd[None, :]
    projection /= np.sqrt(1 + eigenvalues)[:, None]
    reference_total = reference["covariance"] + noise
    np.testing.assert_allclose(
        projection @ reference_total @ projection.T,
        np.eye(len(sd)),
        atol=1e-9,
    )
    alternatives = []
    for comparison in row["comparisons"]:
        fit = comparison["stages"][1]
        if not fit["converged"]:
            raise RuntimeError("unconverged source fit in mode comparison")
        candidate = qu_moments(
            freq,
            replace(model, correlation_pc=comparison["alternative"]),
            method="special",
        )
        mean, signal = transform_source(
            candidate["mean"],
            candidate["covariance"],
            freq,
            model,
            *fit["parameters"],
        )
        projected_mean = projection @ (mean - reference["mean"])
        projected_total = projection @ (signal + noise) @ projection.T
        projected_kl = gaussian_kl(
            np.zeros(len(sd)),
            np.eye(len(sd)),
            projected_mean,
            projected_total,
        )
        # All ten modes must preserve the cached full-distribution comparison.
        np.testing.assert_allclose(projected_kl, fit["kl"], atol=1e-8)
        alternatives.append(
            dict(
                ell=comparison["alternative"],
                mean=projected_mean,
                covariance=projected_total,
                kl=projected_kl,
                parameters=fit["parameters"],
            )
        )

    fig = plt.figure(figsize=(9, 7.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    ax = fig.add_subplot(grid[0, :])
    modes = np.arange(1, len(sd) + 1)
    ax.errorbar(
        modes - 0.2,
        np.zeros(len(sd)),
        yerr=np.ones(len(sd)),
        fmt="o",
        color="0.25",
        capsize=3,
        label="Reference: 0.5 pc",
    )
    limit = 1.0
    for offset, color, marker, label, alternative in zip(
        (0, 0.2),
        ("#0072B2", "#D55E00"),
        ("s", "^"),
        ("Half scale", "Double scale"),
        alternatives,
    ):
        rms = np.sqrt(np.diag(alternative["covariance"]))
        ax.errorbar(
            modes + offset,
            alternative["mean"],
            yerr=rms,
            fmt=marker,
            color=color,
            capsize=3,
            label=f"{label}: {alternative['ell']:g} pc",
        )
        limit = max(limit, np.max(abs(alternative["mean"]) + rms))
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set(
        xticks=modes,
        xlabel="Reference mode (decreasing signal variance)",
        ylabel="Mean offset ± total RMS\n/ reference total RMS",
        ylim=(-1.1 * limit, 1.1 * limit),
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.23), ncol=3, fontsize=9
    )
    residuals = [a["covariance"] - np.eye(len(sd)) for a in alternatives]
    limit = max(np.max(abs(r)) for r in residuals)
    axes = []
    for column, (alternative, residual) in enumerate(
        zip(alternatives, residuals)
    ):
        ax = fig.add_subplot(grid[1, column])
        axes.append(ax)
        image = ax.imshow(
            residual,
            origin="lower",
            extent=(0.5, 10.5, 0.5, 10.5),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.set(
            xticks=modes,
            yticks=modes,
            xlabel="Reference mode",
            ylabel="Reference mode",
            title=(
                f"{alternative['ell']:g} pc: covariance difference"
                f"\n$2D_{{KL}}$ = {2 * alternative['kl']:.3f}"
            ),
        )
    fig.colorbar(
        image,
        ax=axes,
        label="Covariance difference / reference mode RMS products",
    )
    caption = r"""### Figure 2b: what the fitted scale alternatives change

The baseline is $\ell=0.5$ pc at 100 pc, using the same five frequencies
over 45–50 MHz and synthetic noise as Figure 2. The half-scale and double-scale
models use the converged amplitude/angle/spectral-index fits saved in
`assessment.json`; no new optimization or screen draws enter this figure.

Write the ten measurements as
$\mathbf y=(Q_1,\ldots,Q_5,U_1,\ldots,U_5)^T
=\boldsymbol\mu+\mathbf s+\mathbf n$.
The centered screen fluctuation has covariance $S=E[\mathbf s\mathbf s^T]$
and synthetic readout noise has covariance $N=E[\mathbf n\mathbf n^T]$.
Thus “signal” in Figure 2's left panel means $S$, excluding the ensemble mean.
For orthonormal eigenvectors
$$N^{-1/2}S_0N^{-1/2}\mathbf v_a=\lambda_a\mathbf v_a,$$
the mode $z_a=\mathbf v_a^TN^{-1/2}(\mathbf y-\boldsymbol\mu_0)$ has
reference signal variance $\lambda_a$, noise variance one, and total variance
$1+\lambda_a$. Subscript 0 denotes the reference model. Modes are ordered by
decreasing $\lambda_a$; all ten are retained, including noise-dominated modes.

For display, divide each mode by its reference total RMS:
$$W_{a:}=\frac{\mathbf v_a^TN^{-1/2}}{\sqrt{1+\lambda_a}},\qquad
\mathbf t=W(\mathbf y-\boldsymbol\mu_0).$$
The reference has mean zero and covariance $I$. An alternative has
$$\mathbf m=W(\boldsymbol\mu_{\rm alt}-\boldsymbol\mu_0),\qquad
C=W(S_{\rm alt}+N)W^T.$$
The source fit transforms both the mean and screen covariance; the readout
noise $N$ stays fixed. The upper panel shows $m_a\pm\sqrt{C_{aa}}$ alongside
the reference $0\pm1$. Bars are ensemble RMS including readout noise, not
errors on fitted parameters or confidence intervals. The lower panels show
$C-I$: diagonal entries measure variance changes, and off-diagonal entries
show correlations between reference modes introduced by the alternative.
Both matrices share one signed colour scale. All models use the same reference
basis; separately diagonalizing each alternative would hide these correlations.

At this baseline, the alternatives change the first two mode variances and
shift the second mode's mean. The double-scale model also introduces
correlations with modes 3 and 4. The models therefore predict different
distributions, although their marginal RMS ranges overlap. These differences
remain below the declared information criterion after the source fit.

The overlapping marginal RMS bars make the comparison tangible but do not
define a detection or recovery test. The covariance panels retain information
that those bars omit. Because $W$ is invertible, this full ten-mode comparison
preserves the Gaussian-model KL divergence in Figure 2 exactly; the builder
checks it against the saved values. Gaussian distributions here are the
information assessment's approximation, not an assumption established by
matching first and second moments. The plotted RMS follows from those moments
even if the screen readouts are non-Gaussian.
"""
    return save(fig, "gaussian_inference_mode_comparison"), caption


def figures(validation, strong, assessment, inference):
    figures = []
    row = next(r for r in validation["results"] if r["ell"] == 0.5)
    freq = np.array(row["config"]["freq"])
    n = len(freq)
    df = np.diff(freq)[0]
    frequency_edges = [freq[0] - df / 2, freq[-1] + df / 2]
    empirical = np.array(row["moments"]["covariance"]["empirical"])
    expected = np.array(row["moments"]["covariance"]["expected"])
    se = np.array(row["moments"]["covariance"]["se"])
    fig, axes = plt.subplots(3, 3, figsize=(9, 8), layout="constrained")
    for i, (a, b, label) in enumerate(
        [
            (slice(0, n), slice(0, n), "QQ"),
            (slice(0, n), slice(n, 2 * n), "QU"),
            (slice(n, 2 * n), slice(n, 2 * n), "UU"),
        ]
    ):
        vmax = max(abs(empirical[a, b]).max(), abs(expected[a, b]).max())
        for j, value in enumerate(
            [
                empirical[a, b],
                expected[a, b],
                (empirical[a, b] - expected[a, b]) / se[a, b],
            ]
        ):
            limit = row["moments"]["threshold"] if j == 2 else vmax
            image = axes[i, j].imshow(
                value,
                origin="lower",
                extent=frequency_edges * 2,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                aspect="auto",
            )
            axes[i, j].set(
                xlabel="Frequency (MHz)", ylabel=f"{label}: frequency (MHz)"
            )
            if i == 0:
                axes[i, j].set_title(
                    [
                        "Direct screens",
                        "Analytic covariance",
                        "Difference / MC SE",
                    ][j]
                )
            fig.colorbar(
                image,
                ax=axes[i, j],
                label="MC standard errors" if j == 2 else r"$\mathrm{K}^2$",
            )
    figures.append(
        (
            save(fig, "gaussian_inference_covariance"),
            (
                "Joint covariance from the 128-screen baseline ensemble at"
                " 45–50 MHz, with correlation length 0.5 pc at 100 pc. Direct"
                " estimates and analytic predictions share scales within each"
                " row; residuals divide by bootstrap Monte Carlo standard"
                " errors. These are screen-sampling errors, not radiometer"
                " noise. All cells, including Q–U cross terms, enter the gate."
            ),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7), layout="constrained")
    scales = np.array([r["ell"] for r in assessment["recovery"]])
    angular = np.rad2deg(scales / assessment["model"]["distance_pc"]) * 60
    eig = np.array(
        [r["noise_relative_eigenvalues"] for r in assessment["recovery"]]
    )
    for k in range(1, 4):
        axes[0].loglog(angular, eig[:, -k], "o-", label=f"Mode {k}")
    axes[0].axhline(1, color="black", ls=":", label="Signal = noise variance")
    axes[0].set(
        xlabel="Angular correlation scale (arcmin)",
        ylabel="Signal / noise mode variance",
    )
    for k, label in [(0, "Half scale"), (1, "Double scale")]:
        values = [
            2 * r["comparisons"][k]["stages"][1]["kl"]
            for r in assessment["recovery"]
        ]
        axes[1].semilogx(angular, values, "o-", label=label)
    axes[1].axhline(4, color="black", ls=":", label="Recovery criterion")
    axes[1].set(
        xlabel="Angular correlation scale (arcmin)",
        ylabel=r"$2D_{KL}$ after nuisance fit",
    )
    for ax in axes:
        ax.legend(fontsize=8)
    figures.append(
        (
            save(fig, "gaussian_inference_information"),
            (
                "Expected Gaussian-model information at five frequencies in"
                " 45–50 MHz with independent synthetic noise of 5% of"
                " unrotated polarized amplitude per Q/U sample. Left: the"
                " three largest generalized covariance modes. Right: twice the"
                " Kullback–Leibler divergence to half/double correlation"
                " length after profiling amplitude, angle and spectral index."
                " The line marks the declared design criterion, not a"
                " frequentist significance. No tested scale separates both"
                " alternatives by that criterion."
            ),
        )
    )

    figures.append(mode_comparison(assessment))

    model = ScreenModel(**assessment["model"])
    noise = noise_covariance(freq, model)
    corr = expected / np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    # Physical 1-sigma shapes from eigenvectors of N^-1/2 S N^-1/2.
    sd = np.sqrt(np.diag(noise))
    whiten = expected / np.outer(sd, sd)
    ev, vec = np.linalg.eigh(whiten)
    window = np.hanning(n)
    delay = np.fft.fftshift(np.fft.fftfreq(n, d=np.diff(freq)[0] * 1e6))
    transform = np.exp(
        -2j * np.pi * delay[:, None] * (freq - freq.mean())[None, :] * 1e6
    )
    transform *= window[None, :] / np.sqrt(np.sum(window**2))
    complex_cov = (
        expected[:n, :n]
        + expected[n:, n:]
        + 1j * (expected[n:, :n] - expected[:n, n:])
    )
    dcov = transform @ complex_cov @ transform.conj().T
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5), layout="constrained")
    for offset, label in ((0, "Q"), (n, "U")):
        axes[0, 0].plot(
            freq,
            corr[n // 2, slice(offset, offset + n)],
            "o-",
            label=f"Q(47.5 MHz) with {label}",
        )
    axes[0, 0].set(
        xlabel="Frequency (MHz)",
        ylabel="Signed correlation",
        ylim=(-1.05, 1.05),
    )
    axes[0, 0].legend(fontsize=8)
    for k, color in ((1, "#0072B2"), (2, "#D55E00")):
        mode = sd * vec[:, -k] * np.sqrt(ev[-k])
        axes[0, 1].plot(freq, mode[:n], color=color, label=f"Mode {k}: Q")
        axes[0, 1].plot(
            freq, mode[n:], color=color, ls="--", label=f"Mode {k}: U"
        )
    axes[0, 1].set(xlabel="Frequency (MHz)", ylabel="Mode RMS shape (K)")
    axes[0, 1].legend(
        fontsize=8, loc="center left", bbox_to_anchor=(0.03, 0.30)
    )
    vmax = abs(dcov).max()
    for ax, value, label in zip(
        axes[1], (dcov.real, dcov.imag), ("Real", "Imaginary")
    ):
        image = ax.imshow(
            value,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=[
                (delay[0] - np.diff(delay)[0] / 2) * 1e6,
                (delay[-1] + np.diff(delay)[0] / 2) * 1e6,
            ]
            * 2,
        )
        ax.set(xlabel=r"Delay ($\mu$s)", ylabel=r"Delay ($\mu$s)", title=label)
        fig.colorbar(image, ax=ax, label=r"Delay covariance ($\mathrm{K}^2$)")
    figures.append(
        (
            save(fig, "gaussian_inference_modes"),
            (
                "Baseline analytic covariance on the same five-frequency grid."
                " The reference slice retains signed Q–U correlations. Mode"
                " shapes include their RMS amplitudes in K relative to the"
                " declared synthetic noise. Delay covariance uses a Hann"
                " window normalized by sqrt(sum(window²)) on both axes, with"
                " the displayed complex P=Q+iU convention. Both real and"
                " imaginary off-diagonal structure are retained; five samples"
                " provide only a coarse delay representation. This is centered"
                " power; the notebook also records the transformed-mean"
                " contribution to raw delay power."
            ),
        )
    )

    diag = assessment["strong_diagonal"]
    f = np.array([r["freq_mhz"] for r in diag])
    on_power = np.array([r["centered_power"] + r["mean_power"] for r in diag])
    approximate_power = np.array([r["approximate_power"] for r in diag])
    off = qu_moments(
        f, replace(model, mean_depth=0.0, sigma_depth=0.0), method="special"
    )["off"]
    off_power = np.sum(off.reshape(2, -1) ** 2, axis=0)
    fraction = on_power / off_power
    if np.any((fraction < 0) | (fraction > 1 + 1e-12)):
        raise RuntimeError("unphysical surviving power fraction")
    fig = plt.figure(figsize=(9, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    power_ax = fig.add_subplot(grid[0, 0])
    fraction_ax = fig.add_subplot(grid[0, 1], sharex=power_ax)
    residual_ax = fig.add_subplot(grid[1, :], sharex=power_ax)
    for ax, normalization in ((power_ax, 1), (fraction_ax, off_power)):
        ax.semilogy(f, on_power / normalization, "o-", label="Full on power")
        ax.semilogy(
            f,
            approximate_power / normalization,
            "s--",
            label="Local approximation",
        )
    power_ax.semilogy(
        f, off_power, ":", color="0.35", label="Faraday-off power"
    )
    comparison_table = (
        "\n\n| Frequency (MHz) | (Direct − full) / MC SE |"
        " (Direct − local approximation) / MC SE |\n"
        "|---:|---:|---:|\n"
    )
    for label, marker, color, legend in (
        ("strong-validation", "o", "black", "Direct: 20–35 MHz"),
        ("covariance-validation", "D", "#CC79A7", "Direct: 45–50 MHz"),
    ):
        samples, _, entry = load_samples(label, 0.5)
        if ScreenModel(**entry["config"]["model"]) != model:
            raise RuntimeError("power comparison ensembles differ physically")
        sf = np.asarray(entry["config"]["freq"])
        ns = len(sf)
        raw = samples[:, :ns] ** 2 + samples[:, ns:] ** 2
        sample_power = raw.mean(0)
        error = raw.std(0, ddof=1) / np.sqrt(len(raw))
        full = np.asarray(entry["moments"]["second"]["expected"])
        expected_power = np.diag(full)[:ns] + np.diag(full)[ns:]
        sample_off = qu_moments(
            sf,
            replace(model, mean_depth=0.0, sigma_depth=0.0),
            method="special",
        )["off"]
        sample_off_power = np.sum(sample_off.reshape(2, -1) ** 2, axis=0)
        power_ax.errorbar(
            sf,
            sample_power,
            yerr=error,
            fmt=marker,
            color=color,
            label=legend,
        )
        fraction_ax.errorbar(
            sf,
            sample_power / sample_off_power,
            yerr=error / sample_off_power,
            fmt=marker,
            color=color,
            label=legend,
        )
        residual_ax.errorbar(
            sf,
            (sample_power - expected_power) / error,
            yerr=1,
            fmt=marker,
            color=color,
        )
        # Each ensemble retains its own declared simultaneous moment bound.
        threshold = entry["moments"]["threshold"]
        for sign in (-1, 1):
            residual_ax.hlines(
                sign * threshold,
                sf.min(),
                sf.max(),
                color=color,
                linestyles=":",
            )
        if label == "covariance-validation":
            for i, frequency in enumerate(sf):
                local = local_stationary_template(
                    np.array([frequency]), model, frequency
                )
                local_power = np.trace(local["covariance"])
                comparison_table += (
                    f"| {frequency:g} |"
                    f" {(sample_power[i] - expected_power[i])/error[i]:.2f} |"
                    f" {(sample_power[i] - local_power)/error[i]:.2f} |\n"
                )
    fraction_ax.axhline(1, color="0.35", ls=":", label="No depolarization")
    power_ax.set(
        ylabel=r"$\langle |P|^2\rangle$ ($\mathrm{K}^2$)",
        title="Absolute polarized power",
    )
    fraction_ax.set(
        ylabel=r"$D=\langle |P_{\rm on}|^2\rangle/|P_{\rm off}|^2$",
        title="Surviving power fraction",
    )
    for ax in (power_ax, fraction_ax):
        ax.legend(fontsize=8)
        ax.set(xlabel="Frequency (MHz)")
    residual_ax.axhline(0, color="gray")
    residual_ax.set(xlabel="Frequency (MHz)", ylabel="Residual / MC SE")
    figures.append(
        (
            save(fig, "gaussian_inference_template"),
            (
                "Physical polarization power for the baseline"
                " source/beam/screen. Direct-screen points use 128"
                " individually converged realizations per ensemble: black"
                " circles at 20–35 MHz (`strong-validation`) and magenta"
                " diamonds at 45–50 MHz (`covariance-validation`, baseline"
                " scale). Each ensemble contains independent screen draws;"
                " frequencies within an ensemble share the same screens."
                " Errors are screen Monte Carlo errors. Left:"
                " absolute Faraday-on power and the deterministic Faraday-off"
                " power through the same source and beam. Right: both on"
                " predictions and the direct-screen estimates divided by"
                " that off power. This is a power fraction, not a polarized"
                " amplitude fraction. The local"
                " small-separation approximation supplies its own physical"
                " amplitude and approaches the full expression toward strong"
                " rotation. The residual panel compares direct sums with the"
                " full prediction, with each ensemble's own simultaneous"
                " moment bounds drawn over its frequency window. The"
                " high-frequency simulations test the separated curves"
                " directly; the table below reports residuals against both"
                " predictions using the same sample standard errors. These"
                " correlated residuals are diagnostics, not a newly calibrated"
                " model-selection significance. Curves below 20 MHz have"
                " reduced-integral"
                " checks but no direct-screen validation in this run."
                " Narrow-band covariance-shape errors are tabulated"
                " separately."
                "\n\nFor this fixed-angle Gaussian source, the off power is"
                r" $|P_{\rm off}|^2=(\eta P_{\rm ref})^2"
                r"(\nu/\nu_{\rm ref})^{2\beta}$, where $\eta$ is the beam"
                " dilution factor. Defining the surviving fraction as"
                r" $D(\nu)=\langle|P_{\rm on}|^2\rangle/|P_{\rm off}|^2$"
                " separates intrinsic brightening toward low frequency from"
                " stronger fractional depolarization. Their product produces"
                " a minimum in the sampled absolute-power curve near"
                f" {f[np.argmin(on_power)]:g} MHz. This is an ensemble"
                " mean-square minimum for the declared toy parameters, not a"
                " predicted minimum in every individual spectrum.\n\n"
                "Whether this competition produces a measurable minimum in"
                " realistic LuSEE spectra is a candidate follow-up. Its"
                " existence and frequency must be assessed with realistic"
                " emission, screen structure, beam chromaticity and channel"
                " averaging before interpreting a feature in data; no such"
                " validation is supplied here." + comparison_table
            ),
        )
    )

    observation = np.load(OUT / "observation-final-validation-0.5.npz")
    data = observation["data"][0]
    mean = observation["mean"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        layout="constrained",
    )
    for j, label in enumerate(("Q", "U")):
        sl = slice(j * n, (j + 1) * n)
        rms = np.sqrt(np.diag(expected + noise)[sl])
        axes[0, j].fill_between(
            freq,
            mean[sl] - rms,
            mean[sl] + rms,
            color="#0072B2",
            alpha=0.18,
            label="Ensemble RMS",
        )
        axes[0, j].plot(freq, mean[sl], color="#0072B2", label="Ensemble mean")
        axes[0, j].errorbar(
            freq,
            data[sl],
            yerr=sd[sl],
            fmt="o",
            color="black",
            label="One noisy screen",
        )
        axes[0, j].set(ylabel=f"{label} (K)")
        axes[0, j].legend(fontsize=8)
        axes[1, j].errorbar(
            freq, data[sl] - mean[sl], yerr=sd[sl], fmt="o", color="black"
        )
        axes[1, j].plot(freq, rms, color="#0072B2", ls=":")
        axes[1, j].plot(freq, -rms, color="#0072B2", ls=":")
        axes[1, j].axhline(0, color="gray")
        axes[1, j].set(xlabel="Frequency (MHz)", ylabel="Data − mean (K)")
    figures.append(
        (
            save(fig, "gaussian_inference_observation"),
            (
                "The first final-validation screen at ell=0.5 pc, chosen by"
                " seed order, with independent synthetic readout noise. Error"
                " bars show that noise; the band and dotted residual curves"
                " show total ensemble RMS, not a confidence interval for this"
                " particular hidden screen. The inference receives Q/U alone"
                " and never the paired Faraday-off realization."
            ),
        )
    )

    intensity = (
        model.eta * model.i_ref_k * (freq / model.ref_mhz) ** model.beta
    )
    physical = observation["physical_sample"][0]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), layout="constrained")
    curves = [
        ("I", intensity, intensity, np.zeros(n)),
        (
            "Vxx = (I+Q)/2",
            (intensity + physical[:n]) / 2,
            (intensity + mean[:n]) / 2,
            np.sqrt(np.diag(expected)[:n]) / 2,
        ),
        (
            "Vyy = (I−Q)/2",
            (intensity - physical[:n]) / 2,
            (intensity - mean[:n]) / 2,
            np.sqrt(np.diag(expected)[:n]) / 2,
        ),
        (
            "Vxy = U/2",
            physical[n:] / 2,
            mean[n:] / 2,
            np.sqrt(np.diag(expected)[n:]) / 2,
        ),
    ]
    for ax, (label, value, prediction, scatter) in zip(axes.flat, curves):
        ax.fill_between(
            freq,
            prediction - scatter,
            prediction + scatter,
            color="#0072B2",
            alpha=0.18,
        )
        ax.plot(freq, prediction, color="#0072B2", label="Ensemble mean")
        ax.plot(freq, value, "o", color="black", label="One physical screen")
        ax.set(xlabel="Frequency (MHz)", ylabel=label + " (K)")
        ax.legend(fontsize=8)
    figures.append(
        (
            save(fig, "gaussian_inference_readouts"),
            (
                "Ideal x/y products from the same first physical"
                " final-validation screen, before synthetic readout noise. I"
                " is deterministic. The bands show screen RMS about each"
                " ensemble mean. Vxx, Vyy and Vxy are algebraic displays of"
                " I/Q/U, not extra independent observations; only the"
                " nonredundant joint Q/U vector enters inference."
            ),
        )
    )

    m = np.array(row["moments"]["mean"]["expected"])
    raw_second = np.array(row["moments"]["delta_second"]["expected"])
    amplitude = (
        model.eta * model.p_ref_k * (freq / model.ref_mhz) ** model.beta
    )
    off = amplitude * np.exp(2j * np.deg2rad(model.angle_deg))
    delta_mean = m - np.r_[off.real, off.imag]
    fig, axes = plt.subplots(
        1, 2, figsize=(8, 3.5), sharey=True, layout="constrained"
    )
    for j, ax in enumerate(axes):
        sl = slice(j * n, (j + 1) * n)
        ax.semilogy(freq, np.diag(raw_second)[sl], "o-", label="On/off MSE")
        ax.semilogy(
            freq, delta_mean[sl] ** 2, "s--", label="Squared mean shift"
        )
        ax.semilogy(freq, np.diag(expected)[sl], "^:", label="Screen variance")
        ax.set(
            xlabel="Frequency (MHz)",
            ylabel=("Q", "U")[j] + r" power ($\mathrm{K}^2$)",
        )
        ax.legend(fontsize=8)
    figures.append(
        (
            save(fig, "gaussian_inference_mse"),
            (
                "Baseline paired on/off mean-square difference decomposed into"
                " the squared ensemble mean shift and centered screen variance"
                " at 45–50 MHz. All terms use the same beam-averaged K"
                " reference plane. This retains the original diagnostic;"
                " hidden per-screen off truth is not provided to any detection"
                " test."
            ),
        )
    )

    frozen = OUT / "frozen-experiment.json"
    cache = (
        ROOT
        / "generated_data/cache/gaussian_screen_inference"
        / ("calibration-" + hashlib.sha256(frozen.read_bytes()).hexdigest())
    )
    null = {}
    null_depth = {}
    for key in ("known-0", "1-Hc-0"):
        worst = max(
            (r for r in inference["null_tests"] if r["name"] == key),
            key=lambda r: r["interval"][1],
        )
        di = inference["config"]["null_depths"].index(worst["depth"])
        null_depth[key] = worst["depth"]
        chunks = []
        for start in range(
            0, inference["config"]["null_validation_samples"], 128
        ):
            with np.load(cache / f"validation-{di}-{start}.npz") as chunk:
                chunks.append(chunk[key])
        null[key] = np.concatenate(chunks)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), layout="constrained")
    for ax, key, label in zip(
        axes, ("known-0", "1-Hc-0"), ("Known source: H0", "Fitted source: Hc")
    ):
        nullscore = null[key]
        signal = observation[key]
        threshold = inference["thresholds"][key]
        bins = np.linspace(
            min(nullscore.min(), signal.min()),
            max(nullscore.max(), signal.max()),
            45,
        )
        ax.hist(
            nullscore,
            bins=bins,
            density=True,
            histtype="step",
            label=f"Null c={null_depth[key]:g}: {len(nullscore)}",
        )
        ax.hist(
            signal,
            bins=bins,
            density=True,
            histtype="step",
            label=f"Screen: {len(signal)}",
        )
        ax.axvline(
            threshold, color="black", ls=":", label="Calibrated threshold"
        )
        ax.set(
            title=label,
            xlabel="Test statistic",
            ylabel="Probability density",
            yscale="log",
        )
        ax.text(
            0.98,
            0.97,
            f"Above threshold:\nnull {sum(nullscore > threshold)}\nscreen"
            f" {sum(signal > threshold)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        ax.legend(loc="lower left", fontsize=8)
        if key == "known-0":
            inset = ax.inset_axes([0.28, 0.69, 0.36, 0.24])
            inset.hist(nullscore, bins=25, density=True, histtype="step")
            inset.axvline(threshold, color="black", ls=":")
            inset.set_title("Null detail", fontsize=8)
            inset.tick_params(labelsize=7)
    figures.append(
        (
            save(fig, "gaussian_inference_detection"),
            (
                "Held-out null and ell=0.5 pc screen statistics with"
                " thresholds fixed on separate calibration draws. Left: twice"
                " the maximum Gaussian log-likelihood difference over 41"
                " scales against zero rotation with known source. Right:"
                " covariance-weighted residual discrepancy after fitting a"
                " constant-depth screen and source amplitude, angle and beta."
                " It has an empirical null distribution, with no assumed"
                " chi-square law. Displayed counts are final-validation"
                " outcomes. Each panel shows the tested null depth with the"
                " largest false-positive upper bound. The Hc threshold is the"
                " largest of three calibration quantiles. The inset resolves"
                " the narrow known-source null distribution."
            ),
        )
    )
    plots_name, caption = figures[-1]
    failed = [r for r in inference["null_tests"] if not r["passed"]]
    if failed:
        caption += (
            " The fitted-beta full-covariance Hc statistic fails the frozen "
            "null acceptance check; this panel reports its outcome without "
            "claiming validated false-positive control."
        )
    figures[-1] = (plots_name, caption)
    mean_p = np.array(row["moments"]["mean"]["expected"])
    dm = transform @ (mean_p[:n] + 1j * mean_p[n:])
    delay_record = dict(
        delay_us=delay * 1e6,
        centered_power=np.diag(dcov).real,
        raw_power=np.diag(dcov).real + abs(dm) ** 2,
    )
    return figures, delay_record


def main():
    inputs = {
        name: json.loads((OUT / f"{name}.json").read_text())
        for name in (
            "covariance-validation",
            "strong-validation",
            "assessment",
            "inference-results",
            "final-validation",
            "mixture-validation",
        )
    }
    if not inputs["inference-results"].get("complete"):
        raise RuntimeError(
            "calibration incomplete; refuse a completed notebook"
        )
    check_gates()
    frozen = OUT / "frozen-experiment.json"
    result = inputs["inference-results"]
    if (
        hashlib.sha256(frozen.read_bytes()).hexdigest()
        != result["config_sha256"]
    ):
        raise RuntimeError("inference results do not match the frozen design")
    for name, digest in result["config"]["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"stale inference source: {name}")
    if (
        hashlib.sha256((OUT / "assessment.json").read_bytes()).hexdigest()
        != result["config"]["assessment_sha256"]
    ):
        raise RuntimeError("inference assessment changed after freezing")
    FIG.mkdir(parents=True, exist_ok=True)
    plots, delay_record = figures(
        inputs["covariance-validation"],
        inputs["strong-validation"],
        inputs["assessment"],
        inputs["inference-results"],
    )
    cfg = inputs["inference-results"]["config"]
    result = inputs["inference-results"]
    assessment = inputs["assessment"]
    manifest = dict(
        observation_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(OUT.glob("observation-*.npz"))
        },
        renderer_versions={
            name: importlib.metadata.version(name)
            for name in ("matplotlib", "nbformat", "nbclient")
        },
        input_sha256={
            name: (
                hashlib.sha256((OUT / f"{name}.json").read_bytes()).hexdigest()
            )
            for name in inputs
        },
        frozen_config_sha256=hashlib.sha256(
            (OUT / "frozen-experiment.json").read_bytes()
        ).hexdigest(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        figure_sha256={
            f"{name}.{ext}": (
                hashlib.sha256(
                    (FIG / f"{name}.{ext}").read_bytes()
                ).hexdigest()
            )
            for name, _ in plots
            for ext in ("png", "pdf")
        },
        delay=delay_record,
        scope=cfg["scope"],
    )
    write_json(OUT / "notebook-manifest.json", manifest)
    # Compact tracked record contains outcomes, not large cached matrices.
    summary = dict(
        scope=cfg["scope"],
        config=cfg,
        final_screen_provenance={
            label: [
                {k: v for k, v in row.items() if k != "moments"}
                for row in inputs[label]["results"]
            ]
            for label in ("final-validation", "mixture-validation")
        },
        covariance_gates=[
            dict(
                ell=r["ell"],
                samples=r["samples"],
                numerical_pass=r["numerical_pass"],
                moment_pass=r["moments"]["passed"],
                worst_z=r["moments"]["worst_z"],
                threshold=r["moments"]["threshold"],
                max_quadrature_change=r["max_quadrature_change"],
            )
            for r in inputs["covariance-validation"]["results"]
        ],
        strong_gate={
            k: inputs["strong-validation"]["results"][0][k]
            for k in (
                "ell",
                "samples",
                "max_quadrature_change",
                "numerical_pass",
            )
        },
        information=inputs["assessment"]["recovery"],
        assessment={
            k: v
            for k, v in inputs["assessment"].items()
            if k not in ("recovery", "templates")
        },
        templates=[
            {
                k: v
                for k, v in r.items()
                if k not in ("mean", "covariance", "template_covariance")
            }
            for r in inputs["assessment"]["templates"]
        ],
        final_moment_checks=inputs["inference-results"]["final_moment_checks"],
        likelihood_diagnostics=inputs["inference-results"][
            "likelihood_diagnostics"
        ],
        conditioning=inputs["inference-results"]["conditioning"],
        statistic_stability=inputs["inference-results"]["statistic_stability"],
        nuisance_fits=inputs["inference-results"]["nuisance_fits"],
        calibration_tail_probability_interval=inputs["inference-results"][
            "calibration_tail_probability_interval"
        ],
        mixture={
            k: v
            for k, v in inputs["inference-results"]["mixture"].items()
            if not k.endswith("moments")
        },
        null_tests=inputs["inference-results"]["null_tests"],
        power=inputs["inference-results"]["power"],
        null_calibration_passed=inputs["inference-results"][
            "null_calibration_passed"
        ],
        mixture_gaussian_worst_z=inputs["inference-results"]["mixture"][
            "gaussian_moments"
        ]["worst_z"],
        mixture_exact_worst_z=inputs["inference-results"]["mixture"][
            "conditional_mixture_moments"
        ]["worst_z"],
        manifest=manifest,
    )
    sr = inputs["strong-validation"]["results"][0]
    summary["strong_gate"].update(
        freq_mhz=sr["config"]["freq"],
        moment_pass=sr["moments"]["passed"],
        worst_z=sr["moments"]["worst_z"],
        threshold=sr["moments"]["threshold"],
    )
    write_json(ROOT / "docs/gaussian-screen-inference-results.json", summary)
    cells = [
        nbformat.v4.new_markdown_cell(
            """# Gaussian-screen covariance and inference

This executed record reads cached, individually converged direct-screen
ensembles and independently calibrated toy observations. The source and beam
are fixed Gaussian profiles on an East/North tangent plane. Only the foreground
screen changes across ensemble members; each screen is shared by all
frequencies.
Readouts are beam-averaged Q and U in K. Synthetic Gaussian readout noise is
5% of unrotated polarized amplitude per sample. All frequency samples are
monochromatic; no channel response or integration time is assigned. This is
not a LuSEE sensitivity forecast or a validated Galactic residual model.

The companion script is `scripts/gaussian_screen_inference.py`; it regenerates
this notebook and all figures without launching an ensemble. Numerical products
are fingerprinted under `generated_data/gaussian_screen_inference/` and
`generated_data/cache/gaussian_screen_inference/`. The tracked compact result
record is `docs/gaussian-screen-inference-results.json`.
Run `bash scripts/run_gaussian_screen_inference.sh run` to reproduce the
bounded experiment with matching checkpoints; `status` reads its log.
Run `.venv/bin/python scripts/gaussian_screen_inference.py` to rebuild this
notebook and its figures from completed results without launching ensembles.
"""
        )
    ]
    failed = [r for r in result["null_tests"] if not r["passed"]]
    verdict = (
        "The five-scale covariance and separate strong-rotation gates pass. "
        "The information assessment does not support factor-of-two scale "
        "recovery with free source nuisances. "
    )
    if failed:
        verdict += (
            f"{len(result['null_tests'])-len(failed)} of"
            f" {len(result['null_tests'])} null acceptance checks pass. "
        )
        for r in failed:
            verdict += (
                f"Test `{r['name']}` at c={r['depth']:g} rad/m² has "
                f"{r['detections']}/{r['n']} false positives "
                f"({100*r['rate']:.2f}%), with simultaneous upper bound "
                f"{100*r['interval'][1]:.2f}% above the frozen 6.5% limit. "
            )
        verdict += (
            "This is an unresolved calibration acceptance check. Its reported "
            "detection fractions are diagnostics for the frozen test, not a "
            "validated significance claim. No tolerance or sample count was "
            "changed after seeing these draws."
        )
    cells.insert(1, nbformat.v4.new_markdown_cell(verdict))
    phase = np.asarray(assessment["rms_phase"])
    attenuation = np.asarray(assessment["attenuation"])
    grid_checks = assessment["grid_checks"]
    cells.append(
        nbformat.v4.new_markdown_cell(
            f"The frozen window uses {len(cfg['freq_mhz'])} frequencies over"
            f" {min(cfg['freq_mhz']):g}–{max(cfg['freq_mhz']):g} MHz. The RMS"
            f" polarization phase is {phase.min():.2f}–{phase.max():.2f} rad"
            f" and changes by {np.ptp(phase):.2f} rad across the band; mean"
            " attenuation spans"
            f" {attenuation.min():.3f}–{attenuation.max():.3f}. A narrow phase"
            " change does not mean weak absolute rotation. At equal total"
            " inverse-noise-variance budget, profiling the baseline against"
            f" double scale gives KL={grid_checks[0]['kl']:.3f} with five"
            f" samples and {grid_checks[-1]['kl']:.3f} with seventeen. The"
            " five-sample choice and synthetic noise were frozen before final"
            " mocks."
        )
    )
    validation = inputs["covariance-validation"]
    table = (
        "| ell (pc) | Screens | Maximum grid shift / unrotated P | Worst"
        " moment discrepancy / MC SE | Gate |\n|---:|---:|---:|---:|:---|\n"
    )
    for r in validation["results"]:
        table += (
            f"| {r['ell']:g} | {r['samples']} |"
            f" {r['max_quadrature_change']:.2g} |"
            f" {r['moments']['worst_z']:.2f} |"
            f" {r['numerical_pass'] and r['moments']['passed']} |\n"
        )
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Numerical and covariance validation\n\n" + table + """
All Q/U cells, raw second moments and squared on/off differences are checked
with a screen bootstrap and a simultaneous normal diagnostic. Numerical grids
are doubled twice while the Fourier field remains fixed; cutoff truncation
retains coefficients. Domain/spectrum checks use the generator's exact discrete
covariance and tail bounds. The initial large-scale pilot failed to resolve the
source profile. The first 128-screen sweep then found coarse-grid shifts of
1.09e-4 and 1.83e-4 at ell=0.5 and 1.5 pc, respectively. These exceeded the
1e-4 criterion; both grids were doubled before the frozen experiment.
The initial ledger is retained as `covariance-validation-initial.json`.
The production grid resolves both the source profile and phase variation.
Domain enlargement changes the Fourier basis: this run checks its exact
ensemble spectral covariance and weight tails rather than comparing paired
maps from different boxes.
The normal moment diagnostic is an approximate ensemble check, not exact
coverage.
"""
        )
    )
    for name, caption in plots:
        cells.append(
            nbformat.v4.new_code_cell(
                "from IPython.display import Image, display\n"
                f"display(Image(filename='report/figures/{name}.png'))"
            )
        )
        cells.append(nbformat.v4.new_markdown_cell(caption))
    templates = inputs["assessment"]["templates"]
    table = (
        "| Band (MHz) | Max diagonal relative error | Max correlation error |"
        " Fitted covariance relative error |\n|---|---:|---:|---:|\n"
    )
    for r in templates:
        f = r["freq_mhz"]
        table += (
            f"| {f[0]:g}–{f[-1]:g} | {r['max_diagonal_relative_error']:.3g} |"
            f" {r['max_correlation_error']:.3g} |"
            f" {r['fitted_covariance_relative_error']:.3g} |\n"
        )
    table += (
        "\n| Band (MHz) | Fitted power multiplier | Whitened mean error | "
        "Whitened covariance error |\n|---|---:|---:|---:|\n"
    )
    for r in templates:
        f = r["freq_mhz"]
        table += (
            f"| {f[0]:g}–{f[-1]:g} | {r['fitted_amplitude_power']:.3g} | "
            f"{r['whitened_errors'][0]:.3g} | "
            f"{r['whitened_errors'][1]:.3g} |\n"
        )
    cells.append(
        nbformat.v4.new_markdown_cell(
            r"""## What the physical template comparison establishes

The existing `noise.faraday_signal_covariance` on template commit
`adb40246a56965ea1b7247fe1865771c8a939384` uses the signed, unit-normalized
depth-power transform $S(x,x')=\int d\phi\,\widehat H(\phi)
\exp[2i\phi(x-x')]$. For Gaussian $\widehat H$ with mean m and variance
$\sigma_\phi^2$, its shape is the numerator below. This comparison uses that
Gaussian toy depth distribution, not the saved Galactic histogram.

For x=lambda², small-separation Gaussian-screen coherence gives the raw
same-branch kernel proportional to

$$\frac{\exp[2im(x-x')-2\sigma_\phi^2(x-x')^2]}
{1+8\sigma_\phi^2w^2xx'/\ell^2}.$$

Freezing xx' at the band center produces the stationary template. In strong
rotation, mean and opposite-branch terms are negligible; this limit is checked
against the full covariance, rather than imposed at all frequencies. The
normalization comes from physical source/beam/screen scales. A freely fitted
normalization is shown separately to isolate shape error. The absolute
polarization-power normalization at reference x0 is
$\eta^2 P_{\rm ref}^2/[1+8\sigma_\phi^2w^2x_0^2/\ell^2]$, multiplied by the
source spectral factors on both frequency axes. No numerical pixel area enters.
The Q/U assembly retains both branches; the approximation sets the relation
moment and mean to zero only as a tested strong-rotation limit.

"""
            + table
            + """
Whitened errors use the full model's S+N: Euclidean norm for the omitted
mean and spectral norm for covariance error. The 17-point comparisons keep
the same total inverse-noise-variance budget as the five-point benchmark.
Diagonal errors
use absolute physical amplitude; the fitted multiplier is applied only in the
separate Frobenius-norm shape comparison.

These are conditional toy comparisons. The arbitrary depth histogram in the
existing real-sky template does not inherit this normalization or covariance
validity. The 9.9–10.1 MHz row has reduced-integral validation only.
"""
        )
    )
    power = inputs["inference-results"]["power"]
    table = (
        "| Ensemble | ell (pc) | Test | Detection fraction | Simultaneous"
        " interval |\n|---|---:|---|---:|---|\n"
    )
    for r in power:
        if r["name"] in (
            "known-0",
            "0-H0-0",
            "0-Hc-0",
            "1-H0-0",
            "1-Hc-0",
            "1-Hc-1",
        ):
            table += (
                f"| {r['label']} | {r['ell']} | {r['name']} | {r['rate']:.3f}"
                f" | [{r['interval'][0]:.3f}, {r['interval'][1]:.3f}] |\n"
            )
    cells.append(
        nbformat.v4.new_markdown_cell(
            """## Calibrated detection and limitations

Test names `known-0`, `0-Hc-0`, `1-Hc-0`, `1-Hc-1` denote known-source
likelihood scoring, fitted amplitude/angle against a constant-depth screen,
additionally fitted beta, and the last test with cross-frequency signal
covariance removed. The ablation retains same-frequency Q–U covariance and
uses the same physical draws and unchanged noise. Each statistic has its own
null calibration. Adding nuisance freedom changes both the residual statistic
and its null threshold; detection fractions need not decrease monotonically.
H0 in a test name
substitutes zero rotation for Hc, with
the same source nuisance freedom. Detection of zero-rotation departures and
of constant-screen
departures are distinct conclusions.

"""
            + table
            + f"""
The simultaneous null-calibration gate passed:
**{result['null_calibration_passed']}**.
Null rates apply only at the declared source amplitude/angle/spectral index
and constant depths −0.04, 0, +0.04 rad/m²; fitted nuisance ranges do not imply
uniform calibration throughout those ranges. Noise is independent across Q/U
and frequency in this controlled experiment.

The finite-model assessment does not support factor-of-two scale recovery
with the declared noise and free source nuisances. No confidence intervals for
correlation length are claimed. The mean is independent of ell; at small ell/w,
covariance amplitude scales approximately as P_ref²(ell/w)². Changing source
width while holding ell/w and beam-diluted polarized amplitude fixed gives an
exact ambiguity in Q/U alone; additional intensity constraints could help.

The scale-mixture stress test uses A²=0.5 or 1.5 with equal probability,
independently per screen: E[A²]=1 preserves ensemble RM covariance. Its phase
statistics need not match Gaussian depths. Comparison with the conditional
mixture moments separates physical misspecification from non-Gaussian readout
likelihood effects. Gaussian scores here are empirically calibrated statistics;
whitened residual diagnostics alone do not establish a Gaussian likelihood.
"""
        )
    )
    table = (
        "| Test | Null depth (rad/m²) | False-positive fraction | "
        "Simultaneous interval | Nuisance boundary fraction |\n"
        "|---|---:|---:|---|---:|\n"
    )
    for r in result["null_tests"]:
        table += (
            f"| {r['name']} | {r['depth']:g} | {r['detections']}/{r['n']} "
            f"({r['rate']:.3f}) | [{r['interval'][0]:.3f}, "
            f"{r['interval'][1]:.3f}] | "
            f"{r.get('nuisance_boundary_fraction', 0):.3f} |\n"
        )
    tail = result["calibration_tail_probability_interval"]
    flips = sum(r["detection_flips"] for r in result["statistic_stability"])
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Calibration precision and numerical stability\n\nEach depth"
            " has 4096 threshold-calibration draws and 4096 independent"
            " validation draws. The threshold is the 95th percentile (higher"
            " order statistic); Hc uses the maximum across three depths. Exact"
            " binomial intervals use a Bonferroni correction across 18 null"
            " comparisons. The acceptance requirement is an upper bound at or"
            " below 0.065. This tests control of false positives; a"
            " conservative rate can be lower than the nominal 0.05.\n\n"
            + table
            + "\nBefore taking the maximum over depths, the calibration"
            f" quantile's tail-probability interval is [{tail[0]:.3f},"
            f" {tail[1]:.3f}] with the same simultaneous correction."
            " Boundary fractions include the physical polarization cap,"
            " fitted beta and fitted constant-depth bounds; the tracked"
            " record separates these.\n\n"
            + "Re-evaluating every final statistic on the preceding angular"
            " grid changed"
            f" {flips}"
            " decisions across the 30 reported tests. Maximum changes are"
            " recorded per test; the frozen tolerance is 0.001 times max(1,"
            " absolute threshold) and at most one decision per test. No"
            " covariance jitter or eigenvalue clipping is used. The"
            " source-profiled tests fit each null and use a fixed ell=0.5"
            " pc covariance-weighted residual discrepancy. They are"
            " empirically calibrated tests, not a claimed profile"
            " likelihood ratio between Hs and Hc. Full Hs"
            " nuisance-likelihood intervals were not pursued after the"
            " information gate rejected scale recovery."
        )
    )
    boundary_table = (
        "| Ensemble | ell (pc) | Null | Beta boundary fraction | Depth"
        " boundary fraction |\n|---|---:|---|---:|---:|\n"
    )
    for r in result["nuisance_fits"]:
        if r["stage"] == 1:
            boundary_table += (
                f"| {r['label']} | {r['ell']:g} | {r['null']} | "
                f"{r['boundaries'].get('beta', 0):.3f} | "
                f"{r['boundaries'].get('depth', 0):.3f} |\n"
            )
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Dependence on source restrictions\n\n"
            + boundary_table
            + "\nFrequent beta-boundary hits show that detection is"
            " conditional on the frozen intrinsic spectral-index interval"
            " [-4, -1]. Rejection of these fitted nulls does not exclude a"
            " source with a more flexible intrinsic spectrum. The table"
            " reports every final ensemble; a larger source model requires a"
            " separately frozen calibration."
        )
    )
    table = (
        "| Ensemble | ell (pc) | Max absolute whitened skew | "
        "Max absolute excess kurtosis | Max 3σ tail fraction |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    for r in result["likelihood_diagnostics"]:
        table += (
            f"| {r['label']} | {r['ell']:g} | "
            f"{np.max(np.abs(r['skew'])):.2f} | "
            f"{np.max(np.abs(r['excess_kurtosis'])):.2f} | "
            f"{max(r['tail_counts_3sigma'])/r['n']:.3f} |\n"
        )
    mixture_row = inputs["mixture-validation"]["results"][0]
    rm_z = np.max(
        abs(
            (
                np.array(mixture_row["spatial_empirical"])
                - np.array(mixture_row["spatial_expected"])
            )
            / np.array(mixture_row["spatial_se"])
        )
    )
    mixture = result["mixture"]
    cells.append(
        nbformat.v4.new_markdown_cell(
            f"The mixture's measured RM covariance agrees within {rm_z:.2f} "
            "sampling standard errors at separations 0, 0.5, 1 and 2 times "
            "ell. "
            "Each screen retains its drawn amplitude; none is renormalized."
        )
    )
    exact = mixture["conditional_mixture_moments"]
    gaussian = mixture["gaussian_moments"]
    error = mixture["gaussian_model_whitened_error"]
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Distribution and matched-covariance stress test\n\n"
            + table
            + "\nThese diagnostic maxima are descriptive and correlated; no"
            " normal-tail significance is assigned. Coverage of scale"
            " intervals is not claimed. The mixture rows are whitened with"
            " the Gaussian-screen prediction, so they include physical model"
            " mismatch.\n\n"
            + "For the mixture's full moments, the worst Gaussian-screen"
            f" residual is {gaussian['worst_z']:.2f} MC standard errors"
            " against the simultaneous threshold"
            f" {gaussian['threshold']:.2f}. Conditioning on A instead gives"
            f" {exact['worst_z']:.2f}; its moment gate passes:"
            f" **{exact['passed']}**. The analytic physical mismatch has"
            f" whitened mean norm {error[0]:.3f} and covariance spectral"
            f" norm {error[1]:.3f}, using Gaussian S+N. Matching RM"
            " two-point covariance therefore does not by itself fix the"
            " observable phase moments. The detection fractions above"
            " quantify this one stress test, without establishing"
            " robustness to general non-Gaussian or Galactic screens."
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            "import json\nfrom pathlib import Path\n"
            "path = Path('docs/gaussian-screen-inference-results.json')\n"
            "record = json.loads(path.read_text())\n"
            "print(json.dumps({'scope':"
            " record['scope'], 'inputs': record['manifest']['input_sha256'],"
            " 'null_calibration_passed': record['null_calibration_passed']},"
            " indent=2))"
        )
    )
    notebook_path = ROOT / "notebooks/gaussian_screen_inference.ipynb"
    kernelspec = {
        "display_name": "Python 3 (project .venv)",
        "language": "python",
        "name": "python3",
    }
    if notebook_path.exists():
        previous = nbformat.read(notebook_path, as_version=4)
        kernelspec = previous.metadata.get("kernelspec", kernelspec)
        existing_ids = {
            (cell.cell_type, cell.source): cell.id for cell in previous.cells
        }
        for cell in cells:
            cell.id = existing_ids.get((cell.cell_type, cell.source), cell.id)
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={"kernelspec": kernelspec},
    )
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    nbformat.write(notebook, notebook_path)
    print(
        f"Wrote notebook, {len(plots)} figures and tracked result record",
        flush=True,
    )


if __name__ == "__main__":
    main()
