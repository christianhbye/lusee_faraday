"""Verify predictions and generate report tables, macros and captions."""

import common  # noqa: F401
import argparse
import hashlib
import json
import importlib.metadata
import subprocess

import numpy as np

from common import FIG_DIR, REPO
from paper_common import digest, manifest, write_json
from lusee_faraday import channelization as ch
from lusee_faraday.config import BETA_QU, parent_centers
from lusee_faraday.conventions import lambda_squared

RECORDS = [
    "faraday-horizon-results",
    "depth-depolarization-results",
    "paper-kernel-results",
    "mixed-depth-results",
    "mixed-depth-validation",
    "receiver-noise-validation",
    "separability-backend-validation",
    "observing-exposure",
    "observing-validation",
    "local-input-audit-results",
    "matched-geometry-results",
    "measurement-statistics-validation",
    "channel-mixture-validation",
    "visibility-draw-validation",
    "native-clock-reference-validation",
    "strong-screen-results",
    "residual-signal-results",
    "residual-signal-validation",
    "chromatic-leakage-results",
    "measured-system-results",
    "measured-system-validation",
    "moment-diagonal-validation",
    "paper-integration-validation",
    "science-product-results",
    "local-medium-sensitivity-results",
    "local-medium-joint-validation",
    "local-medium-separability-results",
    "faraday-separability-results",
    "common-observing-results",
]
FIGURES = [
    ("faraday_instrument_leakage", "Instrument and diffuse leakage"),
    ("paper_depth_depolarization", "Depth-profile benchmark"),
    ("faraday_redistribution_spin2", "Real-instrument redistribution"),
    ("faraday_measured_spectra", "Simulated measured spectra"),
    ("faraday_channel_covariance", "Frequency correlations"),
    ("faraday_channel_covariance_amplitude", "Q/U covariance amplitudes"),
    ("faraday_separability", "Conditional separability"),
    ("faraday_repeatability", "Repeated sidereal-phase measurements"),
    ("faraday_angular_and_fringe", "Angular and observed fringe spectra"),
]


def latex(value):
    mapping = {
        "_": r"\_",
        "^": r"\textasciicircum{}",
        "|": r"\textbar{}",
        "~": r"\textasciitilde{}",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "\\": r"\textbackslash{}",
        "²": r"$^2$",
        "³": r"$^3$",
        "±": r"$\pm$",
        "−": "-",
        "≈": r"$\simeq$",
        "φ": r"$\phi$",
        "Φ": r"$\Phi$",
        "λ": r"$\lambda$",
        "θ": r"$\theta$",
        "σ": r"$\sigma$",
        "ℓ": r"$\ell$",
        "°": r"$^\circ$",
        "×": r"$\times$",
        "√": "sqrt",
        "–": "--",
        "—": "---",
        "μ": r"$\mu$",
    }
    value = str(value).replace("/m^2", "/m²").replace("K^2", "K²")
    return "".join(mapping.get(c, c) for c in value)


def number(value, digits=2):
    if value is None:
        return r"\text{undefined}"
    if not np.isfinite(value):
        raise ValueError("unmarked nonfinite report number")
    if isinstance(value, (int, np.integer)):
        return str(value)
    if value == 0:
        return "0"
    if 0.01 <= abs(value) < 1000:
        return f"{value:.{digits}g}"
    mantissa, exponent = f"{value:.{digits-1}e}".split("e")
    return mantissa + r"\times10^{" + str(int(exponent)) + "}"


def array_records(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        for field, path in value.items():
            if field.endswith("_path"):
                hash_field = field[:-5] + "_sha256"
                if hash_field in value:
                    yield dict(path=path, sha256=value[hash_field])
        for name, expected in value.get("input_sha256", {}).items():
            yield dict(path=name, sha256=expected)
        for child in value.values():
            yield from array_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from array_records(child)


def verify():
    records, hashes, arrays = {}, {}, {}
    for name in RECORDS:
        path = REPO / "docs" / (name + ".json")
        r = json.loads(path.read_text())
        for gate in ("all_passed", "all_numerical_gates_passed", "passed"):
            if gate in r and not r[gate]:
                raise ValueError(f"retained failed gate: {name}")
        if name == "paper-integration-validation":
            for field in ("sources", "tests"):
                for source, expected in r[field].items():
                    if digest(REPO / source) != expected:
                        raise ValueError("stale integration check: " + source)
        provenance = r.get("provenance", r.get("manifest"))
        if provenance is not None and "sources" in provenance:
            for source, expected in provenance["sources"].items():
                if digest(REPO / source) != expected:
                    raise ValueError(f"stale source in {name}: {source}")
            fingerprint = hashlib.sha256(
                json.dumps(
                    provenance,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            if fingerprint != r["fingerprint"]:
                raise ValueError("manifest fingerprint mismatch: " + name)
            _, current_key = manifest(
                provenance["configuration"], list(provenance["sources"])
            )
            if current_key != fingerprint:
                raise ValueError("runtime dependencies changed: " + name)
        for entry in array_records(r):
            p = REPO / entry["path"]
            label = entry["path"]
            if label not in arrays:
                arrays[label] = digest(p)
            if arrays[label] != entry["sha256"]:
                raise ValueError("corrupt product: " + str(p))
        records[name] = r
        hashes[str(path.relative_to(REPO))] = digest(path)
    for name, _ in FIGURES:
        path = FIG_DIR / (name + ".pdf")
        figure_manifest = json.loads(
            path.with_suffix(".manifest.json").read_text()
        )
        expected = figure_manifest.get(
            "pdf_sha256", figure_manifest.get("figure_sha256")
        )
        if digest(path) != expected:
            raise ValueError("figure hash mismatch: " + name)
        caption = path.with_suffix(".caption.txt").read_text().strip()
        if figure_manifest.get("caption", caption) != caption:
            raise ValueError("figure caption differs: " + name)
        generator = figure_manifest.get("generator")
        if (
            generator
            and digest(REPO / generator) != figure_manifest["generator_sha256"]
        ):
            raise ValueError("stale figure generator: " + name)
        helper = figure_manifest.get("render_helper")
        if (
            helper
            and digest(REPO / helper)
            != figure_manifest["render_helper_sha256"]
        ):
            raise ValueError("stale figure rendering helper: " + name)
        for source, expected in figure_manifest.get(
            "input_records", {}
        ).items():
            if digest(REPO / source) != expected:
                raise ValueError("stale figure input: " + name)
        legacy = {
            "depth_depolarization": (
                "depth-depolarization-results",
                "scripts/depth_depolarization.py",
            ),
            "faraday_redistribution_spin2": (
                "mixed-depth-results",
                "scripts/plot_mixed_depth.py",
            ),
        }
        if name in legacy:
            record_name, plot_source = legacy[name]
            if (
                figure_manifest["record_sha256"]
                != hashes["docs/" + record_name + ".json"]
            ):
                raise ValueError("stale legacy figure record: " + name)
            plot_hash = figure_manifest.get("plot_source_sha256")
            if plot_hash and digest(REPO / plot_source) != plot_hash:
                raise ValueError("stale legacy plotting source: " + name)
        hashes[str(path.relative_to(REPO))] = digest(path)
    return records, hashes, arrays


def table(path, caption, label, columns, rows):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(columns) + r"\\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r"\\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def generate():
    records, hashes, arrays = verify()
    report = REPO / "report"
    revision = subprocess.check_output(
        ["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO, text=True
    ).strip()
    # Sources must be committed before building; use the clean report worktree
    # launcher when the shared checkout contains unrelated notebook edits.
    for name in (
        "scripts/paper_products.py",
        "report/paper-implementation.tex",
    ):
        committed = subprocess.check_output(
            ["git", "show", "HEAD:" + name], cwd=REPO
        )
        if hashlib.sha256(committed).hexdigest() != digest(REPO / name):
            raise ValueError(
                "commit report source before regeneration: " + name
            )
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO,
        text=True,
    ).strip():
        raise ValueError("build from the clean report worktree launcher")
    signal = records["residual-signal-results"]
    decisions = records["faraday-separability-results"]
    local = records["local-input-audit-results"]
    last_plasma = local["lunar_plasma"][-1]
    physical = records["local-medium-separability-results"]
    sensitivity = records["local-medium-sensitivity-results"]
    horizon_config = sensitivity["provenance"]["configuration"]["horizon"]
    normalization_cases = physical["emissivity_normalization_sensitivity"]
    values = dict(
        MaxSpeckleNoise=physical["largest_speckle_to_noise_amplitude"],
        MaxTotalNoise=physical[
            "largest_total_polarization_to_noise_amplitude"
        ],
        MaxEmissivityTotalNoise=max(
            row["largest_total_polarization_to_noise_amplitude"]
            for row in normalization_cases
        ),
        ReferenceSpeckleNoise=decisions["largest_speckle_to_noise_amplitude"],
        PhysicalCaseCount=len(sensitivity["results"]),
        PhysicalRms=physical["intrinsic_rms_k"],
        EmittingPathPc=horizon_config["emitting_path_pc"],
        UnitEmissivity=horizon_config["emissivity_ref_k_per_pc"],
        MeanCaseCount=len(sensitivity["means"]),
        MaxMeanDepth=max(abs(r["mean_depth"]) for r in sensitivity["means"]),
        LunarFaradayBound=last_plasma["faraday_bound_rad_m2"],
        LunarPhaseBound=last_plasma["complex_polarization_phase_bound_rad"][0],
        ParentResidualDof=signal["results"][0]["fit"][0]["residual_dof"],
        ZoomResidualDof=signal["results"][0]["fit"][1]["residual_dof"],
        ResidualCaseCount=len(signal["results"]),
    )
    report.joinpath("paper-numbers.tex").write_text(
        r"\newcommand{\PaperSourceRevision}{"
        + revision
        + "}\n"
        + "\n".join(
            "\\newcommand{\\" + key + "}{$" + number(value) + "$}"
            for key, value in values.items()
        )
        + "\n"
    )
    write_json(report / "paper-numbers.json", values)
    horizon = []
    original = records["faraday-horizon-results"]["rows"]
    for scenario in dict.fromkeys(r["scenario"] for r in original):
        chosen = [
            r
            for r in original
            if r["scenario"] == scenario
            and r["absolute_field_projection"] == 1
            and r["phase_limit_rad"] == 1
        ]
        horizon.append(
            [latex(scenario)]
            + ["$" + number(r["horizon_pc"]) + "$" for r in chosen]
        )
    table(
        report / "paper-horizons.tex",
        "Conditional aligned-field one-radian horizons (pc). Zero field "
        "projection is unbounded in every scenario.",
        "tab:horizon",
        ["Scenario", "10 MHz", "30 MHz", "50 MHz"],
        horizon,
    )
    exposure = []
    for row in records["observing-exposure"]["results"]:
        exposure.append(
            [
                str(row["lunations"]),
                str(row["sidereal_turn"]),
                str(row["eligible_visits"]),
                (
                    "$" + number(row["exposure_s"], 4) + "$"
                    if row["available"]
                    else "Unavailable"
                ),
            ]
        )
    table(
        report / "paper-exposure.tex",
        "Illustrative actual-site nightside exposure; the electronics duty "
        "is applied to eligible complete bins.",
        "tab:exposure",
        ["Lunations", "Sidereal turn", "Visits", "Integration (s)"],
        exposure,
    )
    inputs = records["mixed-depth-results"]["provenance"]["configuration"][
        "covariance"
    ]
    noise_inputs = records["measured-system-results"]["provenance"][
        "configuration"
    ]["noise"]

    def scales(name):
        return " / ".join(f"{x:g}" for x in noise_inputs[name])

    intrinsic = inputs["intrinsic"]
    dispersions = " / ".join(
        str(row["parameters"]["sigma_depth"]) for row in inputs["screens"]
    )
    duty = records["observing-exposure"]["provenance"]["configuration"][
        "electronics_duty_cycle"
    ]
    input_rows = [
        [
            "Reference / volume RMS",
            f"{intrinsic['p_ref_k']:g} / "
            f"{physical['intrinsic_rms_k']:g} K at "
            f"{intrinsic['ref_mhz']:g} MHz",
            "Illustrative reference / emissivity times full path",
        ],
        [
            "Volume emissivity / path",
            f"{horizon_config['emissivity_ref_k_per_pc']:g} K pc$^{{-1}}$ / "
            f"{horizon_config['emitting_path_pc']:g} pc",
            "Assumed polarized RMS; depth integration supplies suppression",
        ],
        [
            "Intrinsic angular shape",
            f"ell scale {intrinsic['ell_scale']:g}; EE "
            f"{intrinsic['ee_fraction']:g}, EB {intrinsic['eb_fraction']:g}",
            "Gaussian-decaying spectrum; BB is the remaining fraction",
        ],
        [
            "Intrinsic frequency shape",
            f"Index {BETA_QU:g}; turnover "
            f"{intrinsic['turnover_mhz']:g} MHz",
            "Assumed absorbed power law, not a measured local spectrum",
        ],
        [
            "Angular RM structure",
            f"theta0={inputs['screens'][0]['parameters']['theta0_rad']:g} "
            "rad; alpha="
            f"{inputs['screens'][0]['parameters']['alpha']:.4g}",
            "Assumed powered-chord covariance, fixed across mean columns",
        ],
        [
            "Local dispersion",
            dispersions + " rad m$^{-2}$",
            "Conditional angular screens; no column-map substitution",
        ],
        [
            "Voltage scales",
            scales("voltage_asd_nv_rt_hz") + " nV Hz$^{-1/2}$",
            r"Adopted EM/requirement scales; "
            r"\href{https://doi.org/10.1029/2023RS007925}{Tamura et al.}",
        ],
        [
            "Current scales",
            scales("current_asd_pa_rt_hz") + " pA Hz$^{-1/2}$",
            "Assumed; independent input sources",
        ],
        [
            "Moon / loss temperatures",
            f"{noise_inputs['moon_temperature_k']:g} K / "
            f"{noise_inputs['antenna_temperature_k']:g} K",
            "Illustrative nightside values",
        ],
        [
            "Emissivity audit",
            "Total I, paths below 1 kpc",
            r"\href{https://arxiv.org/html/1811.10310v1}{Polderman et al.}; "
            "no local polarized prior",
        ],
        [
            "Nearby RM/DM",
            "Historical J0437-4715 values",
            r"\href{https://arxiv.org/pdf/1105.4213}{Yan et al.}; "
            "path averages",
        ],
        [
            "Lunar plasma",
            "Assumed density/field/path envelope",
            r"\href{https://doi.org/10.1029/2011JA017141}{Ando et al.}; "
            "no universal nightside limit",
        ],
        [
            "Observing duty",
            f"{100*duty:g} percent",
            r"Assumed as in \href{https://arxiv.org/html/2508.16773v2}"
            "{Camacho et al.}; actual-site visibility calculated",
        ],
    ]
    report.joinpath("paper-inputs.tex").write_text(
        r"\begin{table}[htbp]\centering\small"
        + "\n"
        + r"\caption{Adopted inputs and their status.}\label{tab:inputs}"
        + "\n"
        + r"\begin{tabular}{p{0.22\linewidth}p{0.26\linewidth}"
        + r"p{0.43\linewidth}}\toprule"
        + "\n"
        + "Quantity & Input & Source/status"
        + r"\\\midrule"
        + "\n"
        + "\n".join(" & ".join(r) + r"\\" for r in input_rows)
        + "\n"
        + r"\bottomrule\end{tabular}\end{table}"
        + "\n"
    )
    noise = records["measured-system-results"]["noise"][0]
    with np.load(REPO / noise["path"]) as data:
        bandwidth = data["bandwidth_hz"]
    parents = parent_centers(noise["center_mhz"])
    parent_count = len(parents)
    parent_spacing = np.diff(parents).mean() * 1e6
    table(
        report / "paper-channels.tex",
        "Noise-equivalent bandwidth from actual folded power weights. "
        "Ranges span channel positions, not uncertainty intervals.",
        "tab:channels",
        ["Type", "Bins", "Nominal spacing (Hz)", "Effective bandwidth (Hz)"],
        [
            [
                "Parent",
                str(parent_count),
                f"{parent_spacing:g}",
                "$" + number(bandwidth[:3].min(), 4) + "$",
            ],
            [
                "Zoom",
                str(len(bandwidth) - parent_count),
                f"{ch.ZOOM_STEP_HZ:g}",
                "$["
                + number(bandwidth[3:].min(), 4)
                + ","
                + number(bandwidth[3:].max(), 4)
                + "]$",
            ],
        ],
    )
    resolution_rows = []
    for frequency in records["faraday-horizon-results"]["manifest"][
        "configuration"
    ]["frequencies_mhz"]:
        values = []
        for width_hz in (parent_spacing, ch.ZOOM_STEP_HZ):
            low, high = (
                frequency - width_hz * 0.5e-6,
                frequency + width_hz * 0.5e-6,
            )
            delta = float((lambda_squared(low) - lambda_squared(high))[0])
            values.append("$" + number(np.pi / delta) + "$")
        resolution_rows.append([f"{frequency:g}"] + values)
    table(
        report / "paper-resolution.tex",
        "Nominal narrow-channel top-hat first-zero Faraday depth "
        r"$\pi/\Delta(\lambda^2)$, in rad m$^{-2}$. This is a resolution "
        "scale; actual predictions use the measured folded power weights "
        "rather than this top-hat approximation.",
        "tab:resolution",
        ["MHz", "Parent", "Zoom"],
        resolution_rows,
    )
    with np.load(REPO / physical["arrays"]["path"]) as data:
        grid = {k: data[k] for k in data.files}
    axis = physical["axes"]
    phase = axis["phase"].index(0.0)
    lunation = axis["lunations"].index(max(axis["lunations"]))
    decision_rows = []
    for center in (30.0, 50.0):
        for group, kind in enumerate(axis["channel_type"]):
            for profile, label in enumerate(
                ("f=0", "f=0.6", "f=1", "slab", "Gaussian")
            ):
                selected = [
                    i
                    for i, row in enumerate(physical["coordinates"])
                    if row["center_mhz"] == center
                    and row["profile_index"] == profile
                ]
                residual = grid["residual_speckle_rms_k"][
                    selected, phase, group
                ]
                speckle = grid["speckle_to_noise_amplitude"][
                    selected, phase, group, :, lunation
                ]
                total = grid["total_to_noise_amplitude"][
                    selected, phase, group, :, lunation
                ]
                retained = grid["retained_speckle_power"][
                    selected, phase, group
                ]
                retained = retained[np.isfinite(retained)]
                decision_rows.append(
                    [f"{center:g}", kind, label]
                    + [
                        "$" + number(x) + "$"
                        for x in (
                            residual[:, 0].max(),
                            residual[:, 1].max(),
                            np.nanmax(speckle),
                            np.nanmax(total),
                        )
                    ]
                    + [
                        "$"
                        + number(retained.max() if len(retained) else None)
                        + "$"
                    ],
                )
    table(
        report / "paper-decisions.tex",
        "Full signed-column range: largest residual Q/U speckle RMS (K), "
        "speckle/noise and total-polarization/noise RMS ratios, and retained "
        "speckle power at phase 0 over the longest illustrative exposure. "
        "Maxima can come from different conditional cases. The full arrays "
        "retain each coordinate and unavailable exposure.",
        "tab:decisions",
        [
            "MHz",
            "Type",
            "Profile",
            "Q (K)",
            "U (K)",
            "S/noise",
            "Total/noise",
            "Power kept",
        ],
        decision_rows,
    )
    table(
        report / "paper-emissivity.tex",
        "Conditional HII total-emissivity endpoints with assumed polarized "
        "fractions, converted to the reference frequency using the declared "
        "audit spectral index. Ratios rescale the same angular/depth model "
        "and its full emitting volume. These are not local measurements or "
        "a prior distribution.",
        "tab:emissivity",
        [
            "I at 74 MHz (K/pc)",
            "Polarized fraction",
            "P at 30 MHz (K/pc)",
            "Max. total/noise",
        ],
        [
            [
                "$" + number(v) + "$"
                for v in (
                    row["total_intensity_at_ref_k_per_pc"],
                    row["assumed_polarized_fraction"],
                    row["polarized_emissivity_at_reference_k_per_pc"],
                    row["largest_total_polarization_to_noise_amplitude"],
                )
            ]
            for row in normalization_cases
        ],
    )
    leakage_rows = []
    for center in (30.0, 50.0):
        for kind in ("parent", "zoom"):
            for component in ("Q", "U"):
                rows = [
                    d
                    for d in decisions["decisions"]
                    if d["center_mhz"] == center
                    and d["channel_type"] == kind
                    and d["component"] == component
                    and d["phase_index"] == phase
                ]
                leakage_rows.append(
                    [f"{center:g}", kind, component]
                    + [
                        "$" + number(max(d[key] for d in rows)) + "$"
                        for key in (
                            "asbuilt_linear_rms_k",
                            "asbuilt_interpolation_difference_rms_k",
                            "worst_common_calibration_mismatch_rms_k",
                        )
                    ]
                )
    table(
        report / "paper-leakage.tex",
        "Phase-0 chromatic I-only residual RMS (K) through the common fit: "
        "as-built linear model, linear/PCHIP difference and worst ablation "
        "mismatch using the same as-built calibration. These model spreads "
        "are not measured beam-error probabilities.",
        "tab:leakage",
        ["MHz", "Type", "Stokes", "As-built", "Interpolation", "Mismatch"],
        leakage_rows,
    )
    figures = []
    for name, title in FIGURES:
        caption = FIG_DIR.joinpath(name + ".caption.txt").read_text().strip()
        figures.extend(
            [
                r"\begin{figure}[p]\centering",
                r"\includegraphics[width=6.4in]{figures/" + name + ".pdf}",
                r"\caption{" + latex(title) + ". " + latex(caption) + "}",
                r"\end{figure}",
            ]
        )
    report.joinpath("paper-figures.tex").write_text("\n".join(figures) + "\n")
    provenance = dict(
        source_revision=revision,
        record_sha256=hashes,
        array_sha256=arrays,
        generator_sha256=digest(__file__),
        runtime_packages={
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
    )
    write_json(report / "paper-provenance.json", provenance)
    report.joinpath("paper-provenance.tex").write_text(
        "The report source and products generator were committed before "
        "regeneration. The build used a clean detached worktree at source "
        "revision " + r"\texttt{" + revision + "}. Numerical records contain "
        "complete source and dependency manifests; the report manifest "
        "checksums every record, array and figure it reads. Large files "
        "remain regenerable. Existing unrelated notebook edits in the shared "
        "workspace are preserved and are not report inputs.\n"
    )
    print("Verified", len(records), "records and", len(arrays), "artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
    else:
        generate()
