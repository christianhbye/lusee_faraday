"""Export compact all-16-product predictions and verify their Stokes map."""

import common  # noqa: F401
import argparse
import json

import numpy as np

from common import CACHE_DIR, REPO
from faraday_covariance import projector
from local_medium_sensitivity import setup
from paper_common import digest, load_record, manifest, write_json
from paper_kernels import load as load_kernel
from residual_signal import common_fit
from lusee_faraday.channelization import ZOOM_STEP_HZ
from lusee_faraday.conventions import PRODUCT_LABELS
from lusee_faraday.depth_channel_covariance import DepthChannelProjector
from lusee_faraday.observing import sidereal_bin_window
from lusee_faraday.receiver_noise import (
    channel_radiometer_covariance,
    science_quadratic_matrices,
)
from lusee_faraday.spin_covariance import AngularScreen
from lusee_faraday.visibility_covariance import real_readout_kernels


def load(item):
    path = REPO / item["path"]
    if digest(path) != item["sha256"]:
        raise ValueError("corrupt science-product input")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def compute():
    config = json.loads(
        (REPO / "configs/paper/local-medium-sensitivity.json").read_text()
    )
    sep, _, _, base, kr, _, local_key, _ = setup(config)
    local_path = REPO / "docs/local-medium-sensitivity-results.json"
    local = load_record(local_path, local_key)
    joint_path = REPO / "docs/local-medium-joint-validation.json"
    joint_record = json.loads(joint_path.read_text())
    system_path = REPO / "docs/measured-system-results.json"
    system = json.loads(system_path.read_text())
    if not local["all_passed"] or not system["all_passed"]:
        raise ValueError("science export requires completed numerical gates")
    provenance, fp = manifest(
        dict(
            local_record_sha256=digest(local_path),
            system_record_sha256=digest(system_path),
            joint_record_sha256=digest(joint_path),
            local_configuration=local["provenance"]["configuration"],
        ),
        [
            "scripts/science_product_predictions.py",
            "scripts/faraday_covariance.py",
            "scripts/residual_signal.py",
            "scripts/paper_common.py",
            "src/lusee_faraday/visibility_covariance.py",
            "src/lusee_faraday/depth_channel_covariance.py",
            "src/lusee_faraday/receiver_noise.py",
        ],
    )
    directory = CACHE_DIR / "science_predictions" / fp
    directory.mkdir(parents=True, exist_ok=True)
    signatures = {
        (r["parameters"]["theta0_rad"], r["parameters"]["alpha"])
        for r in base["screens"]
    }
    if len(signatures) != 1:
        raise ValueError("compact export requires a shared angular shape")
    rows = []
    for center in sorted({r["center_mhz"] for r in local["results"]}):
        target = directory / f"{center:g}.npz"
        if target.exists():
            sidecar = load_record(target.with_suffix(".json"), fp)
            if digest(target) != sidecar["sha256"]:
                raise ValueError("corrupt science-product checkpoint")
            rows.append(sidecar["result"])
            continue
        kernel = kr["arrays"][f"{center:g}"]
        data = load_kernel(REPO / kernel["path"], kr["fingerprint"])
        science_operator = data["science_operator"]
        stokes_operator = data["stokes_operator"]
        mapping, _, rank, _ = np.linalg.lstsq(
            science_operator.T, stokes_operator.T, rcond=None
        )
        mapping = mapping.T
        if rank != len(PRODUCT_LABELS):
            raise ValueError("science basis does not span the 16 products")
        operator_error = float(
            abs(mapping @ science_operator - stokes_operator).max()
            / abs(stokes_operator).max()
        )
        full_mapping = np.kron(np.eye(len(base["sidereal_turns"])), mapping)
        local_weights = load(local["weights"][f"{center:g}"])
        degree = local_weights["weights"].shape[1] - 1
        kernels = real_readout_kernels(data["pair_alms"], science_operator)
        kernels *= sidereal_bin_window(kernels.shape[-2] - 1)
        screen = AngularScreen(**base["screens"][0]["parameters"])
        solver = DepthChannelProjector(
            projector(kernels, base, refinement=2), screen, degree
        )
        projected = np.einsum(
            "ap,kmpq,bq->kmab",
            full_mapping,
            solver.weights,
            full_mapping,
            optimize=True,
        )
        expected = local_weights["weights"]
        errors = []
        mode_psd = []
        for mode in range(degree + 1):
            scale = max(float(abs(expected[:, mode]).max()), 1e-300)
            errors.append(
                float(
                    abs(projected[:, mode] - expected[:, mode]).max() / scale
                )
            )
            basis = np.zeros((2, degree + 1), complex)
            basis[:, mode] = [1, 1j]
            covariance = solver.moments(basis)["total"].reshape(
                2 * solver.size, -1
            )
            minimum = float(
                np.linalg.eigvalsh((covariance + covariance.T) / 2)[0]
            )
            scale = max(float(abs(covariance).max()), 1e-300)
            mode_psd.append(
                dict(
                    mode=mode, minimum=minimum, passed=minimum >= -1e-8 * scale
                )
            )
        fine, w, nominal, residual, _ = common_fit(
            center, sep["smooth_degree"]
        )
        item = next(
            r
            for r in system["leakage"]
            if r["center_mhz"] == center and r["response"] == "asbuilt"
        )
        intensity = load(item)
        readout = science_quadratic_matrices() / float(data["normalization"])
        sky_psd = intensity["fine_sky_covariance"]
        science_fine_means = np.einsum("pab,ftba->ftp", readout, sky_psd).real
        means = np.einsum("bf,ftp->btp", w, science_fine_means)
        projected_means = np.einsum("ap,btp->bta", mapping, means).reshape(
            len(nominal), -1
        )
        mean_error = float(
            abs(projected_means - intensity["linear_stokes_k"]).max()
            / abs(projected_means).max()
        )
        selected = list(range(3))
        for offset in (0, 1, 16, 32):
            frequency = center + offset * ZOOM_STEP_HZ * 1e-6
            matches = np.flatnonzero(
                np.isclose(nominal[3:], frequency, rtol=0, atol=1e-10)
            )
            if len(matches) != 1:
                raise ValueError("science reference coordinate is not unique")
            selected.append(int(matches[0] + 3))
        selected = np.array(selected)
        (
            raw_noise,
            residual_noise,
            amplifiers,
            noise_errors,
            noise_coordinates,
        ) = ([], [], [], [], [])
        for item in [r for r in system["noise"] if r["center_mhz"] == center]:
            noise = load(item)
            psd = sky_psd + noise["thermal_psd"] + noise["amplifier_psd"]
            raw, filtered = [], []
            for phase in range(len(base["sidereal_turns"])):
                for output, operator, name in (
                    (raw, w, "covariance_per_second"),
                    (filtered, residual @ w, "residual_covariance_per_second"),
                ):
                    covariance = channel_radiometer_covariance(
                        psd[:, phase],
                        operator[selected],
                        np.diff(fine).mean() * 1e6,
                        1.0,
                        readout,
                    )
                    output.append(covariance)
                    mapped = np.einsum(
                        "ap,bpcq,dq->bacd",
                        mapping,
                        covariance,
                        mapping,
                        optimize=True,
                    )
                    reference = noise[name][phase][selected][:, :, selected]
                    noise_errors.append(
                        float(
                            abs(mapped - reference).max()
                            / abs(reference).max()
                        )
                    )
            raw_noise.append(raw)
            residual_noise.append(filtered)
            amplifiers.append(noise["amplifier_psd"])
            noise_coordinates.append(
                {k: item[k] for k in ("voltage_nv_rt_hz", "current_pa_rt_hz")}
            )
        case = next(
            r
            for r in local["results"]
            if r["center_mhz"] == center
            and r["mean_depth_rad_m2"]
            == min(m["mean_depth"] for m in local["means"])
            and r["screen_index"] == 0
            and r["profile_index"] == 2
            and r["model"] == "gaussian"
        )
        reference_item = next(
            r
            for r in joint_record["results"]
            if r["case_id"] == case["case_id"]
        )
        reference_file = REPO / reference_item["path"]
        if digest(reference_file) != reference_item["sha256"]:
            raise ValueError("corrupt joint reference sidecar")
        reference = load(json.loads(reference_file.read_text())["arrays"])
        case_arrays = load(case)
        selected_signal, signal_errors = {}, []
        for label, coefficients in (
            ("raw", case_arrays["coefficients"][0]),
            ("residual", case_arrays["residual_coefficients"][0]),
        ):
            moments = solver.moments(coefficients[selected])
            for component, covariance in moments.items():
                selected_signal[label + "_" + component] = covariance
                mapped = np.einsum(
                    "ap,bpcq,dq->bacd",
                    full_mapping,
                    covariance,
                    full_mapping,
                    optimize=True,
                )
                expected = reference[label + "_" + component][selected][
                    :, :, selected
                ]
                signal_errors.append(
                    float(
                        abs(mapped - expected).max()
                        / max(abs(expected).max(), 1e-300)
                    )
                )
        passed = bool(
            operator_error < 1e-10
            and max(errors) < 1e-9
            and mean_error < 1e-10
            and max(noise_errors) < 1e-9
            and max(signal_errors) < 1e-9
            and all(m["passed"] for m in mode_psd)
        )
        np.savez_compressed(
            target,
            **selected_signal,
            weights=solver.weights,
            tail_norm=solver.tail_norm,
            science_to_stokes=mapping,
            science_operator=science_operator,
            readout=readout,
            fine_frequency_mhz=fine,
            channel_operator=w,
            residual_operator=residual,
            nominal_mhz=nominal,
            sidereal_turns=base["sidereal_turns"],
            product_labels=PRODUCT_LABELS,
            intensity_means_k=means,
            intensity_residual_k=np.einsum("bc,ctp->btp", residual, means),
            fine_sky_psd=sky_psd,
            thermal_psd=noise["thermal_psd"],
            amplifier_psd=amplifiers,
            selected_bins=selected,
            selected_noise_per_second=raw_noise,
            selected_residual_noise_per_second=residual_noise,
        )
        row = dict(
            center_mhz=center,
            path=str(target.relative_to(REPO)),
            sha256=digest(target),
            passed=passed,
            operator_relative_error=operator_error,
            maximum_mode_stokes_projection_error=max(errors),
            intensity_relative_error=mean_error,
            maximum_noise_stokes_projection_error=max(noise_errors),
            maximum_signal_stokes_projection_error=max(signal_errors),
            selected_signal_case_id=case["case_id"],
            mode_psd=mode_psd,
            noise_coordinates=noise_coordinates,
        )
        write_json(
            target.with_suffix(".json"),
            dict(
                fingerprint=fp,
                sha256=digest(target),
                result=row,
            ),
        )
        print(center, "PASS" if passed else "FAIL", flush=True)
        rows.append(row)
    write_json(
        REPO / "docs/science-product-results.json",
        dict(
            fingerprint=fp,
            provenance=provenance,
            results=rows,
            all_passed=all(r["passed"] for r in rows),
            scope=(
                "All 16 real science products in PRODUCT_LABELS order, common "
                "loaded enclosure-K normalization. Compact four-branch mode "
                "weights combine with any saved local-grid coefficients to "
                "reconstruct joint signal moments without a dense cube. "
                "I-only means cover all bins/phases; all nine noise models "
                "export selected parent/zoom covariances plus fine PSD and "
                "operators for arbitrary requested bin pairs. Projection back "
                "to Stokes checks operators, every signal mode, means and "
                "noise."
            ),
        ),
    )
    if not all(r["passed"] for r in rows):
        raise SystemExit("retained science-product projection failure")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    compute()
