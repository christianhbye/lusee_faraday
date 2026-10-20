"""Reconstruct and validate complete C/J for a saved local-medium case."""

import common  # noqa: F401

import numpy as np

from common import REPO
from local_medium_sensitivity import setup
from paper_common import digest, load_record, manifest, write_json
from residual_signal import mixture_moments
from lusee_faraday.depth_channel_covariance import DepthChannelProjector
from lusee_faraday.gaussian_screen import real_covariance


def compare(reference, alternative, off, config):
    result = {}
    for name, a in reference.items():
        dimension = a.shape[0] * a.shape[1]
        a, b = a.reshape(dimension, dimension), alternative[name].reshape(
            dimension, dimension
        )
        rms = np.sqrt(np.maximum(np.diag(a), 0))
        off_rms = np.sqrt(np.maximum(off.ravel(), 0))
        allowed = config["relative_covariance_tolerance"] * np.outer(
            rms, rms
        ) + (
            config["absolute_off_covariance_tolerance"]
            * np.outer(off_rms, off_rms)
        )
        result[name] = float(np.max(abs(a - b) / np.maximum(allowed, 1e-300)))
    return result


def compute(config, record_path, case_id):
    _, _, _, base, _, _, key, directory = setup(config)
    record = load_record(record_path, key)
    if not record["all_passed"]:
        raise ValueError("sensitivity gates must pass before joint export")
    row = next(r for r in record["results"] if r["case_id"] == case_id)
    weight_file = record["weights"][f"{row['center_mhz']:g}"]
    for item in (row, weight_file):
        if digest(REPO / item["path"]) != item["sha256"]:
            raise ValueError("corrupt joint input")
    with np.load(REPO / row["path"]) as data:
        saved = {k: data[k] for k in data.files}
    with np.load(REPO / weight_file["path"]) as data:
        weights = {k: data[k] for k in data.files}
    alternatives = {
        "frequency": saved["coarse_coefficients"],
        "depth": saved["depth_coefficients"],
    }
    occupied = np.any(abs(saved["coefficients"]) > 0, axis=(0, 1))
    for value in alternatives.values():
        occupied |= np.any(abs(value) > 0, axis=(0, 1))
    degree = int(np.flatnonzero(occupied)[-1]) if occupied.any() else 0
    probabilities = saved["probabilities"]
    values = saved["coefficients"][:, :, : degree + 1]
    r = weights["operator"]

    def solver(name):
        p = DepthChannelProjector.__new__(DepthChannelProjector)
        p.weights = weights[name][:, : degree + 1]
        p.degree = degree
        p.size = p.weights.shape[-1]
        return p

    def moments(operator, a):
        if len(a) == 1:
            return operator.moments(a[0])
        return mixture_moments([operator] * len(a), a, probabilities)

    operator = solver("weights")
    reference = {
        "raw": moments(operator, values),
        "residual": moments(operator, [r @ a for a in values]),
    }
    checks = {}
    for label in ("raw", "residual"):
        checks[label] = {}
        for name, field in (
            ("angular", "angular_weights"),
            ("bandlimit", "cut_weights"),
        ):
            a = values if label == "raw" else [r @ x for x in values]
            checks[label][name] = compare(
                reference[label],
                moments(solver(field), a),
                saved["off"],
                config,
            )
        for name, value in alternatives.items():
            a = value[:, :, : degree + 1]
            if label == "residual":
                a = [r @ x for x in a]
            checks[label][name] = compare(
                reference[label], moments(operator, a), saved["off"], config
            )
    eigenvalues = {}
    outputs = {}
    dimension = values.shape[1] * operator.size
    count = operator.size // 4
    mapping = np.zeros((count, operator.size), complex)
    for phase in range(count):
        mapping[phase, 4 * phase + 1] = 1
        mapping[phase, 4 * phase + 2] = 1j
    marginal_error = 0.0
    for label, moment in reference.items():
        c = moment["total"].reshape(dimension, dimension)
        minimum = float(np.linalg.eigvalsh((c + c.T) / 2)[0])
        scale = max(float(abs(c).max()), 1e-300)
        eigenvalues[label] = dict(
            minimum=minimum,
            tolerance=1e-8 * scale,
            passed=minimum >= -1e-8 * scale,
        )
        for component, covariance in moment.items():
            outputs[label + "_" + component] = covariance
            diagonal = np.diag(
                covariance.reshape(dimension, dimension)
            ).reshape(values.shape[1], operator.size)
            error = float(
                np.max(
                    abs(diagonal - saved[label + "_" + component])
                    / np.maximum(saved["off"], 1e-300)
                )
            )
            marginal_error = max(marginal_error, error)
            pc = np.einsum(
                "ap,bpcq,dq->bacd",
                mapping,
                covariance,
                mapping.conj(),
                optimize=True,
            )
            pj = np.einsum(
                "ap,bpcq,dq->bacd", mapping, covariance, mapping, optimize=True
            )
            outputs[label + "_" + component + "_polarization_covariance"] = pc
            outputs[label + "_" + component + "_polarization_relation"] = pj
            q = (
                np.arange(values.shape[1])[:, None] * operator.size
                + np.arange(count) * 4
                + 1
            ).ravel()
            u = q + 1
            full = covariance.reshape(dimension, dimension)
            expected = np.block(
                [
                    [full[np.ix_(q, q)], full[np.ix_(q, u)]],
                    [full[np.ix_(u, q)], full[np.ix_(u, u)]],
                ]
            )
            n = values.shape[1] * count
            if not np.allclose(
                real_covariance(pc.reshape(n, n), pj.reshape(n, n)),
                expected,
                rtol=1e-10,
                atol=1e-12 * abs(expected).max(),
            ):
                raise ValueError("joint complex/real moment round trip failed")
    passed = (
        all(
            v <= 1
            for label in checks.values()
            for group in label.values()
            for v in group.values()
        )
        and all(v["passed"] for v in eigenvalues.values())
        and marginal_error <= 1e-10
    )
    target = directory / (case_id + "_joint.npz")
    np.savez_compressed(
        target,
        **outputs,
        nominal_mhz=weights["nominal_mhz"],
        sidereal_turns=base["sidereal_turns"],
    )
    provenance, fp = manifest(
        dict(
            case_id=case_id,
            input_record_sha256=digest(record_path),
            case_sha256=row["sha256"],
            weights_sha256=weight_file["sha256"],
        ),
        [
            "scripts/local_medium_joint.py",
            "scripts/local_medium_sensitivity.py",
            "scripts/residual_signal.py",
            "src/lusee_faraday/depth_channel_covariance.py",
            "src/lusee_faraday/gaussian_screen.py",
            "scripts/paper_common.py",
        ],
    )
    result = dict(
        fingerprint=fp,
        provenance=provenance,
        case=row,
        checks=checks,
        eigenvalues=eigenvalues,
        marginal_error_over_off=marginal_error,
        all_passed=bool(passed),
        arrays=dict(path=str(target.relative_to(REPO)), sha256=digest(target)),
        scope=(
            "Signal-only full joint I/Q/U/V covariance and complex measured "
            "P=Q+iU covariance/relation; add the separate I-only means and "
            "radiometer noise for observations."
        ),
    )
    write_json(target.with_suffix(".json"), result)
    print(case_id, "PASS" if passed else "FAIL", checks, flush=True)
    if not passed:
        raise SystemExit("Retained full joint sensitivity gate failure")
    return result
