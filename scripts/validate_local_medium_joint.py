"""Validate full joint moments at predeclared signed-column corners."""

import common  # noqa: F401
import argparse
import json
from pathlib import Path

from common import REPO
from local_medium_joint import compute
from paper_common import digest, manifest, write_json


def validate(config, record_path, output):
    record = json.loads(record_path.read_text())
    means = [row["mean_depth"] for row in record["means"]]
    cases = [
        row
        for row in record["results"]
        if row["mean_depth_rad_m2"] in (min(means), max(means))
        and row["profile"]["kind"] in ("screen", "slab")
        and (
            row["profile"]["kind"] == "slab" or row["profile"]["fraction"] == 1
        )
        and row["sigma_depth_rad_m2"]
        == min(r["sigma_depth_rad_m2"] for r in record["results"])
        and row["model"] == "gaussian"
    ]
    if len(cases) != 8:
        raise ValueError("expected eight predeclared signed-column corners")
    results = []
    for row in cases:
        result = compute(config, record_path, row["case_id"])
        path = Path(REPO / row["path"]).with_name(
            row["case_id"] + "_joint.json"
        )
        results.append(
            dict(
                case_id=row["case_id"],
                path=str(path.relative_to(REPO)),
                sha256=digest(path),
                checks=result["checks"],
                eigenvalues=result["eigenvalues"],
                marginal_error_over_off=result["marginal_error_over_off"],
                passed=result["all_passed"],
            )
        )
    provenance, key = manifest(
        dict(config, input_record_sha256=digest(record_path)),
        [
            "scripts/validate_local_medium_joint.py",
            "scripts/local_medium_joint.py",
            "scripts/local_medium_sensitivity.py",
            "src/lusee_faraday/moment_diagonal.py",
        ],
    )
    write_json(
        output,
        dict(
            fingerprint=key,
            provenance=provenance,
            results=results,
            all_passed=all(row["passed"] for row in results),
            scope=(
                "Full cross-channel/cross-phase covariance and complex "
                "relation at both extreme signed mean columns, both bands, "
                "screen f=1/slab profiles and the smaller dispersion. "
                "All other cases retain exact diagonal moments and complete "
                "latent phase coefficients for requested joint exports."
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/paper/local-medium-sensitivity.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPO / "docs/local-medium-sensitivity-results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs/local-medium-joint-validation.json",
    )
    args = parser.parse_args()
    validate(json.loads(args.config.read_text()), args.record, args.output)
