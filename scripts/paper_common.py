"""Shared provenance and atomic outputs for the paper calculation stages."""

import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path

from common import REPO


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def manifest(config, sources):
    dependencies = {}
    for package, module in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("lusee", "lusee"),
        ("croissant-sim", "croissant"),
    ):
        spec = importlib.util.find_spec(module)
        path = (
            str(Path(spec.origin).resolve()) if spec and spec.origin else None
        )
        entry = dict(version=importlib.metadata.version(package), path=path)
        if package in ("lusee", "croissant-sim") and path:
            result = subprocess.run(
                ["git", "-C", str(Path(path).parent), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            entry["revision"] = result.stdout.strip() or None
        dependencies[package] = entry
    value = dict(
        configuration=config,
        sources={name: digest(REPO / name) for name in sorted(sources)},
        dependencies=dependencies,
    )
    key = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return value, key


def load_record(path, expected_key):
    record = json.loads(Path(path).read_text())
    if record["fingerprint"] != expected_key:
        raise ValueError("result provenance differs; rerun the compute stage")
    return record
