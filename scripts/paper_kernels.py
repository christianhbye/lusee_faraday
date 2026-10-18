"""Fingerprint and cache full native-channel four-port harmonic kernels."""

import common  # noqa: F401
import json
from pathlib import Path

import numpy as np

from common import CACHE_DIR, REPO, moon_location, times
from paper_common import digest, load_record, manifest, write_json
from lusee_faraday import engine, polarimeter, response
from lusee_faraday.visibility_covariance import (
    loaded_readout_operator,
    real_readout_kernels,
)

SOURCES = [
    "scripts/common.py",
    "scripts/paper_common.py",
    "scripts/paper_kernels.py",
    "src/lusee_faraday/config.py",
    "src/lusee_faraday/conventions.py",
    "src/lusee_faraday/engine.py",
    "src/lusee_faraday/response.py",
    "src/lusee_faraday/instrument.py",
    "src/lusee_faraday/polarimeter.py",
    "src/lusee_faraday/visibility_covariance.py",
    "src/lusee_faraday/gaussian_screen.py",
]


def setup(parameters):
    from lusee.ReceiverImpedance import JFETReceiver

    config = dict(parameters)
    path = Path(config["response_path"])
    if not path.is_absolute():
        path = REPO / path
    receiver = JFETReceiver(**config["receiver_parameters"])
    config["resolved_receiver_parameters"] = {
        name: np.asarray(value).tolist()
        for name, value in receiver.params.items()
    }
    config["receiver_channel_map"] = list(receiver.channel_map)
    config["response_sha256"] = digest(path)
    provenance, key = manifest(config, SOURCES)
    directory = CACHE_DIR / "paper_kernels" / key
    return path, receiver, provenance, key, directory


def load(path, key):
    check = load_record(Path(path).with_suffix(".json"), key)
    if digest(path) != check["array_sha256"]:
        raise ValueError("corrupt instrument kernel cache")
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def compute(parameters, record_path):
    path, receiver, provenance, key, directory = setup(parameters)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    resp = None
    for center in parameters["centers_mhz"]:
        array_path = directory / f"{center:g}mhz.npz"
        if array_path.exists():
            load(array_path, key)
            print(f"verified kernel cache {center:g} MHz", flush=True)
        else:
            if resp is None:
                print("loading response", flush=True)
                resp = response.load_response(path)
            print(f"transforming all pairs at {center:g} MHz", flush=True)
            lmax = parameters["lmax"]
            beam = response.four_port_pair_alms(resp, center, lmax)
            beam = engine.response_in_mepa(
                beam, times()[0], moon_location(), lmax
            )
            x, y, _ = polarimeter.zenith_vectors(
                resp, receiver, center, mode="ortho"
            )
            stokes = loaded_readout_operator(
                resp, receiver, center, x_vec=x, y_vec=y
            )
            science = loaded_readout_operator(
                resp, receiver, center, kind="science", x_vec=x, y_vec=y
            )
            np.savez_compressed(
                array_path,
                pair_alms=beam,
                x_vec=x,
                y_vec=y,
                stokes_operator=stokes["operator"],
                science_operator=science["operator"],
                stokes_kernels=real_readout_kernels(beam, stokes["operator"]),
                normalization=stokes["normalization_v2_per_hz_per_k"],
            )
            write_json(
                array_path.with_suffix(".json"),
                dict(
                    fingerprint=key,
                    array_sha256=digest(array_path),
                    provenance=provenance,
                ),
            )
            print(f"saved kernels {center:g} MHz", flush=True)
        files[f"{center:g}"] = dict(
            path=str(array_path.relative_to(REPO)), sha256=digest(array_path)
        )
    write_json(
        record_path,
        dict(
            fingerprint=key,
            provenance=provenance,
            arrays=files,
            scope=(
                "Complex MEPA kernels, frozen beam and receiver per band. "
                "Kelvin uses a shared loaded 1 K enclosure pseudo-I reference."
            ),
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/paper/kernels.json"
    )
    parser.add_argument(
        "--record", type=Path, default=REPO / "docs/paper-kernel-results.json"
    )
    args = parser.parse_args()
    compute(json.loads(args.config.read_text()), args.record)
