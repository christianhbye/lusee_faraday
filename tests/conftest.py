from pathlib import Path

import numpy as np
import pytest

import lusee_faraday as ld

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NSIDE = 32  # low resolution for fast tests


@pytest.fixture
def data_dir():
    return DATA_DIR


@pytest.fixture
def spec_response_path(data_dir):
    return data_dir / "spectrometer_bin_response.txt"


@pytest.fixture
def spec_response(spec_response_path):
    return ld.SpectrometerResponse.from_file(spec_response_path)


@pytest.fixture
def short_dipole():
    beam = ld.Beam.short_dipole(nside=NSIDE)
    beam.precompute_weights()
    return beam


@pytest.fixture
def healpix_grid():
    return ld.HealpixGrid(nside=NSIDE, horizon=True)


@pytest.fixture
def healpix_grid_full():
    return ld.HealpixGrid(nside=NSIDE, horizon=False)
