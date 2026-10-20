"""The brightness export reads the right Stokes slots and rejects bad input."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def export(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("polarized_brightness")


def layout(phases=2, bins=5):
    """(bin, 4*phase+Stokes) array whose entries encode their slot."""
    a = np.zeros((bins, 4 * phases))
    for t in range(phases):
        for s in range(4):
            a[:, 4 * t + s] = 10 * (t + 1) + s
    return a


def test_rms_reads_q_and_u_of_the_requested_phase_parent_bins(export):
    a = layout()
    a[3:] = 1e9  # zoom bins must not enter the parent-channel RMS
    np.testing.assert_allclose(
        export.variance_rms(a, 1), np.sqrt([21.0, 22.0])
    )
    np.testing.assert_allclose(export.linear_rms(a, 0), [11.0, 12.0])
    assert export.stokes_i(a, 1) == 20.0


def test_negative_variance_is_clipped_not_propagated(export):
    a = layout()
    a[:, 1] = -1.0
    assert export.variance_rms(a, 0)[0] == 0.0


def test_load_rejects_corrupt_arrays(tmp_path, monkeypatch, export):
    monkeypatch.setattr(export, "REPO", tmp_path)
    path = tmp_path / "case.npz"
    np.savez(path, raw_total=layout())
    row = dict(path="case.npz", sha256=export.digest(path))
    assert export.load(row)["raw_total"].shape == (5, 8)
    np.savez(path, raw_total=layout() + 1)
    with pytest.raises(ValueError, match="corrupt"):
        export.load(row)
