"""Real-beam audit checks with tiny synthetic inputs."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "audit/scripts/shuffle_control_bgl.py"
)
spec = importlib.util.spec_from_file_location("bgl_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture(scope="module", autouse=True)
def numerical_stack():
    audit.load_numerics(1)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("batch", [7, 500])
def test_batched_complex_visibility_matches_direct_sum(sign, batch):
    rng = np.random.default_rng(417)
    rm = rng.uniform(-12, 20, 43)
    weight = rng.normal(size=43) + 1j * rng.normal(size=43)
    lam2 = np.linspace(97, 101, 19)
    keep = np.arange(rm.size) % 3 != 0
    expected = weight[keep] @ np.exp(sign * 2j * 0.1 * rm[keep, None] * lam2)
    actual = audit.visibility(
        rm, weight, 0.1, lam2, sign, batch, threads=1, mask=keep
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)


def test_complex_cross_pair_reconstructs_rotated_qu():
    rng = np.random.default_rng(19)
    kq = rng.normal(size=9) + 1j * rng.normal(size=9)
    ku = rng.normal(size=9) + 1j * rng.normal(size=9)
    p = rng.normal(size=9) + 1j * rng.normal(size=9)
    rm = rng.uniform(-8, 10, 9)
    freq = np.array([29.99, 30.0, 30.01])
    phase = audit.conv.faraday_phase_cosmo(rm, freq)
    rotated = p[:, None] * phase
    direct = (kq[:, None] * rotated.real + ku[:, None] * rotated.imag).sum(
        axis=0
    )
    weights = audit.branch_weights(kq, ku, p)
    actual = sum(
        audit.visibility(
            rm,
            weights[branch],
            1.0,
            audit.conv.lambda_squared(freq),
            sign,
            4,
            threads=1,
        )
        for branch, sign in audit.BRANCH_SIGNS.items()
    )
    np.testing.assert_allclose(actual, direct, rtol=2e-7, atol=2e-7)
    assert not np.allclose(weights["negative"], weights["positive"].conj())


def test_auto_branches_are_conjugate_at_every_frequency():
    p = np.array([1 + 2j, -3 + 1j, 4 - 2j])
    weights = audit.branch_weights(
        np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 4.0]), p
    )
    rm, lam2 = np.array([2.0, -4.0, 7.0]), np.array([99.0, 100.0, 101.0])
    pos = audit.visibility(rm, weights["positive"], 1.0, lam2, 1, 2, threads=1)
    neg = audit.visibility(
        rm, weights["negative"], 1.0, lam2, -1, 2, threads=1
    )
    np.testing.assert_allclose(neg, pos.conj(), rtol=2e-7, atol=2e-7)


def test_effective_count_uses_magnitudes_not_complex_cancellation():
    weight = np.array([1, -1, 1j, -1j], dtype=complex)
    stats = audit.weight_statistics(weight, 3)
    assert stats["n_eff_amplitude"] == pytest.approx(4)
    assert stats["unrotated_coherence"] == pytest.approx(0)
    mask = np.array([True, True, False, False])
    masked = audit.weight_statistics(weight, 3, mask)
    assert masked["pixels"] == 2
    assert masked["n_eff_amplitude"] == pytest.approx(2)


def test_disk_weights_match_independent_transport_across_chunk_boundaries(
    tmp_path,
):
    from scipy.spatial.transform import Rotation
    from lusee_faraday.pixel_arm import GalacticGrid, transport

    class Kernel:
        def sample(self, theta, phi):
            out = np.zeros((10, 4, theta.size), dtype=complex)
            out[:, 1] = np.sin(theta) + 1j * np.cos(phi)
            out[:, 2] = np.cos(theta) + 2j * np.sin(phi)
            return out

    ns = 4
    grid = GalacticGrid(ns)
    rotation = Rotation.from_rotvec([0.3, 0.7, -0.4]).as_matrix()
    theta, phi, alpha, up = transport(rotation, grid)
    q = np.linspace(-1, 1, grid.npix)
    u = np.linspace(2, -1, grid.npix)
    kernel = Kernel()
    selection, theta_up, weights = audit.pair_weights(
        kernel, ns, rotation, q, u, 13, tmp_path
    )
    np.testing.assert_array_equal(selection, np.flatnonzero(up))
    np.testing.assert_allclose(theta_up, theta[up], atol=1e-14)
    p = (q[up] + 1j * u[up]) * np.exp(2j * alpha[up])
    sampled = kernel.sample(theta[up], phi[up])
    for pair in audit.PAIR_INDICES:
        expected = audit.branch_weights(sampled[pair, 1], sampled[pair, 2], p)
        for branch in audit.BRANCH_SIGNS:
            assert isinstance(weights[pair][branch], np.memmap)
            np.testing.assert_allclose(
                weights[pair][branch], expected[branch], atol=1e-14
            )


def test_ledger_rejects_changed_provenance_and_keeps_checkpoint(tmp_path):
    args = audit.arguments(["--result-path", str(tmp_path / "ledger.json")])
    metadata = {"fingerprint": "one", "schema_version": audit.SCHEMA_VERSION}
    result = audit.load_results(args, metadata)
    result["runs"]["finished"] = {"real": 12}
    audit.write_results(result, args.result_path)
    assert audit.load_results(args, metadata)["runs"]["finished"]["real"] == 12
    with pytest.raises(RuntimeError, match="incompatible provenance"):
        audit.load_results(args, {"fingerprint": "two"})
    assert (
        json.loads(args.result_path.read_text())["runs"]["finished"]["real"]
        == 12
    )
    with audit.ledger_lock(args.result_path):
        with pytest.raises(RuntimeError, match="another job"):
            with audit.ledger_lock(args.result_path):
                pass


def test_preflight_needs_no_numerical_imports_or_files(tmp_path):
    result = tmp_path / "untouched" / "ledger.json"
    code = (
        "import runpy,sys; "
        f"sys.argv=[{str(SCRIPT)!r},'--plan',"
        f"'--result-path',{str(result)!r}]; "
        f"runpy.run_path({str(SCRIPT)!r},run_name='__main__'); "
        "assert not any(n in sys.modules "
        "for n in ('numpy','healpy','lusee','jax'))"
    )
    child = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True
    )
    assert child.returncode == 0, child.stderr
    assert "requested entries: 144" in child.stdout
    assert not result.parent.exists()


def test_legacy_result_path_is_protected():
    with pytest.raises(SystemExit):
        audit.arguments(
            ["--result-path", str(audit.OUT / "shuffle_bgl_results.json")]
        )


def test_small_full_driver_checkpoints_spectra_and_resumes(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation

    class Kernel:
        def sample(self, theta, phi):
            out = np.zeros((10, 4, theta.size), dtype=complex)
            out[:, 1] = np.sin(theta) + 1j * np.cos(phi)
            out[:, 2] = np.cos(theta) + 2j * np.sin(phi)
            return out

    rng = np.random.default_rng(23)
    maps = tuple(rng.normal(size=192) for _ in range(3))
    args = audit.arguments(
        [
            "--result-path",
            str(tmp_path / "ledger.json"),
            "--cache-dir",
            str(tmp_path / "scratch"),
            "--save-spectra",
            "--snapshots",
            "0",
            "--chunk",
            "17",
            "--nufft-chunk",
            "29",
        ]
    )
    args.nsides = [4]  # exercise the real driver with a tiny synthetic sphere
    monkeypatch.setattr(
        audit,
        "fingerprint",
        lambda: {
            "fingerprint": "synthetic",
            "schema_version": 2,
            "response_path": "fake",
        },
    )
    monkeypatch.setattr(audit, "sky_alms", lambda: ())
    monkeypatch.setattr(audit, "target_maps", lambda alms, ns: maps)
    monkeypatch.setattr(audit.rsp, "load_response", lambda path: None)
    monkeypatch.setattr(
        audit.rsp, "FixedChannelKernel", lambda resp, freq: Kernel()
    )
    monkeypatch.setattr(
        audit,
        "cfg",
        SimpleNamespace(
            fine_freqs=lambda center: np.linspace(29.9, 30.1, 19),
            times=lambda: [0],
            moon_location=lambda: None,
        ),
    )
    rotation = Rotation.from_rotvec([0.3, 0.7, -0.4]).as_matrix()
    monkeypatch.setattr(
        audit.conv, "topo_rotation_matrix", lambda time, loc: rotation
    )
    audit.calculate(args)
    saved = json.loads(args.result_path.read_text())
    assert len(saved["runs"]) == 24
    assert len(list(tmp_path.glob("*.npz"))) == 12
    assert saved["executions"][0]["completed_utc"]
    for entry in saved["runs"].values():
        assert entry["phase_sign"] == audit.BRANCH_SIGNS[entry["branch"]]
        assert 1 <= entry["n_eff_amplitude"] <= entry["pixels"] + 1e-12
    for path in tmp_path.glob("*.npz"):
        with np.load(path) as d:
            assert d["real"].shape == (19,)
            assert np.all(d["real"] >= 0)

    def no_sky_load():
        pytest.fail("a fully completed run must skip sky loading")

    monkeypatch.setattr(audit, "sky_alms", no_sky_load)
    audit.calculate(args)
    assert json.loads(args.result_path.read_text())["runs"] == saved["runs"]
