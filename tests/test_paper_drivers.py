"""Changed inputs and altered arrays must not reuse cached predictions."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def paper_modules(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    common = importlib.import_module("paper_common")
    depth = importlib.import_module("depth_depolarization")
    return common, depth


def test_manifest_refuses_changed_config_and_source(
    tmp_path, monkeypatch, paper_modules
):
    common, _ = paper_modules
    monkeypatch.setattr(common, "REPO", tmp_path)
    source = tmp_path / "equation.py"
    source.write_text("phase = 2\n")
    _, key = common.manifest({"sigma": 0.02}, ["equation.py"])
    record = tmp_path / "record.json"
    common.write_json(record, dict(fingerprint=key))
    assert common.load_record(record, key)["fingerprint"] == key
    _, changed = common.manifest({"sigma": 0.01}, ["equation.py"])
    with pytest.raises(ValueError, match="provenance"):
        common.load_record(record, changed)
    source.write_text("phase = -2\n")
    _, changed = common.manifest({"sigma": 0.02}, ["equation.py"])
    assert changed != key


def test_depth_driver_cache_rejects_corruption(
    tmp_path, monkeypatch, paper_modules
):
    common, driver = paper_modules
    monkeypatch.setattr(driver, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(driver, "REPO", tmp_path)
    config = dict(
        scope="test",
        frequencies_mhz=[45.0, 50.0],
        screen_model={},
        sigma_depth_values=[0.02],
        profiles=[dict(kind="slab")],
        depth_numerical=dict(order=8, max_order=128),
        separation_numerical={},
    )
    record = tmp_path / "result.json"
    first = driver.compute(config, record)
    second = driver.compute(config, record)
    assert first["fingerprint"] == second["fingerprint"]
    assert first["rows"] == second["rows"]
    cache = next(tmp_path.rglob("*.npz"))
    cache.write_bytes(b"altered covariance")
    with pytest.raises(ValueError, match="corrupt cached"):
        driver.compute(config, record)
