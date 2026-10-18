"""Reusable disposable-source fault audit for scientific unit tests."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from common import GEN_DIR, REPO
from paper_common import digest, write_json


def run_faults(faults, stem, runner):
    results = []
    sources = set()
    tests = set()
    for index, (name, source, before, after, test) in enumerate(faults):
        sources.add(source)
        tests.add(test.split("::")[0])
        with tempfile.TemporaryDirectory(prefix=stem + "_fault_") as folder:
            work = Path(folder)
            for subdir in ("src", "tests"):
                shutil.copytree(
                    REPO / subdir,
                    work / subdir,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            path = work / source
            original = path.read_text()
            if original.count(before) != 1:
                raise ValueError(f"fault target not unique: {name}")
            path.write_text(original.replace(before, after))
            run = subprocess.run(
                [
                    str(REPO / ".venv/bin/python"),
                    "-m",
                    "pytest",
                    test,
                    "-q",
                    "-o",
                    "addopts=",
                ],
                cwd=work,
                env=dict(
                    os.environ,
                    PYTHONPATH=str(work / "src"),
                    PYTHONDONTWRITEBYTECODE="1",
                ),
                capture_output=True,
                text=True,
                timeout=120,
            )
            log = GEN_DIR / f"{stem}_fault_{index:02d}.log"
            log.write_text(run.stdout + run.stderr)
            caught = (
                run.returncode == 1
                and "FAILED" in run.stdout
                and "ERROR collecting" not in run.stdout
            )
            results.append(
                dict(
                    fault=name,
                    source=source,
                    test=test,
                    caught=caught,
                    returncode=run.returncode,
                    log_sha256=digest(log),
                )
            )
            print(name, "caught" if caught else "FAILED CHECK", flush=True)
    record = dict(
        faults=results,
        all_caught=all(row["caught"] for row in results),
        source_sha256={p: digest(REPO / p) for p in sorted(sources)},
        test_sha256={p: digest(REPO / p) for p in sorted(tests)},
        runner_sha256=digest(runner),
        helper_sha256=digest(Path(__file__)),
    )
    write_json(GEN_DIR / f"{stem}_fault_checks.json", record)
    if not record["all_caught"]:
        raise SystemExit(1)
    return record
