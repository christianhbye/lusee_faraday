"""Production shared-scale combination faults in disposable script copies."""

import common  # noqa: F401
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from common import GEN_DIR, REPO
from paper_common import digest, write_json


def compute():
    target = "scripts/residual_signal.py"
    test = "tests/test_global_phase_mixture.py"
    faults = [
        (
            "omit global scale mean variation",
            'solver.moments(a)["speckle"] + solver.moments(delta)["smooth"]',
            'solver.moments(a)["speckle"]',
        ),
        (
            "square mixture probabilities",
            "speckle += p * (",
            "speckle += p**2 * (",
        ),
        (
            "use unweighted conditional mean",
            "sum(p * a[:, 0] for p, a in zip(probabilities, values))",
            "sum(a[:, 0] for p, a in zip(probabilities, values))/len(values)",
        ),
    ]
    rows = []
    for i, (name, before, after) in enumerate(faults):
        with tempfile.TemporaryDirectory(
            prefix="global_mixture_fault_"
        ) as folder:
            root = Path(folder)
            for directory in ("src", "scripts", "tests"):
                shutil.copytree(
                    REPO / directory,
                    root / directory,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            source = root / target
            text = source.read_text()
            if text.count(before) != 1:
                raise ValueError("nonunique fault: " + name)
            source.write_text(text.replace(before, after))
            result = subprocess.run(
                [
                    str(REPO / ".venv/bin/python"),
                    "-m",
                    "pytest",
                    test,
                    "-q",
                    "-o",
                    "addopts=",
                ],
                cwd=root,
                env=dict(os.environ, PYTHONPATH=str(root / "src")),
                capture_output=True,
                text=True,
                timeout=120,
            )
            log = GEN_DIR / f"global_phase_mixture_fault_{i}.log"
            log.write_text(result.stdout + result.stderr)
            caught = result.returncode == 1 and "FAILED" in result.stdout
            rows.append(
                dict(fault=name, caught=caught, log_sha256=digest(log))
            )
            print(name, caught, flush=True)
    write_json(
        GEN_DIR / "global_phase_mixture_fault_checks.json",
        dict(
            source_sha256=digest(REPO / target),
            test_sha256=digest(REPO / test),
            runner_sha256=digest(__file__),
            faults=rows,
            all_caught=all(r["caught"] for r in rows),
        ),
    )
    if not all(r["caught"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    compute()
