"""Build the report from committed sources in a clean temporary worktree."""

import common  # noqa: F401
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from common import GEN_DIR, REPO
from paper_common import digest, write_json


def build():
    pending = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "src",
            "scripts",
            "configs/paper",
            "docs",
            "report/paper-implementation.tex",
        ],
        cwd=REPO,
        text=True,
    ).strip()
    if pending:
        raise ValueError(
            "Commit completed sources and numerical records before the "
            "clean report build: " + pending
        )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    with tempfile.TemporaryDirectory(prefix="lusee_report_") as folder:
        work = Path(folder) / "worktree"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(work), revision],
            cwd=REPO,
            check=True,
        )
        try:
            for name in (".venv", "data", "generated_data"):
                (work / name).symlink_to(REPO / name, target_is_directory=True)
            (work / "report/figures").symlink_to(
                REPO / "report/figures", target_is_directory=True
            )
            subprocess.run(
                [str(work / ".venv/bin/python"), "scripts/paper_products.py"],
                cwd=work,
                check=True,
            )
            log = GEN_DIR / "paper_report_latex.log"
            pass_logs = []
            for index in (1, 2):
                pass_log = GEN_DIR / f"paper_report_latex_pass{index}.log"
                with pass_log.open("w") as output:
                    subprocess.run(
                        [
                            "pdflatex",
                            "-interaction=nonstopmode",
                            "-halt-on-error",
                            "paper-implementation.tex",
                        ],
                        cwd=work / "report",
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                pass_logs.append(pass_log.read_text())
            log.write_text("\n".join(pass_logs))
            text = pass_logs[-1]
            shutil.copy2(
                work / "report/paper-implementation.pdf",
                GEN_DIR / "paper_report_preview.pdf",
            )
            forbidden = [
                p
                for p in (
                    "Overfull ",
                    "undefined references",
                    "There were undefined",
                    "LaTeX Error",
                    "Float too large for page",
                )
                if p in text
            ]
            if forbidden:
                raise RuntimeError(
                    "report layout/reference errors: " + str(forbidden)
                )
            # The temporary worktree isolates the user's unrelated notebook
            # edits. Numerical inputs are linked and checksummed, not copied.
            for path in (work / "report").glob("paper-*"):
                if path.suffix in (".tex", ".json", ".pdf"):
                    shutil.copy2(path, REPO / "report" / path.name)
            pdf = REPO / "report/paper-implementation.pdf"
            write_json(
                REPO / "docs/paper-report-build.json",
                dict(
                    source_revision=revision,
                    clean_source_worktree=True,
                    pdf_sha256=digest(pdf),
                    latex_log_sha256=digest(log),
                    generator_sha256=digest(
                        REPO / "scripts/paper_products.py"
                    ),
                    builder_sha256=digest(__file__),
                    report_source_sha256=digest(
                        REPO / "report/paper-implementation.tex"
                    ),
                ),
            )
            print("Built", pdf, flush=True)
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(work)],
                cwd=REPO,
                check=True,
            )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
