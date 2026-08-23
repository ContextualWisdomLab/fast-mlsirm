"""Regression coverage for PR queue snapshot JSON-helper isolation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_pr_queue_snapshot.py"


def test_capture_fails_closed_without_repository_bounded_json_helper(tmp_path):
    """A copied script cannot silently fall back to a weaker JSON decoder."""
    isolated_script = tmp_path / "capture_pr_queue_snapshot.py"
    isolated_script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    probe = "\n".join(
        [
            "import importlib.util",
            f"path = {str(isolated_script)!r}",
            "spec = importlib.util.spec_from_file_location('isolated_capture', path)",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode != 0
    assert "_bounded_json" in completed.stderr
