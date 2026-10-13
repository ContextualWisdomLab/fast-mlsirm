"""Execute the recorded timing shell contract without native work or builds."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "billing-snapshots/perf_0114_regression_20260920.json"


@pytest.mark.parametrize("alias", ["v113", "v114", "fix"])
@pytest.mark.parametrize("failed_call", [0, 1, 2, 3])
def test_repetition_commands_preserve_cwd_and_failures(tmp_path, alias, failed_call):
    """Every declared repetition must run in its tree and fail closed on error."""
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    declared = f"/tmp/fmls-{alias}-perf"
    selected = [
        command
        for command in snapshot["reproduce_commands"]
        if command.startswith(f"(cd {declared} && for i in 1 2 3;")
    ]
    assert len(selected) == 1
    caller = tmp_path / "caller"
    tree = tmp_path / alias
    binaries = tmp_path / "bin"
    for directory in (caller, tree, binaries):
        directory.mkdir()
    calls = tmp_path / "calls.jsonl"
    cargo = binaries / "cargo"
    cargo.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "path = pathlib.Path(os.environ['PERF_CALLS'])\n"
        "number = len(path.read_text().splitlines()) + 1 if path.exists() else 1\n"
        "with path.open('a') as out:\n"
        "    out.write(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:], "
        "'jobs': os.environ.get('CARGO_BUILD_JOBS')}) + '\\n')\n"
        "sys.exit(23 if number == int(os.environ['PERF_FAIL_CALL']) else 0)\n",
        encoding="utf-8",
    )
    cargo.chmod(0o700)
    timer = binaries / "timer"
    timer.write_text(
        '#!/bin/bash\n[ "$1" = "-lp" ] || exit 91\nshift\nexec "$@"\n',
        encoding="utf-8",
    )
    timer.chmod(0o700)
    command = selected[0].replace(declared, shlex.quote(str(tree)))
    command = command.replace("/usr/bin/time", shlex.quote(str(timer)))
    assert "git " not in command
    completed = subprocess.run(
        ["/bin/bash", "-c", command],
        cwd=caller,
        env={
            **os.environ,
            "PATH": str(binaries) + os.pathsep + os.environ.get("PATH", ""),
            "PERF_CALLS": str(calls),
            "PERF_FAIL_CALL": str(failed_call),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    expected_exit = 23 if failed_call else 0
    assert completed.returncode == expected_exit, (
        f"{alias}: repetition {failed_call} failed but its status was lost: "
        f"exit={completed.returncode}, stderr={completed.stderr}"
    )
    observed = [json.loads(row) for row in calls.read_text().splitlines()]
    assert len(observed) == (failed_call or 3)
    assert all(row["cwd"] == str(tree) for row in observed)
    assert all(row["jobs"] == "2" for row in observed)
    expected_args = [
        "test", "--release", "-p", "mlsirm-core", "--test",
        "two_tier_reduces_to_bifactor", "--", "--exact",
        "two_tier_with_single_primary_matches_bifactor_fit",
    ]
    assert all(row["args"] == expected_args for row in observed)
