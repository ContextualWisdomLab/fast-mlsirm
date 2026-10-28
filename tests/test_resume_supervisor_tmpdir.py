"""Fresh-process checks for the supervisor test's optional temporary roots."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SETTLEMENT_TEST = ROOT / "tests/test_resume_supervisor_settlement.py"
PROBE = """
import json
from pathlib import Path
import runpy
import sys
import tempfile

# Resolve Python's default from this child's environment, not a parent cache.
tempfile.tempdir = None
namespace = runpy.run_path(sys.argv[1])
observed = {'evidence': str(namespace['EVIDENCE'])}
if sys.argv[2] == 'fixture':
    import unittest
    case = namespace['SettlementTests']('test_root_nonzero_status_is_preserved_under_safe_sigchld')
    result = unittest.TestResult()
    case.run(result)
    observed.update(
        tests_run=result.testsRun,
        successful=result.wasSuccessful(),
        failures=[detail for _, detail in result.failures],
        errors=[detail for _, detail in result.errors],
        skipped=[reason for _, reason in result.skipped],
        temporary_root=str(case.root) if hasattr(case, 'root') else None,
        temporary_exists=case.root.exists() if hasattr(case, 'root') else None,
        evidence_written=(namespace['EVIDENCE'] / (case._testMethodName + '.json')).is_file(),
    )
print(json.dumps(observed))
"""


def _probe(tmp_path, *, tmpdir_set=False, evidence_override=False, exercise_fixture=False):
    fallback = tmp_path / "fallback"
    configured = tmp_path / "configured"
    evidence = tmp_path / "evidence"
    for path in (fallback, configured, evidence):
        path.mkdir()
    env = dict(os.environ)
    env.pop("TMPDIR", None)
    env.pop("SUPERVISOR_EVIDENCE", None)
    env["TMP"] = env["TEMP"] = str(fallback)
    env["SUPERVISOR_SOURCE"] = str(ROOT)
    if tmpdir_set:
        env["TMPDIR"] = str(configured)
    if evidence_override:
        env["SUPERVISOR_EVIDENCE"] = str(evidence)
    expected_temporary = configured if tmpdir_set else fallback
    expected = evidence if evidence_override else expected_temporary
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", PROBE, str(SETTLEMENT_TEST),
         "fixture" if exercise_fixture else "import"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    assert observed["evidence"] == str(expected)
    if exercise_fixture:
        assert observed["tests_run"] == 1, observed
        assert observed["successful"], json.dumps(observed, indent=2)
        assert observed["failures"] == observed["errors"] == observed["skipped"] == [], observed
        assert Path(observed["temporary_root"]).parent == expected_temporary
        assert observed["temporary_exists"] is False
        assert observed["evidence_written"] is True


@pytest.mark.parametrize("evidence_override", [False, True], ids=["default", "evidence-override"])
def test_module_loads_without_tmpdir(tmp_path, evidence_override):
    _probe(tmp_path, evidence_override=evidence_override)


@pytest.mark.parametrize("tmpdir_set", [False, True], ids=["unset", "set"])
@pytest.mark.parametrize("evidence_override", [False, True], ids=["default", "evidence-override"])
def test_real_supervisor_fixture_uses_optional_tmpdir(tmp_path, tmpdir_set, evidence_override):
    _probe(tmp_path, tmpdir_set=tmpdir_set, evidence_override=evidence_override, exercise_fixture=True)
