"""Offline contracts for the PyPI gap guard's release inventory."""

from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import textwrap

import pytest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/pypi-gap-guard.yml"


def _inventory_step() -> str:
    """Read the actual inventory step, including its declared environment."""
    text = WORKFLOW.read_text(encoding="utf-8")
    inventory = text.split("      - name: Resolve control-plane commit and PyPI inventory\n", 1)[1]
    return inventory.split("\n      - name:", 1)[0]


def test_inventory_declares_the_ephemeral_github_token() -> None:
    """Inventory must not borrow persistent authentication from its runner."""
    environment = _inventory_step().split("        env:\n", 1)[1].split("        run: |\n", 1)[0]
    assert re.search(r"(?m)^          GH_TOKEN: \$\{\{ github.token \}\}$", environment)


def _run_inventory(
    tmp_path: Path, *, event_name: str, release_exit: int = 0,
    release_tags: str = "", hint: str = "", pypi_payload: str = '{"releases":{}}',
    curl_exit: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Execute unchanged workflow Bash with synthetic, network-free adapters."""
    script = textwrap.dedent(_inventory_step().split("        run: |\n", 1)[1])
    binaries = tmp_path / "bin"
    binaries.mkdir()
    adapters = {
        "git": "#!/bin/sh\n[ \"$*\" = 'rev-parse HEAD' ] || exit 98\n"
               "printf '%s\\n' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
        "curl": "#!/bin/sh\nwhile [ \"$#\" -gt 0 ]; do\n"
                "  if [ \"$1\" = --output ]; then\n"
                "    printf '%s' \"$FIXTURE_PYPI_PAYLOAD\" > \"$2\"\n"
                "    exit \"$FIXTURE_CURL_EXIT\"\n"
                "  fi\n  shift\ndone\nexit 99\n",
        "gh": "#!/bin/sh\n"
              '[ "$GH_TOKEN" = synthetic-ephemeral-token ] || exit 97\n'
              '[ "$*" = \'release list --repo fixture/repo --limit 100 --json tagName --jq .[].tagName\' ] || exit 96\n'
              'printf invoked > gh-invoked\n'
              'printf \'%s\' "$FIXTURE_RELEASE_TAGS"\n'
              'if [ "$FIXTURE_RELEASE_EXIT" -ne 0 ]; then\n'
              "  printf '%s\\n' inventory-failure >&2\nfi\n"
              'exit "$FIXTURE_RELEASE_EXIT"\n',
    }
    for name, source in adapters.items():
        binary = binaries / name
        binary.write_text(source, encoding="utf-8")
        binary.chmod(0o755)
    (binaries / "python3").symlink_to(sys.executable)
    environment = {
        "PATH": f"{binaries}{os.pathsep}{os.defpath}",
        "EVENT_NAME": event_name,
        "RELEASE_TAG_HINT": hint,
        "GITHUB_REPOSITORY": "fixture/repo",
        "GITHUB_OUTPUT": str(tmp_path / "github-output"),
        "FIXTURE_RELEASE_EXIT": str(release_exit),
        "FIXTURE_RELEASE_TAGS": release_tags,
        "FIXTURE_PYPI_PAYLOAD": pypi_payload,
        "FIXTURE_CURL_EXIT": str(curl_exit),
        "GH_TOKEN": "synthetic-ephemeral-token",
        "TMPDIR": str(tmp_path),
    }
    bash = shutil.which("bash")
    assert bash is not None, "workflow contract requires Bash"
    return subprocess.run(
        [bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=15,
    )


@pytest.mark.parametrize("event_name", ["schedule", "workflow_dispatch"])
@pytest.mark.parametrize("release_exit", [0, 1, 7])
def test_inventory_failure_cannot_report_no_gaps(
    tmp_path: Path, event_name: str, release_exit: int
) -> None:
    """A failed listing stops inventory; an empty successful list is allowed."""
    result = _run_inventory(tmp_path, event_name=event_name, release_exit=release_exit)
    output = (tmp_path / "github-output").read_text(encoding="utf-8")
    assert result.returncode == release_exit, (result.stdout, result.stderr)
    if release_exit:
        assert "inventory-failure" in result.stderr
        assert "no PyPI gaps detected" not in result.stdout
        assert "gap_count=" not in output
        assert not (tmp_path / "gap_tags.txt").exists()
    else:
        assert "no PyPI gaps detected" in result.stdout
        assert "gap_count=0\n" in output
    assert not list(tmp_path.glob("tmp.*"))


@pytest.mark.parametrize("event_name", ["schedule", "workflow_dispatch", "release"])
def test_inventory_preserves_canonical_missing_version_selection(
    tmp_path: Path, event_name: str
) -> None:
    """Already published and noncanonical tags never become publication gaps."""
    result = _run_inventory(
        tmp_path, event_name=event_name,
        release_tags="v1.2.3\nv2.0.0\nv02.0.0\nnotes\n\n",
        hint="v2.0.0", pypi_payload='{"releases":{"1.2.3":[]}}',
        release_exit=7 if event_name == "release" else 0,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert (tmp_path / "gap_tags.txt").read_text() == "v2.0.0\n"
    assert "gap_count=1\n" in (tmp_path / "github-output").read_text()
    assert (tmp_path / "gh-invoked").exists() is (event_name != "release")
    assert not list(tmp_path.glob("tmp.*"))


@pytest.mark.parametrize("event_name", ["schedule", "workflow_dispatch", "release"])
def test_pypi_inventory_failure_stops_before_release_listing(
    tmp_path: Path, event_name: str
) -> None:
    """PyPI fetch errors cannot admit gaps or use the downstream release API."""
    result = _run_inventory(tmp_path, event_name=event_name, hint="v2.0.0", curl_exit=22)
    assert result.returncode == 22, (result.stdout, result.stderr)
    assert not (tmp_path / "gh-invoked").exists()
    assert not (tmp_path / "gap_tags.txt").exists()
    assert "gap_count=" not in (tmp_path / "github-output").read_text()
    assert not list(tmp_path.glob("tmp.*"))
