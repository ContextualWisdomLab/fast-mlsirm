"""Actual transport policy over synthetic ZIPs; no network or install."""
import hashlib
import io
from pathlib import Path
import runpy
import zipfile

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/ci/release_artifact_transport.py"))


def _zip(name="wheel.whl", data=b"synthetic"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, data)
    return buffer.getvalue()


def _selection(payload):
    return [{"id": 41, "name": "dist-wheel-leg", "digest": "sha256:" + hashlib.sha256(payload).hexdigest()}]


def test_actual_id_download_and_member_binding(tmp_path):
    payload = _zip()
    selected = _selection(payload)
    calls = []
    def fetch(repo, artifact_id):
        calls.append((repo, artifact_id))
        return payload
    receipt = MODULE["materialize"](selected, "owner/repo", tmp_path / "d", fetch)
    assert calls == [("owner/repo", 41)]
    MODULE["verify_materialized"](selected, receipt, tmp_path / "d")
    receipt[0]["id"] = 42
    with pytest.raises(ValueError, match="immutable IDs"):
        MODULE["verify_materialized"](selected, receipt, tmp_path / "d")
    receipt[0]["id"] = 41
    (tmp_path / "d/dist-wheel-leg/wheel.whl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="members changed"):
        MODULE["verify_materialized"](selected, receipt, tmp_path / "d")


@pytest.mark.parametrize("artifact_id", [None, True, 0, -1, "41"])
def test_invalid_id_never_downloads(tmp_path, artifact_id):
    selected = _selection(_zip())
    selected[0]["id"] = artifact_id
    with pytest.raises(ValueError, match="artifact ID"):
        MODULE["materialize"](selected, "owner/repo", tmp_path, lambda *_: pytest.fail("unexpected fetch"))


def test_digest_mismatch_stops_before_extract(tmp_path):
    with pytest.raises(ValueError, match="ZIP digest"):
        MODULE["materialize"](_selection(_zip()), "owner/repo", tmp_path, lambda *_: _zip(data=b"other"))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["../escape", "/absolute", "dir/file", ".."])
def test_unsafe_zip_member_is_never_written(tmp_path, name):
    payload = _zip(name)
    with pytest.raises(ValueError, match="ZIP member"):
        MODULE["materialize"](_selection(payload), "owner/repo", tmp_path, lambda *_: payload)
    assert list((tmp_path / "dist-wheel-leg").iterdir()) == []


def test_download_failure_propagates(tmp_path):
    def fail(*_):
        raise RuntimeError("download failed")
    with pytest.raises(RuntimeError, match="download failed"):
        MODULE["materialize"](_selection(_zip()), "owner/repo", tmp_path, fail)
