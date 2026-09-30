"""Artifact-file readiness only: tiny fixtures, no numerical package imports."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
import time

import pytest

_PATH = Path(__file__).resolve().parents[1] / "python/fast_mlsirm/_artifact_readiness.py"
_SPEC = importlib.util.spec_from_file_location("_artifact_readiness_test_leaf", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = MODULE
_SPEC.loader.exec_module(MODULE)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _pin(policy):
    return hashlib.sha256(b"fast-mlsirm/artifact-policy/1\0" + _canonical(policy)).hexdigest()


@pytest.fixture
def installation(tmp_path):
    package = tmp_path / "fast_mlsirm"
    package.mkdir()
    (package / "__init__.py").write_bytes(b"raise RuntimeError('must never execute')\n")
    (package / "_core.fixture.so").write_bytes(b"approved native-file fixture, not a loaded binary\n")
    descriptor = {"sys_platform": sys.platform, "machine": platform.machine(), "python_abi": sysconfig.get_config_var("SOABI")}
    members = [
        {"path": p.relative_to(tmp_path).as_posix(), "size_bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(package.iterdir())
    ]
    policy = {
        "schema": "worker_artifact_policy/1.0",
        "distribution": {"name": "fast-mlsirm", "version": "0.11.4"},
        "source_commit": "a" * 40,
        "installation_profile": "host_immutable_install/1.0",
        "targets": [{"target_id": "fixture", "descriptor": descriptor, "archive_sha256": "b" * 64,
                     "package_roots": ["fast_mlsirm"], "extension_member": "fast_mlsirm/_core.fixture.so", "members": members}],
    }
    startup = {
        "distribution": dict(policy["distribution"]), "descriptor": dict(descriptor),
        "loaded_origin": str(package / "_core.fixture.so"), "worker_pid": os.getpid(),
        "startup_policy_digest": _pin(policy), "startup_installation_epoch": "epoch1",
        "current_installation_epoch": "epoch1", "immutable_profile": "host_immutable_install/1.0",
    }
    return tmp_path, policy, startup


def _verify(installation, *, policy=None, startup=None, pin=None, deadline=None):
    root, original, observation = installation
    policy = original if policy is None else policy
    startup = observation if startup is None else startup
    return MODULE.verify_artifact_readiness(
        json.dumps(policy), pinned_policy_digest=_pin(original) if pin is None else pin,
        installation_root=root, startup_observation=startup,
        deadline_monotonic=time.monotonic() + 10.0 if deadline is None else deadline,
    )


def test_a01_missing_approval_is_unknown_without_file_access(installation, monkeypatch):
    monkeypatch.setattr(MODULE.os, "open", lambda *a, **k: pytest.fail("unexpected file access"))
    report = MODULE.verify_artifact_readiness(None, pinned_policy_digest=None,
        installation_root=installation[0], startup_observation=None, deadline_monotonic=time.monotonic() + 10.0)
    assert (report.state, report.reason_code) == ("unknown", "approval_missing")
    assert report.observed_members_sha256 is None


@pytest.mark.parametrize("raw", ['{"schema":NaN}', '{"schema":Infinity}', '{"schema":1e999}', '{"schema":1,"schema":2}', '{'])
def test_a02_invalid_json_rejected(raw, installation):
    with pytest.raises(ValueError):
        MODULE.verify_artifact_readiness(raw, pinned_policy_digest="c" * 64,
            installation_root=installation[0], startup_observation=None, deadline_monotonic=time.monotonic() + 10.0)


@pytest.mark.parametrize("change", ["bool_size", "extra_field", "reverse_members", "bad_source"])
def test_a02_exact_policy_shape(installation, change):
    policy = copy.deepcopy(installation[1])
    if change == "bool_size":
        policy["targets"][0]["members"][0]["size_bytes"] = True
    elif change == "extra_field":
        policy["host_pin_override"] = "c" * 64
    elif change == "reverse_members":
        policy["targets"][0]["members"].reverse()
    else:
        policy["source_commit"] = "d" * 64
    with pytest.raises(ValueError):
        _verify(installation, policy=policy, pin=_pin(policy))


def test_a02_unknown_schema_does_not_qualify(installation):
    policy = {"schema": "worker_artifact_policy/9.0"}
    report = _verify(installation, policy=policy, pin=_pin(policy))
    assert (report.state, report.reason_code) == ("unknown", "policy_schema_unsupported")


def test_a03_host_pin_mismatch_and_identical_approved_bytes(installation):
    policy = copy.deepcopy(installation[1])
    policy["source_commit"] = "c" * 40
    assert _verify(installation, policy=policy).reason_code == "policy_pin_mismatch"
    # Content equality with the independently configured pin is allowed, not origin authentication.
    assert _verify(installation, policy=copy.deepcopy(installation[1])).state == "ready"


@pytest.mark.parametrize("field", ["sys_platform", "machine", "python_abi"])
def test_a04_runtime_target_mismatch(installation, field):
    startup = copy.deepcopy(installation[2])
    startup["descriptor"][field] = "wrong"
    assert _verify(installation, startup=startup).reason_code == "target_mismatch"


@pytest.mark.parametrize("change,reason", [("missing", "member_missing"), ("size", "member_size_mismatch"), ("digest", "member_digest_mismatch")])
def test_a05_missing_or_changed_member(installation, change, reason):
    path = installation[0] / "fast_mlsirm/__init__.py"
    if change == "missing":
        path.unlink()
    elif change == "size":
        path.write_bytes(b"short")
    else:
        original = path.read_bytes()
        path.write_bytes(b"x" * len(original))
    assert _verify(installation).reason_code == reason


@pytest.mark.parametrize("origin,reason", [(None, "startup_missing"), ("/outside/_core.so", "origin_mismatch")])
def test_a06_missing_or_unapproved_origin(installation, origin, reason):
    startup = copy.deepcopy(installation[2])
    startup["loaded_origin"] = origin
    assert _verify(installation, startup=startup).reason_code == reason


@pytest.mark.parametrize("path", ["../escape.py", "/absolute.py", "file://code.py", "a\\b.py", "fast_mlsirm/../code.py"])
def test_a07_unsafe_policy_paths(installation, path):
    policy = copy.deepcopy(installation[1])
    policy["targets"][0]["members"][0]["path"] = path
    with pytest.raises(ValueError):
        _verify(installation, policy=policy, pin=_pin(policy))


@pytest.mark.parametrize("relative", ["extra.py", "__pycache__/unapproved.pyc", "metadata.txt"])
def test_a07_closed_member_set_including_bytecode(installation, relative):
    path = installation[0] / "fast_mlsirm" / relative
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"unapproved")
    assert _verify(installation).reason_code == "member_unapproved"


def test_a07_symlink_parent_is_not_followed(installation):
    root, policy, startup = installation
    external = root / "outside"
    (root / "fast_mlsirm").rename(external)
    (root / "fast_mlsirm").symlink_to(external, target_is_directory=True)
    assert _verify(installation).reason_code == "unsafe_file"


def test_a07_hardlink_alias_is_rejected(installation):
    root, policy, startup = installation
    os.link(root / "fast_mlsirm/__init__.py", root / "fast_mlsirm/alias.py")
    policy = copy.deepcopy(policy)
    data = (root / "fast_mlsirm/alias.py").read_bytes()
    policy["targets"][0]["members"].append({"path": "fast_mlsirm/alias.py", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    policy["targets"][0]["members"].sort(key=lambda x: x["path"])
    startup = copy.deepcopy(startup)
    startup["startup_policy_digest"] = _pin(policy)
    assert _verify(installation, policy=policy, startup=startup, pin=_pin(policy)).reason_code == "unsafe_file"


@pytest.mark.parametrize("raw", ['{"pad":"' + "x" * 1048576 + '"}', "[" * 65 + "0" + "]" * 65])
def test_a08_json_bounds(installation, raw):
    with pytest.raises(ValueError):
        MODULE.verify_artifact_readiness(raw, pinned_policy_digest="a" * 64,
            installation_root=installation[0], startup_observation=None, deadline_monotonic=time.monotonic() + 10.0)


def test_a08_installed_byte_budget(installation):
    policy = copy.deepcopy(installation[1])
    policy["targets"][0]["members"][0]["size_bytes"] = 1073741825
    with pytest.raises(ValueError):
        _verify(installation, policy=policy, pin=_pin(policy))


@pytest.mark.parametrize("field,value,reason", [("worker_pid", 999999, "pid_mismatch"), ("current_installation_epoch", "new", "epoch_changed"), ("startup_policy_digest", "c" * 64, "startup_policy_changed")])
def test_a09_startup_invalidation(installation, field, value, reason):
    startup = copy.deepcopy(installation[2])
    startup[field] = value
    assert _verify(installation, startup=startup).reason_code == reason


def test_a09_in_place_mutation_during_hash(installation, monkeypatch):
    original = MODULE.os.read
    changed = False
    def mutate(fd, size):
        nonlocal changed
        data = original(fd, size)
        if data and not changed:
            changed = True
            path = installation[0] / "fast_mlsirm/__init__.py"
            with path.open("ab") as stream:
                stream.write(b"changed")
        return data
    monkeypatch.setattr(MODULE.os, "read", mutate)
    assert _verify(installation).reason_code == "file_changed"


def test_a09_elapsed_deadline_is_unknown(installation):
    assert _verify(installation, deadline=1.0).reason_code == "deadline_exceeded"


@pytest.mark.parametrize("deadline", [True, 1, 0.0, float("nan"), float("inf")])
def test_a09_invalid_deadline(installation, deadline):
    with pytest.raises(ValueError):
        _verify(installation, deadline=deadline)


def test_a10_unknown_profile_and_unavailable_safe_open(installation, monkeypatch):
    startup = copy.deepcopy(installation[2])
    startup["immutable_profile"] = None
    assert _verify(installation, startup=startup).reason_code == "profile_unknown"
    monkeypatch.delattr(MODULE.os, "O_NOFOLLOW")
    assert _verify(installation).reason_code == "safe_io_unavailable"


def test_a11_stdlib_leaf_and_artifact_only_result(installation):
    tree = ast.parse(_PATH.read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in imports:
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
        assert all(name.split(".")[0] in sys.stdlib_module_names or name == "__future__" for name in names)
        assert not isinstance(node, ast.ImportFrom) or node.level == 0
    report = _verify(installation)
    assert (report.state, report.reason_code, report.scope) == ("ready", "artifact_files_verified", "artifact_files_only")
    assert not hasattr(report, "effective_device") and not hasattr(report, "numerical_evidence")
    expected = hashlib.sha256(b"fast-mlsirm/artifact-members/1\0" + _canonical(installation[1]["targets"][0]["members"])).hexdigest()
    assert report.observed_members_sha256 == expected


@pytest.mark.parametrize("change,match", [("targets", "targets"), ("members", "member count"), ("roots", "package roots"), ("path_bytes", "text")])
def test_a08_collection_and_utf8_path_limits(installation, change, match):
    policy = copy.deepcopy(installation[1])
    target = policy["targets"][0]
    if change == "targets":
        policy["targets"] = [copy.deepcopy(target) for _ in range(17)]
    elif change == "members":
        target["members"] = [{"path": f"fast_mlsirm/f{i:04}.py", "size_bytes": 0, "sha256": "a" * 64} for i in range(4097)]
    elif change == "roots":
        target["package_roots"] = [f"p{i:02}" for i in range(17)]
    else:
        target["members"][0]["path"] = "fast_mlsirm/" + "é" * 251
    with pytest.raises(ValueError, match=match):
        _verify(installation, policy=policy, pin=_pin(policy))


def test_a08_scan_budget_stays_unknown(installation, monkeypatch):
    monkeypatch.setattr(MODULE, "_MAX_ENTRIES", 1)
    assert _verify(installation).reason_code == "scan_limit_exceeded"


def test_a09_path_replacement_during_read(installation, monkeypatch):
    original = MODULE.os.read
    paths = {_identity.st_ino: p for p in (installation[0] / "fast_mlsirm").iterdir() for _identity in [p.stat()]}
    replaced = False
    def read(fd, count):
        nonlocal replaced
        data = original(fd, count)
        if data and not replaced:
            replaced = True
            path = paths[os.fstat(fd).st_ino]
            replacement = installation[0] / "replacement"
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        return data
    monkeypatch.setattr(MODULE.os, "read", read)
    assert _verify(installation).reason_code == "file_changed"


def test_a09_earlier_verified_file_changed_while_next_file_is_read(installation, monkeypatch):
    original = MODULE.os.read
    paths = {p.stat().st_ino: p for p in (installation[0] / "fast_mlsirm").iterdir()}
    completed = None
    changed = False
    def read(fd, count):
        nonlocal completed, changed
        path = paths[os.fstat(fd).st_ino]
        data = original(fd, count)
        if not data and completed is None:
            completed = path
        elif data and completed is not None and path != completed and not changed:
            changed = True
            completed.write_bytes(b"x" * completed.stat().st_size)
        return data
    monkeypatch.setattr(MODULE.os, "read", read)
    assert _verify(installation).reason_code == "file_changed"


@pytest.mark.parametrize("mutation", ["added", "deleted", "replaced"])
def test_a09_new_file_between_scan_and_final_inventory(installation, monkeypatch, mutation):
    original = MODULE._open
    visits = 0
    def open_file(parent, name, *, directory=False):
        nonlocal visits
        if directory and name == "fast_mlsirm":
            visits += 1
            if visits == 2:
                package = installation[0] / "fast_mlsirm"
                if mutation == "added":
                    (package / "late.py").write_bytes(b"not approved")
                elif mutation == "deleted":
                    (package / "__init__.py").unlink()
                else:
                    replacement = installation[0] / "replacement"
                    replacement.write_bytes((package / "__init__.py").read_bytes())
                    replacement.replace(package / "__init__.py")
        return original(parent, name, directory=directory)
    monkeypatch.setattr(MODULE, "_open", open_file)
    assert _verify(installation).state == "mismatch"


def test_a09_nested_directory_added_after_its_hash_pass(installation, monkeypatch):
    root, policy, startup = installation
    nested = root / "fast_mlsirm/nested"
    nested.mkdir()
    member = nested / "approved.txt"
    member.write_bytes(b"nested fixture")
    policy = copy.deepcopy(policy)
    policy["targets"][0]["members"].append({"path": "fast_mlsirm/nested/approved.txt", "size_bytes": 14,
        "sha256": hashlib.sha256(member.read_bytes()).hexdigest()})
    policy["targets"][0]["members"].sort(key=lambda m: m["path"])
    observation = copy.deepcopy(startup)
    observation["startup_policy_digest"] = _pin(policy)
    original = MODULE._open
    visits = 0
    def open_file(parent, name, *, directory=False):
        nonlocal visits
        if directory and name == "fast_mlsirm":
            visits += 1
            if visits == 2:
                (nested / "unlisted.pyc").write_bytes(b"not approved")
        return original(parent, name, directory=directory)
    monkeypatch.setattr(MODULE, "_open", open_file)
    assert _verify(installation, policy=policy, startup=observation, pin=_pin(policy)).state == "mismatch"


@pytest.mark.parametrize("refusal", ["mismatch", "deadline"])
def test_a09_all_owned_descriptors_close_on_refusal(installation, monkeypatch, refusal):
    opened, closed = [], []
    original_open, original_close = MODULE.os.open, MODULE.os.close
    def open_file(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd
    def close_file(fd):
        closed.append(fd)
        return original_close(fd)
    monkeypatch.setattr(MODULE.os, "open", open_file)
    monkeypatch.setattr(MODULE.os, "close", close_file)
    monkeypatch.setattr(MODULE, "_safe_io_available", lambda: True)
    if refusal == "mismatch":
        (installation[0] / "fast_mlsirm/__init__.py").write_bytes(b"wrong")
    else:
        def expired(*args):
            raise MODULE._Refusal("unknown", "deadline_exceeded")
        monkeypatch.setattr(MODULE, "_hash_fd", expired)
    assert _verify(installation).state != "ready"
    assert opened and sorted(opened) == sorted(closed)


def test_a12_streams_more_than_one_chunk_without_candidate_execution(installation):
    root, original, startup = installation
    policy = copy.deepcopy(original)
    path = root / "fast_mlsirm/_core.fixture.so"
    data = b"fixture" * 10000
    path.write_bytes(data)
    member = policy["targets"][0]["members"][1]
    member.update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    observation = copy.deepcopy(startup)
    observation["startup_policy_digest"] = _pin(policy)
    assert _verify(installation, policy=policy, startup=observation, pin=_pin(policy)).state == "ready"


@pytest.mark.parametrize("state,reason,scope", [("ready", "anything", "artifact_files_only"), ("unknown", "approval_missing", "numeric_verified"), ("unknown", "artifact_files_verified", "artifact_files_only")])
def test_readiness_result_codes_and_scope_are_closed(state, reason, scope):
    with pytest.raises(ValueError):
        MODULE.ArtifactReadiness(state, reason, scope=scope)


@pytest.mark.parametrize("root", [True, None, "file://not-an-installation"])
def test_invalid_host_root_has_documented_validation_error(installation, root):
    with pytest.raises(ValueError):
        MODULE.verify_artifact_readiness(json.dumps(installation[1]), pinned_policy_digest=_pin(installation[1]),
            installation_root=root, startup_observation=installation[2], deadline_monotonic=time.monotonic() + 10.0)


@pytest.mark.parametrize("descriptor,archive", [({"sys_platform": "linux", "machine": "x86_64", "python_abi": "cpython-314-x86_64-linux-gnu"}, "c" * 64), ({"sys_platform": "darwin", "machine": "arm64", "python_abi": "cpython-314-darwin"}, "d" * 64)])
def test_a12_approved_cross_target_fixtures(installation, monkeypatch, descriptor, archive):
    policy = copy.deepcopy(installation[1])
    policy["targets"][0]["descriptor"] = descriptor
    policy["targets"][0]["archive_sha256"] = archive
    startup = copy.deepcopy(installation[2])
    startup["descriptor"] = descriptor
    startup["startup_policy_digest"] = _pin(policy)
    monkeypatch.setattr(MODULE, "_runtime_descriptor", lambda: descriptor)
    report = _verify(installation, policy=policy, startup=startup, pin=_pin(policy))
    assert report.state == "ready" and report.archive_sha256 == archive
    assert report.source_commit == "a" * 40
