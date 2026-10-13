"""Internal artifact-file verification under a trusted immutable startup profile.

Approval and startup observations are host configuration, not authenticated by
this helper. The host must keep the installation root and its ancestors
immutable: no-follow opening protects the root's final component, not arbitrary
ancestor replacement. Final inventory checks are non-atomic and cannot prevent
mutation after an entry's last check. Matching disk files do not prove bytes
already loaded or numerical execution. No candidate code is imported. Deadlines are cooperative; the caller
must supervise a hard deadline around potentially blocking filesystem calls.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import sysconfig
import time
import unicodedata

_POLICY_SCHEMA = "worker_artifact_policy/1.0"
_PROFILE = "host_immutable_install/1.0"
_POLICY_DOMAIN = b"fast-mlsirm/artifact-policy/1\0"
_MEMBERS_DOMAIN = b"fast-mlsirm/artifact-members/1\0"
_MAX_JSON = 1 << 20
_MAX_DEPTH = 64
_MAX_MEMBERS = 4096
_MAX_ENTRIES = 8192
_MAX_BYTES = 1 << 30
_CHUNK = 64 << 10
_DESCRIPTOR = {"sys_platform", "machine", "python_abi"}
_DISTRIBUTION = {"name", "version"}
_REASONS = {
    "ready": frozenset({"artifact_files_verified"}),
    "unknown": frozenset({"approval_missing", "startup_missing", "policy_schema_unsupported", "profile_unknown",
                          "safe_io_unavailable", "io_unavailable", "deadline_exceeded", "scan_limit_exceeded"}),
    "mismatch": frozenset({"policy_pin_mismatch", "target_mismatch", "distribution_mismatch", "pid_mismatch",
                           "startup_policy_changed", "epoch_changed", "origin_mismatch", "member_missing",
                           "member_unapproved", "member_size_mismatch", "member_digest_mismatch", "unsafe_file", "file_changed"}),
}


@dataclass(frozen=True, slots=True)
class ArtifactReadiness:
    state: str
    reason_code: str
    scope: str = "artifact_files_only"
    artifact_policy_digest: str | None = None
    target_id: str | None = None
    source_commit: str | None = None
    archive_sha256: str | None = None
    observed_members_sha256: str | None = None
    worker_pid: int | None = None
    installation_epoch: str | None = None

    def __post_init__(self):
        if (type(self) is not ArtifactReadiness or type(self.state) is not str or self.state not in _REASONS
                or type(self.reason_code) is not str or self.reason_code not in _REASONS[self.state]
                or type(self.scope) is not str or self.scope != "artifact_files_only"):
            raise ValueError("invalid artifact readiness outcome")
        for value in (self.artifact_policy_digest, self.archive_sha256, self.observed_members_sha256):
            if value is not None:
                _hex(value)
        if self.source_commit is not None:
            _hex(self.source_commit, 40)
        for value in (self.target_id, self.installation_epoch):
            if value is not None:
                _text(value)
        if self.worker_pid is not None and (type(self.worker_pid) is not int or self.worker_pid <= 0):
            raise ValueError("invalid artifact readiness process")
        if self.state == "ready" and any(value is None for value in (
                self.artifact_policy_digest, self.target_id, self.source_commit, self.archive_sha256,
                self.observed_members_sha256, self.worker_pid, self.installation_epoch)):
            raise ValueError("ready artifact outcome lacks observations")
        if self.state != "ready" and self.observed_members_sha256 is not None:
            raise ValueError("incomplete artifact outcome cannot claim a complete inventory")


class _Refusal(Exception):
    def __init__(self, state: str, reason: str):
        self.state = state
        self.reason = reason


def _fields(value, keys):
    if type(value) is not dict or len(value) != len(keys) or any(type(k) is not str for k in value) or set(value) != keys:
        raise ValueError("invalid artifact record fields")


def _text(value, maximum=128):
    if (type(value) is not str or not value or value.strip() != value
            or len(value.encode("utf-8")) > maximum
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or unicodedata.normalize("NFC", value) != value):
        raise ValueError("invalid artifact record text")
    return value


def _hex(value, length=64):
    if type(value) is not str or len(value) != length or re.fullmatch(r"[0-9a-f]+", value) is None:
        raise ValueError("invalid artifact identity")
    return value


def _path(value):
    _text(value, 512)
    path = PurePosixPath(value)
    if (path.is_absolute() or path.as_posix() != value or "\\" in value or ":" in value
            or any(part in (".", "..") for part in path.parts) or not path.parts or len(path.parts) > _MAX_DEPTH):
        raise ValueError("invalid artifact member path")
    return value


def _inside(path, root):
    return path.startswith(root + "/")


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _reject_number(_value):
    raise ValueError("artifact policy has no floating-point fields")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate artifact policy field")
        result[key] = value
    return result


def _load_policy(raw):
    if type(raw) not in (str, bytes) or len(raw) > _MAX_JSON:
        raise ValueError("invalid artifact policy size or type")
    try:
        text = raw.decode("utf-8") if type(raw) is bytes else raw
        if len(text.encode("utf-8")) > _MAX_JSON:
            raise ValueError("artifact policy is oversized")
        depth = 0
        quoted = escaped = False
        for char in text:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "[{":
                depth += 1
                if depth > _MAX_DEPTH:
                    raise ValueError("artifact policy is too deep")
            elif char in "]}":
                depth -= 1
        policy = json.loads(text, object_pairs_hook=_pairs, parse_constant=_reject_number, parse_float=_reject_number)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid artifact policy JSON") from exc
    if type(policy) is not dict or "schema" not in policy:
        raise ValueError("artifact policy must be a versioned object")
    _text(policy["schema"])
    if policy["schema"] != _POLICY_SCHEMA:
        raise _Refusal("unknown", "policy_schema_unsupported")
    _fields(policy, {"schema", "distribution", "source_commit", "installation_profile", "targets"})
    _fields(policy["distribution"], _DISTRIBUTION)
    if _text(policy["distribution"]["name"]) != "fast-mlsirm":
        raise ValueError("invalid artifact distribution")
    _text(policy["distribution"]["version"], 64)
    _hex(policy["source_commit"], 40)
    _text(policy["installation_profile"])
    targets = policy["targets"]
    if type(targets) is not list or not 1 <= len(targets) <= 16:
        raise ValueError("invalid artifact targets")
    ids, descriptors, member_count = [], set(), 0
    for target in targets:
        _fields(target, {"target_id", "descriptor", "archive_sha256", "package_roots", "extension_member", "members"})
        ids.append(_text(target["target_id"]))
        _fields(target["descriptor"], _DESCRIPTOR)
        descriptor = tuple(_text(target["descriptor"][k]) for k in sorted(_DESCRIPTOR))
        if descriptor in descriptors:
            raise ValueError("duplicate artifact target descriptor")
        descriptors.add(descriptor)
        _hex(target["archive_sha256"])
        roots = target["package_roots"]
        if type(roots) is not list or not 1 <= len(roots) <= 16:
            raise ValueError("invalid artifact package roots")
        for root in roots:
            _path(root)
        if roots != sorted(set(roots)) or any(_inside(a, b) for a in roots for b in roots if a != b):
            raise ValueError("overlapping or duplicate artifact roots")
        extension = _path(target["extension_member"])
        members = target["members"]
        if type(members) is not list or not members:
            raise ValueError("invalid artifact member set")
        member_count += len(members)
        if member_count > _MAX_MEMBERS:
            raise ValueError("artifact member count exceeds limit")
        names, total = [], 0
        for member in members:
            _fields(member, {"path", "size_bytes", "sha256"})
            name = _path(member["path"])
            if not any(_inside(name, root) for root in roots):
                raise ValueError("artifact member is outside package roots")
            names.append(name)
            size = member["size_bytes"]
            if type(size) is not int or not 0 <= size <= _MAX_BYTES:
                raise ValueError("invalid artifact member size")
            total += size
            _hex(member["sha256"])
        if total > _MAX_BYTES or names != sorted(set(names)) or extension not in names:
            raise ValueError("invalid artifact member closure or byte budget")
    if ids != sorted(set(ids)):
        raise ValueError("artifact targets must be unique and sorted")
    return policy


def _runtime_descriptor():
    return {"sys_platform": sys.platform, "machine": platform.machine(), "python_abi": sysconfig.get_config_var("SOABI")}


def _startup(value):
    _fields(value, {"distribution", "descriptor", "loaded_origin", "worker_pid", "startup_policy_digest",
                    "startup_installation_epoch", "current_installation_epoch", "immutable_profile"})
    for field, keys in (("distribution", _DISTRIBUTION), ("descriptor", _DESCRIPTOR)):
        _fields(value[field], keys)
        for item in value[field].values():
            if item is not None:
                _text(item)
    for field in ("startup_installation_epoch", "current_installation_epoch", "immutable_profile"):
        if value[field] is not None:
            _text(value[field])
    if value["loaded_origin"] is not None:
        _text(value["loaded_origin"], 4096)
    if value["startup_policy_digest"] is not None:
        _hex(value["startup_policy_digest"])
    pid = value["worker_pid"]
    if pid is not None and (type(pid) is not int or not 0 < pid < 1 << 63):
        raise ValueError("invalid artifact startup process")
    # Freeze caller-owned primitive observations before doing any I/O.
    return {**value, "distribution": dict(value["distribution"]), "descriptor": dict(value["descriptor"])}


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise _Refusal("unknown", "deadline_exceeded")


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _identity(info):
    return info.st_dev, info.st_ino


def _check_path(parent, name, info):
    current = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if _stamp(current) != _stamp(info):
        raise _Refusal("mismatch", "file_changed")


def _safe_io_available():
    return (all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"))
            and os.open in os.supports_dir_fd and os.stat in os.supports_dir_fd
            and os.stat in os.supports_follow_symlinks and os.scandir in os.supports_fd)


def _open(parent, name, *, directory=False):
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    valid = stat.S_ISDIR(before.st_mode) if directory else stat.S_ISREG(before.st_mode)
    if not valid:
        raise _Refusal("mismatch", "unsafe_file")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= os.O_DIRECTORY
    fd = os.open(name, flags, dir_fd=parent)
    try:
        info = os.fstat(fd)
        if _stamp(info) != _stamp(before):
            raise _Refusal("mismatch", "file_changed")
        return fd, info
    except BaseException:
        os.close(fd)
        raise


def _hash_fd(fd, expected, deadline):
    digest, count = hashlib.sha256(), 0
    while True:
        _deadline(deadline)
        chunk = os.read(fd, min(_CHUNK, expected + 1 - count))
        if not chunk:
            break
        count += len(chunk)
        digest.update(chunk)
        if count > expected:
            raise _Refusal("mismatch", "file_changed")
    return count, digest.hexdigest()


def _close_owned(fds):
    """Attempt each owned close once before propagating the first I/O error."""
    first_error = None
    for fd in fds:
        try:
            os.close(fd)
        except OSError as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _verify_files(root, target, deadline):
    expected = {m["path"]: m for m in target["members"]}
    observed, identities, snapshots, directories = {}, set(), {}, {}
    entries = 0
    rechecking = False
    final_names = set()

    def walk(fd, prefix, depth):
        nonlocal entries
        _deadline(deadline)
        if depth > _MAX_DEPTH:
            raise _Refusal("unknown", "scan_limit_exceeded")
        before_directory = os.fstat(fd)
        if rechecking and directories.get(prefix) != _stamp(before_directory):
            raise _Refusal("mismatch", "file_changed")
        if not rechecking:
            directories[prefix] = _stamp(before_directory)
        with os.scandir(fd) as iterator:
            for entry in iterator:
                _deadline(deadline)
                entries += 1
                if entries > _MAX_ENTRIES:
                    raise _Refusal("unknown", "scan_limit_exceeded")
                name = entry.name
                path = prefix + "/" + name
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    child, initial = _open(fd, name, directory=True)
                    try:
                        walk(child, path, depth + 1)
                        if _stamp(os.fstat(child)) != _stamp(initial):
                            raise _Refusal("mismatch", "file_changed")
                        _check_path(fd, name, initial)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    if path not in expected:
                        raise _Refusal("mismatch", "member_unapproved")
                    if _identity(info) in identities:
                        raise _Refusal("mismatch", "unsafe_file")
                    member = expected[path]
                    if rechecking:
                        file_fd, final = _open(fd, name)
                        try:
                            if snapshots.get(path) != _stamp(final):
                                raise _Refusal("mismatch", "file_changed")
                            _check_path(fd, name, final)
                            final_names.add(path)
                            identities.add(_identity(final))
                        finally:
                            os.close(file_fd)
                        continue
                    file_fd, initial = _open(fd, name)
                    try:
                        if initial.st_size != member["size_bytes"]:
                            raise _Refusal("mismatch", "member_size_mismatch")
                        count, digest = _hash_fd(file_fd, initial.st_size, deadline)
                        if _stamp(os.fstat(file_fd)) != _stamp(initial):
                            raise _Refusal("mismatch", "file_changed")
                        _check_path(fd, name, initial)
                        if count != initial.st_size:
                            raise _Refusal("mismatch", "file_changed")
                        if digest != member["sha256"]:
                            raise _Refusal("mismatch", "member_digest_mismatch")
                        identities.add(_identity(initial))
                        snapshots[path] = _stamp(initial)
                        observed[path] = {"path": path, "size_bytes": count, "sha256": digest}
                    finally:
                        os.close(file_fd)
                else:
                    raise _Refusal("mismatch", "unsafe_file")
        if _stamp(os.fstat(fd)) != _stamp(before_directory):
            raise _Refusal("mismatch", "file_changed")

    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        initial_root = os.fstat(root_fd)
        for package_root in target["package_roots"]:
            opened, bindings = [], []
            parent = root_fd
            try:
                for part in package_root.split("/"):
                    child, initial = _open(parent, part, directory=True)
                    opened.append(child)
                    bindings.append((parent, part, initial))
                    parent = child
                walk(parent, package_root, len(package_root.split("/")))
                for parent_fd, part, initial in bindings:
                    _check_path(parent_fd, part, initial)
            finally:
                _close_owned(reversed(opened))
        if set(observed) != set(expected):
            raise _Refusal("mismatch", "member_missing")
        # One bounded final closure pass; never retry a mutating installation.
        rechecking = True
        identities.clear()
        entries = 0
        for package_root in target["package_roots"]:
            opened, bindings = [], []
            parent = root_fd
            try:
                for part in package_root.split("/"):
                    child, initial = _open(parent, part, directory=True)
                    opened.append(child)
                    bindings.append((parent, part, initial))
                    parent = child
                walk(parent, package_root, len(package_root.split("/")))
                for parent_fd, part, initial in bindings:
                    _check_path(parent_fd, part, initial)
            finally:
                _close_owned(reversed(opened))
        if final_names != set(expected):
            raise _Refusal("mismatch", "member_missing")
        if _stamp(os.fstat(root_fd)) != _stamp(initial_root) or _identity(os.stat(root, follow_symlinks=False)) != _identity(initial_root):
            raise _Refusal("mismatch", "file_changed")
        _deadline(deadline)
        return hashlib.sha256(_MEMBERS_DOMAIN + _canonical([observed[k] for k in sorted(observed)])).hexdigest()
    finally:
        os.close(root_fd)


def verify_artifact_readiness(policy_json, *, pinned_policy_digest, installation_root,
                              startup_observation, deadline_monotonic) -> ArtifactReadiness:
    """Verify files only; never authenticate the caller or emit numeric evidence.

    Invalid supplied records raise ValueError. Known contradictory facts return
    mismatch; missing or unsupported facts return unknown. No candidate is
    loaded. The caller owns approval, immutable startup and hard I/O deadlines.
    """
    if type(deadline_monotonic) is not float or not math.isfinite(deadline_monotonic) or deadline_monotonic <= 0:
        raise ValueError("artifact readiness needs a positive finite monotonic deadline")
    report = ArtifactReadiness("unknown", "approval_missing")
    try:
        _deadline(deadline_monotonic)
        if policy_json is None or pinned_policy_digest is None:
            return report
        _hex(pinned_policy_digest)
        policy = _load_policy(policy_json)
        digest = hashlib.sha256(_POLICY_DOMAIN + _canonical(policy)).hexdigest()
        if digest != pinned_policy_digest:
            raise _Refusal("mismatch", "policy_pin_mismatch")
        report = replace(report, artifact_policy_digest=digest, source_commit=policy["source_commit"])
        if startup_observation is None:
            raise _Refusal("unknown", "startup_missing")
        startup = _startup(startup_observation)
        if policy["installation_profile"] != _PROFILE or startup["immutable_profile"] != _PROFILE:
            raise _Refusal("unknown", "profile_unknown")
        required = (startup["loaded_origin"], startup["worker_pid"], startup["startup_policy_digest"],
                    startup["startup_installation_epoch"], startup["current_installation_epoch"],
                    *startup["distribution"].values(), *startup["descriptor"].values())
        runtime = _runtime_descriptor()
        if any(v is None for v in required) or any(v is None for v in runtime.values()):
            raise _Refusal("unknown", "startup_missing")
        if startup["worker_pid"] != os.getpid():
            raise _Refusal("mismatch", "pid_mismatch")
        if startup["descriptor"] != runtime:
            raise _Refusal("mismatch", "target_mismatch")
        if startup["distribution"] != policy["distribution"]:
            raise _Refusal("mismatch", "distribution_mismatch")
        if startup["startup_policy_digest"] != digest:
            raise _Refusal("mismatch", "startup_policy_changed")
        if startup["startup_installation_epoch"] != startup["current_installation_epoch"]:
            raise _Refusal("mismatch", "epoch_changed")
        target = next((t for t in policy["targets"] if t["descriptor"] == runtime), None)
        if target is None:
            raise _Refusal("mismatch", "target_mismatch")
        if type(installation_root) not in (str, type(Path())):
            raise ValueError("invalid artifact installation root")
        root_text = os.fspath(installation_root)
        _text(root_text, 4096)
        if "://" in root_text:
            raise ValueError("artifact installation root must not be a URI")
        root = Path(os.path.abspath(root_text))
        if Path(startup["loaded_origin"]) != root / target["extension_member"]:
            raise _Refusal("mismatch", "origin_mismatch")
        report = replace(report, target_id=target["target_id"], archive_sha256=target["archive_sha256"],
                         worker_pid=os.getpid(), installation_epoch=startup["current_installation_epoch"])
        if not _safe_io_available():
            raise _Refusal("unknown", "safe_io_unavailable")
        _deadline(deadline_monotonic)
        observed = _verify_files(root, target, deadline_monotonic)
        return replace(report, state="ready", reason_code="artifact_files_verified", observed_members_sha256=observed)
    except _Refusal as exc:
        return replace(report, state=exc.state, reason_code=exc.reason)
    except FileNotFoundError:
        return replace(report, state="mismatch", reason_code="member_missing")
    except OSError:
        return replace(report, state="unknown", reason_code="io_unavailable")
