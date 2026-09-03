from __future__ import annotations

import importlib.util
import stat
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "enterprise_due_diligence_gate.py"
SOURCE_COMMIT = "a" * 40


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("enterprise_due_diligence_gate_atomic", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load_module()


def _descriptor_write_prerequisites() -> dict[str, bool]:
    """Report each OS primitive required by descriptor-relative replacement."""
    return {
        "posix": GATE.os.name == "posix",
        "open_dir_fd": GATE.os.open in GATE.os.supports_dir_fd,
        "mkdir_dir_fd": GATE.os.mkdir in GATE.os.supports_dir_fd,
        "rename_dir_fd": GATE.os.rename in GATE.os.supports_dir_fd,
        "unlink_dir_fd": GATE.os.unlink in GATE.os.supports_dir_fd,
        "stat_dir_fd": GATE.os.stat in GATE.os.supports_dir_fd,
        "fchmod": hasattr(GATE.os, "fchmod"),
        "o_directory": hasattr(GATE.os, "O_DIRECTORY"),
        "o_nofollow": hasattr(GATE.os, "O_NOFOLLOW"),
    }


def _descriptor_writes_supported() -> bool:
    """Return whether every descriptor-relative replacement primitive exists."""
    return all(_descriptor_write_prerequisites().values())


def _assert_descriptor_write_capability() -> bool:
    """Prove unsupported platforms lack a named prerequisite instead of skipping."""
    prerequisites = _descriptor_write_prerequisites()
    supported = all(prerequisites.values())
    if not supported:
        missing = tuple(name for name, available in prerequisites.items() if not available)
        assert missing
    return supported


def test_descriptor_write_capability_fails_closed_when_prerequisite_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing atomic-write primitives must be failing evidence, never a passing return."""
    monkeypatch.setattr(
        __import__(__name__),
        "_descriptor_write_prerequisites",
        lambda: {"posix": False, "open_dir_fd": True},
    )

    with pytest.raises(AssertionError, match="posix"):
        _assert_descriptor_write_capability()


def _permissions(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_descriptor_write_failure_preserves_existing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed descriptor write must not truncate the accepted manifest."""
    if not _assert_descriptor_write_capability():
        return

    monkeypatch.chdir(tmp_path)
    output_path = Path("secure") / "gate.json"
    output_path.parent.mkdir()
    output_path.write_text("previous-manifest\n", encoding="utf-8")
    manifest = GATE.build_gate_manifest(source_commit=SOURCE_COMMIT)
    original_fdopen = GATE.os.fdopen

    class FailingStream:
        def __init__(self, fd: int, *args: object, **kwargs: object) -> None:
            self._stream = original_fdopen(fd, *args, **kwargs)

        def __enter__(self) -> "FailingStream":
            self._stream.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._stream.__exit__(*args)

        def write(self, content: str) -> int:
            del content
            raise OSError("injected descriptor write failure")

    monkeypatch.setattr(GATE.os, "fdopen", FailingStream)

    with pytest.raises(ValueError, match="manifest output could not be written"):
        GATE.write_gate_manifest(manifest, output_path)

    assert output_path.read_text(encoding="utf-8") == "previous-manifest\n"
    assert sorted(path.name for path in output_path.parent.iterdir()) == ["gate.json"]


def test_descriptor_replacement_preserves_existing_manifest_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Atomic replacement must retain an existing manifest's access contract."""
    if not _assert_descriptor_write_capability():
        return

    monkeypatch.chdir(tmp_path)
    output_path = Path("secure") / "gate.json"
    output_path.parent.mkdir()
    output_path.write_text("previous-manifest\n", encoding="utf-8")
    output_path.chmod(0o640)

    GATE.write_gate_manifest(
        GATE.build_gate_manifest(source_commit=SOURCE_COMMIT),
        output_path,
    )

    assert _permissions(output_path) == 0o640


def test_descriptor_new_manifest_uses_normal_creation_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new atomic manifest must retain ordinary file-creation permissions."""
    if not _assert_descriptor_write_capability():
        return

    monkeypatch.chdir(tmp_path)
    baseline = Path("baseline.json")
    baseline.write_text("baseline\n", encoding="utf-8")
    expected_permissions = _permissions(baseline)
    output_path = Path("secure") / "gate.json"

    GATE.write_gate_manifest(
        GATE.build_gate_manifest(source_commit=SOURCE_COMMIT),
        output_path,
    )

    assert _permissions(output_path) == expected_permissions
