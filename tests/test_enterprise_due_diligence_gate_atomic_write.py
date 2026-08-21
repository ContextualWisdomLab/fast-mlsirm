from __future__ import annotations

import importlib.util
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


def _descriptor_writes_supported() -> bool:
    return (
        GATE.os.name == "posix"
        and GATE.os.open in GATE.os.supports_dir_fd
        and GATE.os.mkdir in GATE.os.supports_dir_fd
        and GATE.os.rename in GATE.os.supports_dir_fd
        and GATE.os.unlink in GATE.os.supports_dir_fd
        and hasattr(GATE.os, "O_DIRECTORY")
        and hasattr(GATE.os, "O_NOFOLLOW")
    )


def test_descriptor_write_failure_preserves_existing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed descriptor write must not truncate the accepted manifest."""
    if not _descriptor_writes_supported():
        pytest.skip("descriptor-relative atomic replacement is unavailable")

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
