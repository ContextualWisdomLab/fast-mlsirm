"""Executable contracts for strict SPDX 2.3 release-SBOM validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_release_spdx.py"


def _validator_module():
    """Load the repository-owned release-SBOM validator by path."""
    assert _SCRIPT.is_file(), "release SBOM validation must be repository-owned"
    spec = importlib.util.spec_from_file_location("validate_release_spdx", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_document() -> dict[str, object]:
    """Return the minimal governed SPDX 2.3 document-creation envelope."""
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "fast-mlsirm release source",
        "documentNamespace": "https://github.com/ContextualWisdomLab/fast-mlsirm/sbom/test",
        "creationInfo": {
            "created": "2026-09-05T13:00:00Z",
            "creators": ["Tool: syft-v1.51.1"],
        },
    }


def test_valid_spdx_23_document_creation_envelope_is_accepted(tmp_path: Path):
    """A complete SPDX 2.3 document-creation envelope is admissible."""
    module = _validator_module()
    path = tmp_path / "sbom.json"
    path.write_text(json.dumps(_valid_document()), encoding="utf-8")
    module.validate_release_spdx(path)


def test_duplicate_member_and_nonfinite_json_fail_closed(tmp_path: Path):
    """Ambiguous or non-standard JSON cannot become release evidence."""
    module = _validator_module()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"spdxVersion":"SPDX-2.3","spdxVersion":"SPDX-2.3"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON member"):
        module.validate_release_spdx(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"spdxVersion":"SPDX-2.3","x":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        module.validate_release_spdx(nonfinite)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("spdxVersion", "SPDX-2.2", "spdxVersion"),
        ("dataLicense", "MIT", "dataLicense"),
        ("SPDXID", "SPDXRef-OTHER", "SPDXID"),
        ("name", "", "name"),
        ("name", "fast-mlsirm\nrelease source", "name"),
        ("documentNamespace", "relative/path", "documentNamespace"),
        (
            "documentNamespace",
            "https://github.com/ContextualWisdomLab/fast-mlsirm/sbom/test#fragment",
            "documentNamespace",
        ),
        (
            "documentNamespace",
            "https://github.com/ContextualWisdomLab/fast-mlsirm/sbom/%ZZ",
            "documentNamespace",
        ),
        (
            "documentNamespace",
            "https://github.com/ContextualWisdomLab/fast-mlsirm/sbom/%",
            "documentNamespace",
        ),
        (
            "documentNamespace",
            "https://github.com/ContextualWisdomLab/fast-mlsirm/sbom/측정",
            "documentNamespace",
        ),
        (
            "documentNamespace",
            "https://github.com/ContextualWisdomLab/fast-mlsirm/sbom\\test",
            "documentNamespace",
        ),
        (
            "documentNamespace",
            "https://github.com/ContextualWisdomLab/fast-mlsirm/sbom/\x00test",
            "documentNamespace",
        ),
    ],
)
def test_required_document_identity_fields_fail_closed(
    tmp_path: Path, field: str, value: str, message: str
):
    """SPDX version and document identity requirements are explicit gates."""
    module = _validator_module()
    document = _valid_document()
    document[field] = value
    path = tmp_path / f"{field}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        module.validate_release_spdx(path)


@pytest.mark.parametrize(
    "creation_info",
    [
        None,
        {},
        {"created": "2026-09-05T13:00:00Z", "creators": []},
        {"created": "2026-09-05T13:00:00Z", "creators": [""]},
        {"created": "2026-09-05T13:00:00Z", "creators": ["syft-v1.51.1"]},
        {"created": "2026-09-05T13:00:00Z", "creators": ["Tool: syft\nv1.51.1"]},
        {"created": "2026-09-05T13:00:00+09:00", "creators": ["Tool: syft"]},
        {"created": "2026-09-05Z", "creators": ["Tool: syft"]},
        {"created": "2026-09-05 13:00:00Z", "creators": ["Tool: syft"]},
        {"created": "2026-09-05T13:00Z", "creators": ["Tool: syft"]},
        {"created": "2026-09-05T13:00:00.123Z", "creators": ["Tool: syft"]},
    ],
)
def test_creation_information_is_complete_and_exact_spdx_utc(
    tmp_path: Path, creation_info: object
):
    """Creation evidence must use SPDX's single-line creator and exact UTC form."""
    module = _validator_module()
    document = _valid_document()
    document["creationInfo"] = creation_info
    path = tmp_path / "creation.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="creationInfo"):
        module.validate_release_spdx(path)
