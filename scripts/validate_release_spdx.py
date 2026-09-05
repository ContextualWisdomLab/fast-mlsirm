#!/usr/bin/env python3
"""Fail closed unless a release SBOM has the required SPDX 2.3 document envelope."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from urllib.parse import urlsplit


def _reject_nonfinite(token: str) -> None:
    """Reject Python JSON decoder extensions that are not interoperable JSON."""
    raise ValueError(f"non-standard JSON constant in release SBOM: {token}")


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate object members instead of accepting decoder last-wins behavior."""
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON member in release SBOM: {key}")
        document[key] = value
    return document


def _nonblank_string(document: dict[str, object], field: str) -> str:
    """Return one required non-blank string field or fail with release-owned context."""
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"release SBOM {field} must be a non-blank string")
    return value


def _validate_namespace(namespace: str) -> None:
    """Require the SPDX document namespace to be an absolute URI without whitespace."""
    if any(character.isspace() for character in namespace):
        raise ValueError("release SBOM documentNamespace must be an absolute URI")
    parsed = urlsplit(namespace)
    if not parsed.scheme:
        raise ValueError("release SBOM documentNamespace must be an absolute URI")


def _validate_creation_info(value: object) -> None:
    """Validate the required SPDX creator and UTC creation-time evidence."""
    if not isinstance(value, dict):
        raise ValueError("release SBOM creationInfo must be an object")

    created = value.get("created")
    if not isinstance(created, str) or not created.endswith("Z"):
        raise ValueError("release SBOM creationInfo.created must be an ISO-8601 UTC timestamp")
    try:
        parsed_created = datetime.fromisoformat(created[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(
            "release SBOM creationInfo.created must be an ISO-8601 UTC timestamp"
        ) from exc
    if parsed_created.utcoffset() is None or parsed_created.utcoffset().total_seconds() != 0:
        raise ValueError("release SBOM creationInfo.created must be an ISO-8601 UTC timestamp")

    creators = value.get("creators")
    if not isinstance(creators, list) or not creators:
        raise ValueError("release SBOM creationInfo.creators must be a non-empty list")
    if any(not isinstance(creator, str) or not creator.strip() for creator in creators):
        raise ValueError("release SBOM creationInfo.creators must contain non-blank strings")


def validate_release_spdx(path: Path) -> None:
    """Validate strict JSON plus the required SPDX 2.3 document-creation fields."""
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(
            stream,
            parse_constant=_reject_nonfinite,
            object_pairs_hook=_reject_duplicates,
        )

    if not isinstance(document, dict):
        raise ValueError("release SBOM root must be a JSON object")

    expected_identity = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
    }
    for field, expected in expected_identity.items():
        actual = document.get(field)
        if actual != expected:
            raise ValueError(
                f"release SBOM {field} must be {expected!r}, got {actual!r}"
            )

    _nonblank_string(document, "name")
    namespace = _nonblank_string(document, "documentNamespace")
    _validate_namespace(namespace)
    _validate_creation_info(document.get("creationInfo"))


def main(argv: list[str] | None = None) -> int:
    """Validate one release SBOM path from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="SPDX 2.3 JSON SBOM to validate")
    args = parser.parse_args(argv)
    try:
        validate_release_spdx(args.path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
