"""Apply the exact accepted review fixes for pull request 533."""

from __future__ import annotations

from pathlib import Path


IMPLEMENTATION = Path(
    "python/fast_mlsirm/scoring/enterprise_issue/explicit_values.py"
)
TESTS = Path("tests/test_scoring_enterprise_explicit_values.py")
DOCS = Path("docs/enterprise_issue_evidence_contracts.md")
FRAGMENT = Path("docs/changelog.d/enterprise-explicit-value-parser.md")


def replace_once(path: Path, before: str, after: str, label: str) -> None:
    """Replace exactly one expected block or fail without partial ambiguity."""
    text = path.read_text(encoding="utf-8")
    count = text.count(before)
    if count != 1:
        raise SystemExit(f"expected one {label} block in {path}, found {count}")
    path.write_text(text.replace(before, after, 1), encoding="utf-8")


def patch_implementation() -> None:
    """Align parser typing, offsets, and successive-pair iteration."""
    replace_once(
        IMPLEMENTATION,
        "import hashlib\nimport json\nimport re\n"
        "from typing import Any, Protocol, runtime_checkable",
        "import hashlib\nfrom itertools import pairwise\nimport json\nimport re\n"
        "from typing import Any, NoReturn, Protocol, runtime_checkable",
        "standard-library imports",
    )
    replace_once(
        IMPLEMENTATION,
        "from .contracts import (\n"
        "    EnterpriseAssertionKind,\n"
        "    EnterpriseSourceRecord,\n"
        "    EvidenceSpanRecord,\n"
        ")",
        "from .contracts import (\n"
        "    MAX_ENTERPRISE_SOURCE_CHARACTERS,\n"
        "    EnterpriseAssertionKind,\n"
        "    EnterpriseSourceRecord,\n"
        "    EvidenceSpanRecord,\n"
        ")",
        "enterprise contract imports",
    )
    replace_once(
        IMPLEMENTATION,
        '        object.__setattr__(self, "start_offset", _offset(self.start_offset, "start_offset"))\n'
        '        object.__setattr__(self, "end_offset", _offset(self.end_offset, "end_offset"))\n'
        "        if self.end_offset <= self.start_offset:\n",
        '        object.__setattr__(self, "start_offset", _offset(self.start_offset, "start_offset"))\n'
        '        object.__setattr__(self, "end_offset", _offset(self.end_offset, "end_offset"))\n'
        "        if self.start_offset > MAX_ENTERPRISE_SOURCE_CHARACTERS:\n"
        "            raise assessment_error(\n"
        '                "invalid_start_offset",\n'
        '                "$.start_offset",\n'
        "                (\n"
        '                    "start_offset must be between 0 and "\n'
        '                    f"{MAX_ENTERPRISE_SOURCE_CHARACTERS}"\n'
        "                ),\n"
        "            )\n"
        "        if self.end_offset > MAX_ENTERPRISE_SOURCE_CHARACTERS:\n"
        "            raise assessment_error(\n"
        '                "invalid_end_offset",\n'
        '                "$.end_offset",\n'
        "                (\n"
        '                    "end_offset must be between 0 and "\n'
        '                    f"{MAX_ENTERPRISE_SOURCE_CHARACTERS}"\n'
        "                ),\n"
        "            )\n"
        "        if self.end_offset <= self.start_offset:\n",
        "record offset bounds",
    )
    replace_once(
        IMPLEMENTATION,
        "    def _payload_error(message: str) -> None:\n",
        "    def _payload_error(message: str) -> NoReturn:\n",
        "payload error annotation",
    )
    text = IMPLEMENTATION.read_text(encoding="utf-8")
    old_loop = "for previous, current in zip(ordered, ordered[1:]):"
    if text.count(old_loop) != 2:
        raise SystemExit("expected exactly two successive-pair loops")
    IMPLEMENTATION.write_text(
        text.replace(old_loop, "for previous, current in pairwise(ordered):"),
        encoding="utf-8",
    )


def patch_tests() -> None:
    """Make malformed payload and offset assertions contract-specific."""
    replace_once(
        TESTS,
        "def _normalized(records: tuple[ExplicitValueRecord, ...]) -> set[tuple[str, str]]:\n"
        '    """Return kind/payload pairs independent of source offsets."""\n'
        "    return {\n"
        '        (record.value_kind.value, repr(record.to_dict()["normalized_payload"]))\n'
        "        for record in records\n"
        "    }\n",
        "def _normalized(\n"
        "    records: tuple[ExplicitValueRecord, ...],\n"
        ") -> set[tuple[str, tuple[tuple[str, Any], ...]]]:\n"
        '    """Return kind/payload pairs independent of offsets and key order."""\n'
        "    return {\n"
        "        (\n"
        "            record.value_kind.value,\n"
        '            tuple(sorted(record.to_dict()["normalized_payload"].items())),\n'
        "        )\n"
        "        for record in records\n"
        "    }\n",
        "normalized payload helper",
    )

    text = TESTS.read_text(encoding="utf-8")
    start = text.index('@pytest.mark.parametrize(\n    ("kind", "payload"),')
    end = text.index(
        '\n\n@pytest.mark.parametrize(\n    ("changes", "error"),', start
    )
    replacement = '''@pytest.mark.parametrize(
    ("kind", "payload", "expected_code"),
    (
        (ExplicitValueKind.CALENDAR_DATE, [], "invalid_metadata"),
        (ExplicitValueKind.CALENDAR_DATE, {}, "invalid_normalized_payload"),
        (
            ExplicitValueKind.CALENDAR_DATE,
            {"calendar_date": 1},
            "invalid_calendar_date",
        ),
        (
            ExplicitValueKind.CALENDAR_DATE,
            {"calendar_date": "2026-02-30"},
            "invalid_calendar_date",
        ),
        (
            ExplicitValueKind.MONEY_AMOUNT,
            {"currency_code": "usd", "decimal_amount": "1"},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.MONEY_AMOUNT,
            {"currency_code": "USD", "decimal_amount": 1},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.MONEY_AMOUNT,
            {"currency_code": "USD", "decimal_amount": "abc"},
            "invalid_decimal_amount",
        ),
        (
            ExplicitValueKind.MONEY_AMOUNT,
            {"currency_code": "USD", "decimal_amount": "-1"},
            "invalid_decimal_amount",
        ),
        (
            ExplicitValueKind.MONEY_AMOUNT,
            {"currency_code": "USD", "decimal_amount": "1.00"},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.FREQUENCY_COUNT,
            {"frequency_count": True, "frequency_period": "month"},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.FREQUENCY_COUNT,
            {"frequency_count": 1_000_000_001, "frequency_period": "month"},
            "invalid_frequency_count",
        ),
        (
            ExplicitValueKind.FREQUENCY_COUNT,
            {"frequency_count": 1, "frequency_period": []},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.FREQUENCY_COUNT,
            {"frequency_count": 1, "frequency_period": "decade"},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.CUSTOMER_IDENTIFIER,
            {},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.CUSTOMER_IDENTIFIER,
            {"identifier_fingerprint": "bad"},
            "invalid_identifier_fingerprint",
        ),
    ),
)
def test_kind_specific_payloads_fail_closed(
    kind: ExplicitValueKind,
    payload: Any,
    expected_code: str,
) -> None:
    """Noncanonical payloads fail with their stable structured code."""
    with pytest.raises(AssessmentSpecError) as captured:
        _record(kind, payload=payload)
    assert captured.value.code == expected_code
'''
    TESTS.write_text(text[:start] + replacement + text[end:], encoding="utf-8")

    replace_once(
        TESTS,
        '        ({"start_offset": -1}, "invalid_start_offset"),\n'
        '        ({"end_offset": True}, "invalid_end_offset"),\n'
        '        ({"end_offset": 2}, "invalid_explicit_value_offsets"),',
        '        ({"start_offset": -1}, "invalid_start_offset"),\n'
        '        ({"start_offset": 100_000_001}, "invalid_start_offset"),\n'
        '        ({"end_offset": True}, "invalid_end_offset"),\n'
        '        ({"end_offset": 100_000_001}, "invalid_end_offset"),\n'
        '        ({"end_offset": 2}, "invalid_explicit_value_offsets"),',
        "offset validation cases",
    )

    replace_once(
        TESTS,
        "\ndef test_private_normalizers_cover_nonpublic_exception_boundaries() -> None:\n",
        '''
def test_payload_defensive_guards_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive payload invariants fail closed if upstream helpers regress."""
    with monkeypatch.context() as patch:
        patch.setattr(parser_module, "freeze_metadata", lambda value: value)
        patch.setattr(parser_module, "thaw_json_value", lambda value: value)
        with pytest.raises(AssessmentSpecError) as captured:
            _record(ExplicitValueKind.CALENDAR_DATE, payload=[])
        assert captured.value.code == "invalid_normalized_payload"

    with monkeypatch.context() as patch:
        patch.setattr(
            parser_module,
            "_calendar_date",
            lambda _value, _path: "2026-09-30",
        )
        with pytest.raises(AssessmentSpecError) as captured:
            _record(
                ExplicitValueKind.CALENDAR_DATE,
                payload={"calendar_date": "2026-09-29"},
            )
        assert captured.value.code == "invalid_normalized_payload"

    for kind in (
        ExplicitValueKind.MONEY_AMOUNT,
        ExplicitValueKind.FREQUENCY_COUNT,
    ):
        with pytest.raises(AssessmentSpecError) as captured:
            _record(kind, payload={})
        assert captured.value.code == "invalid_normalized_payload"


def test_private_normalizers_cover_nonpublic_exception_boundaries() -> None:
''',
        "defensive payload coverage tests",
    )


def patch_documentation() -> None:
    """Document the parser's whole-source rejection semantics and offset bound."""
    replace_once(
        DOCS,
        "Deadline matches supersede the calendar-date match embedded inside the same\n"
        "marked deadline. Any other accepted overlap fails closed rather than multiplying\n"
        "one occurrence into several evidence records. Output order and parser revision\n"
        "identity are deterministic and independent of caller currency-code ordering.\n",
        "Deadline matches supersede the calendar-date match embedded inside the same\n"
        "marked deadline. Any other accepted overlap fails closed rather than multiplying\n"
        "one occurrence into several evidence records. Validation is whole-source rather\n"
        "than per-span: any date-shaped candidate that is not a real Gregorian date, or\n"
        "any labeled customer/account identifier that is empty, malformed, or oversized,\n"
        "rejects the complete parse. Output order and parser revision identity are\n"
        "deterministic and independent of caller currency-code ordering.\n",
        "whole-source rejection documentation",
    )
    replace_once(
        FRAGMENT,
        "  Provider-owned records are reconstructed as fresh canonical instances, and\n"
        "  deterministic candidate producers stop at the configured limit plus one rather\n"
        "  than exhausting unexpectedly prolific iterators.\n",
        "  Provider-owned records are reconstructed as fresh canonical instances, manual\n"
        "  offsets share the enterprise source-character bound, and deterministic candidate\n"
        "  producers stop at the configured limit plus one rather than exhausting\n"
        "  unexpectedly prolific iterators.\n",
        "offset-bound changelog note",
    )


def main() -> None:
    """Apply all accepted findings atomically within one checked-out worktree."""
    patch_implementation()
    patch_tests()
    patch_documentation()


if __name__ == "__main__":
    main()
