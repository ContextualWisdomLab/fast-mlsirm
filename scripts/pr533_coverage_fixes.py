"""Close deterministic branch-coverage gaps in the PR 533 review patch."""

from __future__ import annotations

from pathlib import Path


IMPLEMENTATION = Path(
    "python/fast_mlsirm/scoring/enterprise_issue/explicit_values.py"
)
TESTS = Path("tests/test_scoring_enterprise_explicit_values.py")


def replace_once(path: Path, before: str, after: str, label: str) -> None:
    """Replace one exact block or stop before producing ambiguous output."""
    text = path.read_text(encoding="utf-8")
    count = text.count(before)
    if count != 1:
        raise SystemExit(f"expected one {label} block, found {count}")
    path.write_text(text.replace(before, after, 1), encoding="utf-8")


def main() -> None:
    """Remove one unreachable branch and cover all reachable payload branches."""
    replace_once(
        IMPLEMENTATION,
        '            normalized = _calendar_date(\n'
        '                payload["calendar_date"], "$.normalized_payload"\n'
        '            )\n'
        '            if normalized != payload["calendar_date"]:\n'
        '                self._payload_error("calendar_date must already be canonical")\n',
        '            _calendar_date(payload["calendar_date"], "$.normalized_payload")\n',
        "unreachable calendar canonicalization",
    )

    replace_once(
        TESTS,
        '        (\n'
        '            ExplicitValueKind.MONEY_AMOUNT,\n'
        '            {"currency_code": "usd", "decimal_amount": "1"},\n'
        '            "invalid_normalized_payload",\n'
        '        ),\n',
        '        (\n'
        '            ExplicitValueKind.MONEY_AMOUNT,\n'
        '            {"currency_code": "USD"},\n'
        '            "invalid_normalized_payload",\n'
        '        ),\n'
        '        (\n'
        '            ExplicitValueKind.MONEY_AMOUNT,\n'
        '            {"currency_code": 1, "decimal_amount": "1"},\n'
        '            "invalid_normalized_payload",\n'
        '        ),\n'
        '        (\n'
        '            ExplicitValueKind.MONEY_AMOUNT,\n'
        '            {"currency_code": "usd", "decimal_amount": "1"},\n'
        '            "invalid_normalized_payload",\n'
        '        ),\n',
        "money key and currency-type cases",
    )
    replace_once(
        TESTS,
        '        (\n'
        '            ExplicitValueKind.MONEY_AMOUNT,\n'
        '            {"currency_code": "USD", "decimal_amount": "-1"},\n'
        '            "invalid_decimal_amount",\n'
        '        ),\n',
        '        (\n'
        '            ExplicitValueKind.MONEY_AMOUNT,\n'
        '            {"currency_code": "USD", "decimal_amount": "-1"},\n'
        '            "invalid_decimal_amount",\n'
        '        ),\n'
        '        (\n'
        '            ExplicitValueKind.MONEY_AMOUNT,\n'
        '            {"currency_code": "USD", "decimal_amount": "NaN"},\n'
        '            "invalid_decimal_amount",\n'
        '        ),\n',
        "nonfinite decimal case",
    )
    replace_once(
        TESTS,
        '        (\n'
        '            ExplicitValueKind.FREQUENCY_COUNT,\n'
        '            {"frequency_count": True, "frequency_period": "month"},\n'
        '            "invalid_normalized_payload",\n'
        '        ),\n',
        '        (\n'
        '            ExplicitValueKind.FREQUENCY_COUNT,\n'
        '            {"frequency_count": 1},\n'
        '            "invalid_normalized_payload",\n'
        '        ),\n'
        '        (\n'
        '            ExplicitValueKind.FREQUENCY_COUNT,\n'
        '            {"frequency_count": True, "frequency_period": "month"},\n'
        '            "invalid_normalized_payload",\n'
        '        ),\n',
        "frequency key case",
    )


if __name__ == "__main__":
    main()
