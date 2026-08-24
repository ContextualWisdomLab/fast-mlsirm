#!/usr/bin/env python
"""Build a currency-explicit enterprise due-diligence gate manifest.

The gate is evidence-oriented and amount-neutral by name. A monetary scenario
may be attached for procurement review, but the resulting manifest always
states that it is not a valuation claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence


CANONICAL_GATE_NAME = "enterprise_due_diligence_gate"
DEFAULT_CURRENCY_CODE = "KRW"
DEFAULT_SCENARIO_AMOUNT = 2_000_000_000
SCHEMA_VERSION = "1.0.0"
LEGACY_GATE_ALIASES = frozenset(
    {
        "20b",
        "20b_product",
        "20b_product_readiness",
        "require_20b_product",
    }
)


def _contains_control_character(value: str) -> bool:
    """Return whether *value* contains an ASCII control character."""

    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def normalize_gate_name(value: str) -> str:
    """Return the canonical gate name and warn for a supported legacy alias."""

    normalized = value.strip().lower().replace("-", "_")
    if normalized == CANONICAL_GATE_NAME:
        return CANONICAL_GATE_NAME
    if normalized in LEGACY_GATE_ALIASES:
        warnings.warn(
            (
                f"{value!r} is deprecated; use {CANONICAL_GATE_NAME!r}. "
                "The legacy alias describes neither a currency nor a valuation contract."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return CANONICAL_GATE_NAME
    raise ValueError(f"unsupported due-diligence gate name: {value!r}")


def validate_currency_code(value: str) -> str:
    """Validate and normalize an ISO-4217-style three-letter currency code."""

    normalized = value.strip().upper()
    if len(normalized) != 3 or not normalized.isascii() or not normalized.isalpha():
        raise ValueError("currency_code must be exactly three ASCII letters")
    return normalized


def validate_scenario_amount(value: int) -> int:
    """Validate a positive procurement scenario amount without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("scenario_amount must be a positive integer")
    return value


def validate_source_commit(value: str) -> str:
    """Validate a bounded, printable source-commit identifier."""

    normalized = value.strip()
    if not normalized:
        raise ValueError("source_commit must not be empty")
    if len(normalized) > 128:
        raise ValueError("source_commit must not exceed 128 characters")
    if _contains_control_character(normalized):
        raise ValueError("source_commit must not contain control characters")
    return normalized


def build_gate_manifest(
    *,
    source_commit: str,
    gate_name: str = CANONICAL_GATE_NAME,
    currency_code: str = DEFAULT_CURRENCY_CODE,
    scenario_amount: int = DEFAULT_SCENARIO_AMOUNT,
    valuation_claim: bool = False,
) -> dict[str, Any]:
    """Build the deterministic public contract for an enterprise evidence gate."""

    if not isinstance(valuation_claim, bool):
        raise ValueError("valuation_claim must be a boolean")
    if valuation_claim:
        raise ValueError("enterprise due-diligence evidence must not be a valuation claim")

    canonical_gate_name = normalize_gate_name(gate_name)
    normalized_currency = validate_currency_code(currency_code)
    normalized_amount = validate_scenario_amount(scenario_amount)
    normalized_commit = validate_source_commit(source_commit)
    scenario_name = f"{normalized_currency.lower()}_{normalized_amount}_procurement_scenario"

    return {
        "currency_code": normalized_currency,
        "gate_name": canonical_gate_name,
        "legacy_gate_aliases": sorted(LEGACY_GATE_ALIASES),
        "scenario_amount": normalized_amount,
        "scenario_name": scenario_name,
        "schema_version": SCHEMA_VERSION,
        "source_commit": normalized_commit,
        "valuation_claim": False,
    }


def write_gate_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Write a gate manifest as deterministic UTF-8 JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the gate manifest utility."""

    parser = argparse.ArgumentParser(
        description="Build a currency-explicit enterprise due-diligence gate manifest."
    )
    parser.add_argument(
        "--gate-name",
        default=CANONICAL_GATE_NAME,
        help="Canonical gate name or a supported legacy alias during deprecation.",
    )
    parser.add_argument(
        "--currency-code",
        default=DEFAULT_CURRENCY_CODE,
        help="Three-letter procurement-scenario currency code.",
    )
    parser.add_argument(
        "--scenario-amount",
        type=int,
        default=DEFAULT_SCENARIO_AMOUNT,
        help="Positive integer procurement-scenario amount.",
    )
    parser.add_argument(
        "--source-commit",
        required=True,
        help="Commit or immutable source identifier represented by the evidence.",
    )
    parser.add_argument(
        "--out",
        default="enterprise_due_diligence_gate.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--require-enterprise-due-diligence",
        action="store_true",
        help="Require the canonical evidence gate. Retained for orchestration symmetry.",
    )
    parser.add_argument(
        "--require-20b-product",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit status."""

    parser = build_parser()
    args = parser.parse_args(argv)
    selected_gate_name = args.gate_name
    if args.require_20b_product:
        warnings.warn(
            (
                "--require-20b-product is deprecated; use "
                "--require-enterprise-due-diligence."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        selected_gate_name = "require_20b_product"
    try:
        manifest = build_gate_manifest(
            source_commit=args.source_commit,
            gate_name=selected_gate_name,
            currency_code=args.currency_code,
            scenario_amount=args.scenario_amount,
            valuation_claim=False,
        )
        output_path = Path(args.out)
        write_gate_manifest(manifest, output_path)
    except ValueError as exc:
        print(json.dumps({"error": str(exc), "status": "failed"}, sort_keys=True), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "gate_name": manifest["gate_name"],
                "out": str(output_path),
                "status": "ok",
                "valuation_claim": manifest["valuation_claim"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
