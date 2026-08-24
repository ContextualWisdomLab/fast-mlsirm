from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "enterprise_due_diligence_gate.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("enterprise_due_diligence_gate", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load_module()


def test_build_gate_manifest_is_currency_explicit_and_not_a_valuation_claim() -> None:
    manifest = GATE.build_gate_manifest(source_commit="abc123")

    assert manifest == {
        "currency_code": "KRW",
        "gate_name": "enterprise_due_diligence_gate",
        "legacy_gate_aliases": [
            "20b",
            "20b_product",
            "20b_product_readiness",
            "require_20b_product",
        ],
        "scenario_amount": 2_000_000_000,
        "scenario_name": "krw_2000000000_procurement_scenario",
        "schema_version": "1.0.0",
        "source_commit": "abc123",
        "valuation_claim": False,
    }


def test_normalize_gate_name_accepts_canonical_hyphenated_form() -> None:
    assert GATE.normalize_gate_name(" enterprise-due-diligence-gate ") == GATE.CANONICAL_GATE_NAME


@pytest.mark.parametrize(
    "legacy_alias",
    ["20B", "20b-product", "20b_product_readiness", "require_20b_product"],
)
def test_normalize_gate_name_maps_legacy_aliases_with_warning(legacy_alias: str) -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert GATE.normalize_gate_name(legacy_alias) == GATE.CANONICAL_GATE_NAME


def test_normalize_gate_name_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unsupported due-diligence gate name"):
        GATE.normalize_gate_name("usd_20b_readiness")


@pytest.mark.parametrize("currency_code", ["krw", " usd "])
def test_validate_currency_code_normalizes_ascii_letters(currency_code: str) -> None:
    assert GATE.validate_currency_code(currency_code) == currency_code.strip().upper()


@pytest.mark.parametrize("currency_code", ["KR", "KRWW", "K1W", "₩RW"])
def test_validate_currency_code_rejects_invalid_values(currency_code: str) -> None:
    with pytest.raises(ValueError, match="three ASCII letters"):
        GATE.validate_currency_code(currency_code)


@pytest.mark.parametrize("scenario_amount", [True, False, 0, -1, 2.5])
def test_validate_scenario_amount_rejects_non_positive_integer_values(
    scenario_amount: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GATE.validate_scenario_amount(scenario_amount)


def test_validate_scenario_amount_accepts_positive_integer() -> None:
    assert GATE.validate_scenario_amount(1) == 1


@pytest.mark.parametrize(
    "source_commit, message",
    [
        ("   ", "must not be empty"),
        ("a" * 129, "must not exceed 128"),
        ("abc\n123", "control characters"),
        ("abc\x7f123", "control characters"),
    ],
)
def test_validate_source_commit_rejects_unsafe_values(source_commit: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GATE.validate_source_commit(source_commit)


def test_validate_source_commit_trims_printable_identifier() -> None:
    assert GATE.validate_source_commit("  abc123  ") == "abc123"


@pytest.mark.parametrize("valuation_claim", [True, "false"])
def test_build_gate_manifest_rejects_valuation_claims(valuation_claim: object) -> None:
    with pytest.raises(ValueError, match="valuation_claim|valuation claim"):
        GATE.build_gate_manifest(
            source_commit="abc123",
            valuation_claim=valuation_claim,
        )


def test_write_gate_manifest_is_deterministic(tmp_path: Path) -> None:
    manifest = GATE.build_gate_manifest(source_commit="abc123", currency_code="usd", scenario_amount=25)
    output_path = tmp_path / "nested" / "gate.json"

    GATE.write_gate_manifest(manifest, output_path)

    assert output_path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest
    assert output_path.read_text(encoding="utf-8").splitlines()[1].strip().startswith('"currency_code"')


def test_main_writes_canonical_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_path = tmp_path / "gate.json"

    exit_code = GATE.main(
        [
            "--source-commit",
            "abc123",
            "--currency-code",
            "usd",
            "--scenario-amount",
            "25",
            "--require-enterprise-due-diligence",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["gate_name"] == GATE.CANONICAL_GATE_NAME
    stdout = json.loads(capsys.readouterr().out)
    assert stdout == {
        "gate_name": GATE.CANONICAL_GATE_NAME,
        "out": str(output_path),
        "status": "ok",
        "valuation_claim": False,
    }


def test_main_supports_deprecated_flag_during_migration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "legacy.json"

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        exit_code = GATE.main(
            [
                "--source-commit",
                "abc123",
                "--require-20b-product",
                "--out",
                str(output_path),
            ]
        )

    assert exit_code == 0
    assert len(captured) == 2
    assert all(item.category is DeprecationWarning for item in captured)
    assert json.loads(output_path.read_text(encoding="utf-8"))["valuation_claim"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_main_returns_stable_failure_payload_for_invalid_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "invalid.json"

    exit_code = GATE.main(
        [
            "--source-commit",
            "abc123",
            "--currency-code",
            "invalid",
            "--out",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": "currency_code must be exactly three ASCII letters",
        "status": "failed",
    }
    assert not output_path.exists()
