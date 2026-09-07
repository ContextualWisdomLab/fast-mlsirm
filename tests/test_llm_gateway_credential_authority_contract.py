"""Contracts for the current LLM orchestration and credential authority."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT_POLICY_DOCS = (
    ROOT / "docs" / "TRD.md",
    ROOT / "ARCHITECTURE.md",
    ROOT / "docs" / "GOVERNANCE_INDEX.md",
    ROOT / "docs" / "multilevel_multiple_membership_longitudinal_rfc.md",
)
OLD_ADR = ROOT / "docs" / "adr" / "0010-llm-orchestration-and-credentials.md"
NEW_ADR = ROOT / "docs" / "adr" / "0101-contextual-orchestrator-gateway-credential-boundary.md"
ADR_INDEX = ROOT / "docs" / "adr" / "README.md"


def test_current_policy_does_not_assign_provider_credentials_to_fast_mlsirm() -> None:
    """Provider credentials belong to contextual-orchestrator, not this consumer."""
    for path in CURRENT_POLICY_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "NVIDIA_NIM_API_KEY" not in text, path
        assert "OPENROUTER_API_KEY" not in text, path
        assert "OPENAI_API_KEY" not in text, path
        assert "BYTEZ_API_KEY" not in text, path


def test_trd_requires_released_gateway_contract_and_free_route() -> None:
    """Model-backed Actions use only the released gateway boundary and free route."""
    text = (ROOT / "docs" / "TRD.md").read_text(encoding="utf-8")
    assert "released" in text[text.index("### 4.13 LLM/provider tests and automation") :]
    assert "`contextual-orchestrator`" in text
    assert "`orchestrator/free`" in text
    assert "gateway token" in text.lower()
    assert "provider/model/group/paid fallback" in text


def test_legacy_credential_adr_is_explicitly_superseded() -> None:
    """The historical direct-provider credential decision cannot remain Accepted."""
    old = OLD_ADR.read_text(encoding="utf-8")
    assert "Status: **Superseded**" in old
    assert "ADR-0101" in old
    assert NEW_ADR.is_file()

    index = ADR_INDEX.read_text(encoding="utf-8")
    assert "| [0010](0010-llm-orchestration-and-credentials.md) | Superseded |" in index
    assert "| [0101](0101-contextual-orchestrator-gateway-credential-boundary.md) | Accepted |" in index
