"""Source-document contracts; no worker, server or numerical acceptance."""
from pathlib import Path
import re

DOCUMENT = Path(__file__).resolve().parents[1] / "docs/orchestration/remote-family-io-equivalence-2072.md"


def _section(text: str, heading: str) -> str:
    """Read a complete Markdown section, including wrapped paragraphs."""
    assert text.count(heading) == 1
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


def test_status_does_not_deny_present_valkey_adapters():
    """Do not describe an implemented adapter as absent in current status."""
    text = DOCUMENT.read_text(encoding="utf-8")
    status = " ".join(text.split("\n\n")[1].split())
    assert "does not implement valkey transport" not in status.lower()
    assert "implement valkey transport" not in status.lower()
    assert "Does **not** close #2001" in status
    assert "A4 completion" in status


def test_identity_layer_count_matches_the_complete_table():
    """The declared layer count must match all four named identity rows."""
    contract = _section(DOCUMENT.read_text(encoding="utf-8"), "## Contract")
    rows = re.findall(r"^\| (Envelope|Payload cohort|Cohort manifest|Output) \|", contract, re.MULTILINE)
    assert rows == ["Envelope", "Payload cohort", "Cohort manifest", "Output"]
    assert "carries four identity layers" in contract
    assert "carries three identity layers" not in contract


def test_nonclaims_describe_injected_adapters_without_server_certification():
    """Separate adapter presence from real-server and distributed acceptance."""
    nonclaims = " ".join(_section(DOCUMENT.read_text(encoding="utf-8"), "## Explicit non-claims").split())
    assert "No Valkey/Redis Streams transport" not in nonclaims
    assert "`ValkeyStreamsOutcomeStore`" in nonclaims
    assert "`ValkeyStreamsBackend`" in nonclaims
    assert "injected-client contracts" in nonclaims
    assert "recording-client tests do not certify real-server operation" in nonclaims
    assert "cross-host always-on evidence remains unverified" in nonclaims
    assert "Does not close #2001 or #2071" in nonclaims


def test_existing_device_and_failure_nonclaims_are_preserved():
    """Keep declaration-only provenance and rejected-versus-FAILED distinction."""
    text = DOCUMENT.read_text(encoding="utf-8")
    assert 'identity_verification="declared_unverified"' in text
    assert "not evidence of actual CPU execution" in text
    assert "actual device and float-path acceptance remain outstanding" in text
    assert "Only documented unobserved-category or empty-item `ValueError`" in text
    assert "other exceptions propagate and the worker serializes them as `FAILED`" in text
