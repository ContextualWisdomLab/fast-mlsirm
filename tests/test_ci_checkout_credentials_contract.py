"""Security contracts for repository CI checkout credential handling."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CHECKOUT = "uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"


def test_ci_checkouts_do_not_persist_credentials() -> None:
    """PR-controlled CI steps must not inherit a persisted checkout token."""

    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    checkout_tails = workflow.split(CHECKOUT)[1:]

    assert checkout_tails
    for checkout_tail in checkout_tails:
        checkout_block = checkout_tail.split("\n      - ", maxsplit=1)[0]
        assert "\n        with:\n" in checkout_block
        assert "\n          persist-credentials: false" in checkout_block
