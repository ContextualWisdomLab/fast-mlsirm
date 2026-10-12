"""Keep the isolated runner label visible to normal actionlint discovery.

A source label declaration does not attest registered capacity or isolation.
"""
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def test_actionlint_declares_only_the_required_custom_runner_label() -> None:
    """Declare the custom label without disabling any workflow lint checks."""
    config = _ROOT / ".github" / "actionlint.yaml"
    assert config.is_file(), "normal actionlint configuration is missing"
    assert config.read_text(encoding="utf-8") == (
        "# Custom label syntax only; runner capacity and isolation need operator evidence.\n"
        "self-hosted-runner:\n"
        "  labels:\n"
        "    - cwlab-ci-isolated\n"
    )
