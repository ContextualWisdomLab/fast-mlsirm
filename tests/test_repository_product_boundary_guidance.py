"""Regression contracts for the fast-mlsirm / Psychometrics Commons boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _guidance(name: str) -> str:
    """Return one authoritative repository-guidance document."""
    return (ROOT / name).read_text(encoding="utf-8")


def _normalized_guidance(name: str) -> str:
    """Return guidance with insignificant Markdown wrapping collapsed."""
    return " ".join(_guidance(name).split())


def test_agents_declares_psychometrics_commons_as_downstream_product() -> None:
    """AGENTS must not direct agents to rebuild the hosted product in this repo."""
    agents = _guidance("AGENTS.md")
    assert "ContextualWisdomLab/psychometrics-commons" in agents
    assert "downstream consumer" in agents


def test_guidance_excludes_legacy_r_packages_from_product_dependencies() -> None:
    """Current guidance must not advertise legacy R packages as product internals."""
    agents = _guidance("AGENTS.md")
    claude = _guidance("CLAUDE.md")
    assert "`kaefa`, `aFIPC`, and `nonnest2`" in agents
    assert "runtime, build, CI-oracle, or release dependencies" in agents
    assert "aFIPC fixed-item calibration + kaefa item-fit" not in claude
    assert "incorporates aFIPC Fixed-Item Parameter Calibration and kaefa" not in agents


def test_claude_summary_uses_the_same_repository_boundary() -> None:
    """Claude guidance must preserve the canonical upstream/downstream direction."""
    claude = _normalized_guidance("CLAUDE.md")
    assert "ContextualWisdomLab/psychometrics-commons" in claude
    assert "hosted product" in claude
    assert "does not depend on Psychometrics Commons" in claude


def test_guidance_forbids_recreating_hosted_runtime_in_fast_mlsirm() -> None:
    """Both agent guides must keep the hosted runtime out of this repository."""
    agents = _normalized_guidance("AGENTS.md")
    claude = _normalized_guidance("CLAUDE.md")
    runtime_path = "`services/assessment_runtime`"

    assert runtime_path in agents
    assert "must not be recreated" in agents
    assert runtime_path in claude
    assert "must not be recreated" in claude
