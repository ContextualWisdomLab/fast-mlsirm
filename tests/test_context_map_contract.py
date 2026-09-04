"""Architecture fitness tests for the fast-mlsirm bounded-context map."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_MAP = ROOT / "docs" / "context-map.md"
DOCS_INDEX = ROOT / "docs" / "README.md"


def _section(document: str, heading: str, next_heading: str) -> str:
    """Return one exact Markdown section so ownership claims cannot drift."""
    marker = f"{heading}\n"
    assert document.count(marker) == 1
    _, _, remainder = document.partition(marker)
    end_marker = f"\n{next_heading}\n"
    assert remainder.count(end_marker) == 1
    section, _, _ = remainder.partition(end_marker)
    return section


def test_context_map_declares_internal_bounded_contexts() -> None:
    """Keep numerical ownership and public binding responsibilities explicit."""
    context_map = CONTEXT_MAP.read_text(encoding="utf-8")
    purpose = _section(context_map, "## Purpose", "## Internal bounded contexts")
    internal = _section(
        context_map,
        "## Internal bounded contexts",
        "## Foreign bounded contexts and relationship contracts",
    )

    expected_rows = (
        "| `Model Specification` | Typed model/formulation identity, parameter blocks, supported/research-candidate/unsupported status, identification and recovery requirements. | Likelihood arithmetic, provider routing, event ontology. |",
        "| `Estimation` | Rust likelihood/gradient/integration/optimization kernels and estimator-local convergence contracts for admitted specifications. | Hosted workflow/session state, TEPP temporal semantics. |",
        "| `Scoring` | Rust-owned score/information/EAP or other explicitly supported scoring kernels over released model contracts. | Participant lifecycle, UI interpretation, causal/business outcome claims. |",
        "| `Diagnostics` | Fit, dependence, DIF/invariance/fairness and other measurement diagnostics whose estimands are explicitly defined. | EA facts, temporal event meaning, generic product analytics. |",
        "| `Simulation-Recovery` | Known-truth simulation, deterministic seed manifests, bias/MAE/RMSE/coverage, Monte Carlo uncertainty, identifiability/recovery evidence and reproducibility gates. | Production respondent/session records or architecture inventory. |",
        "| `Compute Backend` | Rust CPU parallelism, GPU kernels where promoted, CPU/GPU parity, deterministic execution/resource contracts. | Model semantics or provider/model routing. |",
        "| `Public Binding` | Stable Rust/PyO3/Python API contracts, validation, immutable marshalling, installed-package behavior, reporting and compatibility/version surfaces. | Independent production statistical arithmetic or hidden fallback estimators; governed explicit reference/parity calculations remain permitted. |",
    )
    for row in expected_rows:
        assert row in internal

    assert "Production psychometric arithmetic is Rust-owned" in purpose
    assert (
        "Python is limited to validation, immutable marshalling, reporting, and binding ergonomics"
        in purpose
    )


def test_context_map_declares_foreign_owner_relationships() -> None:
    """Prevent temporal, product, and architecture authorities from leaking inward."""
    context_map = CONTEXT_MAP.read_text(encoding="utf-8")
    tepp = _section(
        context_map,
        "### TEPP — temporal/event authority",
        "### psychometrics-commons — hosted product",
    )
    hosted_product = _section(
        context_map,
        "### psychometrics-commons — hosted product",
        "### context-graph-contracts — Context Fabric Shared Kernel",
    )
    context_graph = _section(
        context_map,
        "### context-graph-contracts — Context Fabric Shared Kernel",
        "### enterprise-architecture-core — EA Decision Plane",
    )
    ea = _section(
        context_map,
        "### enterprise-architecture-core — EA Decision Plane",
        "## Integration invariants",
    )

    assert "TEPP owns temporal/event composition and semantics" in tepp
    assert "cross-service SQL is prohibited" in tepp
    assert "psychometrics-commons is a downstream hosted-product consumer" in hosted_product
    assert "context-graph-contracts is the contract-only Shared Kernel" in context_graph
    assert "immutable released context-graph-contracts contract" in context_graph
    assert "enterprise-architecture-core is the authoritative EA Decision Plane" in ea
    assert (
        "Estimator values, latent scores, DIF/fit diagnostics, and scientific-validity evidence are not EA-authoritative facts"
        in ea
    )


def test_active_context_map_is_linked_from_canonical_documentation_index() -> None:
    """Keep the active Context Map discoverable from the documentation authority."""
    docs_index = DOCS_INDEX.read_text(encoding="utf-8")
    canonical = _section(
        docs_index,
        "## Canonical architecture package",
        "## Authority and status",
    )

    assert (
        "| [`context-map.md`](context-map.md) | DDD bounded-context ownership, dependency direction, and foreign-context relationship contracts |"
        in canonical
    )
