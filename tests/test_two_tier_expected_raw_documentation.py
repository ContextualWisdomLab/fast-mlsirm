"""Source-only scoring documentation contracts; no native computation.

Cai (2015, pp. 542–543, Eqs. 14–17) describes conditional likelihoods
and dimension reduction. This source contract checks the verified locator
and separates a fixture-specific numerical comparison from a universal bound.

Reference:
Cai, L. (2015). Lord–Wingersky algorithm version 2.0 for hierarchical item
factor models with applications in test scoring, scale alignment, and model
fit testing. Psychometrika, 80(2), 535–559.
https://doi.org/10.1007/s11336-014-9411-3
"""
from pathlib import Path
import ast


SOURCE = Path(__file__).resolve().parents[1] / 'python' / 'fast_mlsirm' / 'two_tier_grm.py'


def scoring_docstring():
    """Read the actual function docstring without importing the package."""
    tree = ast.parse(SOURCE.read_bytes())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == 'expected_raw_two_tier_grm')
    docstring = ast.get_docstring(function)
    assert docstring is not None
    return ' '.join(docstring.split())


def test_scoring_citation_uses_verified_equation_pages():
    """Equation 14 is on p. 542; equations 15–17 are on p. 543."""
    doc = scoring_docstring()
    assert '(Cai, 2015, Eqs. 14-17, pp. 542-543)' in doc
    assert '(Cai, 2015, Eqs. 14-17, pp. 543-544)' not in doc


def test_numerical_agreement_is_scoped_to_retained_fixtures():
    """Retained comparisons do not establish a universal floating-point bound."""
    doc = scoring_docstring()
    assert 'within 1e-12 on the retained comparison fixtures' in doc
    assert 'not a universal floating-point error bound' in doc
    assert 'identical to the Lord-Wingersky mean to 1e-12' not in doc


def test_plugin_estimand_and_caller_quadrature_are_retained():
    """Documentation must retain plug-in primaries and caller-owned integration."""
    doc = scoring_docstring()
    assert 'Primary coordinates are fixed at ``fit.theta_p_eap``' in doc
    assert '``q_specific`` is REQUIRED (no default' in doc
    assert 'does not reintegrate ``Phi``' in doc


def test_changelog_scopes_numerical_agreement_to_fixtures():
    """The user-facing changelog must not make a broader claim than the API."""
    changelog = SOURCE.parents[2] / 'docs' / 'changelog.d' / 'two-tier-expected-raw-gpu.md'
    text = ' '.join(changelog.read_text().split())
    assert 'within 1e-12 on the retained comparison fixtures' in text
    assert 'not a universal floating-point error bound' in text
    assert 'The two agree to 1e-12.' not in text


def test_rust_expected_raw_docs_state_paper_scope_and_full_reference():
    """Rust API docs must cite the verified two-tier derivation, not just an equation number."""
    rust = SOURCE.parents[2] / 'crates' / 'mlsirm-core' / 'src' / 'two_tier_recursion.rs'
    text = rust.read_text()
    start = text.index('/// Expected raw total at plug-in primary coordinates')
    end = text.index('pub fn two_tier_expected_raw(', start)
    docs = text[start:end]
    assert 'Cai (2015, pp. 542–543, Eqs. 14–17)' in docs
    assert 'not a universal floating-point error bound' in docs
    assert 'Cai, L. (2015).' in docs
    assert 'Psychometrika, 80(2), 535–559.' in docs
    assert 'https://doi.org/10.1007/s11336-014-9411-3' in docs
