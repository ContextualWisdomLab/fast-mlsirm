"""Contracts for evidence-bounded marginal-estimator performance claims."""

import re
from pathlib import Path


MARGINAL_SOURCE = (
    Path(__file__).parents[1] / "python" / "fast_mlsirm" / "estimators" / "marginal.py"
)


def test_marginal_source_avoids_unmeasured_numeric_speedup_claims() -> None:
    """Require a benchmark artifact before publishing fixed speedup factors."""
    source = MARGINAL_SOURCE.read_text(encoding="utf-8")

    assert re.search(r"~\d+(?:\.\d+)?x speedup", source, re.IGNORECASE) is None

def test_marginal_distance_keeps_allocation_reduced_reduction() -> None:
    """Bind the production distance path to the allocation-reduced reduction."""
    source = MARGINAL_SOURCE.read_text(encoding="utf-8")

    assert 'np.einsum("ij,ij->i", diff, diff)' in source
    assert "np.sum(diff * diff, axis=1)" not in source
