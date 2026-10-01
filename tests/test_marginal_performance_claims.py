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
