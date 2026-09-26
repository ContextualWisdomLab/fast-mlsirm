"""Same-input checks for Rust-owned Monte Carlo rank diagnostics."""

import numpy as np
import pytest

from fast_mlsirm import (
    binomial_interval_coverage,
    binomial_quantile,
    linear_percentile,
    mc_rank_interval,
)


def test_rank_interval_and_percentile_match_order_statistics():
    draws = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    result = mc_rank_interval(draws, 0.5, 0.8)
    assert (result["count_low"], result["count_high"]) == (1, 4)
    assert (result["lo"], result["hi"]) == (1.0, 5.0)
    assert result["attained_coverage"] == pytest.approx(0.9375)
    assert linear_percentile(draws, 0.125) == pytest.approx(
        np.quantile(draws, 0.125, method="linear")
    )
    assert binomial_quantile(10, 0.5, 0.025) == 2
    assert binomial_interval_coverage(10, 0.5, 2, 8) == pytest.approx(0.978515625)


def test_mc_inputs_fail_closed():
    with pytest.raises(ValueError, match="finite"):
        mc_rank_interval(np.array([1.0, np.nan]), 0.5, 0.95)
    with pytest.raises(ValueError, match="confidence"):
        mc_rank_interval(np.array([1.0, 2.0]), 0.5, 1.0)
    with pytest.raises(ValueError, match="increase B"):
        mc_rank_interval(np.array([1.0, 2.0]), 0.5, 0.95)
    with pytest.raises(ValueError, match="probability"):
        binomial_quantile(10, 0.5, np.nan)
