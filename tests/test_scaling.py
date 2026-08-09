import numpy as np
import pytest

from fast_mlsirm.scaling import lsr_rankings


def test_lsr_rankings_basic():
    # Test with a simple set of rankings (strongly connected to avoid ValueError)
    # Items: 0, 1, 2
    # Rankings:
    # - [0, 1] (0 beats 1)
    # - [1, 2] (1 beats 2)
    # - [2, 0] (2 beats 0)
    # - [0, 2] (0 beats 2) - breaking symmetry so 0 is uniquely ranked highest

    rankings = [[0, 1], [1, 2], [2, 0], [0, 2]]
    n = 3

    res = lsr_rankings(rankings, n)

    # 0 has 2 wins, 1 has 1 win, 2 has 1 win
    assert res.params[0] > res.params[1]

    # Should be centered
    assert np.isclose(np.mean(res.params), 0.0)
    assert res.iterations == 1


def test_lsr_rankings_partial():
    # Test with partial rankings (more than 2 items)
    rankings = [[0, 1, 2], [2, 1, 0]]
    n = 3

    res = lsr_rankings(rankings, n)

    # Due to symmetry, params should be equal for 0 and 2
    assert np.isclose(res.params[0], res.params[2])
    # Total sum should be ~0 due to centering
    assert np.isclose(np.mean(res.params), 0.0)


def test_lsr_rankings_alpha():
    # Test with alpha parameter
    rankings = [[0, 1], [1, 2]]
    n = 3

    # With alpha=0, [0,1], [1,2] is not strongly connected
    # Alpha > 0 ensures strong connectivity.
    res_alpha = lsr_rankings(rankings, n, alpha=0.1)

    assert len(res_alpha.params) == n
    assert res_alpha.params[0] > res_alpha.params[1]
    assert res_alpha.params[1] > res_alpha.params[2]


def test_lsr_rankings_errors():
    # Test invalid inputs
    n = 3

    # Empty rankings list
    with pytest.raises(ValueError, match="at least one ranking is required"):
        lsr_rankings([], n)

    # Ranking with only 1 item
    with pytest.raises(ValueError, match="ranking 0 has fewer than 2 items"):
        lsr_rankings([[0]], n)

    # Disconnected graph without alpha
    with pytest.raises(
        ValueError, match="stationary distribution could not be computed"
    ):
        lsr_rankings([[0, 1], [1, 2]], n, alpha=0.0)
