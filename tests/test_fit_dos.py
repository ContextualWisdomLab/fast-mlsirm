import pytest
import numpy as np

from fast_mlsirm.fit import _line_search

def test_line_search_unbound():
    def bad_obj(x):
        return np.inf, np.zeros_like(x), -np.inf

    x = np.array([0.0])
    direction = np.array([1.0])

    accepted, candidate, next_obj, next_grad, next_loglik = _line_search(
        objective=bad_obj,
        x=x,
        direction=direction,
        obj=0.0,
        slope=-1.0,
    )
    assert not accepted
    assert np.isinf(next_obj)

def test_line_search_unbound():
    def bad_obj(x):
        return np.inf, np.zeros_like(x), -np.inf

    x = np.array([0.0])
    direction = np.array([1.0])
    from fast_mlsirm.fit import _line_search

    accepted, candidate, next_obj, next_grad, next_loglik = _line_search(
        objective=bad_obj,
        x=x,
        direction=direction,
        obj=0.0,
        slope=-1.0,
    )
    assert not accepted
    assert np.isinf(next_obj)
