import numpy as np
import pytest
from fast_mlsirm.config import FitConfig
from fast_mlsirm.fit import fit, _adam, _lbfgs, _unpack
from fast_mlsirm.types import MLSIRMParams
from fast_mlsirm.objective import neg_loglik_and_grad

def test_adam_nan():
    def obj(x):
        return np.nan, np.zeros_like(x), np.nan
    x, _, _, status = _adam(np.zeros(1), obj, FitConfig(), 10)
    assert status == "nan_or_inf"

def test_lbfgs_line_search_failed():
    def obj(x):
        return 10.0, np.ones_like(x), -10.0
    x, _, _, status = _lbfgs(np.zeros(1), obj, FitConfig(), 10)
    assert status == "line_search_failed"
