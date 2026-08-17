"""Cross-component IRT response shape contract."""

from __future__ import annotations

import re

import numpy as np
import pytest
from fast_mlsirm.irt_contract import validate_irt_response_matrix


def test_dichotomous_contract_requires_multiple_items() -> None:
    matrix = validate_irt_response_matrix(
        [[0, 1], [1, np.nan]],
        "dichotomous",
    )
    assert matrix.shape == (2, 2)
    with pytest.raises(ValueError, match="at least two item columns"):
        validate_irt_response_matrix([[1]], "dichotomous")


def test_dichotomous_contract_rejects_non_binary_observations() -> None:
    with pytest.raises(ValueError, match="0, 1"):
        validate_irt_response_matrix([[0, 2]], "dichotomous")
    with pytest.raises(ValueError, match="integer"):
        validate_irt_response_matrix([[0.5, 1]], "dichotomous")


def test_polytomous_contract_requires_explicit_categories_and_multiple_items() -> None:
    matrix = validate_irt_response_matrix(
        [[0, 2], [1, np.nan]],
        "polytomous",
        n_categories=3,
    )
    assert matrix.shape == (2, 2)
    with pytest.raises(ValueError, match="n_categories"):
        validate_irt_response_matrix([[0, 1]], "polytomous")
    with pytest.raises(ValueError, match="at least two item columns"):
        validate_irt_response_matrix([[1]], "polytomous", n_categories=3)


def test_polytomous_contract_rejects_invalid_categories_and_shape() -> None:
    with pytest.raises(ValueError, match=re.escape("0..2")):
        validate_irt_response_matrix([[0, 3]], "polytomous", n_categories=3)
    with pytest.raises(ValueError, match="n_categories is only valid"):
        validate_irt_response_matrix([[0, 1]], "dichotomous", n_categories=2)
    with pytest.raises(ValueError, match=re.escape("2..")):
        validate_irt_response_matrix([[0, 1]], "polytomous", n_categories=1)
    with pytest.raises(ValueError, match="2-D"):
        validate_irt_response_matrix([0, 1], "polytomous", n_categories=3)
    with pytest.raises(ValueError, match="finite"):
        validate_irt_response_matrix([[0, np.inf]], "polytomous", n_categories=3)


if __name__ == "__main__":
    test_dichotomous_contract_requires_multiple_items()
    test_dichotomous_contract_rejects_non_binary_observations()
    test_polytomous_contract_requires_explicit_categories_and_multiple_items()
    test_polytomous_contract_rejects_invalid_categories_and_shape()
    print("ok")
