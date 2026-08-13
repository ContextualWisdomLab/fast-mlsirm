"""Cross-component IRT response shape contract."""

from __future__ import annotations

import re

import numpy as np
import pytest

from fast_mlsirm.config import FitConfig
from fast_mlsirm.fit import fit as fit_binary
from fast_mlsirm.grm import fit_grm
from fast_mlsirm.polytomous import fit_polytomous
from fast_mlsirm.rsm import fit_rsm
from fast_mlsirm.twopl import fit_2pl
from fast_mlsirm.irt_contract import (
    validate_irt_experiment_readiness,
    validate_irt_response_matrix,
)


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


def test_irt_experiment_readiness_enforces_min_persons_items_and_variation() -> None:
    matrix = [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]
    assert validate_irt_experiment_readiness(matrix, "dichotomous", min_persons=5).shape == (
        5,
        2,
    )
    with pytest.raises(ValueError, match="at least .* persons"):
        validate_irt_experiment_readiness(
            [[0, 1], [1, 0]],
            "dichotomous",
            min_persons=5,
        )
    with pytest.raises(ValueError, match="at least .* non-missing"):
        validate_irt_experiment_readiness(
            [[np.nan, 1], [np.nan, 0], [1, 1], [np.nan, np.nan], [0, 1]],
            "dichotomous",
            min_observed_per_item=3,
        )


def test_irt_experiment_readiness_requires_observed_item_variation() -> None:
    with pytest.raises(ValueError, match="constant"):
        validate_irt_experiment_readiness(
            [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
            "dichotomous",
            min_persons=5,
        )


def test_irt_experiment_readiness_checks_factor_coverage() -> None:
    matrix = [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]
    validate_irt_experiment_readiness(
        matrix,
        "dichotomous",
        min_persons=6,
        factor_ids=("g", "f"),
        min_items_per_factor=1,
    )
    with pytest.raises(ValueError, match="factor_ids"):
        validate_irt_experiment_readiness(
            matrix,
            "dichotomous",
            min_persons=6,
            factor_ids=("g",),
        )
    with pytest.raises(ValueError, match="under-covered"):
        validate_irt_experiment_readiness(
            matrix,
            "dichotomous",
            min_persons=6,
            factor_ids=("g", "g"),
            min_items_per_factor=3,
        )


def test_irt_experiment_readiness_rejects_unsafe_controls_and_factor_shapes() -> None:
    matrix = [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]
    with pytest.raises(ValueError, match="min_items_per_factor"):
        validate_irt_experiment_readiness(
            matrix,
            "dichotomous",
            min_persons=5,
            factor_ids=np.array(["a", "b"]),
            min_items_per_factor=True,
        )
    with pytest.raises(ValueError, match="factor_ids"):
        validate_irt_experiment_readiness(
            matrix,
            "dichotomous",
            min_persons=5,
            factor_ids="ab",
        )
    with pytest.raises(ValueError, match="cannot exceed n_categories"):
        validate_irt_experiment_readiness(
            matrix,
            "polytomous",
            n_categories=2,
            min_persons=5,
            min_item_distinct_values=3,
        )


@pytest.mark.parametrize(
    "fit_call",
    [
        lambda y: fit_binary(y, np.array([0]), FitConfig(backend="numpy", max_iter=1)),
        lambda y: fit_2pl(y, q=7, max_iter=1),
        lambda y: fit_grm(y, n_cat=2, q=7, max_iter=1),
        lambda y: fit_polytomous(y, n_cat=2, q_theta=7, max_iter=1),
        lambda y: fit_rsm(y, n_cat=2, q_theta=7, max_iter=1),
    ],
)
def test_public_irt_fitters_reject_one_item_results(fit_call) -> None:
    with pytest.raises(ValueError, match="at least two item columns"):
        fit_call(np.array([[0.0], [1.0], [0.0]]))


if __name__ == "__main__":
    test_dichotomous_contract_requires_multiple_items()
    test_dichotomous_contract_rejects_non_binary_observations()
    test_polytomous_contract_requires_explicit_categories_and_multiple_items()
    test_polytomous_contract_rejects_invalid_categories_and_shape()
    print("ok")
