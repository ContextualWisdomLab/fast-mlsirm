"""Fixed-seed, synthetic-only orientation checks against the installed Rust core."""
import numpy as np
import pytest

from fast_mlsirm.scale_codebook import (
    OrderedItem, ScaleCodebook, prepare_scale_responses,
)
from fast_mlsirm.polytomous import (
    fit_polytomous, polytomous_category_probabilities,
    polytomous_expected_response, score_polytomous,
)


@pytest.mark.parametrize("seed", [733, 991, 1729])
def test_declared_keys_recover_positive_grm_trait_in_rust(seed):
    """Key recovery is exact; statistical recovery is bounded, not universal."""
    import fast_mlsirm._core as core
    assert hasattr(core, "fit_poly_unidim")
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=600)
    slopes = np.linspace(1.2, 2.0, 7)
    survival = 1.0 / (1.0 + np.exp(-(
        theta[:, None, None] * slopes[None, :, None]
        + np.array([1.3, -0.1, -1.5])[None, None, :]
    )))
    probabilities = np.concatenate([
        1.0 - survival[:, :, :1],
        survival[:, :, :-1] - survival[:, :, 1:],
        survival[:, :, -1:],
    ], axis=2)
    keyed = (rng.random((600, 7, 1)) > probabilities.cumsum(axis=2)[:, :, :-1]).sum(axis=2)
    raw = keyed.copy()
    raw[:, 1::2] = 3 - raw[:, 1::2]
    ids = tuple(f"I{j + 1}" for j in range(7))
    book = ScaleCodebook(
        scale_id="synthetic-grm", high_score_meaning="more target propensity",
        source_ref="synthetic://samejima-grm/fixed-fixtures/v1",
        items=tuple(OrderedItem(name, (3, 2, 1, 0) if j % 2 else (0, 1, 2, 3))
                    for j, name in enumerate(ids)),
    )
    prepared = prepare_scale_responses(raw, ids, book)
    np.testing.assert_array_equal(prepared.responses, keyed)
    fit = fit_polytomous(prepared.responses, n_cat=4, model="grm", q_theta=21,
                         max_iter=400, tol=1e-6)
    assert fit.converged, fit.termination_reason
    assert np.all(fit.slope > 0)
    scored = score_polytomous(prepared.responses, fit, q_theta=31)
    error = scored["theta_eap"] - theta
    assert abs(error.mean()) < 0.2
    assert np.mean(np.abs(error)) < 0.6
    assert np.sqrt(np.mean(error**2)) < 0.7
    assert np.corrcoef(theta, scored["theta_eap"])[0, 1] > 0.75
    grid = np.linspace(-4, 4, 81)
    expected = polytomous_expected_response(fit, grid)
    assert np.all(np.diff(expected, axis=0) >= -1e-12)
    cells = polytomous_category_probabilities(fit, grid)
    tails = np.cumsum(cells[:, :, ::-1], axis=2)[:, :, ::-1][:, :, 1:]
    assert np.all(np.diff(tails, axis=0) >= -1e-12)
    uniform_patterns = np.repeat(np.arange(4)[:, None], 7, axis=1)
    uniform_scores = score_polytomous(uniform_patterns, fit, q_theta=31)["theta_eap"]
    assert np.all(np.diff(uniform_scores) > 0)
