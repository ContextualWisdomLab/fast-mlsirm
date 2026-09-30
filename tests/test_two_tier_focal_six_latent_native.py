"""Installed native six-latent bridge check against a Rust-produced fixture.

Cai (2010), DOI 10.1007/s11336-010-9178-0, pp.589-590 eqs.11-12,15-16
and pp.608-609 Appendices A/B. The fixture producer compares the actual Rust
scorer with independent physical full-product integration at the same nodes.
Synthetic parameters and five-point quadrature are diagnostic choices, not
study estimates or evidence of integration accuracy or population recovery.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from fast_mlsirm import (
    expected_total_score_two_tier_given_primary,
    score_two_tier_grm_orthogonal,
)


@pytest.mark.parametrize("cache_item_tables", [False, True])
def test_six_latent_native_matches_rust_full_product_fixture(cache_item_tables):
    """Check joint shared G/W, four specifics, four categories and missingness.

    Cai (2010), p.609 Appendix B supplies the moments definition. Fixture
    provenance records actual kernel/producer hashes and independent oracle.
    A missing native export fails; no numerical fallback replaces this test.
    """
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/two_tier_focal/six_latent_same_node.json"
        ).read_text()
    )
    y = np.array(fixture["responses"], dtype=float).reshape(3, 16)
    observed = np.array(fixture["observed"], dtype=bool).reshape(3, 16)
    y[~observed] = np.nan
    scores = score_two_tier_grm_orthogonal(
        responses=y,
        primary_map=np.array(fixture["primary_map"], dtype=bool).reshape(16, 2),
        specific_map=np.array(fixture["specific_map"], dtype=np.int64),
        a_primary=np.array(fixture["a_primary"]).reshape(16, 2),
        a_specific=np.array(fixture["a_specific"]),
        threshold=np.array(fixture["threshold"]).reshape(16, 3),
        latent_mean=np.array(fixture["latent_mean"]),
        latent_sd=np.array(fixture["latent_sd"]),
        n_cat=4,
        n_primary=2,
        n_specific=4,
        q_primary=5,
        q_specific=5,
        cache_item_tables=cache_item_tables,
    )
    for actual, key in (
        (scores.mean, "person_mean"),
        (scores.second, "person_second"),
        (scores.sd, "person_sd"),
    ):
        np.testing.assert_allclose(
            actual, np.array(fixture[key]).reshape(3, 6), rtol=0, atol=1e-10
        )
    np.testing.assert_allclose(scores.loglik, fixture["loglik"], rtol=0, atol=1e-10)


def test_native_focal_g_scores_use_explicit_reference_expected_score_curve():
    """Connect native focal EAPs to a separate reference-nuisance curve.

    Cai (2010), p.609 Appendix B defines person posterior moments. The
    expected-score API cites STAT414 lesson26.1's independent-normal theorem
    and MGF proof, read at https://online.stat.psu.edu/stat414/Lesson26.
    Choosing the standard-normal reference here is a synthetic fixture
    contract; it is distinct from the supplied focal latent distribution.
    This bridge check does not fit reference items or establish study recovery.
    """
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/two_tier_focal/six_latent_same_node.json")
        .read_text()
    )
    y = np.array(fixture["responses"], dtype=float).reshape(3, 16)
    y[~np.array(fixture["observed"], dtype=bool).reshape(3, 16)] = np.nan
    bank = dict(
        a_primary=np.array(fixture["a_primary"]).reshape(16, 2),
        a_specific=np.array(fixture["a_specific"]),
        threshold=np.array(fixture["threshold"]).reshape(16, 3),
        specific_map=np.array(fixture["specific_map"], dtype=np.int64),
    )
    focal = score_two_tier_grm_orthogonal(
        responses=y,
        primary_map=np.array(fixture["primary_map"], dtype=bool).reshape(16, 2),
        **bank,
        latent_mean=np.array(fixture["latent_mean"]),
        latent_sd=np.array(fixture["latent_sd"]),
        n_cat=4, n_primary=2, n_specific=4, q_primary=5, q_specific=5,
    )
    reference = dict(
        **bank, theta_focal=focal.mean[:, 0], focal_primary=0,
        nuisance_prior="independent_standardized",
        primary_ref_mean=np.zeros(2), primary_ref_sd=np.ones(2),
        specific_ref_mean=np.zeros(4), specific_ref_sd=np.ones(4),
    )
    low = expected_total_score_two_tier_given_primary(**reference, q_nuisance=121)
    high = expected_total_score_two_tier_given_primary(**reference, q_nuisance=241)
    np.testing.assert_array_equal(low.theta_focal, focal.mean[:, 0])
    np.testing.assert_allclose(low.expected_total, high.expected_total, atol=1e-8, rtol=0)
    substituted = expected_total_score_two_tier_given_primary(
        **{**reference,
           "primary_ref_mean": np.array(fixture["latent_mean"][:2]),
           "primary_ref_sd": np.array(fixture["latent_sd"][:2]),
           "specific_ref_mean": np.array(fixture["latent_mean"][2:]),
           "specific_ref_sd": np.array(fixture["latent_sd"][2:])},
        q_nuisance=241,
    )
    assert not np.allclose(high.expected_total, substituted.expected_total, atol=1e-8, rtol=0)
