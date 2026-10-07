"""Expected-raw person scoring for fitted two-tier GRM models."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from fast_mlsirm.two_tier_grm import expected_raw_two_tier_grm, fit_two_tier_grm

N_PERSONS = 80
N_ITEMS = 6
N_CAT = 3
N_PRIMARY = 2
N_SPECIFIC = 2
SEED = 0xC0FFEE


PRIMARY_MAP = np.array(
    [
        [True, False],
        [True, False],
        [True, True],
        [True, True],
        [True, False],
        [True, False],
    ],
    dtype=bool,
)
SPECIFIC_MAP = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)


def _simulate(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, N_CAT, size=(N_PERSONS, N_ITEMS), dtype=np.int64)


def _fit(responses: np.ndarray):
    return fit_two_tier_grm(
        responses,
        PRIMARY_MAP,
        SPECIFIC_MAP,
        N_CAT,
        N_PRIMARY,
        N_SPECIFIC,
        q_primary=21,
        q_specific=21,
        max_iter=300,
        tol=1e-5,
        n_starts=1,
        seed=SEED,
    )


def test_expected_raw_two_tier_grm_smoke() -> None:
    y = _simulate(SEED)
    fit = _fit(y)
    scores = expected_raw_two_tier_grm(fit, SPECIFIC_MAP, q_specific=21)
    assert scores.shape == (N_PERSONS,)
    assert np.all(np.isfinite(scores))
    lo = 0.0
    hi = float(N_ITEMS * (N_CAT - 1))
    assert float(scores.min()) >= lo - 1e-9
    assert float(scores.max()) <= hi + 1e-9


def test_expected_raw_requires_q_specific() -> None:
    y = _simulate(SEED + 1)
    fit = _fit(y)
    with pytest.raises(TypeError):
        expected_raw_two_tier_grm(fit, SPECIFIC_MAP)  # type: ignore[call-arg]


def test_expected_raw_two_tier_grm_is_lord_wingersky_eap_plugin_not_joint_posterior_mean() -> None:
    """Lord-Wingersky conditional expected raw at the primary EAP plug-in.

    ``expected_raw_two_tier_grm`` fixes primary coordinates at
    ``fit.theta_p_eap``, integrates each item-block specific factor at
    ``q_specific`` nodes, and returns the Lord-Wingersky conditional expected
    raw total on the observed category scale. This estimand is **not** the mean
    of the joint posterior of the primary dimensions (G, W): the scorer does
    not accept ``phi`` and does not reintegrate ``fit.phi`` over primaries.
    """
    sig = inspect.signature(expected_raw_two_tier_grm)
    assert "phi" not in sig.parameters
    assert set(sig.parameters) == {"fit", "specific_map", "q_specific"}

    y = _simulate(SEED + 2)
    fit = _fit(y)
    scores = expected_raw_two_tier_grm(fit, SPECIFIC_MAP, q_specific=21)

    fit_phi_perturbed = replace(
        fit,
        phi=np.array([[1.0, 0.95], [0.95, 1.0]], dtype=np.float64),
    )
    scores_phi = expected_raw_two_tier_grm(
        fit_phi_perturbed, SPECIFIC_MAP, q_specific=21
    )
    np.testing.assert_allclose(scores, scores_phi)

    fit_theta_shifted = replace(
        fit, theta_p_eap=fit.theta_p_eap + np.array([0.15, -0.1])
    )
    scores_theta = expected_raw_two_tier_grm(
        fit_theta_shifted, SPECIFIC_MAP, q_specific=21
    )
    assert not np.allclose(scores, scores_theta)


@pytest.mark.parametrize(
    "bad_map",
    [
        SPECIFIC_MAP.astype(np.float64) + 0.9,
        np.where(SPECIFIC_MAP == 0, np.nan, 1.0),
        np.where(SPECIFIC_MAP == 0, 2, SPECIFIC_MAP),
        np.where(SPECIFIC_MAP == 0, -2, SPECIFIC_MAP),
        np.where(SPECIFIC_MAP == 0, 2**32, SPECIFIC_MAP),
    ],
)
def test_expected_raw_rejects_non_integer_or_out_of_range_specific_map(
    bad_map: np.ndarray,
) -> None:
    fit = _fit(_simulate(SEED + 3))
    with pytest.raises(ValueError, match="specific_map"):
        expected_raw_two_tier_grm(fit, bad_map, q_specific=21)


@pytest.mark.parametrize(
    "bad_map",
    [
        np.array([1.9, -1.7, 0, 0, 1, 1], dtype=object),
        np.array([2**64 - 1, 0, 0, 1, 1, 1], dtype=np.uint64),
    ],
)
def test_expected_raw_rejects_non_float_dtypes_that_astype_would_truncate(
    bad_map: np.ndarray,
) -> None:
    fit = _fit(_simulate(SEED + 4))
    with pytest.raises(ValueError, match="specific_map"):
        expected_raw_two_tier_grm(fit, bad_map, q_specific=21)


def test_expected_raw_rejects_transposed_item_parameters() -> None:
    fit = _fit(_simulate(SEED + 5))
    with pytest.raises(ValueError, match="threshold"):
        expected_raw_two_tier_grm(
            replace(fit, threshold=fit.threshold.T), SPECIFIC_MAP, q_specific=21
        )
    with pytest.raises(ValueError, match="a_primary"):
        expected_raw_two_tier_grm(
            replace(fit, a_primary=fit.a_primary.T), SPECIFIC_MAP, q_specific=21
        )


def test_expected_raw_rejects_bool_complex_maps_and_malformed_fit_arrays() -> None:
    fit = _fit(_simulate(SEED + 6))
    for bad_map in (
        np.array([True, False, True, False, True, False], dtype=object),
        SPECIFIC_MAP.astype(np.complex128) + 2j,
        SPECIFIC_MAP.astype(bool),
    ):
        with pytest.raises(ValueError, match="specific_map"):
            expected_raw_two_tier_grm(fit, bad_map, q_specific=21)
    with pytest.raises(ValueError, match="a_specific"):
        expected_raw_two_tier_grm(
            replace(fit, a_specific=fit.a_specific[None, :]), SPECIFIC_MAP, q_specific=21
        )
    with pytest.raises(ValueError, match="theta_p_eap"):
        expected_raw_two_tier_grm(
            replace(fit, theta_p_eap=np.full_like(fit.theta_p_eap, np.nan)),
            SPECIFIC_MAP,
            q_specific=21,
        )


def _storage_layout_fit():
    """Create finite monotonic four-item parameters without fitting."""
    from types import SimpleNamespace

    return SimpleNamespace(
        a_primary=np.array([[.75,-.3],[1.1,.25],[-.4,.9],[.6,-1.2]],dtype=np.float64),
        a_specific=np.array([.7,-.6,.8,1.1],dtype=np.float64),
        threshold=np.array([[1.2,.1,-1.0],[.8,-.2,-1.1],[1.5,.3,-.7],[.9,-.1,-1.4]],dtype=np.float64),
        theta_p_eap=np.array([[-1.7,.2],[.0,-.9],[.45,1.3],[1.8,-.6],[-.7,-1.2]],dtype=np.float64),
        n_primary=2,n_specific=2,n_cat=4,
    )


@pytest.mark.parametrize('field',['a_primary','a_specific','threshold','theta_p_eap'])
@pytest.mark.parametrize('storage',['strided','unaligned'])
def test_expected_raw_accepts_valid_noncontiguous_and_unaligned_arrays(field,storage):
    """Compare logical-value twins through the actual native score consumer."""
    from types import SimpleNamespace

    fit=_storage_layout_fit();smap=np.array([0,0,1,1],dtype=np.int64)
    base=expected_raw_two_tier_grm(fit,smap,q_specific=121)
    source=getattr(fit,field)
    if storage=='strided':
        backing=np.empty(source.size*2,dtype=np.float64)
        view=backing[::2].reshape(source.shape)
        assert not view.flags.c_contiguous
    else:
        backing=bytearray(source.nbytes+1)
        view=np.ndarray(source.shape,dtype=np.float64,buffer=backing,offset=1)
        assert view.flags.c_contiguous and not view.flags.aligned
    view[...]=source
    before=view.tobytes()
    trial=SimpleNamespace(**{**vars(fit),field:view})
    result=expected_raw_two_tier_grm(trial,smap,q_specific=121)
    np.testing.assert_array_equal(result,base)
    assert view.tobytes()==before
