"""Boundary tests for the Python validation layer in :mod:`scaling`.

The numerical kernels already have an independent Rust test suite.  These
tests keep the smaller Python-to-Rust trust boundary honest: malformed values
must fail before unsigned conversion or allocation, while representative
object, integer, and float inputs must still reach the real core.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

import fast_mlsirm.scaling as scaling
from fast_mlsirm.estimators.mmle import fit_mmle_2pl
from fast_mlsirm.rubric import EvidenceMode
import fast_mlsirm.rubric.generation as generation


def _raises(call, match: str | None = None) -> None:
    """Assert that a boundary call raises the package-owned ``ValueError``."""
    with pytest.raises(ValueError, match=match):
        call()


def _games(dtype=float):
    """Return one finite two-player game in the requested dtype."""
    return np.array([[0, 0, 1, 1]], dtype=dtype)


class _BrokenIterator:
    """Iterator whose ordinary exception must be redacted by a wrapper."""

    def __iter__(self):
        """Raise a private error when iteration begins."""
        raise RuntimeError("private_scaling_payload")


def test_ranking_and_top1_remaining_iterator_boundaries(monkeypatch) -> None:
    """Exercise rejected outer/inner iterators and final CSR-budget checks."""
    _raises(lambda: scaling._rankings_to_csr("probe", _BrokenIterator(), 3), "iteration")
    _raises(
        lambda: scaling._rankings_to_csr("probe", (_BrokenIterator(),), 3),
        "iteration",
    )
    monkeypatch.setattr(scaling, "MAX_RANKING_CSR_BYTES", 24)
    _raises(lambda: scaling._rankings_to_csr("probe", ((0, 1),), 3), "byte limit")

    original_iter = builtins.iter

    def fail_outer(_value):
        """Raise from a Python iterator shim so the re-raise path is traced."""
        raise RuntimeError("private_outer_iterator")

    monkeypatch.setattr(builtins, "iter", fail_outer)
    _raises(lambda: scaling._rankings_to_csr("probe", (), 3), "iteration")
    _raises(lambda: scaling._top1_to_csr("probe", (), 3), "iteration")

    def fail_inner(value):
        """Return the outer iterator, then fail while creating its inner one."""
        fail_inner.calls += 1
        if fail_inner.calls == 2:
            raise RuntimeError("private_inner_iterator")
        return original_iter(value)

    fail_inner.calls = 0
    monkeypatch.setattr(builtins, "iter", fail_inner)
    _raises(lambda: scaling._rankings_to_csr("probe", ((0, 1),), 3), "iteration")

    def fail_process(_value):
        """Raise a process-control signal from the iterator shim."""
        raise KeyboardInterrupt()

    monkeypatch.setattr(builtins, "iter", fail_process)
    with pytest.raises(KeyboardInterrupt):
        scaling._rankings_to_csr("probe", (), 3)
    with pytest.raises(KeyboardInterrupt):
        scaling._top1_to_csr("probe", (), 3)
    def fail_process_inner(value):
        """Raise a process-control signal only for the inner ranking."""
        fail_process_inner.calls += 1
        if fail_process_inner.calls == 2:
            raise KeyboardInterrupt()
        return original_iter(value)

    fail_process_inner.calls = 0
    monkeypatch.setattr(builtins, "iter", fail_process_inner)
    with pytest.raises(KeyboardInterrupt):
        scaling._rankings_to_csr("probe", ((0, 1),), 3)
    monkeypatch.setattr(builtins, "iter", original_iter)

    _raises(lambda: scaling._top1_to_csr("probe", _BrokenIterator(), 3), "iteration")
    _raises(
        lambda: scaling._top1_to_csr("probe", [], 3),
        "at least one observation",
    )
    _raises(
        lambda: scaling._top1_to_csr("probe", ((0, (1,)),), 3),
        "byte limit",
    )

    monkeypatch.setattr(scaling, "MAX_RANKING_CSR_BYTES", 16)
    _raises(lambda: scaling._top1_to_csr("probe", ((0, (1,)),), 3), "byte limit")

    class _ProcessObservation:
        """Observation whose unpacking raises a process-control signal."""

        def __iter__(self):
            """Raise during tuple unpacking."""
            raise KeyboardInterrupt()

    class _ProcessLosers:
        """Loser iterable whose construction raises a process-control signal."""

        def __iter__(self):
            """Raise while creating the loser iterator."""
            raise KeyboardInterrupt()

    class _ProcessOuter:
        """Top-1 outer iterator whose next item raises a process signal."""

        def __iter__(self):
            """Return this iterator."""
            return self

        def __next__(self):
            """Raise when the adapter requests the next observation."""
            raise KeyboardInterrupt()

    monkeypatch.setattr(scaling, "MAX_RANKING_CSR_BYTES", 8 * 1024 * 1024)
    with pytest.raises(KeyboardInterrupt):
        scaling._top1_to_csr("probe", (_ProcessObservation(),), 3)
    with pytest.raises(KeyboardInterrupt):
        scaling._top1_to_csr("probe", ((0, _ProcessLosers()),), 3)
    with pytest.raises(KeyboardInterrupt):
        scaling._top1_to_csr("probe", _ProcessOuter(), 3)

    class _BadObservation:
        """Observation that cannot be unpacked by the top-1 adapter."""

        def __iter__(self):
            """Raise an ordinary caller error during tuple unpacking."""
            raise RuntimeError("private_observation_payload")

    monkeypatch.setattr(scaling, "MAX_RANKING_CSR_BYTES", 8 * 1024 * 1024)
    _raises(lambda: scaling._top1_to_csr("probe", (_BadObservation(),), 3), "iteration")

    class _BadLosers:
        """Loser collection with a private iteration failure."""

        def __iter__(self):
            """Raise an ordinary caller error while creating the iterator."""
            raise RuntimeError("private_loser_payload")

    _raises(lambda: scaling._top1_to_csr("probe", ((0, _BadLosers()),), 3), "loser")


@pytest.mark.parametrize(
    "fn",
    (scaling.elo_rating, scaling.glicko_rating, scaling.glicko2_rating, scaling.fide_rating),
)
def test_two_player_rating_wrappers_reject_common_game_boundaries(fn) -> None:
    """Shared game validation rejects malformed numeric and index inputs."""
    name = fn.__name__
    _raises(lambda: fn(np.array([[0, 0, 1]], dtype=float), 2), "games must be")
    _raises(lambda: fn(np.empty((0, 4)), 2), "at least one game")
    _raises(lambda: fn(np.array([[0, 0, 1, np.nan]]), 2), "non-finite")
    _raises(lambda: fn(np.array([[0.5, 0, 1, 1.0]]), 2), "period")
    _raises(lambda: fn(np.array([[-1, 0, 1, 1.0]]), 2), "period")
    _raises(lambda: fn(np.array([[0, -1, 1, 1.0]]), 2), "nonnegative")
    _raises(lambda: fn(_games(), 2.5), "n_players")
    _raises(lambda: fn(_games(), 2, gamma=1 + 2j), "gamma")
    _raises(lambda: fn(_games(), 2, gamma="x"), "gamma")
    _raises(lambda: fn(_games(), 2, gamma=[1.0, 2.0]), "gamma")
    assert fn(_games(), 2, gamma=np.array([0.0])).ratings.shape == (2,)
    assert fn(np.array([[0, 0, 1, 1]], dtype=object), 2).ratings.shape == (2,)
    assert name.endswith("rating")


def test_elo_float_period_fidelity_and_gamma_scalar() -> None:
    """Elo rejects ambiguous float periods and accepts an object scalar gamma."""
    _raises(lambda: scaling.elo_rating(np.array([[2**53, 0, 1, 1.0]], dtype=float), 2), "representable")
    result = scaling.elo_rating(np.array([[0, 0, 1, 1]], dtype=object), 2, gamma=0.0)
    assert result.ratings.shape == (2,)


def test_glicko_boundary_variants_and_init_contract() -> None:
    """Glicko's pair-valued init and numeric conversion paths are explicit."""
    for fn, prefix in (
        (scaling.glicko_rating, "glicko_rating"),
        (scaling.glicko2_rating, "glicko2_rating"),
    ):
        _raises(lambda fn=fn: fn([['x', 0, 1, 1]], 2), "not numeric")
        _raises(lambda fn=fn: fn([[0, 0, 1, 1]], 2, init=("x", 300)), "init")
        _raises(lambda fn=fn: fn([[0, 0, 1, 1]], 2, init=(1 + 2j, 300)), "init")
        _raises(lambda fn=fn: fn([[0, 0, 1, 1]], 2, init=(2200,)), "init")
        if fn is scaling.glicko_rating:
            _raises(lambda: fn([[0, 0, 1, 1]], 2, init=([1, 2], [3])), "length")
            _raises(lambda: fn([[0, 0, 1, 1]], 2, gamma="x"), "gamma")
        else:
            _raises(lambda: fn([[0, 0, 1, 1]], 2, init=(2200, 300, "x")), "init")
            _raises(lambda: fn([[0, 0, 1, 1]], 2, init=(1 + 2j, 300, 0.15)), "init")
            _raises(lambda: fn([[0, 0, 1, 1]], 2, gamma="x"), "gamma")
        assert prefix in fn.__name__


def test_glicko_integer_and_object_period_paths() -> None:
    """Lossless integer and object period representations reach both kernels."""
    for fn in (scaling.glicko_rating, scaling.glicko2_rating):
        result = fn(_games(dtype=np.int64), 2)
        assert result.ratings.shape == (2,)
        result = fn(np.array([[0, 0, 1, 1]], dtype=object), 2, gamma=0.0)
        assert result.games.shape == (2,)
        _raises(lambda fn=fn: fn([[0, 0, 1, 1]], 2, gamma=[1.0, 2.0]), "gamma")


def test_stephenson_validation_and_success_paths() -> None:
    """Stephenson validates inherited state vectors before invoking Rust."""
    valid = _games(dtype=np.int64)
    assert scaling.stephenson_rating(valid, 2).ratings.shape == (2,)
    assert scaling.stephenson_rating(valid, 2, gamma=0.0).ratings.shape == (2,)
    assert scaling.stephenson_rating(np.array([[0, 0, 1, 1]], dtype=object), 2).games.shape == (2,)
    _raises(lambda: scaling.stephenson_rating(np.empty((0, 4)), 2), "at least one game")
    _raises(lambda: scaling.stephenson_rating([[0, -1, 1, 1]], 2), "nonnegative")
    _raises(lambda: scaling.stephenson_rating(valid, 2, gamma="x"), "gamma")
    _raises(lambda: scaling.stephenson_rating(valid, 2, gamma=[1.0, 2.0]), "gamma")
    _raises(lambda: scaling.stephenson_rating(valid, 2, init=(1 + 2j, 300)), "init")
    _raises(lambda: scaling.stephenson_rating(valid, 2, init=("x", 300)), "init")
    _raises(lambda: scaling.stephenson_rating(valid, 2, init=(1, 2, 3)), "init")
    _raises(lambda: scaling.stephenson_rating(valid, 2, init_games="x"), "init_games")
    _raises(lambda: scaling.stephenson_rating(valid, 2, init_games=[1]), "init_games")
    _raises(lambda: scaling.stephenson_rating(valid, 2, init_games=[1j, 0]), "real")
    _raises(lambda: scaling.stephenson_rating(valid, 2, init_games=[-1, 0]), "nonnegative")
    _raises(lambda: scaling.stephenson_rating(valid, 2, init_games=[2**53, 0]), "2\\*\\*53")
    _raises(lambda: scaling.stephenson_rating(valid, 2, cval="x"), "not numeric")
    _raises(lambda: scaling.stephenson_rating(valid, 1), "two players")
    _raises(lambda: scaling.stephenson_rating(valid, 2.5), "n_players")


def test_elom_validation_and_success_paths() -> None:
    """EloM accepts both K-factor modes and rejects malformed event state."""
    periods = np.array([0], dtype=np.int64)
    players = np.array([[0, 1]], dtype=np.int64)
    scores = np.array([[1.0, 0.0]])
    kwargs = {"base": (30.0, -30.0)}
    assert scaling.elom_rating(periods, players, scores, 2, kfac=10.0, **kwargs).ratings.shape == (2,)
    assert scaling.elom_rating(periods, players, scores, 2, kfac=("kriichi", 400.0, 0.2), **kwargs).ratings.shape == (2,)

    _raises(lambda: scaling.elom_rating(periods, players, scores, 2.5, **kwargs), "n_players")
    _raises(lambda: scaling.elom_rating(periods, players, scores, "x", **kwargs), "not an integer")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 10001, **kwargs), "10000")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 1, **kwargs), "two players")
    _raises(lambda: scaling.elom_rating(periods, np.empty((0, 2)), np.empty((0, 2)), 2, **kwargs), "at least one event")
    _raises(lambda: scaling.elom_rating(periods, [0, 1], scores, 2, **kwargs), "array")
    _raises(lambda: scaling.elom_rating(periods, [[0.5, 1]], scores, 2, **kwargs), "integral")
    _raises(lambda: scaling.elom_rating(periods, [[float(2**53), 1]], scores, 2, **kwargs), "representable")
    _raises(lambda: scaling.elom_rating(periods, [[0, -2]], scores, 2, **kwargs), ">= -1")
    _raises(lambda: scaling.elom_rating(periods, [[0, "x"]], scores, 2, **kwargs), "not numeric")
    _raises(lambda: scaling.elom_rating(periods, [[0, 1]], np.array([[1 + 2j, 0]]), 2, **kwargs), "scores")
    _raises(lambda: scaling.elom_rating(periods, [[0, 1]], [[1, "x"]], 2, **kwargs), "scores")
    _raises(lambda: scaling.elom_rating(periods, players, [[1.0]], 2, **kwargs), "match players")
    _raises(lambda: scaling.elom_rating([0.5], players, scores, 2, **kwargs), "period")
    _raises(lambda: scaling.elom_rating([1 + 2j], players, scores, 2, **kwargs), "periods")
    _raises(lambda: scaling.elom_rating(["x"], players, scores, 2, **kwargs), "periods")
    _raises(lambda: scaling.elom_rating([], players, scores, 2, **kwargs), "length")
    _raises(lambda: scaling.elom_rating([-1], players, scores, 2, **kwargs), "nonnegative")
    _raises(lambda: scaling.elom_rating([float(2**53)], players, scores, 2, **kwargs), "representable")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 2, base="x"), "base")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 2, base=1 + 2j), "base")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 2, base=(1.0,)), "base")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 2, init="x", **kwargs), "init")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 2, init=1 + 2j, **kwargs), "init")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 2, init=[1.0], **kwargs), "init")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 2, init_games=[1j, 0], **kwargs), "real")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 2, init_games=[object(), 0], **kwargs), "numeric")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 2, init_games=[1], **kwargs), "init_games")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 2, init_games=[-1, 0], **kwargs), "nonnegative")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 2, init_games=[2**53, 0], **kwargs), "2\\*\\*53")
    valid_counts = {"init_games": [1, 0], "init_lag": [0, 1], "init_places": [[1, 0], [0, 1]]}
    assert scaling.elom_rating(periods, players, scores, 2, **kwargs, **valid_counts).ratings.shape == (2,)
    _raises(lambda: scaling.elom_rating(periods, players, scores, 2, kfac=("bad", 1, 1), **kwargs), "kfac")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 2, kfac=("kriichi", "x", 1), **kwargs), "numeric")
    _raises(lambda: scaling.elom_rating(periods, players, scores, 2, kfac="x", **kwargs), "kfac")


def test_elom_object_and_float_paths() -> None:
    """Object and float event arrays use their explicit fidelity branches."""
    periods = np.array([0], dtype=object)
    players = np.array([[0, 1]], dtype=object)
    scores = np.array([[1.0, 0.0]])
    result = scaling.elom_rating(periods, players, scores, 2, base=(30.0, -30.0), kfac=10.0)
    assert result.places.shape == (2, 2)
    assert scaling.elom_rating(
        periods,
        players,
        scores,
        2,
        base=(30.0, -30.0),
        init=[1500.0, 1500.0],
        kfac=10.0,
    ).ratings.shape == (2,)
    _raises(
        lambda: scaling.elom_rating(
            np.array([2**53], dtype=float), players, scores, 2, base=(30.0, -30.0)
        ),
        "representable",
    )


def test_metrics_and_fide_object_conversion_boundaries() -> None:
    """Metrics and FIDE wrappers reject opaque objects and preserve valid paths."""
    _raises(lambda: scaling.metrics_rating([1.0], [object()]), "numeric")
    _raises(lambda: scaling.metrics_rating([1.0], [[1.0], [0.0]]), "rows")

    valid = _games(dtype=np.int64)
    assert scaling.fide_rating(valid, 2).ratings.shape == (2,)
    assert scaling.fide_rating(np.array([[0, 0, 1, 1]], dtype=object), 2, gamma=0.0).games.shape == (2,)
    _raises(lambda: scaling.fide_rating([[object(), 0, 1, 1]], 2), "not numeric")
    _raises(lambda: scaling.fide_rating(np.empty((0, 4)), 2), "at least one game")
    _raises(lambda: scaling.fide_rating([[0, 0, 1, np.nan]], 2), "non-finite")
    _raises(lambda: scaling.fide_rating([[0, -1, 1, 1]], 2), "nonnegative")
    _raises(lambda: scaling.fide_rating([[-1, 0, 1, 1]], 2), "period")
    _raises(lambda: scaling.fide_rating([[float(2**53), 0, 1, 1]], 2), "representable")
    _raises(lambda: scaling.fide_rating(valid, 2, gamma=1 + 2j), "gamma")
    _raises(lambda: scaling.fide_rating(valid, 2, gamma="x"), "gamma")
    _raises(lambda: scaling.fide_rating(valid, 2, gamma=[1.0, 2.0]), "gamma")
    assert scaling.fide_rating(valid, 2, gamma=np.array([0.0])).ratings.shape == (2,)
    _raises(lambda: scaling.fide_rating(valid, 2, kv=object()), "kv")
    _raises(lambda: scaling.fide_rating(valid, 2, kv=(1.0, 2.0)), "triple")
    assert scaling.fide_rating(valid, 2, kv=np.array([10.0, 15.0, 30.0])).ratings.shape == (2,)
    _raises(lambda: scaling.fide_rating(valid, 2, init="x"), "init")


@pytest.mark.parametrize(
    ("helper", "bad_values"),
    (
        (
            scaling._predict_int_index_array,
            (np.ma.array([0]), np.array([True]), np.array([1 + 2j]), [None], [object()], np.array([[0]]), [np.inf], [0.5]),
        ),
        (
            lambda value: scaling._predict_float_array(value, "values", "probe", False),
            (np.ma.array([0.0]), np.array([True]), np.array([1 + 2j]), [object()], np.array([[0.0]]), [np.inf], [np.nan]),
        ),
    ),
)
def test_prediction_array_helpers_reject_each_invalid_shape(helper, bad_values) -> None:
    """Prediction helpers reject masks, nonnumeric values, shape, and finiteness errors."""
    for value in bad_values:
        _raises(lambda value=value: helper(value, "values", "probe"), None) if helper is scaling._predict_int_index_array else _raises(lambda value=value: helper(value), None)


def test_prediction_scalar_and_count_helpers() -> None:
    """Scalar/count helpers retain exact integer semantics at their boundaries."""
    _raises(lambda: scaling._predict_scalar(1 + 2j, "x", "probe"), "complex")
    _raises(lambda: scaling._predict_scalar(object(), "x", "probe"), "numeric")
    _raises(lambda: scaling._predict_scalar(np.array(True, dtype=object), "x", "probe"), "bool")
    _raises(lambda: scaling._predict_scalar(np.inf, "x", "probe"), "finite")
    _raises(lambda: scaling._predict_games_u64(np.ma.array([1]), "probe"), "masked")
    _raises(lambda: scaling._predict_games_u64([1 + 2j], "probe"), "complex")
    _raises(lambda: scaling._predict_games_u64([None], "probe"), "non-numeric")
    _raises(lambda: scaling._predict_games_u64([object()], "probe"), "numeric")
    _raises(lambda: scaling._predict_games_u64([[1], [2, 3]], "probe"), "numeric")
    _raises(lambda: scaling._predict_games_u64(np.array([[1]], dtype=int), "probe"), "one-dimensional")
    _raises(lambda: scaling._predict_games_u64(np.array([-1], dtype=int), "probe"), "nonnegative")
    _raises(lambda: scaling._predict_games_u64([1.5], "probe"), "nonnegative")
    _raises(lambda: scaling._predict_games_u64([float(2**53)], "probe"), "representable")
    assert scaling._predict_games_u64(np.array([2**53], dtype=np.int64), "probe")[0] == 2**53
    assert scaling._predict_games_u64(np.array([1], dtype=object), "probe")[0] == 1
    _raises(lambda: scaling._predict_tng_u64(True, "probe"), "bool")
    _raises(lambda: scaling._predict_tng_u64(-1, "probe"), "nonnegative")
    _raises(lambda: scaling._predict_tng_u64(2**64, "probe"), "unsigned")
    _raises(lambda: scaling._predict_tng_u64(1.5, "probe"), "nonnegative")
    _raises(lambda: scaling._predict_tng_u64(float(2**53), "probe"), "representable")
    assert scaling._predict_tng_u64(np.array(3, dtype=np.int64), "probe") == 3
    assert scaling._predict_tng_u64(3.0, "probe") == 3


def test_prediction_public_wrappers_cover_optional_branches() -> None:
    """Two-player and multiplayer prediction wrappers reach the Rust core."""
    ratings = np.array([1500.0, 1500.0])
    games = np.array([20, 20], dtype=np.int64)
    white = np.array([0])
    black = np.array([1])
    assert scaling.predict_rating(ratings, games, white, black).shape == (1,)
    assert scaling.predict_rating(
        ratings,
        games,
        white,
        black,
        deviations=np.array([300.0, 300.0]),
        trat=(1500.0, 300.0),
        gamma=np.array([0.0]),
        thresh=0.5,
    ).shape == (1,)
    _raises(lambda: scaling.predict_rating(ratings, games, white, black, gamma=1 + 2j), "gamma")
    assert scaling.predict_rating(ratings, games, white, black, trat=[1500.0]).shape == (1,)
    _raises(lambda: scaling.predict_rating(ratings, [20], white, black), "one entry")
    _raises(lambda: scaling.predict_rating(ratings, games, white, black, gamma=[0.0, 1.0]), "gamma")
    _raises(lambda: scaling.predict_rating(ratings, games, white, black, deviations=[300, 300], trat=1500), "pair")
    _raises(lambda: scaling.predict_rating(ratings, games, white, black, trat=[1, 2]), "scalar")

    players = np.array([[0, 1]], dtype=np.int64)
    assert scaling.predict_rating_multi(ratings, games, players).shape == (1, 2)
    assert scaling.predict_rating_multi(ratings, games, players, placing=True).shape == (1, 2)
    _raises(lambda: scaling.predict_rating_multi(np.array([1500.0]), np.array([20]), players), "2..=10000")
    _raises(lambda: scaling.predict_rating_multi(ratings, [20], players), "one entry")
    _raises(lambda: scaling.predict_rating_multi(ratings, games, np.ma.array([[0, 1]])), "masked")
    _raises(lambda: scaling.predict_rating_multi(ratings, games, np.array([[0]])), "seats")
    _raises(lambda: scaling.predict_rating_multi(ratings, games, players, placing=1), "bool")


def test_mmle_skips_singular_newton_update_when_every_item_is_nonfinite() -> None:
    """A malformed observed cell must not turn a singular Newton step into a crash."""
    with np.errstate(invalid="ignore"):
        result = fit_mmle_2pl(
            np.array([[np.nan]], dtype=float),
            np.array([[True]]),
            n_nodes=11,
            max_iter=1,
            ridge_a=0.0,
            ridge_b=0.0,
        )
    assert result["n_iter"] == 1


def test_generation_rejects_an_unknown_evidence_mode() -> None:
    """The source-cardinality guard fails closed for future enum values."""

    class UnknownEvidenceMode:
        """Enum-shaped value that is not currently supported."""

        value = "unknown_mode"

    with pytest.raises(ValueError, match="source cardinality"):
        generation._validate_source_cardinality(UnknownEvidenceMode(), 0)

    generation._validate_source_cardinality(EvidenceMode.UNANSWERABLE, 1)
