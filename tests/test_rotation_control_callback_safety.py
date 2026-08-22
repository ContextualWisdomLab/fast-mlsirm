from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.rotation as rotation
import fast_mlsirm.rotation_selection as selection


class HostileText(str):
    def __new__(cls, value: str):
        instance = super().__new__(cls, value)
        instance.calls = 0
        return instance

    def strip(self, *args: object, **kwargs: object) -> str:
        self.calls += 1
        raise AssertionError("caller text callback executed")


class HostileIndex:
    def __init__(self) -> None:
        self.calls = 0

    def __index__(self) -> int:
        self.calls += 1
        raise AssertionError("caller __index__ callback executed")


class HostileFloat:
    def __init__(self) -> None:
        self.calls = 0

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("caller __float__ callback executed")


class HostileBool:
    def __init__(self) -> None:
        self.calls = 0

    def __bool__(self) -> bool:
        self.calls += 1
        raise AssertionError("caller __bool__ callback executed")


class CoreReached(AssertionError):
    pass


def _loadings() -> np.ndarray:
    return np.asarray([[0.8, 0.2], [0.1, 0.7], [0.5, -0.4]], dtype=np.float64)


def _forbid_core() -> object:
    raise CoreReached("compiled rotation core discovered before control admission")


def test_rotation_criterion_rejects_str_subclass_without_text_callback() -> None:
    criterion = HostileText("geomin")

    with pytest.raises(ValueError, match="criterion must be a non-empty string"):
        rotation.rotate_factor_loadings(_loadings(), criterion)

    assert criterion.calls == 0


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("normalize", HostileBool),
        ("n_starts", HostileIndex),
        ("seed", HostileIndex),
        ("max_iter", HostileIndex),
        ("tolerance", HostileFloat),
        ("function_window", HostileIndex),
        ("max_line_search", HostileIndex),
        ("basin_tolerance", HostileFloat),
        ("max_threads", HostileIndex),
        ("kappa", HostileFloat),
        ("gamma", HostileFloat),
        ("delta", HostileFloat),
        ("simplimax_zeros", HostileIndex),
    ],
)
def test_direct_rotation_controls_fail_before_callback_or_core(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    factory: type[object],
) -> None:
    value = factory()
    monkeypatch.setattr(rotation, "rotation_core", _forbid_core)

    with pytest.raises(ValueError):
        rotation.rotate_factor_loadings(_loadings(), "geomin", **{name: value})

    assert value.calls == 0


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("kappa", HostileFloat),
        ("gamma", HostileFloat),
        ("delta", HostileFloat),
        ("simplimax_zeros", HostileIndex),
    ],
)
def test_gradient_controls_fail_before_callback_or_core(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    factory: type[object],
) -> None:
    value = factory()
    monkeypatch.setattr(rotation, "rotation_core", _forbid_core)

    with pytest.raises(ValueError):
        rotation.rotation_criterion_value_gradient(
            _loadings(), "geomin", **{name: value}
        )

    assert value.calls == 0


def test_selection_policy_and_candidate_reject_str_subclasses_without_callbacks() -> None:
    policy = HostileText("fully_exploratory")
    candidate = HostileText("varimax")

    with pytest.raises(ValueError, match="policy must be a non-empty string"):
        selection.select_rotation_criterion(
            _loadings(), ["varimax", "geomin"], policy=policy
        )
    assert policy.calls == 0

    with pytest.raises(ValueError, match="criterion must be a non-empty string"):
        selection.select_rotation_criterion(
            _loadings(), [candidate, "geomin"]
        )
    assert candidate.calls == 0


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("normalize", HostileBool),
        ("n_starts", HostileIndex),
        ("seed", HostileIndex),
        ("max_iter", HostileIndex),
        ("tolerance", HostileFloat),
        ("function_window", HostileIndex),
        ("max_line_search", HostileIndex),
        ("basin_tolerance", HostileFloat),
        ("max_threads", HostileIndex),
        ("kappa", HostileFloat),
        ("gamma", HostileFloat),
        ("delta", HostileFloat),
        ("simplimax_zeros", HostileIndex),
    ],
)
def test_selection_controls_fail_before_callback_or_core(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    factory: type[object],
) -> None:
    value = factory()
    monkeypatch.setattr(selection, "rotation_core", _forbid_core)

    with pytest.raises(ValueError):
        selection.select_rotation_criterion(
            _loadings(), ["varimax", "geomin"], **{name: value}
        )

    assert value.calls == 0


def test_concrete_numpy_controls_are_normalized_before_direct_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, tuple[object, ...]] = {}

    class Probe:
        def rotate_factor_loadings(self, *args: object) -> object:
            captured["args"] = args
            raise CoreReached("probe")

    monkeypatch.setattr(rotation, "rotation_core", lambda: Probe())

    with pytest.raises(CoreReached, match="probe"):
        rotation.rotate_factor_loadings(
            _loadings(),
            "geomin",
            normalize=np.bool_(True),
            n_starts=np.int16(4),
            seed=np.uint32(7),
            max_iter=np.int32(100),
            tolerance=np.float32(1e-5),
            function_window=np.uint8(5),
            max_line_search=np.int16(8),
            basin_tolerance=np.float64(1e-8),
            max_threads=np.uint8(1),
            kappa=np.float32(0.1),
            gamma=np.float64(0.2),
            delta=np.float32(0.03),
            simplimax_zeros=np.int16(2),
        )

    args = captured["args"]
    assert type(args[3]) is bool
    for index in (4, 5, 6, 8, 9, 11, 15):
        assert type(args[index]) is int
    for index in (7, 10, 12, 13, 14):
        assert type(args[index]) is float
