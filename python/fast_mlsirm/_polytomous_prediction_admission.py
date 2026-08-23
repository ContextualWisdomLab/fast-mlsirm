"""Callback-free public evidence admission for GRM/GPCM predictions."""

from __future__ import annotations

from types import ModuleType

import numpy as np


_NUMPY_INTEGER_SCALAR_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intp,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uintp,
    np.ulonglong,
)
_NUMPY_FLOAT_SCALAR_TYPES = (np.float16, np.float32, np.float64, np.longdouble)
_NUMPY_COMPLEX_SCALAR_TYPES = (np.complex64, np.complex128, np.clongdouble)
_TRUSTED_SCALAR_TYPES = (
    bool,
    int,
    float,
    complex,
    np.bool_,
    *_NUMPY_INTEGER_SCALAR_TYPES,
    *_NUMPY_FLOAT_SCALAR_TYPES,
    *_NUMPY_COMPLEX_SCALAR_TYPES,
)
_MAX_PREDICTION_CELLS = 20_000_000
_MARKER = "__fast_mlsirm_polytomous_prediction_admission__"


def _preflight(value: object, name: str, max_ndim: int) -> None:
    """Reject callback-bearing, over-rank, and over-budget evidence before NumPy."""

    error = f"{name} must be a trusted NumPy array or built-in sequence"
    rank_error = f"{name} must be at most {max_ndim}-D"
    resource_error = (
        f"{name} exceeds the {_MAX_PREDICTION_CELLS:,}-cell prediction evidence budget"
    )
    # frame = [item, next_child_index, logical_cells, subtree_rank, entered]
    frames: list[list[object]] = [[value, 0, 0, 1, False]]
    active_container_ids: set[int] = set()
    subtree_metrics: dict[int, tuple[int, int]] = {}

    def add_to_parent(count: int, subtree_rank: int) -> None:
        if count > _MAX_PREDICTION_CELLS:
            raise ValueError(resource_error)
        if not frames:
            return
        parent = frames[-1]
        subtotal = int(parent[2])
        if count > _MAX_PREDICTION_CELLS - subtotal:
            raise ValueError(resource_error)
        parent[2] = subtotal + count
        parent[3] = max(int(parent[3]), 1 + subtree_rank)

    while frames:
        frame = frames[-1]
        item = frame[0]
        item_type = type(item)
        depth = len(frames) - 1

        if item_type is np.ndarray:
            if depth + int(item.ndim) > max_ndim:
                raise ValueError(rank_error)
            frames.pop()
            add_to_parent(int(item.size), int(item.ndim))
            continue

        if any(item_type is scalar_type for scalar_type in _TRUSTED_SCALAR_TYPES):
            if depth > max_ndim:
                raise ValueError(rank_error)
            frames.pop()
            add_to_parent(1, 0)
            continue

        if item_type is not list and item_type is not tuple:
            raise ValueError(error)
        if depth + 1 > max_ndim:
            raise ValueError(rank_error)

        item_id = id(item)
        if not bool(frame[4]):
            cached = subtree_metrics.get(item_id)
            if cached is not None:
                count, subtree_rank = cached
                if depth + subtree_rank > max_ndim:
                    raise ValueError(rank_error)
                frames.pop()
                add_to_parent(count, subtree_rank)
                continue
            if item_id in active_container_ids:
                raise ValueError(error)
            active_container_ids.add(item_id)
            frame[4] = True

        child_index = int(frame[1])
        if child_index < len(item):
            frame[1] = child_index + 1
            frames.append([item[child_index], 0, 0, 1, False])
            continue

        count = int(frame[2])
        subtree_rank = int(frame[3])
        active_container_ids.remove(item_id)
        subtree_metrics[item_id] = (count, subtree_rank)
        frames.pop()
        add_to_parent(count, subtree_rank)


def _real_numeric_array(value: object, name: str, max_ndim: int) -> np.ndarray:
    """Admit trusted evidence and verify lossless float64 representation."""

    _preflight(value, name, max_ndim)
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    if raw.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError(f"{name} must be a numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = raw.astype(np.float64, copy=False)
        round_tripped = converted.astype(raw.dtype, copy=False)
    if raw.dtype.kind == "f":
        lossless = np.array_equal(raw, round_tripped, equal_nan=True)
    else:
        lossless = np.array_equal(raw, round_tripped)
    if not lossless:
        raise ValueError(f"{name} must be exactly representable as float64")
    return converted


def install(module: ModuleType) -> None:
    """Install trusted prediction evidence admission on the polytomous module."""

    current = module._polytomous_predictions
    if getattr(current, _MARKER, False):
        return

    def guarded_predictions(fit, theta):
        if type(fit) is not module.PolytomousFit:
            raise TypeError("fit must be a PolytomousFit")

        trusted_theta = _real_numeric_array(theta, "theta", 1)
        trusted_slope = _real_numeric_array(fit.slope, "fit.slope", 1)
        trusted_cat_params = _real_numeric_array(fit.cat_params, "fit.cat_params", 2)

        if (
            trusted_theta.ndim == 1
            and trusted_slope.ndim == 1
            and trusted_cat_params.ndim == 2
            and trusted_slope.size > 0
            and trusted_cat_params.shape[0] == trusted_slope.size
            and trusted_cat_params.shape[1] >= 1
        ):
            n_cat = int(trusted_cat_params.shape[1]) + 1
            cells = int(trusted_theta.size) * int(trusted_slope.size) * n_cat
            if cells > _MAX_PREDICTION_CELLS:
                raise ValueError(
                    f"prediction grid of {cells:,} cells exceeds the "
                    f"{_MAX_PREDICTION_CELLS:,} prediction-cell limit"
                )

        trusted_theta = np.ascontiguousarray(trusted_theta)
        trusted_slope = np.ascontiguousarray(trusted_slope)
        trusted_cat_params = np.ascontiguousarray(trusted_cat_params)
        trusted_fit = module.PolytomousFit(
            fit.model,
            trusted_slope,
            trusted_cat_params,
            fit.loglik,
            fit.n_iter,
        )
        return current(trusted_fit, trusted_theta)

    setattr(guarded_predictions, _MARKER, True)
    module._polytomous_predictions = guarded_predictions
