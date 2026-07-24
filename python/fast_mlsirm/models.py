"""IRT model specifications shared by dimension-agnostic item-family APIs.

The public fitting functions use one ``model=`` argument, following the R
``mirt`` convention: a number denotes an exploratory factor count, while a
confirmatory specification declares the loading pattern.  The current Rust
estimators implement unrestricted exploratory estimation only for one factor;
multidimensional exploratory requests fail explicitly until rotation and
identification are implemented.

References (APA 7th ed.):
    Chalmers, R. P. (2012). mirt: A multidimensional item response theory
        package for the R environment. *Journal of Statistical Software, 48*(6),
        1-29. https://doi.org/10.18637/jss.v048.i06
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ConfirmatoryModel",
    "ExploratoryModel",
    "IrtModel",
    "confirmatory",
    "exploratory",
]


@dataclass(frozen=True)
class ExploratoryModel:
    """An exploratory model identified by its number of latent dimensions."""

    dimensions: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, (int, np.integer))
            or int(self.dimensions) < 1
        ):
            raise ValueError("exploratory dimensions must be a positive integer")
        object.__setattr__(self, "dimensions", int(self.dimensions))

    @property
    def n_dims(self) -> int:
        """Derived latent dimension count."""

        return self.dimensions


@dataclass(frozen=True, eq=False)
class ConfirmatoryModel:
    """A confirmatory model defined by an items-by-dimensions loading pattern."""

    loading_pattern: np.ndarray

    def __post_init__(self) -> None:
        raw = np.asarray(self.loading_pattern)
        if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] < 1:
            raise ValueError(
                "confirmatory loading_pattern must be a non-empty 2-D items x dimensions array"
            )
        if not np.issubdtype(raw.dtype, np.number) and not np.issubdtype(
            raw.dtype, np.bool_
        ):
            raise ValueError(
                "confirmatory loading_pattern entries must be numeric 0 or 1"
            )
        if np.iscomplexobj(raw):
            raise ValueError("confirmatory loading_pattern entries must be real 0 or 1")
        numeric = raw.astype(np.float64)
        if not np.all(np.isfinite(numeric)) or not np.all(
            (numeric == 0) | (numeric == 1)
        ):
            raise ValueError(
                "confirmatory loading_pattern entries must be finite and exactly 0 or 1"
            )
        pattern = numeric.astype(np.int64)
        pattern.setflags(write=False)
        object.__setattr__(self, "loading_pattern", pattern)

    @property
    def n_dims(self) -> int:
        """Derived latent dimension count."""

        return int(self.loading_pattern.shape[1])


IrtModel = ExploratoryModel | ConfirmatoryModel


def exploratory(dimensions: int = 1) -> ExploratoryModel:
    """Build an exploratory model specification."""

    return ExploratoryModel(dimensions)


def confirmatory(loading_pattern: np.ndarray) -> ConfirmatoryModel:
    """Build a confirmatory model specification from a binary loading pattern."""

    return ConfirmatoryModel(loading_pattern)


def _resolve_model(
    model: int | IrtModel,
    n_items: int,
) -> tuple[IrtModel, np.ndarray]:
    """Normalize a public model argument to a specification and core loading pattern."""

    if isinstance(model, bool):
        raise TypeError("model must be a factor count or an IRT model specification")
    if isinstance(model, (int, np.integer)):
        model = ExploratoryModel(int(model))
    if isinstance(model, ExploratoryModel):
        if model.dimensions != 1:
            raise NotImplementedError(
                "multidimensional exploratory loading estimation is not implemented; "
                "use models.confirmatory(...) for an identified loading structure"
            )
        return model, np.ones((n_items, 1), dtype=np.int64)
    if isinstance(model, ConfirmatoryModel):
        if model.loading_pattern.shape[0] != n_items:
            raise ValueError(
                "confirmatory model must have one loading-pattern row per item"
            )
        return model, model.loading_pattern
    raise TypeError("model must be a factor count or an IRT model specification")
