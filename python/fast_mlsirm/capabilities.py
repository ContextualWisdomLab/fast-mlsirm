"""Machine-readable production fitting capability manifest.

The manifest is intentionally derived from the validated public ``FitConfig``
vocabulary rather than maintaining a second hand-written compatibility table.
It describes dispatch support only; numerical ownership remains in the Rust
backend and this module performs no psychometric arithmetic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from .config import FitConfig, VALID_ESTIMATORS, VALID_MODELS


FIT_CAPABILITY_SCHEMA_VERSION = "1.0"
PRODUCTION_NUMERIC_OWNER = "rust"


def _accepted_estimators(model: str) -> tuple[str, ...]:
    accepted: list[str] = []
    for estimator in sorted(VALID_ESTIMATORS):
        try:
            FitConfig(model=model, estimator=estimator)
        except ValueError:
            continue
        accepted.append(estimator)
    return tuple(accepted)


@dataclass(frozen=True, slots=True)
class FitCapability:
    """One canonical model and its admitted production estimators."""

    model: str
    estimators: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject forged or non-canonical public capability records."""
        if type(self.model) is not str or self.model not in VALID_MODELS:
            raise ValueError("FitCapability must match a canonical model-estimator capability")
        if type(self.estimators) is not tuple or any(
            type(estimator) is not str for estimator in self.estimators
        ):
            raise ValueError("FitCapability must match a canonical model-estimator capability")
        if self.estimators != _accepted_estimators(self.model):
            raise ValueError("FitCapability must match a canonical model-estimator capability")


_FIT_CAPABILITY_ROWS = tuple(
    (model, _accepted_estimators(model)) for model in sorted(VALID_MODELS)
)


def fit_capabilities() -> tuple[FitCapability, ...]:
    """Return fresh canonical 1.0 model-by-estimator capability value objects."""
    return tuple(
        FitCapability(model=model, estimators=estimators)
        for model, estimators in _FIT_CAPABILITY_ROWS
    )


def fit_capability_manifest() -> dict[str, object]:
    """Return a fresh JSON-shaped production capability manifest."""
    return {
        "schema_version": FIT_CAPABILITY_SCHEMA_VERSION,
        "production_numeric_owner": PRODUCTION_NUMERIC_OWNER,
        "models": [
            {
                "model": model,
                "estimators": list(estimators),
            }
            for model, estimators in _FIT_CAPABILITY_ROWS
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Print the production capability manifest as JSON for operator tooling."""
    parser = argparse.ArgumentParser(
        prog="python -m fast_mlsirm.capabilities",
        description="Print the bounded fast-mlsirm production fitting capability manifest.",
    )
    parser.parse_args(argv)
    print(json.dumps(fit_capability_manifest(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
