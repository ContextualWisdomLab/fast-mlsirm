"""Extension-independent #2001 L3 public-result provenance transport contracts.

The binding stand-in exercises only Python argument/result transport. These
fixtures do not attest native execution, GPU readback, or numerical recovery.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

bifactor = importlib.import_module("fast_mlsirm.bifactor_grm")
fitstats = importlib.import_module("fast_mlsirm.fitstats")


@pytest.mark.parametrize(
    ("requested_device", "effective_device", "shards"),
    [
        ("cpu", "cpu", [{"device": "cpu", "person_start": 0, "person_end": 2}]),
        ("gpu", "gpu", [{"device": "gpu", "person_start": 0, "person_end": 2}]),
        (
            "split",
            "cpu+gpu",
            [
                {"device": "cpu", "person_start": 0, "person_end": 1},
                {"device": "gpu", "person_start": 1, "person_end": 2},
            ],
        ),
        ("split", "cpu", [{"device": "cpu", "person_start": 0, "person_end": 2}]),
    ],
)
def test_public_result_preserves_binding_provenance(
    monkeypatch: pytest.MonkeyPatch,
    requested_device: str,
    effective_device: str,
    shards: list[dict[str, object]],
) -> None:
    """Transport observed metadata without substituting the requested device."""
    raw = {
        "a_general": [1.0, 1.0],
        "a_specific": [1.0, 1.0],
        "threshold": [0.0, 0.0],
        "theta_g_eap": [0.0, 0.0],
        "theta_g_sd": [1.0, 1.0],
        "category_counts": [1, 1, 1, 1],
        "loglik_trace": [-4.0, -3.0],
        "n_iter": 1,
        "converged": False,
        "termination_reason": "max_iter_reached",
        "final_loglik_change": 1.0,
        "best_start": 2,
        "n_parameters": 6,
        "effective_device": effective_device,
        "estep_shards": shards,
    }
    calls: list[tuple[object, ...]] = []

    def binding(*args: object) -> dict[str, object]:
        """Capture one transport call without loading or invoking native code."""
        calls.append(args)
        return raw

    monkeypatch.setattr(
        fitstats, "_core_module", lambda: SimpleNamespace(fit_bifactor_grm=binding)
    )
    result = bifactor.fit_bifactor_grm(
        np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
        n_cat=2,
        n_specific=1,
        q_general=121,
        q_specific=121,
        max_iter=1,
        tol=1e-6,
        n_starts=3,
        seed=20261002,
        device=requested_device,
        split_at_person=1 if requested_device == "split" else None,
    )
    assert len(calls) == 1
    assert calls[0][-2:] == (
        requested_device,
        1 if requested_device == "split" else None,
    )
    assert result.effective_device == effective_device
    assert result.estep_shards == shards
    assert result.estep_shards is not shards
    assert result.best_start == 2
    assert result.converged is False
    assert result.termination_reason == "max_iter_reached"


@pytest.mark.parametrize(
    "seed",
    [2**53 + 1, 2**64 - 1, np.uint64(2**53 + 1), np.uint64(2**64 - 1)],
)
def test_public_fit_preserves_full_u64_integer_seed(
    monkeypatch: pytest.MonkeyPatch, seed: int | np.uint64,
) -> None:
    """Pass the caller's integer seed unchanged across the Python binding seam."""
    captured: list[int] = []

    class BindingReached(Exception):
        """Stop at the stand-in binding before numerical execution."""

    def binding(*args: object) -> dict[str, object]:
        """Capture the normalized seed and stop without executing a fit."""
        assert type(args[12]) is int
        captured.append(args[12])
        raise BindingReached

    monkeypatch.setattr(
        fitstats, "_core_module", lambda: SimpleNamespace(fit_bifactor_grm=binding)
    )
    with pytest.raises(BindingReached):
        bifactor.fit_bifactor_grm(
            np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            np.asarray([0, 0], dtype=np.int64),
            n_cat=2, n_specific=1, q_general=121, q_specific=121,
            max_iter=1, tol=1e-6, n_starts=1, seed=seed, device="cpu",
        )
    assert captured == [int(seed)]
    assert type(captured[0]) is int


@pytest.mark.parametrize("seed", [-1, 2**64, True, 0.5, float("inf")])
def test_u64_seed_invalid_values_remain_rejected(seed: object) -> None:
    """Reject invalid seeds without relaxing the existing u64 boundary."""
    with pytest.raises(ValueError):
        bifactor._u64_seed(seed)


def test_seed_precision_repair_does_not_add_integer_subclass_callbacks() -> None:
    """Keep the pre-existing float compatibility path for integer subclasses."""
    calls: list[str] = []

    class SeedSubclass(int):
        """Distinguish the old float protocol from a newly introduced int hook."""

        def __float__(self) -> float:
            """Retain the accepted legacy conversion behavior."""
            calls.append("float")
            return 7.0

        def __int__(self) -> int:
            """Reject an integer hook that the previous adapter never invoked."""
            raise AssertionError("new integer callback must not execute")

    assert bifactor._u64_seed(SeedSubclass(7)) == 7
    assert calls == ["float"]
