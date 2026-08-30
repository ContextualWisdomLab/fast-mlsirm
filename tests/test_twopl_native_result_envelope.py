"""Trust-boundary regressions for compensatory 2PL native result admission."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
import pytest

from fast_mlsirm.twopl import fit_2pl


class _HostileNativeResult(Mapping[str, object]):
    """Mapping-shaped provider object whose callbacks must never execute."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getitem__(self, _key: str) -> object:
        self.calls.append("getitem")
        raise AssertionError("native result mapping callback executed")

    def __iter__(self) -> Iterator[str]:
        self.calls.append("iter")
        raise AssertionError("native result iteration callback executed")

    def __len__(self) -> int:
        self.calls.append("len")
        raise AssertionError("native result length callback executed")


class _Core:
    def __init__(self, result: object) -> None:
        self.result = result

    def fit_2pl(self, *_args: object) -> object:
        return self.result


def test_fit_2pl_rejects_non_builtin_native_result_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Foreign mapping protocols cannot run at the Rust-result trust boundary."""

    result = _HostileNativeResult()
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core(result))

    with pytest.raises(ValueError, match="native fit_2pl result must be a built-in dict"):
        fit_2pl(np.array([[0.0, 1.0], [1.0, 0.0]]), max_iter=1)

    assert result.calls == []
