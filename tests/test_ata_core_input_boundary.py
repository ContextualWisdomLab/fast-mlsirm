"""Native ATA input-boundary safety regressions."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm._ata_core_loader import ata_core


def _valid_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return one small valid target-gain call."""
    matrix = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float64)
    candidates = np.array([0, 1], dtype=np.int64)
    target = np.array([2.0, 2.0], dtype=np.float64)
    accumulated = np.array([0.0, 0.0], dtype=np.float64)
    return matrix, candidates, target, accumulated


def test_target_gain_rejects_non_array_matrix_as_value_error() -> None:
    """Python containers must not leak PyO3 ``TypeError`` extraction details."""
    _, candidates, target, accumulated = _valid_inputs()

    with pytest.raises(ValueError, match="information_matrix must be a 2-D float64 NumPy array"):
        ata_core().target_information_gains(
            [[1.0, 2.0], [0.5, 1.5]], candidates, target, accumulated
        )


def test_target_gain_rejects_wrong_matrix_dtype_as_value_error() -> None:
    """Wrong matrix dtype must fail with a stable package-owned error."""
    matrix, candidates, target, accumulated = _valid_inputs()

    with pytest.raises(ValueError, match="information_matrix must be a 2-D float64 NumPy array"):
        ata_core().target_information_gains(
            matrix.astype(np.float32), candidates, target, accumulated
        )


def test_target_gain_rejects_noncontiguous_matrix_as_value_error() -> None:
    """A strided matrix must fail before native slice arithmetic begins."""
    matrix = np.arange(8, dtype=np.float64).reshape(2, 4)[:, ::2]
    candidates = np.array([0, 1], dtype=np.int64)
    target = np.array([4.0, 4.0], dtype=np.float64)
    accumulated = np.zeros(2, dtype=np.float64)
    assert not matrix.flags.c_contiguous

    with pytest.raises(ValueError, match="information_matrix must be C-contiguous and aligned"):
        ata_core().target_information_gains(matrix, candidates, target, accumulated)


def test_target_gain_rejects_wrong_candidate_dtype_as_value_error() -> None:
    """Candidate dtype failures must be normalized to ``ValueError``."""
    matrix, candidates, target, accumulated = _valid_inputs()

    with pytest.raises(ValueError, match="candidates must be a 1-D int64 NumPy array"):
        ata_core().target_information_gains(
            matrix, candidates.astype(np.float64), target, accumulated
        )


def test_target_gain_rejects_noncontiguous_candidates_as_value_error() -> None:
    """Strided candidate arrays must not leak rust-numpy slice errors."""
    matrix, _, target, accumulated = _valid_inputs()
    candidates = np.array([0, 1, 0, 1], dtype=np.int64)[::2]
    assert not candidates.flags.c_contiguous

    with pytest.raises(ValueError, match="candidates must be C-contiguous and aligned"):
        ata_core().target_information_gains(matrix, candidates, target, accumulated)


def test_target_gain_bounds_candidate_count_before_output_allocation() -> None:
    """The native wrapper must reject oversized candidate lists before ``Vec`` allocation."""
    matrix = np.array([[1.0]], dtype=np.float64)
    candidates = np.zeros(100_001, dtype=np.int64)
    target = np.array([1.0], dtype=np.float64)
    accumulated = np.array([0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="candidate count exceeds the 100000 ATA limit"):
        ata_core().target_information_gains(matrix, candidates, target, accumulated)


def test_target_gain_bounds_matrix_item_dimension_before_candidate_conversion() -> None:
    """Oversized matrix dimensions must fail before candidate conversion/allocation."""
    matrix = np.zeros((1, 100_001), dtype=np.float64)
    candidates = np.array([0], dtype=np.int64)
    target = np.array([1.0], dtype=np.float64)
    accumulated = np.array([0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="information matrix item count exceeds the 100000 ATA limit"):
        ata_core().target_information_gains(matrix, candidates, target, accumulated)
