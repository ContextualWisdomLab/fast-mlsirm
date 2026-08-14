"""Fail-first resource contracts for exact-value report materialization."""

from __future__ import annotations

import pytest

from fast_mlsirm.report_exact_values import (
    MAX_EXACT_VALUE_CELLS,
    MAX_EXACT_VALUE_COLUMNS,
    MAX_EXACT_VALUE_ROWS,
    exact_value_csv,
    exact_value_disclosure,
    exact_value_json,
)


def test_exact_value_materializers_reject_row_amplification() -> None:
    """Complete exports must fail closed before an oversized row fan-out."""
    rows = [{"item_id": index} for index in range(MAX_EXACT_VALUE_ROWS + 1)]

    for materialize in (
        lambda: exact_value_disclosure(rows, section_label="Item Fit"),
        lambda: exact_value_json(rows),
        lambda: exact_value_csv(rows),
    ):
        with pytest.raises(ValueError, match="exact-value row limit"):
            materialize()


def test_exact_value_materializers_reject_column_amplification() -> None:
    """One adversarially wide row cannot multiply HTML/JSON/CSV output."""
    row = {f"metric_{index}": index for index in range(MAX_EXACT_VALUE_COLUMNS + 1)}

    for materialize in (
        lambda: exact_value_disclosure([row], section_label="Model Fit"),
        lambda: exact_value_json([row]),
        lambda: exact_value_csv([row]),
    ):
        with pytest.raises(ValueError, match="exact-value column limit"):
            materialize()


def test_exact_value_materializers_reject_total_cell_amplification() -> None:
    """Rows and columns that are individually bounded still have a cell ceiling."""
    column_count = min(MAX_EXACT_VALUE_COLUMNS, 101)
    row_count = MAX_EXACT_VALUE_CELLS // column_count + 1
    assert row_count <= MAX_EXACT_VALUE_ROWS
    rows = [
        {f"metric_{column}": row + column for column in range(column_count)}
        for row in range(row_count)
    ]

    for materialize in (
        lambda: exact_value_disclosure(rows, section_label="Candidate Comparison"),
        lambda: exact_value_json(rows),
        lambda: exact_value_csv(rows),
    ):
        with pytest.raises(ValueError, match="exact-value cell limit"):
            materialize()


def test_exact_value_limit_boundary_remains_complete_not_truncated() -> None:
    """Accepted data remains complete; resource controls reject rather than truncate."""
    rows = [{"item_id": index} for index in range(MAX_EXACT_VALUE_ROWS)]

    disclosure = exact_value_disclosure(rows, section_label="Item Fit")

    assert f"({MAX_EXACT_VALUE_ROWS} rows)" in disclosure
    assert str(MAX_EXACT_VALUE_ROWS - 1) in disclosure
