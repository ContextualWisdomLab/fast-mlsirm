"""Explicit item-order helper for judge-result IRT projection."""

from __future__ import annotations

import math

from .llm_judge import (
    JudgeFormatError,
    LLMJudgeResult,
    MAX_JUDGE_CATEGORIES,
    _category,
    _category_count,
    _score,
)


def _exact_result_mapping(value: object, name: str) -> dict[str, object]:
    """Admit one inert built-in result mapping before any mapping protocol runs."""
    if type(value) is not dict:
        raise JudgeFormatError(f"{name} must be an exact built-in dict")
    if any(type(key) is not str for key in value):
        raise JudgeFormatError(f"{name} keys must be strings")
    return value


def _ordered_criterion_ids(
    result: LLMJudgeResult,
    criterion_order: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Validate one explicit inert criterion order without caller callbacks."""
    if type(criterion_order) not in (list, tuple):
        raise JudgeFormatError("criterion_order must be a list or tuple of strings")
    if any(type(criterion_id) is not str for criterion_id in criterion_order):
        raise JudgeFormatError("criterion_order must contain only strings")
    criterion_ids = tuple(criterion_order)
    if len(criterion_ids) < 2:
        raise JudgeFormatError(
            "IRT output requires multiple criterion items; a scalar judge result is invalid"
        )
    if len(set(criterion_ids)) != len(criterion_ids):
        raise JudgeFormatError("criterion_order must contain unique criterion ids")
    criterion_scores = _exact_result_mapping(
        result.criterion_scores,
        "criterion_scores",
    )
    if set(criterion_ids) != set(criterion_scores):
        raise JudgeFormatError(
            "criterion_order must contain exactly the rubric criterion ids"
        )
    return criterion_ids


def project_row_in_order(
    result: LLMJudgeResult,
    *,
    item_type: str,
    n_categories: int | None,
    criterion_order: list[str] | tuple[str, ...],
) -> tuple[int, ...]:
    """Project one result in an authoritative caller-supplied item order."""
    if type(result) is not LLMJudgeResult:
        raise TypeError("result must be an LLMJudgeResult")
    if type(item_type) is not str or item_type not in {"dichotomous", "polytomous"}:
        raise JudgeFormatError("item_type must be dichotomous or polytomous")
    criterion_ids = _ordered_criterion_ids(result, criterion_order)
    criterion_scores = _exact_result_mapping(
        result.criterion_scores,
        "criterion_scores",
    )
    if result.criterion_categories is not None:
        if result.category_count is None:
            raise JudgeFormatError("criterion categories require category_count")
        criterion_categories = _exact_result_mapping(
            result.criterion_categories,
            "criterion_categories",
        )
        try:
            category_count = _category_count(result.category_count)
        except ValueError as exc:
            raise JudgeFormatError(str(exc)) from exc
        if set(criterion_categories) != set(criterion_ids):
            raise JudgeFormatError(
                "criterion categories must contain exactly the rubric criterion ids"
            )
        if item_type == "dichotomous":
            if category_count != 2 or n_categories is not None:
                raise JudgeFormatError(
                    "dichotomous output requires a two-category judge result"
                )
            return tuple(
                _category(
                    criterion_categories[criterion_id],
                    f"criterion_categories.{criterion_id}",
                    category_count,
                )
                for criterion_id in criterion_ids
            )
        if n_categories is not None:
            try:
                n_categories = _category_count(n_categories)
            except ValueError as exc:
                raise JudgeFormatError(str(exc)) from exc
        if n_categories is not None and n_categories != category_count:
            raise JudgeFormatError("n_categories must match the judge category_count")
        return tuple(
            _category(
                criterion_categories[criterion_id],
                f"criterion_categories.{criterion_id}",
                category_count,
            )
            for criterion_id in criterion_ids
        )
    if item_type == "dichotomous":
        if n_categories is not None:
            raise JudgeFormatError(
                "n_categories is only valid for polytomous IRT output"
            )
        return tuple(
            int(
                _score(
                    criterion_scores[criterion_id],
                    f"criterion_scores.{criterion_id}",
                )
                >= 0.5
            )
            for criterion_id in criterion_ids
        )
    try:
        resolved_categories = _category_count(n_categories)
    except ValueError as exc:
        raise JudgeFormatError(
            f"polytomous IRT output requires n_categories in 2..{MAX_JUDGE_CATEGORIES}"
        ) from exc
    return tuple(
        min(
            resolved_categories - 1,
            max(
                0,
                math.floor(
                    _score(
                        criterion_scores[criterion_id],
                        f"criterion_scores.{criterion_id}",
                    )
                    * resolved_categories
                ),
            ),
        )
        for criterion_id in criterion_ids
    )


__all__ = ["project_row_in_order"]
