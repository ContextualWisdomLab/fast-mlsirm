"""Governed criterion-level many-facet calibration handoff.

The module converts already governed :class:`ScoringRequest`,
:class:`ScoringResult`, and :class:`EngineDescriptor` values into deterministic
criterion-specific ``persons x tasks x raters`` tensors accepted by the existing
Rust-backed :func:`fast_mlsirm.fit_facets` estimator. It performs no
psychometric arithmetic in Python and does not treat structural validity as
evidence of model fit, reliability, fairness, scoreability, or construct
validity.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from typing import Any

import numpy as np

from ._contract_safety import artifact_digest, bounded_values
from ._validation import (
    ASSESSMENT_SCHEMA_VERSION,
    CanonicalContract,
    assessment_error,
    assessment_schema_version,
    descriptive_identifier,
    fingerprint,
    strict_boolean,
)
from .execution import (
    EngineDescriptor,
    ObservationGranularity,
    ObservationStatus,
    ScoringRequest,
    ScoringResult,
)

MAX_SCORING_FACETS_RATINGS = 2_048
MAX_SCORING_FACETS_CELLS = 1_000_000

_RATING_TOKEN = object()
_DESIGN_TOKEN = object()
_BUNDLE_TOKEN = object()


def _score_values(values: Iterable[int]) -> tuple[int, ...]:
    """Return a non-empty ordered score scale without Boolean coercion."""
    materialized = bounded_values(
        values,
        "allowed_scores",
        minimum=2,
        maximum=64,
    )
    normalized: list[int] = []
    for index, value in enumerate(materialized):
        if isinstance(value, bool) or not isinstance(value, int):
            raise assessment_error(
                "invalid_allowed_scores",
                f"$.allowed_scores[{index}]",
                "allowed scores must be exact integers",
            )
        normalized.append(value)
    output = tuple(normalized)
    if output != tuple(sorted(set(output))):
        raise assessment_error(
            "invalid_allowed_scores",
            "$.allowed_scores",
            "allowed scores must be sorted and unique",
        )
    return output


def _optional_score(value: int | None, status: ObservationStatus) -> int | None:
    """Validate the score/status relationship retained by one rating record."""
    if status is ObservationStatus.SCORED:
        if isinstance(value, bool) or not isinstance(value, int):
            raise assessment_error(
                "missing_rating_score",
                "$.score_category",
                "scored ratings require an exact integer score",
            )
        return value
    if value is not None:
        raise assessment_error(
            "unexpected_rating_score",
            "$.score_category",
            "terminal ratings cannot carry a score",
        )
    return None


@dataclass(frozen=True)
class ScoringFacetsRatingRecord(CanonicalContract):
    """One criterion rating bound to exact request, result, and engine identities."""

    assessment_fingerprint: str
    rubric_fingerprint: str
    construct_id: str
    request_fingerprint: str
    result_fingerprint: str
    observation_fingerprint: str
    respondent_id: str
    response_id: str
    task_id: str
    occasion_id: str
    criterion_id: str
    engine_id: str
    engine_family_id: str
    engine_fingerprint: str
    status: ObservationStatus
    score_category: int | None
    allowed_scores: tuple[int, ...]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _rating_token: InitVar[object | None] = None

    def __post_init__(self, _rating_token: object | None) -> None:
        """Reject direct construction and normalize retained provenance."""
        if _rating_token is not _RATING_TOKEN:
            raise assessment_error(
                "unverified_facets_rating",
                "$",
                "use build_scoring_facets_rating_records",
            )
        for field_name in (
            "assessment_fingerprint",
            "rubric_fingerprint",
            "request_fingerprint",
            "result_fingerprint",
            "observation_fingerprint",
            "engine_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                fingerprint(getattr(self, field_name), field_name),
            )
        for field_name in (
            "construct_id",
            "respondent_id",
            "response_id",
            "task_id",
            "occasion_id",
            "criterion_id",
            "engine_id",
            "engine_family_id",
        ):
            object.__setattr__(
                self,
                field_name,
                descriptive_identifier(getattr(self, field_name), field_name),
            )
        if not isinstance(self.status, ObservationStatus):
            try:
                object.__setattr__(self, "status", ObservationStatus(self.status))
            except Exception:
                raise assessment_error(
                    "invalid_observation_status",
                    "$.status",
                    "status must be a governed observation status",
                ) from None
        object.__setattr__(
            self,
            "score_category",
            _optional_score(self.score_category, self.status),
        )
        scores = _score_values(self.allowed_scores)
        if self.status is ObservationStatus.SCORED and self.score_category not in scores:
            raise assessment_error(
                "rating_score_out_of_range",
                "$.score_category",
                "scored rating is outside the request score scale",
            )
        object.__setattr__(self, "allowed_scores", scores)
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical rating content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "assessment_fingerprint": self.assessment_fingerprint,
            "rubric_fingerprint": self.rubric_fingerprint,
            "construct_id": self.construct_id,
            "request_fingerprint": self.request_fingerprint,
            "result_fingerprint": self.result_fingerprint,
            "observation_fingerprint": self.observation_fingerprint,
            "respondent_id": self.respondent_id,
            "response_id": self.response_id,
            "task_id": self.task_id,
            "occasion_id": self.occasion_id,
            "criterion_id": self.criterion_id,
            "engine_id": self.engine_id,
            "engine_family_id": self.engine_family_id,
            "engine_fingerprint": self.engine_fingerprint,
            "status": self.status.value,
            "score_category": self.score_category,
            "allowed_scores": list(self.allowed_scores),
        }

    @property
    def rating_fingerprint(self) -> str:
        """Return SHA-256 over the exact normalized rating record."""
        return artifact_digest(self)

    @property
    def rating_handle(self) -> str:
        """Return a descriptive 128-bit public rating handle."""
        return f"scoring_facets_rating_{self.rating_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical rating content and deterministic identities."""
        return {
            **self._content_dict(),
            "rating_handle": self.rating_handle,
            "rating_fingerprint": self.rating_fingerprint,
        }


@dataclass(frozen=True)
class ScoringFacetsDesign(CanonicalContract):
    """One criterion-specific sparse design for the Rust-backed facets estimator.

    The person axis is ``response_id`` rather than raw response content. The item
    axis is ``task_id`` (for essay scoring, the prompt identity), and the rater
    axis is the exact engine fingerprint, so model or prompt-template revisions
    cannot be silently pooled. Original rubric scores are order-preservingly
    mapped to zero-based categories only at the estimator boundary.
    """

    assessment_fingerprint: str
    rubric_fingerprint: str
    construct_id: str
    occasion_id: str
    criterion_id: str
    category_values: tuple[int, ...]
    response_ids: tuple[str, ...]
    respondent_ids: tuple[str, ...]
    response_task_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    rater_engine_ids: tuple[str, ...]
    rater_engine_family_ids: tuple[str, ...]
    rater_engine_fingerprints: tuple[str, ...]
    rating_records: tuple[ScoringFacetsRatingRecord, ...]
    connected: bool
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _design_token: InitVar[object | None] = None

    def __post_init__(self, _design_token: object | None) -> None:
        """Reject direct construction outside the validated bundle assembler."""
        if _design_token is not _DESIGN_TOKEN:
            raise assessment_error(
                "unverified_facets_design",
                "$",
                "use build_scoring_facets_calibration_bundle",
            )
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical design content without materializing dense arrays."""
        return {
            "schema_version": self.schema_version,
            "assessment_fingerprint": self.assessment_fingerprint,
            "rubric_fingerprint": self.rubric_fingerprint,
            "construct_id": self.construct_id,
            "occasion_id": self.occasion_id,
            "criterion_id": self.criterion_id,
            "category_values": list(self.category_values),
            "response_ids": list(self.response_ids),
            "respondent_ids": list(self.respondent_ids),
            "response_task_ids": list(self.response_task_ids),
            "task_ids": list(self.task_ids),
            "rater_engine_ids": list(self.rater_engine_ids),
            "rater_engine_family_ids": list(self.rater_engine_family_ids),
            "rater_engine_fingerprints": list(self.rater_engine_fingerprints),
            "rating_fingerprints": [
                record.rating_fingerprint for record in self.rating_records
            ],
            "connected": self.connected,
        }

    @property
    def design_fingerprint(self) -> str:
        """Return SHA-256 over the complete sparse calibration design."""
        return artifact_digest(self)

    @property
    def design_handle(self) -> str:
        """Return a descriptive 128-bit public design handle."""
        return f"scoring_facets_design_{self.design_fingerprint[:32]}"

    def _cell_indexes(self) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        """Return deterministic response, task, and rater index maps."""
        return (
            {value: index for index, value in enumerate(self.response_ids)},
            {value: index for index, value in enumerate(self.task_ids)},
            {
                value: index
                for index, value in enumerate(self.rater_engine_fingerprints)
            },
        )

    def responses_array(self) -> np.ndarray:
        """Return a fresh zero-based float tensor accepted by ``fit_facets``."""
        output = np.full(
            (
                len(self.response_ids),
                len(self.task_ids),
                len(self.rater_engine_fingerprints),
            ),
            np.nan,
            dtype=np.float64,
        )
        response_index, task_index, rater_index = self._cell_indexes()
        category_index = {
            value: index for index, value in enumerate(self.category_values)
        }
        for record in self.rating_records:
            if record.status is ObservationStatus.SCORED:
                output[
                    response_index[record.response_id],
                    task_index[record.task_id],
                    rater_index[record.engine_fingerprint],
                ] = category_index[record.score_category]
        return output

    def original_scores_array(self) -> np.ndarray:
        """Return a fresh float tensor on the original rubric score scale."""
        output = np.full(
            (
                len(self.response_ids),
                len(self.task_ids),
                len(self.rater_engine_fingerprints),
            ),
            np.nan,
            dtype=np.float64,
        )
        response_index, task_index, rater_index = self._cell_indexes()
        for record in self.rating_records:
            if record.status is ObservationStatus.SCORED:
                output[
                    response_index[record.response_id],
                    task_index[record.task_id],
                    rater_index[record.engine_fingerprint],
                ] = record.score_category
        return output

    def response_states(
        self,
    ) -> tuple[tuple[tuple[ObservationStatus | None, ...], ...], ...]:
        """Return exact observed/terminal states with ``None`` for absent cells."""
        states: list[list[list[ObservationStatus | None]]] = [
            [
                [None for _rater in self.rater_engine_fingerprints]
                for _task in self.task_ids
            ]
            for _response in self.response_ids
        ]
        response_index, task_index, rater_index = self._cell_indexes()
        for record in self.rating_records:
            states[response_index[record.response_id]][task_index[record.task_id]][
                rater_index[record.engine_fingerprint]
            ] = record.status
        return tuple(
            tuple(tuple(rater_states) for rater_states in task_states)
            for task_states in states
        )

    def to_fit_facets_kwargs(self) -> dict[str, Any]:
        """Return copied arguments accepted by the Rust-backed ``fit_facets``."""
        return {
            "responses": self.responses_array(),
            "n_cat": len(self.category_values),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return canonical sparse design content and deterministic identities."""
        return {
            **self._content_dict(),
            "design_handle": self.design_handle,
            "design_fingerprint": self.design_fingerprint,
        }


@dataclass(frozen=True)
class ScoringFacetsCalibrationBundle(CanonicalContract):
    """Content-addressed collection of criterion-specific facets designs."""

    assessment_fingerprint: str
    rubric_fingerprint: str
    construct_id: str
    occasion_id: str
    category_values: tuple[int, ...]
    designs: tuple[ScoringFacetsDesign, ...]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _bundle_token: InitVar[object | None] = None

    def __post_init__(self, _bundle_token: object | None) -> None:
        """Reject direct construction outside the validated assembler."""
        if _bundle_token is not _BUNDLE_TOKEN:
            raise assessment_error(
                "unverified_facets_bundle",
                "$",
                "use build_scoring_facets_calibration_bundle",
            )
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    @property
    def criterion_ids(self) -> tuple[str, ...]:
        """Return criterion identifiers in design order."""
        return tuple(design.criterion_id for design in self.designs)

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical bundle content without dense response tensors."""
        return {
            "schema_version": self.schema_version,
            "assessment_fingerprint": self.assessment_fingerprint,
            "rubric_fingerprint": self.rubric_fingerprint,
            "construct_id": self.construct_id,
            "occasion_id": self.occasion_id,
            "category_values": list(self.category_values),
            "design_fingerprints": [
                design.design_fingerprint for design in self.designs
            ],
        }

    @property
    def bundle_fingerprint(self) -> str:
        """Return SHA-256 over the exact ordered design collection."""
        return artifact_digest(self)

    @property
    def bundle_handle(self) -> str:
        """Return a descriptive 128-bit public bundle handle."""
        return f"scoring_facets_bundle_{self.bundle_fingerprint[:32]}"

    def design_by_criterion(self) -> dict[str, ScoringFacetsDesign]:
        """Return a fresh criterion-to-design mapping."""
        return {design.criterion_id: design for design in self.designs}

    def to_dict(self) -> dict[str, Any]:
        """Return canonical bundle content and deterministic identities."""
        return {
            **self._content_dict(),
            "criterion_ids": list(self.criterion_ids),
            "bundle_handle": self.bundle_handle,
            "bundle_fingerprint": self.bundle_fingerprint,
        }


def build_scoring_facets_rating_records(
    *,
    request: ScoringRequest,
    result: ScoringResult,
    engine: EngineDescriptor,
) -> tuple[ScoringFacetsRatingRecord, ...]:
    """Project one governed criterion-level result into immutable rating records."""
    if not isinstance(request, ScoringRequest):
        raise assessment_error(
            "invalid_scoring_request",
            "$.request",
            "request must be a ScoringRequest",
        )
    if not isinstance(result, ScoringResult):
        raise assessment_error(
            "invalid_scoring_result",
            "$.result",
            "result must be a ScoringResult",
        )
    if not isinstance(engine, EngineDescriptor):
        raise assessment_error(
            "invalid_engine_descriptor",
            "$.engine",
            "engine must be an EngineDescriptor",
        )
    if request.granularity is not ObservationGranularity.CRITERION_LEVEL:
        raise assessment_error(
            "unsupported_calibration_granularity",
            "$.request.granularity",
            "facets calibration requires criterion-level observations",
        )
    if result.granularity is not ObservationGranularity.CRITERION_LEVEL:
        raise assessment_error(
            "unsupported_calibration_granularity",
            "$.result.granularity",
            "facets calibration requires criterion-level observations",
        )
    if result.request_fingerprint != request.request_fingerprint:
        raise assessment_error(
            "calibration_request_result_mismatch",
            "$.result.request_fingerprint",
            "result is not bound to the supplied scoring request",
        )
    if result.engine_fingerprint != engine.engine_fingerprint:
        raise assessment_error(
            "calibration_engine_result_mismatch",
            "$.result.engine_fingerprint",
            "result is not bound to the supplied engine descriptor",
        )
    records = tuple(
        ScoringFacetsRatingRecord(
            assessment_fingerprint=observation.assessment_fingerprint,
            rubric_fingerprint=observation.rubric_fingerprint,
            construct_id=observation.construct_id,
            request_fingerprint=request.request_fingerprint,
            result_fingerprint=result.result_fingerprint,
            observation_fingerprint=observation.observation_fingerprint,
            respondent_id=request.respondent_id,
            response_id=request.response_id,
            task_id=request.task_id,
            occasion_id=request.occasion_id,
            criterion_id=observation.criterion_id or "",
            engine_id=engine.engine_id,
            engine_family_id=engine.engine_family_id,
            engine_fingerprint=engine.engine_fingerprint,
            status=observation.status,
            score_category=observation.score_category,
            allowed_scores=request.allowed_scores,
            _rating_token=_RATING_TOKEN,
        )
        for observation in result.observations
    )
    return tuple(sorted(records, key=lambda record: record.rating_fingerprint))


def _design_connected(
    task_ids: tuple[str, ...],
    rater_fingerprints: tuple[str, ...],
    records: tuple[ScoringFacetsRatingRecord, ...],
) -> bool:
    """Return connectedness of the observed task-rater bipartite graph."""
    task_nodes = tuple(f"task:{value}" for value in task_ids)
    rater_nodes = tuple(f"rater:{value}" for value in rater_fingerprints)
    adjacency: dict[str, set[str]] = {
        value: set() for value in (*task_nodes, *rater_nodes)
    }
    for record in records:
        if record.status is ObservationStatus.SCORED:
            task_node = f"task:{record.task_id}"
            rater_node = f"rater:{record.engine_fingerprint}"
            adjacency[task_node].add(rater_node)
            adjacency[rater_node].add(task_node)
    pending = [task_nodes[0]]
    visited: set[str] = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(adjacency[node] - visited)
    return len(visited) == len(adjacency)


def _build_criterion_design(
    records: tuple[ScoringFacetsRatingRecord, ...],
    *,
    require_connected: bool,
) -> ScoringFacetsDesign:
    """Build one bounded criterion-specific response tensor contract."""
    first = records[0]
    response_contracts: dict[str, tuple[str, str]] = {}
    rater_contracts: dict[str, tuple[str, str]] = {}
    cells: set[tuple[str, str, str]] = set()
    observed_responses: set[str] = set()
    observed_tasks: set[str] = set()
    observed_raters: set[str] = set()
    observed_categories: set[int] = set()

    for index, record in enumerate(records):
        response_contract = (record.respondent_id, record.task_id)
        previous_response = response_contracts.get(record.response_id)
        if previous_response is not None and previous_response != response_contract:
            raise assessment_error(
                "response_provenance_conflict",
                f"$.records[{index}].response_id",
                "one response identity is bound to conflicting respondent or task provenance",
            )
        response_contracts[record.response_id] = response_contract

        rater_contract = (record.engine_id, record.engine_family_id)
        previous_rater = rater_contracts.get(record.engine_fingerprint)
        if previous_rater is not None and previous_rater != rater_contract:
            raise assessment_error(
                "rater_provenance_conflict",
                f"$.records[{index}].engine_fingerprint",
                "one engine fingerprint is bound to conflicting rater provenance",
            )
        rater_contracts[record.engine_fingerprint] = rater_contract

        cell = (record.response_id, record.task_id, record.engine_fingerprint)
        if cell in cells:
            raise assessment_error(
                "duplicate_facets_rating_cell",
                f"$.records[{index}]",
                "each response-task-rater cell may occur only once per criterion",
            )
        cells.add(cell)
        if record.status is ObservationStatus.SCORED:
            if record.score_category is None:  # pragma: no cover - record invariant
                raise RuntimeError("scored rating category is unavailable")
            observed_responses.add(record.response_id)
            observed_tasks.add(record.task_id)
            observed_raters.add(record.engine_fingerprint)
            observed_categories.add(record.score_category)

    response_ids = tuple(sorted(response_contracts))
    task_ids = tuple(sorted({record.task_id for record in records}))
    rater_fingerprints = tuple(sorted(rater_contracts))
    if len(response_ids) < 2:
        raise assessment_error(
            "insufficient_facets_responses",
            "$.records",
            "facets calibration requires at least two scored responses",
        )
    if len(rater_fingerprints) < 2:
        raise assessment_error(
            "insufficient_facets_raters",
            "$.records",
            "facets calibration requires at least two raters",
        )
    if (
        observed_responses != set(response_ids)
        or observed_tasks != set(task_ids)
        or observed_raters != set(rater_fingerprints)
    ):
        raise assessment_error(
            "unobserved_facets_level",
            "$.records",
            "every response, task, and rater must have at least one scored rating",
        )
    if len(observed_categories) < 2:
        raise assessment_error(
            "single_observed_category",
            "$.records",
            "each criterion design must observe at least two score categories",
        )

    dense_cell_count = len(response_ids) * len(task_ids) * len(rater_fingerprints)
    if dense_cell_count > MAX_SCORING_FACETS_CELLS:
        raise assessment_error(
            "facets_cell_budget_exceeded",
            "$.records",
            "dense facets design exceeds the configured cell budget",
        )

    connected = _design_connected(task_ids, rater_fingerprints, records)
    if require_connected and not connected:
        raise assessment_error(
            "disconnected_facets_design",
            "$.records",
            "observed task-rater graph is disconnected",
        )

    respondent_ids = tuple(
        response_contracts[response_id][0] for response_id in response_ids
    )
    response_task_ids = tuple(
        response_contracts[response_id][1] for response_id in response_ids
    )
    return ScoringFacetsDesign(
        assessment_fingerprint=first.assessment_fingerprint,
        rubric_fingerprint=first.rubric_fingerprint,
        construct_id=first.construct_id,
        occasion_id=first.occasion_id,
        criterion_id=first.criterion_id,
        category_values=first.allowed_scores,
        response_ids=response_ids,
        respondent_ids=respondent_ids,
        response_task_ids=response_task_ids,
        task_ids=task_ids,
        rater_engine_ids=tuple(
            rater_contracts[value][0] for value in rater_fingerprints
        ),
        rater_engine_family_ids=tuple(
            rater_contracts[value][1] for value in rater_fingerprints
        ),
        rater_engine_fingerprints=rater_fingerprints,
        rating_records=records,
        connected=connected,
        _design_token=_DESIGN_TOKEN,
    )


def build_scoring_facets_calibration_bundle(
    records: Iterable[ScoringFacetsRatingRecord],
    *,
    require_connected: bool = True,
) -> ScoringFacetsCalibrationBundle:
    """Assemble governed ratings into criterion-specific Rust facets handoffs.

    All records must share one assessment, rubric, construct, occasion, and
    ordered score scale. Criterion dimensions are calibrated separately rather
    than averaged. This baseline estimates response proficiency, task
    difficulty, and rater severity within each criterion; it does not yet claim a
    cross-criterion general factor or separate criterion effects in one joint
    likelihood.
    """
    connected_required = strict_boolean(require_connected, "require_connected")
    materialized = bounded_values(
        records,
        "records",
        minimum=1,
        maximum=MAX_SCORING_FACETS_RATINGS,
    )
    for index, record in enumerate(materialized):
        if not isinstance(record, ScoringFacetsRatingRecord):
            raise assessment_error(
                "invalid_facets_rating_record",
                f"$.records[{index}]",
                "records must contain ScoringFacetsRatingRecord values",
            )
    ordered = tuple(sorted(materialized, key=lambda value: value.rating_fingerprint))
    if len({record.rating_fingerprint for record in ordered}) != len(ordered):
        raise assessment_error(
            "duplicate_facets_rating_record",
            "$.records",
            "rating records must be unique",
        )

    contract_keys = {
        (
            record.assessment_fingerprint,
            record.rubric_fingerprint,
            record.construct_id,
            record.occasion_id,
            record.allowed_scores,
        )
        for record in ordered
    }
    if len(contract_keys) != 1:
        raise assessment_error(
            "mixed_facets_calibration_contract",
            "$.records",
            "ratings must share one assessment, rubric, construct, occasion, and score scale",
        )

    grouped: dict[str, list[ScoringFacetsRatingRecord]] = {}
    for record in ordered:
        grouped.setdefault(record.criterion_id, []).append(record)
    designs = tuple(
        _build_criterion_design(
            tuple(grouped[criterion_id]),
            require_connected=connected_required,
        )
        for criterion_id in sorted(grouped)
    )
    first = ordered[0]
    return ScoringFacetsCalibrationBundle(
        assessment_fingerprint=first.assessment_fingerprint,
        rubric_fingerprint=first.rubric_fingerprint,
        construct_id=first.construct_id,
        occasion_id=first.occasion_id,
        category_values=first.allowed_scores,
        designs=designs,
        _bundle_token=_BUNDLE_TOKEN,
    )


def fit_scoring_facets_design(
    design: ScoringFacetsDesign,
    *,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
    allow_disconnected: bool = False,
):
    """Fit one design by delegating all numeric work to Rust ``fit_facets``."""
    if not isinstance(design, ScoringFacetsDesign):
        raise assessment_error(
            "invalid_facets_design",
            "$.design",
            "design must be a ScoringFacetsDesign",
        )
    disconnected_allowed = strict_boolean(
        allow_disconnected,
        "allow_disconnected",
    )
    if not design.connected and not disconnected_allowed:
        raise assessment_error(
            "disconnected_facets_design",
            "$.design.connected",
            "disconnected designs require explicit diagnostic opt-in",
        )
    from fast_mlsirm.facets import fit_facets

    return fit_facets(
        **design.to_fit_facets_kwargs(),
        q_theta=q_theta,
        max_iter=max_iter,
        tol=tol,
    )


def fit_scoring_facets_bundle(
    bundle: ScoringFacetsCalibrationBundle,
    *,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
    allow_disconnected: bool = False,
) -> dict[str, Any]:
    """Fit every criterion design through the existing Rust-backed estimator."""
    if not isinstance(bundle, ScoringFacetsCalibrationBundle):
        raise assessment_error(
            "invalid_facets_bundle",
            "$.bundle",
            "bundle must be a ScoringFacetsCalibrationBundle",
        )
    return {
        design.criterion_id: fit_scoring_facets_design(
            design,
            q_theta=q_theta,
            max_iter=max_iter,
            tol=tol,
            allow_disconnected=allow_disconnected,
        )
        for design in bundle.designs
    }


__all__ = [
    "MAX_SCORING_FACETS_CELLS",
    "MAX_SCORING_FACETS_RATINGS",
    "ScoringFacetsCalibrationBundle",
    "ScoringFacetsDesign",
    "ScoringFacetsRatingRecord",
    "build_scoring_facets_calibration_bundle",
    "build_scoring_facets_rating_records",
    "fit_scoring_facets_bundle",
    "fit_scoring_facets_design",
]
