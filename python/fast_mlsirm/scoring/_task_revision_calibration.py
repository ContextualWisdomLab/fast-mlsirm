"""Task-revision-aware assembly for the governed many-facet handoff.

This module changes only provenance validation and tensor indexing.  Numeric
estimation remains delegated to the existing Rust-backed ``fit_facets`` path.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import calibration as _base
from ._contract_safety import artifact_digest, bounded_values, enum_value
from ._validation import (
    ASSESSMENT_SCHEMA_VERSION,
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
    ScoringRequest as LegacyScoringRequest,
    ScoringResult,
)
from ._task_revision_contract import ScoringRequest


@dataclass(frozen=True)
class ScoringFacetsRatingRecord(_base.ScoringFacetsRatingRecord):
    """Criterion rating bound to an exact logical task and content revision."""

    task_revision_fingerprint: Any = field(default=None, kw_only=True)
    task_family_id: Any = field(default=None, kw_only=True)

    def __post_init__(self, _rating_token: object | None) -> None:
        """Replay legacy rating fields and validate revision provenance."""
        super().__post_init__(_rating_token)
        object.__setattr__(
            self,
            "task_revision_fingerprint",
            fingerprint(
                self.task_revision_fingerprint,
                "task_revision_fingerprint",
                "$.task_revision_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "task_family_id",
            descriptive_identifier(
                self.task_family_id,
                "task_family_id",
                "$.task_family_id",
            ),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical rating content including task revision provenance."""
        legacy = super()._content_dict()
        output: dict[str, Any] = {}
        for key, value in legacy.items():
            output[key] = value
            if key == "task_id":
                output["task_revision_fingerprint"] = (
                    self.task_revision_fingerprint
                )
                output["task_family_id"] = self.task_family_id
        return output


@dataclass(frozen=True)
class ScoringFacetsDesign(_base.ScoringFacetsDesign):
    """Criterion design whose estimator item axis is exact task revision."""

    task_revision_fingerprints: tuple[Any, ...] = field(
        default=(),
        kw_only=True,
    )
    task_family_ids: tuple[Any, ...] = field(default=(), kw_only=True)
    response_task_revision_fingerprints: tuple[Any, ...] = field(
        default=(),
        kw_only=True,
    )

    def __post_init__(self, _design_token: object | None) -> None:
        """Replay legacy sealing and normalize revision-indexed audit fields."""
        super().__post_init__(_design_token)
        revisions = tuple(
            fingerprint(
                value,
                "task_revision_fingerprint",
                f"$.task_revision_fingerprints[{index}]",
            )
            for index, value in enumerate(self.task_revision_fingerprints)
        )
        families = tuple(
            descriptive_identifier(
                value,
                "task_family_id",
                f"$.task_family_ids[{index}]",
            )
            for index, value in enumerate(self.task_family_ids)
        )
        response_revisions = tuple(
            fingerprint(
                value,
                "task_revision_fingerprint",
                f"$.response_task_revision_fingerprints[{index}]",
            )
            for index, value in enumerate(
                self.response_task_revision_fingerprints
            )
        )
        if not (
            len(revisions) == len(self.task_ids) == len(families)
        ):
            raise assessment_error(
                "invalid_task_revision_axis",
                "$.task_revision_fingerprints",
                "task revision, logical task, and family axes must align",
            )
        if len(response_revisions) != len(self.response_ids):
            raise assessment_error(
                "invalid_response_task_revision_axis",
                "$.response_task_revision_fingerprints",
                "every response must retain one task revision identity",
            )
        object.__setattr__(self, "task_revision_fingerprints", revisions)
        object.__setattr__(self, "task_family_ids", families)
        object.__setattr__(
            self,
            "response_task_revision_fingerprints",
            response_revisions,
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return revision-indexed sparse design content without dense arrays."""
        task_revisions = [
            {
                "task_revision_fingerprint": revision,
                "task_id": task_id,
                "task_family_id": task_family_id,
            }
            for revision, task_id, task_family_id in zip(
                self.task_revision_fingerprints,
                self.task_ids,
                self.task_family_ids,
                strict=True,
            )
        ]
        respondent_task_responses = [
            {
                "respondent_id": respondent_id,
                "task_id": task_id,
                "task_revision_fingerprint": revision,
                "response_id": response_id,
                "response_content_fingerprint": response_content_fingerprint,
            }
            for (
                respondent_id,
                task_id,
                revision,
                response_id,
                response_content_fingerprint,
            ) in zip(
                self.response_respondent_ids,
                self.response_task_ids,
                self.response_task_revision_fingerprints,
                self.response_ids,
                self.response_content_fingerprints,
                strict=True,
            )
        ]
        return {
            "schema_version": self.schema_version,
            "assessment_fingerprint": self.assessment_fingerprint,
            "rubric_fingerprint": self.rubric_fingerprint,
            "construct_id": self.construct_id,
            "occasion_id": self.occasion_id,
            "criterion_id": self.criterion_id,
            "category_values": list(self.category_values),
            "respondent_ids": list(self.respondent_ids),
            "task_revision_fingerprints": list(
                self.task_revision_fingerprints
            ),
            "task_ids": list(self.task_ids),
            "task_family_ids": list(self.task_family_ids),
            "task_revisions": task_revisions,
            "response_ids": list(self.response_ids),
            "response_respondent_ids": list(self.response_respondent_ids),
            "response_task_ids": list(self.response_task_ids),
            "response_task_revision_fingerprints": list(
                self.response_task_revision_fingerprints
            ),
            "response_content_fingerprints": list(
                self.response_content_fingerprints
            ),
            "respondent_task_responses": respondent_task_responses,
            "rater_engine_ids": list(self.rater_engine_ids),
            "rater_engine_family_ids": list(self.rater_engine_family_ids),
            "rater_engine_fingerprints": list(
                self.rater_engine_fingerprints
            ),
            "rating_fingerprints": [
                record.rating_fingerprint for record in self.rating_records
            ],
            "respondent_task_connected": self.respondent_task_connected,
            "task_rater_connected": self.task_rater_connected,
            "connected": self.connected,
        }

    def _indexes(self) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        """Return respondent, task-revision, and rater index maps."""
        return (
            {value: index for index, value in enumerate(self.respondent_ids)},
            {
                value: index
                for index, value in enumerate(
                    self.task_revision_fingerprints
                )
            },
            {
                value: index
                for index, value in enumerate(
                    self.rater_engine_fingerprints
                )
            },
        )

    def _dense_array(self, *, zero_based: bool) -> np.ndarray:
        """Materialize one fresh respondent-revision-rater tensor."""
        output = np.full(
            (
                len(self.respondent_ids),
                len(self.task_revision_fingerprints),
                len(self.rater_engine_fingerprints),
            ),
            np.nan,
            dtype=np.float64,
        )
        respondent_index, revision_index, rater_index = self._indexes()
        category_index = {
            value: index for index, value in enumerate(self.category_values)
        }
        for record in self.rating_records:
            if record.status is not ObservationStatus.SCORED:
                continue
            score = record.score_category
            if score is None:  # pragma: no cover - guaranteed by rating record
                raise RuntimeError("scored rating category is unavailable")
            output[
                respondent_index[record.respondent_id],
                revision_index[record.task_revision_fingerprint],
                rater_index[record.engine_fingerprint],
            ] = category_index[score] if zero_based else score
        return output

    def response_states(
        self,
    ) -> tuple[tuple[tuple[ObservationStatus | None, ...], ...], ...]:
        """Return assigned states over respondent-revision-rater cells."""
        states: list[list[list[ObservationStatus | None]]] = [
            [
                [None for _rater in self.rater_engine_fingerprints]
                for _revision in self.task_revision_fingerprints
            ]
            for _respondent in self.respondent_ids
        ]
        respondent_index, revision_index, rater_index = self._indexes()
        for record in self.rating_records:
            states[respondent_index[record.respondent_id]][
                revision_index[record.task_revision_fingerprint]
            ][rater_index[record.engine_fingerprint]] = record.status
        return tuple(
            tuple(tuple(rater_states) for rater_states in revision_states)
            for revision_states in states
        )


def _validated_result_observations(
    *,
    request: LegacyScoringRequest,
    result: ScoringResult,
) -> tuple[Any, ...]:
    """Reuse the baseline criterion-coverage validation."""
    return _base._validated_result_observations(
        request=request,
        result=result,
    )


def build_scoring_facets_rating_records(
    *,
    request: LegacyScoringRequest,
    result: ScoringResult,
    engine: EngineDescriptor,
) -> tuple[ScoringFacetsRatingRecord, ...]:
    """Project one matched schema-1.1 execution into revision ratings."""
    if not isinstance(request, LegacyScoringRequest):
        raise assessment_error(
            "invalid_scoring_request",
            "$.request",
            "request must be a ScoringRequest",
        )
    if not isinstance(request, ScoringRequest):
        raise assessment_error(
            "missing_task_revision_identity",
            "$.request.task_revision_fingerprint",
            "facets calibration requires a schema-1.1 scoring request",
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
    observations = _validated_result_observations(
        request=request,
        result=result,
    )
    expected_observation_fields = (
        (
            "request_fingerprint",
            request.request_fingerprint,
            "calibration_observation_request_mismatch",
        ),
        (
            "engine_fingerprint",
            engine.engine_fingerprint,
            "calibration_observation_engine_mismatch",
        ),
        (
            "assessment_fingerprint",
            request.assessment_fingerprint,
            "calibration_observation_assessment_mismatch",
        ),
        (
            "rubric_fingerprint",
            request.rubric_fingerprint,
            "calibration_observation_rubric_mismatch",
        ),
        (
            "construct_id",
            request.construct_id,
            "calibration_observation_construct_mismatch",
        ),
        (
            "granularity",
            request.granularity,
            "calibration_observation_granularity_mismatch",
        ),
    )
    for index, observation in enumerate(observations):
        path = f"$.result.observations[{index}]"
        for field_name, expected, code in expected_observation_fields:
            if getattr(observation, field_name) != expected:
                raise assessment_error(
                    code,
                    f"{path}.{field_name}",
                    "observation provenance does not match the supplied execution",
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
            response_content_fingerprint=request.response_content_fingerprint,
            task_id=request.task_id,
            occasion_id=request.occasion_id,
            criterion_id=observation.criterion_id,
            engine_id=engine.engine_id,
            engine_family_id=engine.engine_family_id,
            engine_fingerprint=engine.engine_fingerprint,
            status=observation.status,
            score_category=observation.score_category,
            allowed_scores=request.allowed_scores,
            task_revision_fingerprint=request.task_revision_fingerprint,
            task_family_id=request.task_family_id,
            _rating_token=_base._RATING_TOKEN,
        )
        for observation in observations
    )
    keyed = [
        (record.criterion_id, record.rating_fingerprint, record)
        for record in records
    ]
    keyed.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in keyed)


def _identity_provenance(
    records: tuple[ScoringFacetsRatingRecord, ...],
) -> tuple[
    dict[str, tuple[str, str, str, str, str]],
    dict[tuple[str, str], tuple[str, str]],
    dict[str, tuple[str, str]],
    dict[str, tuple[str, str]],
]:
    """Validate response, respondent-revision, task-revision, and rater identity."""
    response_contracts: dict[str, tuple[str, str, str, str, str]] = {}
    respondent_revision_contracts: dict[tuple[str, str], tuple[str, str]] = {}
    task_revision_contracts: dict[str, tuple[str, str]] = {}
    rater_contracts: dict[str, tuple[str, str]] = {}
    for index, record in enumerate(records):
        response_contract = (
            record.respondent_id,
            record.task_id,
            record.task_revision_fingerprint,
            record.task_family_id,
            record.response_content_fingerprint,
        )
        previous_response = response_contracts.get(record.response_id)
        if previous_response is not None and previous_response != response_contract:
            fields = (
                "respondent_id",
                "task_id",
                "task_revision_fingerprint",
                "task_family_id",
                "response_content_fingerprint",
            )
            conflict_field = next(
                field_name
                for field_name, left, right in zip(
                    fields,
                    previous_response,
                    response_contract,
                    strict=True,
                )
                if left != right
            )
            raise assessment_error(
                "response_provenance_conflict",
                f"$.records[{index}].{conflict_field}",
                "one response identity has conflicting provenance",
            )
        response_contracts[record.response_id] = response_contract

        revision_contract = (record.task_id, record.task_family_id)
        previous_revision = task_revision_contracts.get(
            record.task_revision_fingerprint
        )
        if previous_revision is not None and previous_revision != revision_contract:
            conflict_field = (
                "task_id"
                if previous_revision[0] != revision_contract[0]
                else "task_family_id"
            )
            raise assessment_error(
                "task_revision_provenance_conflict",
                f"$.records[{index}].{conflict_field}",
                "one task revision has conflicting logical task provenance",
            )
        task_revision_contracts[record.task_revision_fingerprint] = (
            revision_contract
        )

        cell_key = (record.respondent_id, record.task_revision_fingerprint)
        cell_contract = (
            record.response_id,
            record.response_content_fingerprint,
        )
        previous_cell = respondent_revision_contracts.get(cell_key)
        if previous_cell is not None and previous_cell != cell_contract:
            conflict_field = (
                "response_id"
                if previous_cell[0] != cell_contract[0]
                else "response_content_fingerprint"
            )
            raise assessment_error(
                "respondent_task_response_conflict",
                f"$.records[{index}].{conflict_field}",
                "one respondent-task-revision cell has conflicting response provenance",
            )
        respondent_revision_contracts[cell_key] = cell_contract

        rater_contract = (record.engine_id, record.engine_family_id)
        previous_rater = rater_contracts.get(record.engine_fingerprint)
        if previous_rater is not None and previous_rater != rater_contract:
            raise assessment_error(
                "rater_provenance_conflict",
                f"$.records[{index}].engine_fingerprint",
                "one engine fingerprint has conflicting rater provenance",
            )
        rater_contracts[record.engine_fingerprint] = rater_contract
    return (
        response_contracts,
        respondent_revision_contracts,
        task_revision_contracts,
        rater_contracts,
    )


def _bipartite_connected(
    *,
    left_ids: tuple[str, ...],
    right_ids: tuple[str, ...],
    edges: set[tuple[str, str]],
) -> bool:
    """Return whether one bounded bipartite incidence graph is connected."""
    left_nodes = tuple(f"left:{value}" for value in left_ids)
    right_nodes = tuple(f"right:{value}" for value in right_ids)
    adjacency: dict[str, set[str]] = {
        value: set() for value in (*left_nodes, *right_nodes)
    }
    for left_value, right_value in edges:
        left_node = f"left:{left_value}"
        right_node = f"right:{right_value}"
        adjacency[left_node].add(right_node)
        adjacency[right_node].add(left_node)
    pending = [left_nodes[0]]
    visited: set[str] = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(adjacency[node] - visited)
    return len(visited) == len(adjacency)


def _design_connected(
    *,
    respondent_ids: tuple[str, ...],
    task_revision_fingerprints: tuple[str, ...],
    rater_fingerprints: tuple[str, ...],
    records: tuple[ScoringFacetsRatingRecord, ...],
) -> tuple[bool, bool]:
    """Return respondent-revision and revision-rater connectedness."""
    scored = tuple(
        record for record in records if record.status is ObservationStatus.SCORED
    )
    respondent_task_edges = {
        (record.respondent_id, record.task_revision_fingerprint)
        for record in scored
    }
    task_rater_edges = {
        (record.task_revision_fingerprint, record.engine_fingerprint)
        for record in scored
    }
    return (
        _bipartite_connected(
            left_ids=respondent_ids,
            right_ids=task_revision_fingerprints,
            edges=respondent_task_edges,
        ),
        _bipartite_connected(
            left_ids=task_revision_fingerprints,
            right_ids=rater_fingerprints,
            edges=task_rater_edges,
        ),
    )


def _build_criterion_design(
    records: tuple[ScoringFacetsRatingRecord, ...],
    *,
    require_connected: bool,
) -> ScoringFacetsDesign:
    """Build one bounded criterion design indexed by exact task revision."""
    first = records[0]
    (
        response_contracts,
        _,
        task_revision_contracts,
        rater_contracts,
    ) = _identity_provenance(records)
    cells: set[tuple[str, str, str]] = set()
    observed_respondents: set[str] = set()
    observed_revisions: set[str] = set()
    observed_raters: set[str] = set()
    observed_categories: set[int] = set()
    for index, record in enumerate(records):
        cell = (
            record.respondent_id,
            record.task_revision_fingerprint,
            record.engine_fingerprint,
        )
        if cell in cells:
            raise assessment_error(
                "duplicate_facets_rating_cell",
                f"$.records[{index}]",
                "each respondent-task-revision-rater cell may occur once per criterion",
            )
        cells.add(cell)
        if record.status is ObservationStatus.SCORED:
            score = record.score_category
            if score is None:  # pragma: no cover - guaranteed by rating record
                raise RuntimeError("scored rating category is unavailable")
            observed_respondents.add(record.respondent_id)
            observed_revisions.add(record.task_revision_fingerprint)
            observed_raters.add(record.engine_fingerprint)
            observed_categories.add(score)

    respondent_ids = tuple(sorted({record.respondent_id for record in records}))
    revisions = tuple(sorted(task_revision_contracts))
    rater_fingerprints = tuple(sorted(rater_contracts))
    if len(respondent_ids) < 2:
        raise assessment_error(
            "insufficient_facets_respondents",
            "$.records",
            "facets calibration requires at least two respondents",
        )
    if len(revisions) < 2:
        raise assessment_error(
            "insufficient_facets_tasks",
            "$.records",
            "facets calibration requires at least two task revisions",
        )
    if len(rater_fingerprints) < 2:
        raise assessment_error(
            "insufficient_facets_raters",
            "$.records",
            "facets calibration requires at least two raters",
        )
    if (
        observed_respondents != set(respondent_ids)
        or observed_revisions != set(revisions)
        or observed_raters != set(rater_fingerprints)
    ):
        raise assessment_error(
            "unobserved_facets_level",
            "$.records",
            "every respondent, task revision, and rater must have a scored rating",
        )
    if len(observed_categories) < 2:
        raise assessment_error(
            "single_observed_category",
            "$.records",
            "each criterion must observe at least two score categories",
        )
    dense_cell_count = (
        len(respondent_ids) * len(revisions) * len(rater_fingerprints)
    )
    if dense_cell_count > _base.MAX_SCORING_FACETS_CELLS:
        raise assessment_error(
            "facets_cell_budget_exceeded",
            "$.records",
            "dense respondent-task-revision-rater allocation exceeds the configured cell budget",
        )
    respondent_connected, rater_connected = _design_connected(
        respondent_ids=respondent_ids,
        task_revision_fingerprints=revisions,
        rater_fingerprints=rater_fingerprints,
        records=records,
    )
    if require_connected and not respondent_connected:
        raise assessment_error(
            "unidentified_respondent_task_design",
            "$.records",
            "scored respondent-task-revision graph must be connected",
        )
    if require_connected and not rater_connected:
        raise assessment_error(
            "disconnected_task_rater_design",
            "$.records",
            "scored task-revision-rater graph must be connected",
        )

    response_ids = tuple(sorted(response_contracts))
    return ScoringFacetsDesign(
        assessment_fingerprint=first.assessment_fingerprint,
        rubric_fingerprint=first.rubric_fingerprint,
        construct_id=first.construct_id,
        occasion_id=first.occasion_id,
        criterion_id=first.criterion_id,
        category_values=first.allowed_scores,
        respondent_ids=respondent_ids,
        task_ids=tuple(
            task_revision_contracts[revision][0] for revision in revisions
        ),
        response_ids=response_ids,
        response_respondent_ids=tuple(
            response_contracts[value][0] for value in response_ids
        ),
        response_task_ids=tuple(
            response_contracts[value][1] for value in response_ids
        ),
        response_content_fingerprints=tuple(
            response_contracts[value][4] for value in response_ids
        ),
        rater_engine_ids=tuple(
            rater_contracts[value][0] for value in rater_fingerprints
        ),
        rater_engine_family_ids=tuple(
            rater_contracts[value][1] for value in rater_fingerprints
        ),
        rater_engine_fingerprints=rater_fingerprints,
        rating_records=records,
        respondent_task_connected=respondent_connected,
        task_rater_connected=rater_connected,
        connected=respondent_connected and rater_connected,
        task_revision_fingerprints=revisions,
        task_family_ids=tuple(
            task_revision_contracts[revision][1] for revision in revisions
        ),
        response_task_revision_fingerprints=tuple(
            response_contracts[value][2] for value in response_ids
        ),
        _design_token=_base._DESIGN_TOKEN,
    )


def build_scoring_facets_calibration_bundle(
    records: Iterable[ScoringFacetsRatingRecord],
    *,
    require_connected: bool = True,
) -> _base.ScoringFacetsCalibrationBundle:
    """Assemble revision-aware criterion designs into one calibration bundle."""
    connected_required = strict_boolean(
        require_connected,
        "require_connected",
    )
    raw = bounded_values(
        records,
        "records",
        minimum=1,
        maximum=_base.MAX_SCORING_FACETS_RATINGS,
    )
    for index, record in enumerate(raw):
        if not isinstance(record, ScoringFacetsRatingRecord):
            raise assessment_error(
                "invalid_facets_rating_record",
                f"$.records[{index}]",
                "records must contain ScoringFacetsRatingRecord values",
            )
    keyed = [
        (record.criterion_id, record.rating_fingerprint, record)
        for record in raw
    ]
    keyed.sort(key=lambda item: (item[0], item[1]))
    ordered = tuple(item[2] for item in keyed)
    if len({item[1] for item in keyed}) != len(keyed):
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
            "ratings must share assessment, rubric, construct, occasion, and scale",
        )
    _identity_provenance(ordered)
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
    return _base.ScoringFacetsCalibrationBundle(
        assessment_fingerprint=first.assessment_fingerprint,
        rubric_fingerprint=first.rubric_fingerprint,
        construct_id=first.construct_id,
        occasion_id=first.occasion_id,
        category_values=first.allowed_scores,
        designs=designs,
        _bundle_token=_base._BUNDLE_TOKEN,
    )


def install(base_module: Any) -> None:
    """Install revision-aware classes and assemblers into calibration module."""
    if getattr(base_module, "_task_revision_calibration_installed", False):
        return
    base_module.ScoringFacetsRatingRecord = ScoringFacetsRatingRecord
    base_module.ScoringFacetsDesign = ScoringFacetsDesign
    base_module.build_scoring_facets_rating_records = (
        build_scoring_facets_rating_records
    )
    base_module._identity_provenance = _identity_provenance
    base_module._design_connected = _design_connected
    base_module._build_criterion_design = _build_criterion_design
    base_module.build_scoring_facets_calibration_bundle = (
        build_scoring_facets_calibration_bundle
    )
    base_module._task_revision_calibration_installed = True


__all__ = [
    "ScoringFacetsDesign",
    "ScoringFacetsRatingRecord",
    "build_scoring_facets_calibration_bundle",
    "build_scoring_facets_rating_records",
    "install",
]
