# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Behavioral contract for L4 remote execution envelopes and loopback executor."""

from __future__ import annotations

import json

import pytest

from fast_mlsirm.remote_exec import (
    CohortMismatchError,
    ExecutionFloatPath,
    FORBIDDEN_REMOTE_JOB_FAMILIES,
    INDEX_SEED_STEP,
    LoopbackExecutor,
    RemoteJobDeliveryState,
    RemoteJobEnvelope,
    RemoteJobFamily,
    RemoteJobOutcome,
    RemoteRunManifest,
    RemoteWorkerProvenance,
    SEED_DERIVATION_RULE,
    admit_remote_job_family,
    derive_index_seed,
    envelope_fingerprint,
)

_SHA = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def _manifest(*, payload_sha256: str = _SHA, source_sha256: str = _SHA_B) -> RemoteRunManifest:
    return RemoteRunManifest(
        schema_version="1.0",
        library_version="0.11.4",
        source_sha256=source_sha256,
        seed_derivation_rule=SEED_DERIVATION_RULE,
        float_path=ExecutionFloatPath.F64,
        payload_sha256=payload_sha256,
        integration_nodes_sha256=_SHA_C,
    )


def _envelope(
    *,
    family: RemoteJobFamily = RemoteJobFamily.MC_REPLICATE,
    unit_index: int = 3,
    base_seed: int = 20260917,
    manifest: RemoteRunManifest | None = None,
) -> RemoteJobEnvelope:
    man = manifest or _manifest()
    return RemoteJobEnvelope(
        run_id="cp3_bootstrap_001",
        family=family,
        unit_index=unit_index,
        base_seed=base_seed,
        payload_ref=man.payload_sha256,
        manifest=man,
    )


def test_derive_index_seed_matches_bifactor_bootstrap_golden_step() -> None:
    """Replicate seeds must match the in-tree bootstrap driver formula."""
    base_seed = 20260917
    for unit_index in (0, 1, 7, 41):
        expected = int((base_seed + unit_index * INDEX_SEED_STEP) & 0xFFFF_FFFF_FFFF_FFFF)
        assert derive_index_seed(base_seed, unit_index) == expected


@pytest.mark.parametrize(
    "family",
    [
        RemoteJobFamily.FIT_RESTART,
        RemoteJobFamily.SCORING_PERSON,
        RemoteJobFamily.MC_REPLICATE,
    ],
)
def test_legal_job_families_accepted(family: RemoteJobFamily) -> None:
    envelope = _envelope(family=family)
    assert envelope.family is family
    assert admit_remote_job_family(family.value) is family


@pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_REMOTE_JOB_FAMILIES))
def test_forbidden_job_families_rejected(forbidden: str) -> None:
    with pytest.raises(ValueError, match="not a legal remote split unit"):
        admit_remote_job_family(forbidden)


def test_envelope_round_trip_json_is_stable() -> None:
    envelope = _envelope(family=RemoteJobFamily.FIT_RESTART, unit_index=2)
    restored = RemoteJobEnvelope.from_dict(envelope.to_dict())
    assert restored == envelope
    assert envelope_fingerprint(envelope) == envelope_fingerprint(restored)
    round_trip = json.loads(json.dumps(envelope.to_dict()))
    assert RemoteJobEnvelope.from_dict(round_trip) == envelope


def test_loopback_executor_runs_batch_with_index_derived_seeds() -> None:
    manifest = _manifest()
    envelopes = tuple(
        _envelope(
            family=RemoteJobFamily.MC_REPLICATE,
            unit_index=index,
            base_seed=99,
            manifest=manifest,
        )
        for index in range(4)
    )
    seen: list[tuple[int, int, RemoteJobFamily]] = []

    def handler(envelope: RemoteJobEnvelope, unit_seed: int) -> int:
        seen.append((envelope.unit_index, unit_seed, envelope.family))
        return unit_seed + envelope.unit_index

    executor = LoopbackExecutor()
    outcomes = executor.run_batch(
        envelopes,
        handler,
        worker_manifest=manifest,
        requested_device="cpu",
        effective_device="cpu",
    )

    assert len(outcomes) == 4
    assert all(
        outcome.delivery_state is RemoteJobDeliveryState.COMPLETED for outcome in outcomes
    )
    for envelope, outcome in zip(envelopes, outcomes, strict=True):
        assert outcome.unit_seed == envelope.unit_seed()
        assert outcome.result == envelope.unit_seed() + envelope.unit_index
        assert outcome.provenance.requested_device == "cpu"
        assert outcome.provenance.effective_device == "cpu"
        assert outcome.provenance.library_version == manifest.library_version
    assert seen == [
        (index, derive_index_seed(99, index), RemoteJobFamily.MC_REPLICATE)
        for index in range(4)
    ]


def test_cohort_gate_rejects_mismatched_worker_manifest() -> None:
    driver_manifest = _manifest(source_sha256=_SHA_B)
    worker_manifest = _manifest(source_sha256=_SHA_C)
    envelope = _envelope(manifest=driver_manifest)
    executor = LoopbackExecutor()

    with pytest.raises(CohortMismatchError, match="incompatible with envelope cohort"):
        executor.run_batch(
            (envelope,),
            lambda env, seed: seed,
            worker_manifest=worker_manifest,
        )


def test_loopback_executor_records_handler_failure_without_aborting_batch() -> None:
    manifest = _manifest()
    good = _envelope(unit_index=0, manifest=manifest)
    bad = _envelope(unit_index=1, manifest=manifest)

    def handler(envelope: RemoteJobEnvelope, unit_seed: int) -> int:
        if envelope.unit_index == 1:
            raise RuntimeError("simulated non-convergence")
        return unit_seed

    outcomes = LoopbackExecutor().run_batch(
        (good, bad),
        handler,
        worker_manifest=manifest,
    )

    assert outcomes[0].delivery_state is RemoteJobDeliveryState.COMPLETED
    assert outcomes[0].result == good.unit_seed()
    assert outcomes[1].delivery_state is RemoteJobDeliveryState.FAILED
    assert outcomes[1].error_message == "simulated non-convergence"
    assert outcomes[1].result is None


def test_payload_ref_must_match_manifest_payload_sha256() -> None:
    manifest = _manifest(payload_sha256=_SHA)
    with pytest.raises(ValueError, match="payload_ref must equal manifest.payload_sha256"):
        RemoteJobEnvelope(
            run_id="run_a",
            family=RemoteJobFamily.SCORING_PERSON,
            unit_index=0,
            base_seed=1,
            payload_ref=_SHA_B,
            manifest=manifest,
        )


def test_remote_job_outcome_validates_failed_state() -> None:
    provenance = RemoteWorkerProvenance(
        hostname="host",
        architecture="arm64",
        operating_system="Darwin",
        library_version="0.11.4",
        source_sha256=_SHA_B,
        requested_device="cpu",
        effective_device="cpu",
        wall_clock_seconds=0.01,
    )
    with pytest.raises(ValueError, match="failed outcomes require"):
        RemoteJobOutcome(
            run_id="run_a",
            unit_index=0,
            unit_seed=1,
            family=RemoteJobFamily.FIT_RESTART,
            delivery_state=RemoteJobDeliveryState.FAILED,
            result=None,
            error_message=None,
            provenance=provenance,
        )
