"""Retry failure-cache regressions using a recording Streams client.

These cases do not establish real-server pending-entry or concurrency behavior.
"""
from dataclasses import replace

import pytest

from fast_mlsirm.remote_exec import RemoteJobDeliveryState
from test_remote_exec_valkey import (
    FakeValkey, _backend, _envelope, _manifest, _MC_PAYLOAD,
    _record, _failed_outcome, _completed_outcome,
)


@pytest.mark.parametrize("pending", [False, True])
def test_retry_ignores_cached_failure_until_delayed_success(pending):
    class DelayedSuccess(FakeValkey):
        def __init__(self):
            super().__init__()
            self.reply = None
            self.reads = 0

        def xadd(self, name, fields):
            rid = super().xadd(name, fields)
            if name == "jobs":
                self.reply = ("99-0", _record(_completed_outcome(3, {"current": "success"})))
            return rid

        def xreadgroup(self, *args, **kwargs):
            self.reads += 1
            if self.reads == 1:
                return []
            if self.reply:
                reply, self.reply = self.reply, None
                return [("outcomes", [reply])]
            return []

    client = DelayedSuccess()
    backend = _backend(client)
    failed = _failed_outcome(3)
    backend._outcomes._accept_records([("1-0", _record(failed))])
    stored_failure = client.hashes["outcomes:failed"][failed.envelope_fingerprint]
    if pending:
        client.pending.append(("2-0", _record(failed)))
    outcome = backend.run_batch((_envelope(unit_index=3),), worker_manifest=_manifest(), payload=_MC_PAYLOAD)[0]
    assert outcome.delivery_state is RemoteJobDeliveryState.COMPLETED
    assert outcome.result == {"current": "success"}
    assert client.reply is None
    assert len(client.streams["jobs"]) == 1
    assert client.hashes["outcomes:failed"][failed.envelope_fingerprint] == stored_failure
    assert "99-0" in client.acked
    if pending:
        assert "2-0" in client.acked


@pytest.mark.parametrize("reply_kind", ["none", "identical-failure", "different-failure"])
def test_retry_does_not_attribute_ambiguous_failure_to_new_dispatch(reply_kind):
    old = _failed_outcome(3)

    class FailureReply(FakeValkey):
        def xadd(self, name, fields):
            rid = super().xadd(name, fields)
            if name == "jobs" and reply_kind != "none":
                failure = old if reply_kind == "identical-failure" else replace(old, error_message="new failure")
                self.streams.setdefault("outcomes", []).append(("9-0", _record(failure)))
            return rid

    client = FailureReply()
    backend = _backend(client)
    backend._outcomes._accept_records([("1-0", _record(old))])
    before = dict(client.hashes["outcomes:failed"])
    with pytest.raises(TimeoutError, match="no Valkey outcome available"):
        backend.run_batch((_envelope(unit_index=3),), worker_manifest=_manifest(), payload=_MC_PAYLOAD)
    assert client.hashes["outcomes:failed"] == before
    assert len(client.streams["jobs"]) == 1
    if reply_kind != "none":
        assert "9-0" in client.acked


def test_first_dispatch_accepts_failure_persisted_during_publication():
    client = FakeValkey()
    backend = _backend(client)
    original_xadd = client.xadd
    failed = _failed_outcome(3)

    def synchronous_worker(name, fields):
        rid = original_xadd(name, fields)
        if name == "jobs":
            backend._outcomes._accept_records([("7-0", _record(failed))])
        return rid

    client.xadd = synchronous_worker
    outcome = backend.run_batch(
        (_envelope(unit_index=3),), worker_manifest=_manifest(), payload=_MC_PAYLOAD
    )[0]
    assert outcome == failed
    assert client.acked == ["7-0"]


def test_store_wait_without_new_dispatch_keeps_cached_failure_terminal():
    import time

    client = FakeValkey()
    backend = _backend(client)
    failed = _failed_outcome(3)
    backend._outcomes._accept_records([("1-0", _record(failed))])
    found = backend._outcomes.wait_for_terminal(
        (failed.envelope_fingerprint,), deadline=time.monotonic() + 0.2
    )
    assert found == {failed.envelope_fingerprint: failed}
    assert "jobs" not in client.streams


def test_failure_exclusion_applies_only_to_named_fingerprint():
    import time

    client = FakeValkey()
    backend = _backend(client)
    excluded = _failed_outcome(3)
    terminal = _failed_outcome(4)
    backend._outcomes._accept_records([
        ("1-0", _record(excluded)), ("2-0", _record(terminal)),
    ])
    found = backend._outcomes.wait_for_terminal(
        (excluded.envelope_fingerprint, terminal.envelope_fingerprint),
        deadline=time.monotonic() + 0.05,
        prior_failed_fingerprints=(excluded.envelope_fingerprint,),
    )
    assert found == {terminal.envelope_fingerprint: terminal}
    assert len(client.hashes["outcomes:failed"]) == 2


def test_excluded_failure_does_not_hide_success_persisted_at_deadline(monkeypatch):
    import time

    client = FakeValkey()
    backend = _backend(client)
    failed = _failed_outcome(3)
    success = _completed_outcome(3, {"late": True})
    backend._outcomes._accept_records([("1-0", _record(failed))])

    def persist_after_deadline(*, deadline):
        backend._outcomes._accept_records([("99-0", _record(success))])
        time.sleep(max(0.0, deadline - time.monotonic()) + 0.005)

    monkeypatch.setattr(backend._outcomes, "_drain", persist_after_deadline)
    found = backend._outcomes.wait_for_terminal(
        (failed.envelope_fingerprint,), deadline=time.monotonic() + 0.02,
        prior_failed_fingerprints=(failed.envelope_fingerprint,),
    )
    assert found == {success.envelope_fingerprint: success}
    assert client.acked == ["1-0", "99-0"]


def test_malformed_cached_failure_rejected_before_any_new_job():
    client = FakeValkey()
    backend = _backend(client)
    failed = _failed_outcome(3)
    client.hashes["outcomes:failed"] = {failed.envelope_fingerprint: "not-json"}
    with pytest.raises(ValueError):
        backend.run_batch(
            (_envelope(unit_index=3),), worker_manifest=_manifest(), payload=_MC_PAYLOAD
        )
    assert "jobs" not in client.streams
    assert client.hashes["outcomes:failed"] == {failed.envelope_fingerprint: "not-json"}
    assert client.acked == []
