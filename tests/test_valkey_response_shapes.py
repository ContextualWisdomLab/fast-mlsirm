"""Stream response containers must not change outcome admission or ACK order."""
import pytest

from fast_mlsirm.remote_exec import ValkeyStreamsOutcomeStore
from test_remote_exec_valkey import FakeValkey, _completed_outcome, _record


@pytest.mark.parametrize("encoded", [False, True])
def test_native_resp3_fresh_success_is_committed_and_acked(encoded):
    outcome = _completed_outcome(27, {"response": "native-resp3"})
    fields = _record(outcome)
    stream = "outcomes"
    record_id = "27-0"
    if encoded:
        stream = stream.encode()
        record_id = record_id.encode()
        fields = {key.encode(): value.encode() for key, value in fields.items()}

    class NativeRESP3(FakeValkey):
        served = False

        def xreadgroup(self, groupname, consumername, streams, count, block=None):
            self.xreadgroup_blocks.append(block)
            if self.served:
                return {}
            self.served = True
            # redis-py's native RESP3 parser wraps the entry list once.
            return {stream: [[(record_id, fields)]]}

    client = NativeRESP3()
    store = ValkeyStreamsOutcomeStore(
        client, stream="outcomes", group="drivers", consumer="d1", block_ms=0
    )
    assert store.committed_success(outcome.envelope_fingerprint) == outcome
    assert client.acked == [record_id]
    assert len(client.hashes["outcomes:committed"]) == 1
    assert all(block is None for block in client.xreadgroup_blocks)


@pytest.mark.parametrize("shape", ["resp2", "native-resp3", "unified"])
@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("count", [1, 2])
def test_supported_response_shapes_preserve_batch_admission(shape, encoded, count):
    outcomes = [_completed_outcome(30 + index, {"index": index}) for index in range(count)]
    entries = [(f"{index + 1}-0", _record(outcome)) for index, outcome in enumerate(outcomes)]
    stream = "outcomes"
    if encoded:
        stream = stream.encode()
        entries = [
            (rid.encode(), {key.encode(): value.encode() for key, value in fields.items()})
            for rid, fields in entries
        ]

    class ShapedClient(FakeValkey):
        served = False

        def xreadgroup(self, *args, **kwargs):
            if self.served:
                return [] if shape == "resp2" else {}
            self.served = True
            if shape == "resp2":
                return [(stream, entries)]
            return {stream: [entries] if shape == "native-resp3" else entries}

    client = ShapedClient()
    store = ValkeyStreamsOutcomeStore(
        client, stream="outcomes", group="drivers", consumer="d1", block_ms=0
    )
    assert store.consume_available() == {outcome.envelope_fingerprint: outcome for outcome in outcomes}
    assert client.acked == [rid for rid, _ in entries]
    assert len(client.hashes["outcomes:committed"]) == count


@pytest.mark.parametrize("shape", ["resp2", "native-resp3", "unified"])
@pytest.mark.parametrize("encoded", [False, True])
def test_malformed_outcome_stays_unacked_for_each_response_shape(shape, encoded):
    fingerprint = _completed_outcome(50, {}).envelope_fingerprint
    stream, rid = "outcomes", "1-0"
    fields = {"fingerprint": fingerprint, "outcome": "not-json"}
    if encoded:
        stream, rid = stream.encode(), rid.encode()
        fields = {key.encode(): value.encode() for key, value in fields.items()}
    entries = [(rid, fields)]

    class MalformedClient(FakeValkey):
        def xreadgroup(self, *args, **kwargs):
            if shape == "resp2":
                return [(stream, entries)]
            return {stream: [entries] if shape == "native-resp3" else entries}

    client = MalformedClient()
    store = ValkeyStreamsOutcomeStore(
        client, stream="outcomes", group="drivers", consumer="d1", block_ms=0
    )
    with pytest.raises(ValueError):
        store.consume_available()
    assert client.acked == []
    assert client.hashes == {}
