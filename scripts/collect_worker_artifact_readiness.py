"""Target-host artifact-file collector; not numerical or remote attestation.

The independently owned ``fast_mlsirm._artifact_readiness`` helper must already
be available in the caller's trusted package environment. This consumer does
not copy that helper, infer an approval policy, or inspect a remote host.
Startup observations and approval remain caller responsibilities. The caller
must supervise a hard deadline around the helper's cooperative filesystem I/O.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import sys

_MAX_REQUEST_BYTES = 1 << 20
_REQUEST_FIELDS = {
    "policy_json", "pinned_policy_digest", "installation_root",
    "startup_observation", "deadline_monotonic",
}


def _unique_fields(pairs):
    """Reject duplicate JSON members instead of accepting last-key wins."""
    fields = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("duplicate collector field")
        fields[key] = value
    return fields


def _invalid_constant(value):
    """Refuse nonfinite JSON constants at the collector input boundary."""
    raise ValueError("nonfinite collector field")


def collect_worker_artifact_readiness(
    policy_json,
    *,
    pinned_policy_digest,
    installation_root,
    startup_observation,
    deadline_monotonic,
) -> dict[str, object]:
    """Forward explicit host inputs to the existing artifact-only verifier.

    Basis: the internal ``worker_artifact_policy/1.0`` and
    ``host_immutable_install/1.0`` contracts in
    ``fast_mlsirm._artifact_readiness.verify_artifact_readiness``. No scientific
    inference, loaded-binary identity, device measurement or authentication is
    derived from a file-readiness result. The helper retains record validation,
    refusal states and caller-owned monotonic deadline semantics unchanged.
    """
    from fast_mlsirm._artifact_readiness import verify_artifact_readiness

    artifact = verify_artifact_readiness(
        policy_json,
        pinned_policy_digest=pinned_policy_digest,
        installation_root=installation_root,
        startup_observation=startup_observation,
        deadline_monotonic=deadline_monotonic,
    )
    return {
        "schema": "worker_artifact_readiness_report/1.0",
        "artifact": asdict(artifact),
    }


def main() -> int:
    """Read one explicit request and emit artifact-only JSON on this host.

    Exit 0 means files-only ready, 1 means unknown or mismatch, and 2 means
    invalid input or an unavailable approved helper. No state is inferred from
    an outcome, wheel label or remote hostname. The supplied monotonic deadline
    belongs to this process's host clock; it is not a transportable wall time.
    The caller must bound stdin reading and filesystem I/O externally.
    """
    try:
        stream = getattr(sys.stdin, "buffer", sys.stdin)
        raw = stream.read(_MAX_REQUEST_BYTES + 1)
        encoded = raw.encode("utf-8") if isinstance(raw, str) else raw
        if len(encoded) > _MAX_REQUEST_BYTES:
            raise ValueError("collector request exceeds limit")
        request = json.loads(
            encoded,
            object_pairs_hook=_unique_fields,
            parse_constant=_invalid_constant,
        )
        if type(request) is not dict or set(request) != _REQUEST_FIELDS:
            raise ValueError("invalid collector request fields")
        report = collect_worker_artifact_readiness(**request)
        rendered = json.dumps(report, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        print("invalid artifact-readiness request", file=sys.stderr)
        return 2
    except ImportError:
        print("approved artifact-readiness helper unavailable", file=sys.stderr)
        return 2
    print(rendered)
    return 0 if report["artifact"]["state"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
