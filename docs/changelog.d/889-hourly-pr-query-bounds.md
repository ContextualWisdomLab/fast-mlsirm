# Bounded hourly PR queue capture

## Fixed

- Split hourly open-PR identity enumeration from per-PR nested evidence capture so large queues no longer exceed GitHub GraphQL resource limits or publish a false zero-PR snapshot.
- Preserve fail-closed review, merge-state, label, changed-file, body, history, and exact default-branch evidence while excluding pull requests that close during capture.

## Security

- Keep GitHub subprocesses bounded, retry only explicit HTTP 502/503/504 responses, reject malformed or duplicate PR identities, and fail rather than truncate queues above the supported cap.
