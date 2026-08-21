# PR queue malformed-list evidence hardening

## Fixed

- Fail closed when successful GitHub open-PR identity or PR-history list payloads contain non-object entries instead of silently dropping malformed evidence and publishing a complete-looking queue snapshot.
- Preserve valid object evidence while recording the schema failure, and report the raw open-identity list count so malformed entries cannot disappear from queue-size evidence.
