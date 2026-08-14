# Workflow-registry audit transport retry hardening

## Fixed

- Expanded the read-only Actions-registry audit transport's bounded retry classifier to cover transient HTTP 403, 404, 429, and all 5xx responses, while preserving fail-closed exhaustion and immediate failure for non-transient authentication errors such as HTTP 401.
- Added direct transport regression coverage so incident audits do not misclassify one transient GitHub control-plane response as a completed inventory failure.
