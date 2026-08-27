# Release acceptance watchdog budget

## Fixed

- Derive the commercial-release wrapper deadline for `release_acceptance.py` from the authoritative sequential inner acceptance budgets plus a 60-second orchestration margin, so future bounded-stage changes cannot silently reintroduce an outer watchdog that terminates legitimate fail-closed acceptance before the inner operation-specific deadline can report its evidence.
