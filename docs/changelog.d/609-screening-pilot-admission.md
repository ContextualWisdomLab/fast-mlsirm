# Require semantic screening before pilot admission

## Added

- Pilot admission now requires a complete, factory-sealed and pilot-eligible
  `CandidateScreeningResult` bound to the exact generated candidate and audit
  report. Review-required or blocking semantic decisions fail closed, and the
  resulting `PilotCandidateRecord` retains the screening-result fingerprint for
  downstream item-bank provenance. This connects the existing semantic
  screening contract to the governed item-bank path without adding provider
  calls or Python psychometric arithmetic; calibration and validity remain
  downstream Rust-backed evidence gates.
- Screening-bound pilot records now advertise the dedicated
  `schema_version="2.0"`, while the public generated-item audit/admission policy
  is version `2.0.0`. Legacy pilot-record schema `1.0` is rejected explicitly;
  unrelated rubric/blueprint contracts retain their existing shared schema
  version rather than being silently re-versioned.
