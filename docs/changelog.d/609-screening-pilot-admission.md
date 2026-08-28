# Require semantic screening before pilot admission

## Added

- Pilot admission now requires a complete, factory-sealed and pilot-eligible
  `CandidateScreeningResult` bound to the exact generated candidate and audit
  report. Review-required or blocking semantic decisions fail closed, and the
  resulting `PilotCandidateRecord` retains the screening-result fingerprint for
  downstream item-bank provenance. The public pilot record also retains a
  package-owned creation-time seal, so post-construction rebinding of candidate,
  audit, screening, policy, blueprint, rubric, lifecycle, or schema provenance
  fails closed before public identity or serialization can grant new authority.
  This connects the existing semantic screening contract to the governed
  item-bank path without adding provider calls or Python psychometric arithmetic;
  calibration and validity remain downstream Rust-backed evidence gates.
