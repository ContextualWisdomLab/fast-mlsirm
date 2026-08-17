# Support Policy

## Current Pre-1.0 Support Scope

`fast-mlsirm` 0.7.x is the current supported pre-1.0 minor line. Support applies
to behavior that is part of a released artifact and documented public API or CLI
contract. An active pull request, research plan, experimental internal helper,
or roadmap item is not supported merely because it exists in the repository.

Support covers reproducible defects in:

- installation, wheel loading, and the packaged Rust/PyO3 runtime;
- documented public API and CLI behavior;
- deterministic validation, serialization, reporting, and audit artifacts;
- numerical results that disagree with the documented formula or backend-parity
  contract on a supported production path; and
- regressions in documented security, resource, or compatibility behavior.

Production numerical ownership is Rust-first. Explicit NumPy/reference code that
is retained for parity or research may be used to reproduce an investigation,
but its existence does not make it an implicitly selected production backend.

## Support Boundaries

Support does not provide or imply:

- clinical, educational-placement, hiring, or other high-stakes decision
  guarantees;
- construct-validity, fairness, causal-effect, or consequential-use approval;
- certification or conformance attestation for SOC 2, CSAP, ISO, NIST, or another
  governance framework;
- a service-level agreement, uptime guarantee, or hosted incident-response
  commitment for downstream applications;
- hosted dashboards, tenant administration, identity, persistence, or deployment
  services owned by downstream products; or
- support for unreleased, superseded, rejected, or explicitly research-only
  capabilities.

A supported software defect can still occur in a model, dataset, or use case for
which the available scientific evidence is insufficient. Software support must
not be interpreted as authorization for a high-stakes use.

## Requesting Support

Open a GitHub issue with:

- package version and commit SHA;
- operating system, CPU architecture, Python version, Rust version, and install
  method;
- the selected public operation and, where relevant, the resolved Rust device or
  explicitly selected reference/parity path;
- minimal reproduction data or a synthetic reproduction script; and
- expected and observed behavior.

For private datasets, replace the data with a synthetic reproduction before
opening a public issue. For a suspected vulnerability, follow `SECURITY.md` and
use a private GitHub security advisory when available.
