# Security Policy

## Supported Versions

`fast-mlsirm` is pre-1.0. Security support follows the currently released
major/minor line; older pre-1.0 minor lines are not maintained unless a release
notice explicitly says otherwise.

| Version | Supported |
| --- | --- |
| 0.9.x | Yes |
| < 0.9 | No |

## Reporting a Vulnerability

Report suspected vulnerabilities through a private GitHub security advisory for
`ContextualWisdomLab/fast-mlsirm` when available:
https://github.com/ContextualWisdomLab/fast-mlsirm/security/advisories/new

If private advisories are not available to you, open a minimal public issue that
describes the affected area without publishing exploit data or sensitive
datasets.

Please include:

- affected version or commit SHA;
- operating system, Python version, and install method;
- whether the compiled Rust/PyO3 path, CLI, report renderer, or an explicitly
  selected reference/parity path is involved; and
- a minimal reproduction that does not include private response data.

## Security Boundaries

`fast-mlsirm` is a local computation library and CLI. It does not run a network
service, authenticate users, store credentials, or upload response data. Hosts
and downstream applications own transport, identity, tenancy, persistence, and
deployment controls. The package owns bounded local validation, deterministic
failure behavior, numerical/runtime integrity, package-installation integrity,
and the security properties documented for its public artifacts.

Static HTML reports are generated from local diagnostics artifacts. Do not embed
untrusted free-form text into report inputs unless it has been reviewed and
bounded by the calling application.

Security support is not a certification, regulatory approval, service-level
agreement, or guarantee that a model or score is valid for consequential use.
