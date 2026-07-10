# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |

## Reporting a Vulnerability

Report suspected vulnerabilities through a private GitHub security advisory for
`ContextualWisdomLab/fast-mlsirm` when available:
https://github.com/ContextualWisdomLab/fast-mlsirm/security/advisories/new.

If private advisories are not available to you, open a minimal public issue that
describes the affected area without publishing exploit data or sensitive
datasets.

Please include:

- affected version or commit SHA;
- operating system, Python version, and install method;
- whether the NumPy backend, Rust backend, CLI, or report renderer is involved;
- a minimal reproduction that does not include private response data.

## Security Boundaries

`fast-mlsirm` is a local computation library and CLI. It does not run a network
service, authenticate users, store credentials, or upload response data. Users
are responsible for protecting local input/output files and for applying their
own data governance controls before using real assessment data.

Static HTML reports are generated from local diagnostics artifacts. Do not embed
untrusted free-form text into report inputs unless it has been reviewed by the
calling application.

## Disclosure Timeline

We aim to acknowledge vulnerability reports within 7 days, provide a remediation
plan or status update within 30 days, and publish coordinated disclosure notes
after a fix is available or within 90 days when a longer remediation is required.
