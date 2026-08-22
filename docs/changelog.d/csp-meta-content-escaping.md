# CSP meta-tag content over-escaping

## Fixed

- Stopped HTML-escaping the Content-Security-Policy string embedded in every generated standalone HTML report (`fast_mlsirm.report`, the essay HTML report family, and the benchmark/buyer-packet/commercial-release/Figma-evidence-sync/PR-queue-governance/procurement-due-diligence/release-evidence-index builder scripts). `html.escape(..., quote=True)` converted the CSP's literal `'none'`/`'unsafe-inline'` source expressions into `&#x27;none&#x27;`, which browsers do not recognize as valid CSP syntax, silently disabling the meta-delivered policy on every emitted report. Found by Strix (CWE-693, CVSS 6.5) while scanning an unrelated in-progress PR. The policy string is a fixed, non-user-controlled constant in every call site, so interpolating it directly into the `content` attribute is safe; tests that previously asserted the escaped (broken) form now assert the literal, unescaped CSP directive and guard against re-escaping.
