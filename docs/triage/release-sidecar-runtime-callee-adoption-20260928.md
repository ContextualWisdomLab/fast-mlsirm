# Release sidecar runtime callee adoption

Advance the single publication dependency-gate call from central f5864568 to b6cebb36dc11afe409a7fee8a3262827255c029c. Central #2472 pins all three helper checkouts to e45f1b144aef900d734ff4c900f9e0010fd5a32d, authenticating scripts/ci tree 7f902df89a925f89c4fae69a842508406cd0207c and the unchanged requirements blob. The old helper predates the Python runtime fix.

The central adoption passed 58 guard/workflow tests locally and in CI mode; the complete merged tree matched the tested source tree. The helper scripts delta is only the six-line matching-library binding, independently exercised on guest 02 before #2468 merged. Caller input/output contracts, distribution admission, source ancestry, pre-tag verdict and security gates remain intact. Source-notice PRs #2232/#2233 are already merged; no full-wheel grant clearance or publication is claimed.
