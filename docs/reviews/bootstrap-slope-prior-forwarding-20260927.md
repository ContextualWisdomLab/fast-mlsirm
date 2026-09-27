# Bootstrap slope-prior forwarding (2026-09-27)

The integration preserves PR #2092 head f1a6c46fab1a9e07024bdf2c47945669a0705553
and PR #2205 head 4f7f4b5984d7384a5723a1fb5276a3680d73ed5b through an ordinary
merge. Neither original branch is replaced or its review waived.

## Source and application

Efron, B. (1979). Bootstrap methods: Another look at the jackknife.
The Annals of Statistics, 7(1), 1–26. DOI: 10.1214/aos/1176344552.
Actual primary paper downloaded from the University of Helsinki mirror:
https://blogs.helsinki.fi/bk-club/files/2012/05/Efron_Bootstrap_AS1979.pdf
PDF page 3 displays printed page 2; PDF page 4 displays printed page 3.
Both images were directly viewed because text extraction produced no usable
body. Section 2, eq. 2.4 resamples from the empirical distribution and eq. 2.5
applies R to X* and the empirical distribution. The operative definition is
`R* = R(X*, F-hat)`. The same page cautions that approximation quality depends
on the form of R.

Application: a prior supplied to the target MAP fit is an argument of the
estimation procedure, so each replicate must retain it. This is a derived
application of the same-statistic definition, not an Efron prescription for
lognormal GRM priors or hyperparameters. No interval coverage guarantee is
established. The existing PR #2092 fit APIs own the prior kernel and its
scientific/native validation; the bootstrap only reuses their shared validator
and forwards the pair. The runner docstring states this boundary.

Personal and group Zotero queries both timed out after 10 seconds; their
contents were not inferred. The Washington university mirror web request also
timed out. The Helsinki primary-paper download succeeded. No PDF is added to
the repository or remote review request; source byte hashes accompany this note.

## Verification

Before the change, the new real-worker test failed with an unexpected
`slope_prior_mu` keyword. An intermediate test caught missing result metadata
(2 failed, 19 passed). After forwarding and result/error metadata repairs:
22 passed in 0.05 seconds across `test_bootstrap_slope_prior.py` and
`test_bootstrap_failure_receipts.py`. Tests overlay the three candidate Python
modules on an existing native environment and replace only the fit call with
synthetic outcomes: they execute the actual dispatcher and both worker routes,
but do not execute native MAP refits. Omitted priors and supplied priors,
invalid pair admission, all-failed metadata and stratum/failure bookkeeping are
covered. `git diff --check` passes.

Not accepted yet: native recovery/interval behavior, full exact-head hosted
checks, nonauthor approval, merged immutable release and install hash; joint
DT/E/AC/Z/Y refitting/scoring/regression and study replicates. The existing
batch endpoint movement heuristic is not an Andrews–Buchinsky accuracy rule.

## Integrated source correction and final scoped check

An ordinary merge preserves PR #2188 head
eae402c905a2b886f69d5ae1a64e3b605e3fcca1. Its correction removes the unsupported
Andrews–Buchinsky attribution from the endpoint-movement heuristic. The actual
Cowles PDF was opened: PDF page 2 is printed 23 and PDF page 3 is printed 24.
Printed 24 defines percentage deviation from the ideal quantity and its
probability bound, different from successive-batch movement. No stopping
algorithm is replaced in this integration.

At integration head 83a7fcc9e0c146ec96c6b7758c2dab18e9c94fd5, the two new test
files plus the inherited public-doc source-boundary check passed: 23 passed,
0.06 seconds. A first post-merge invocation failed before collection because
the sparse merge removed hydrated Python files; they were restored from that
exact head before the successful invocation. The subsequent evidence-note
append also initially failed on a missing sparse file; the PR opened at the
already-tested code head. The note is now restored and updated in a follow-up.
No native MAP result or active job restart is established by these operations.
