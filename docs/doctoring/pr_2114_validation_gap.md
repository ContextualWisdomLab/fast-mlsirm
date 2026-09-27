# PR #2114 validation gap: evidence ledger

This ledger records observed evidence, not a certification of completed checks.

## Original delivery

[PR #2114](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/2114)
was merged as `2d80e8d85d40c665a8000020b78723f701f878c8` using administrative
bypass with five failed checks. That merge does not prove those gates passed.

## CodeQL dispatch reproduction

[Central run 36315242280](https://github.com/ContextualWisdomLab/.github/actions/runs/36315242280)
targets fast-mlsirm PR #2112 head
`6e45fbf42b8458345a1a87d38566abe48899ae1c`, base
`00f5cb91b417e40102036eb31ab8a4b843c76076`, required run `36239997770`.

| Evidence | Observed result |
| --- | --- |
| Dispatch validation, job 108608601318 | Success |
| Actions analysis, job 108609052914 | Success |
| Python analysis, job 108609052931 | Failure |
| Exact required-run settlement, job 108610599004 | Failure |

The Python job successfully checked out the target head and completed analysis.
Its SARIF gate reported two `py/incomplete-url-substring-sanitization` findings
(security severity 7.8), in `tests/test_architecture_baseline_contract.py:34`
and `tests/test_governance_index_contract.py:32`. Both citation checks accepted
an arbitrary occurrence of `https://doi.org/` instead of verifying a URL host.

[PR #2218](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/2218)
replaced those substring checks with standard-library URL parsing, HTTPS,
exact hostname, and DOI-path checks. Both affected test files passed locally
(four tests). It was administratively merged as
`76826aad7cd9e6a0e33a6a84e38ddc89ef8204c0` while hosted checks were queued;
there was no independent approval. Hosted CodeQL clearance remains unproven.

The same Python job also failed status publication: the target application
token and central workflow token each received HTTP 403. This is a separate
authority failure, not evidence that the analysis did not run. The successful
Actions job retained authenticated scan/SARIF fallback evidence; its success
does not establish that a target commit status was published.

## Remaining proof

The original failed jobs are independently identifiable:

| Gate | Original failed job | Required recovery evidence |
| --- | --- | --- |
| CodeQL Actions | [106839633565](https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/35735298245/job/106839633565) | Current-head analysis and required-wrapper settlement |
| CodeQL Python | [106839633554](https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/35735298245/job/106839633554) | Current-head analysis and required-wrapper settlement |
| Noema | [106842255571](https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/35735295741/job/106842255571) | Published exact-head model verdict and valid continuation authority |
| Strix | [106845076759](https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/35735295779/job/106845076759) | Authoritative scan, trusted binder, and working provider path |
| OpenCode | [106839384622](https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/35735295744/job/106839384622) | Authenticated independent verdict, coverage evidence, and wrapper settlement |

The original CodeQL wrapper reported dispatch pending its terminal verdict.
OpenCode reported that neither APPROVED nor CHANGES_REQUESTED from its
authenticated reviewer identity was present on the head. Noema failed both
preparation and re-dispatch; the latter recorded HTTP 403. Strix recorded
provider unavailability and a missing binder, which need separate repairs.

Central [PR #2429](https://github.com/ContextualWisdomLab/.github/pull/2429)
landed as `24bdfe0b4fcd093cac4047433211d2cd59d53b43`. Its final Noema routing
admits only the central repository, contextual-orchestrator, and fast-mlsirm,
and requires the exact central main workflow identity. The model runner group
received fast-mlsirm access while retaining repository and workflow restrictions.
Ninety affected local tests passed, but the integration used administrative
bypass with hosted checks queued and no independent approval. It is not live
consumer verdict proof.

Central [draft PR #2430](https://github.com/ContextualWisdomLab/.github/pull/2430)
retains the missing Strix fixture runtime carryover. Forty-three relevant local
tests passed; its full shell harness is still pending. This is partial carryover
and does not retire every unique delta in central PR #2291.

The five original failed-check gaps remain open until genuine current-head
consumer runs prove the repaired gates. A queued job, an earlier head's green
check, local test results, preserved SARIF, and an administrative merge each
prove different things; none substitutes for all required terminal results.
Release license and wheel machinery changes in contextual-orchestrator also
do not establish an approved published artifact or deployed runtime version.
