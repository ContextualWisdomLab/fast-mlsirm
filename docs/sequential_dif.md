# Sequential DIF controller

## Scope and availability

```python
from fast_mlsirm.sequential_dif import run_sequential_dif
```

The controller coordinates bounded native whole-family rounds and archives their
outputs. It does not implement estimation, LR statistics, Hessians, multiplicity
adjustment, effect sizes, or removal rankings. The intended native policy tests
all items against the full response matrix, removes at most one current anchor,
and retests every item, including earlier nonanchors, after a removal. This is
an explicit software contract, not a claim of a universally validated anchor
selection procedure.

**Native integration is pending.** At main commit
`338f019e7927a6b134e514be29fd0c4098e8c5e8`, the compiled Python core does not expose
`sequential_dif_capabilities` or `sequential_dif_round_gpu`. The controller rejects
that missing contract before archive creation or fitting. It never falls back to
an E-step-only GPU fitter. This source contribution does not enable strict GPU
DIF, demonstrate numerical recovery, or certify inferential validity.

The original 12 controller tests plus two identity-type regression tests use a
recording synthetic native fixture. Fixture
receipts are test inputs, not GPU dispatch evidence. Native inference, an
installed-package integration test, full-suite acceptance, independent review,
and immutable release adoption remain separate acceptance steps.

## Request and native policy

The caller supplies a JSON-serializable request containing:

- unique nonempty `item_ids` and a same-length Boolean `anchor` mask containing
  at least one anchor;
- `device='gpu'`, `controls.require_gpu=true`, and `studied_item=None`;
- an explicit positive integer `max_rounds`;
- all data, group/model, prior, Hessian, quadrature, identification, convergence,
  and inference settings required by the eventual native implementation.

There are no default quadrature counts or numerical controls here. Each call
passes the request unchanged except for the current anchor mask and the native
one-round limit (`max_rounds=1`, `studied_item=None`). Native validation owns
model-specific settings and their compatibility; forwarding a prior does not
certify an ordinary LR interpretation for a prior-regularized fit.

For a studied item j and current anchor set A, the intended native comparison
uses H0 anchors A union {j} and H1 anchors A minus {j}, leaving other nonanchors
free. The native supplier owns adjustment, deterministic selection, all
identification/convergence checks, and termination reasons. Python validates
inventory order, single-removal transitions, and reported execution scope. A
failed pair prevents adjustment or removal for the family. A round limit is
incomplete, never convergence.

Sparse/unobserved categories preserve the native item/group/category error and
all available item records. The controller does not merge categories, insert
threshold priors, partially equate boundaries, change settings and retry, or
turn a failed family into a clean anchor set.

## Relationship to mirt

The current mirt `DIF.R` implementation documents `scheme='add'`,
`seq_stat='SABIC'`, `p.adjust='holm'`, and `items2test=1:nitems`. Its sequential
schemes can update multiple constraints after a sweep. They are not the
one-removal/full-family-retest contract specified here. No mirt package or R
runtime is a dependency of this controller, a CI oracle, or a source of runtime
statistics.

Chalmers (2012, Section 3.1, pp. 5–7) describes the underlying EM estimation and
bifactor context. That paper does not establish this controller's removal or
archive policy. The mirt source comparison is implementation evidence only;
neither citation is presented as validation of this proposed native contract.
The repository already includes the openly distributed article at
`docs/papers/Chalmers_2012_mirt_JSS.pdf`.

References:

- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package
  for the R environment. *Journal of Statistical Software, 48*(6), 1–29.
  https://doi.org/10.18637/jss.v048.i06
- Chalmers, R. P. (n.d.). *DIF.R* [Source code]. mirt. Retrieved October 25,
  2026, from https://github.com/philchalmers/mirt/blob/master/R/DIF.R

## Required compiled contract

`_core.sequential_dif_capabilities()` must return a dictionary with:

- `schema_version: 1`;
- a nonempty `native_identity` binding the actual native/kernel sources;
- `fit_backend: 'gpu_only'`;
- `full_matrix_single_removal: true` and `all_items_retested: true`.

Only the compiled implementation with both strict E-step and M-step dispatch
paths may advertise these capabilities. Adding flags, mocked counters, or an
E-step-only implementation does not satisfy the contract.

`_core.sequential_dif_round_gpu(json_request) -> json_string` must:

1. Preflight strict GPU availability before any fit. Raise device errors
   without CPU fitting, including failures from losing starts. This is the
   compiled supplier's obligation; the Python controller archives these
   exceptions under the failed-output policy below.
2. Run one full native family round with the exact input settings and complete
   response matrix, using library-owned private fit objects.
3. Return `request_sha256` over the exact input JSON string.
4. Return `fit_receipt` with `backend='gpu_only'`, integer
   `e_step_gpu_dispatches>0`, integer `m_step_gpu_dispatches>0`, and integer
   `cpu_fit_dispatches=0`. Count every fit/start and terminal E sweep, not only
   the winning start. Enforce the no-CPU condition inside the actual call.
5. Return `result.rounds` with exactly one ordered complete item inventory,
   including failures, native adjusted p-values, removal and status, plus
   `result.final_anchor`.

CPU-f64 information/Hessian evaluation is outside the GPU-only **fitting**
scope and must be identified separately. The controller cannot prove device
execution from counters; source-bound native integration tests must establish
that provenance. There is no public backend injection argument. Tests replace
only the private importer.

## Private archive and resume

`run_sequential_dif(request, archive=private_directory)` uses a caller-owned
private directory and a POSIX advisory lock. `manifest.json` binds the complete
request digest and native capability identity. Exclusive `round-NNNN.pending`
creation precedes dispatch; exclusive `round-NNNN.json` creation saves the full
native output and its digest with file fsync.

Completed rounds are validated and reused without refitting. Changed requests
(including prior/Hessian settings), native identity drift, and corrupted output
are refused. A pending round without a completed output requires explicit owner
reconciliation; it is never automatically refit. Native exceptions, including
device errors, are retained as failed output;
the public controller returns `native_contract_failed` with `failed_output`
on the first call and replay, without reconstructing the original exception
class or automatically retrying. The resulting mask must retain at least one
anchor, so removing the final anchor fails validation before another dispatch.

The manifest stores input digests, not response arrays. Native outputs can
contain sensitive fitted values. Keep archives outside public repositories and
review packets; the caller is responsible for an appropriately private existing
directory.

Limits: cooperative same-host/same-directory POSIX locking only. There is no
global GPU lease, distributed archive, hostile same-user protection, atomic
crash-durable publication, directory fsync, kernel cancellation, or wall-clock
timeout. Partial JSON fails closed. Stronger publication and interrupted-worker
settlement require integration with the archive owner before operational use.

## Integration acceptance

- Compose the native whole-family inference supplier, strict GPU E-step and
  M-step implementation, and PyO3 binding; demonstrate actual same-call E+M
  dispatch and no-GPU/CPU-fallback negative controls.
- Exercise actual native synthetic family inference and model-specific
  identification, convergence, Hessian/prior, and sparse-category refusals.
- Bind native source identity and adopt the archive owner's custody, durable
  publication, and interrupted-worker settlement contract.
- Verify the public import from a freshly built candidate package and run the
  full applicable suite and independent review before release adoption.

This contribution is eligible for source-component review and normal CI while
native integration and release adoption remain pending. The implemented
controller and its tests do not establish a native capability that does not
exist.
