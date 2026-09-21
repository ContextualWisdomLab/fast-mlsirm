# AC (SlopePrior lognormal on |a|) integration review — 2026-09-21T04:35Z

Reviewer: Claude session `term_9464f3b6` (reuse, review-only). No source edits, build, refit, or new worker.
The AC worktree was read over ssh only. E PID/shared venv not touched.

## Subject
- Air WT `/Users/seonghobae/orca/workspaces/fast-mlsirm/fmls-ac-slope-prior`, branch `seonghobae/fmls-ac-slope-prior`,
  HEAD `99c228a8` (no commits yet), 16 tracked files modified (+235/−14), untracked `.opencode/` (jobs `[]`).
- Live `git diff --binary` = **29804 B** (matches CO's handoff-0422 count), sha256 `22c61721…f039a9`, copy: `ac_diff_live.patch`.
- No process holds the AC WT: opencode 99948 cwd = `fmls-e-mean-next` (E, protected), 66547 = another repo.
- CO evidence `handoff-0422/ac*` and any test receipts: **not locatable** from this session. No receipt files exist in the WT,
  so **no test run is evidenced**. Everything below is from reading the source.

## What is correct (verified by reading)
- Math: `-log p = ln|a| + z²/2` with `z = (ln|a|-mu)/sd` (constants dropped). `d/da = (1+z/sd)/a`, which is the correct signed chain rule.
  The folded density `p(|a|)/2` is proper on ℝ\{0}. Its Jacobian is a constant ½, so the signed extension is legitimate.
  This answers VERIFY_0302 item 1 in principle, but the argument is not written in the docs.
- 0-boundary (item 2): `a == 0` → `nll = +inf`. The line search rejects it and nothing is clamped. As |a|→0 the objective → +inf (a barrier).
- `validate()` rejects non-finite mu and sd ≤ 0 (Rust, PyO3, and Python). mu and sd must be given as a pair. There are no research defaults.
- `None` is the default everywhere: existing configs got `slope_prior: None`, and the marginal-loglik, brute, Oakes Stage1 and FIPC paths are pinned to `None`.
  Default MML behavior is unchanged as long as `None` is the default.
- FIPC (item 3): fixed anchors never receive the prior. FIPC free items get `None` too, and `BifactorFipcConfig` has no knob for it.
- Unit tests added: FD vs analytic gradient at a = −1.7 (item 4, prior term only), zero → +inf, and hyperparameter rejection.

## Blocking gaps (must fix before a scoped commit)
1. **In MG, the prior never applies to common (anchored) items, and these items are freely estimated.** MG anchored items are estimated freely from pooled counts
   (`bifactor_grm.rs` ~2234, Bock–Zimowski). They are not fixed, yet the diff passes `SlopePrior::None` for them (~2278).
   `anchor=None` means all items are common (~2452), so **for n_groups ≥ 2 with the default anchor, `slope_prior_*` is a silent no-op**.
   For n_groups == 1 the call is delegated to single-group, which applies the prior to all items. The same knob therefore gives different models.
   The in-code comment ("anchors … must not move") confuses MG common items with FIPC fixed anchors.
   Fix: apply `cfg.slope_prior` to the pooled anchored M-step as well (one prior per shared parameter, not one per group).
   Add a test showing that n_groups=1 matches single-group with the prior, and that n_groups≥2 with anchor=None changes the estimates.
2. **SE and provenance mismatch.** The fit result (`BifactorGrmFit`/`BifactorMultigroupFit`, Rust results) does not record `slope_prior`.
   Oakes SE (`bifactor_oakes.rs` Stage1Provider = `None`) will compute likelihood-only information at MAP estimates without any warning.
   Fix: record the prior in the fit output. Then either add the prior Hessian term to the Oakes information, or refuse or flag Oakes SE on a MAP fit.
3. **No end-to-end evidence.** There is no test that a fit with `Lognormal` runs and changes or shrinks slopes, and no test that `None` gives bit-identical results through the new PyO3 signature.
   There is no Python test for the new kwargs, their pairing error, or the finite/positive checks.
   The FD test covers only the isolated prior term, not the `item_neg_ll_grad` gradient with the prior included.
4. **Primary-source/docs (root 03:04).** The docstring cites Chalmers (2012) `lnorm` for positive `a`. It needs to state the folded-density and constant-Jacobian argument, and mark the |a| use as this crate's extension.
   Doc/code mismatch: the doc says `max(|a|,1e-12)`, but the code returns +inf at 0 and has no floor.
   CHANGELOG/docs are untouched. The AGENTS.md paper-first rule applies because this is an estimator change.

## Non-blocking
- rustfmt: the diff adds new unformatted spots, e.g. mis-indented `ridge:` and `slope_prior:` in `bifactor_oakes_calibration.rs` at 2 sites, the long `_ => return Err(...)` lines in `lib.rs`, and the multi-line `assert!` in the unit tests.
  Base files are already not fmt-clean, and CI has no fmt/clippy gate, so this is cosmetic. Still, fix the oakes_calibration indentation.
- No MG anchor-exclusion or FIPC-exclusion regression test yet. This is covered by the fix in gap 1.

## Scoped commit recommendation
One commit on `seonghobae/fmls-ac-slope-prior`, containing only the 16 listed files (exclude `.opencode/`), **after** gaps 1–4.
Suggested message: `feat(bifactor): opt-in lognormal |a| slope prior for bifactor/MG GRM MAP`.
Required verification (to be run by the AC owner, not this review):
`cargo test --workspace`, `cargo test --manifest-path crates/fast-mlsirm-py/Cargo.toml`, and an editable install followed by `pytest tests/` (bifactor subset plus the rust-default assertion).
Also include the new MG/n_groups=1 equivalence test and an explicit `None` bit-identity test.
Oakes SE on prior fits must be guarded or extended before any research consumer uses the SEs.
