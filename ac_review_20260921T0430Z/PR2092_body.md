## Summary

Opt-in lognormal `|a|` slope prior (MAP) for the polytomous bifactor GRM, single-group and multigroup, with a defined MAP uncertainty for `bifactor_oakes_se`.

- `e9f5aa53`: the preserved original AC patch, committed unchanged (29804-byte tracked diff, sha256 `22c61721…`).
- `fe51d0bc`: the fixes from the integration review (below).
- `a7e2bf82`: **independent-review P2 fix.** The shared slope-prior validator (single-group, multigroup, Oakes) now rejects `np.bool_`, complex/`np.complexfloating` and `ndarray` before `float()` coercion. Before, `np.bool_(True)` became 1.0 and `np.complex128(0.1+3j)` became 0.1. Python-only (`bifactor_grm.py` +5/−2, test +5). The new before-core tests fail on `db7ae8fd` (12 cases) and pass here (forwarding tier, 42 + 10 changelog = 52 local). Rust is untouched, so the native results below still apply to the Rust code.
- `db7ae8fd`: a docstring-only follow-up (Python `final_loglik_change` meaning under a prior). No code change; the native results below are for the `fe51d0bc` tree.

Refs #1932 (slope-magnitude guard / F069 `|a|` track). There is no dedicated issue for the slope prior.

## Review defects fixed in `fe51d0bc`

1. **MG prior was a silent no-op.** Common (anchored) MG items are estimated from pooled counts but got `SlopePrior::None`, so under the default `anchor=None` the prior never applied. Common items now carry the prior once per shared parameter, and free items once per group.
2. **MAP EM used a likelihood-only guard.** Native run 1 rejected the MG prior fits with "EM observed-data log-likelihood decreased". Generalized EM under a prior ascends the log **posterior**, so the guard, the `tol` test, `final_loglik_change`, and multi-start ranking now use log-likelihood + log prior.
   - `loglik_trace` keeps its meaning.
   - The maximized objective is exposed as `em_objective_trace`. Without a prior it is bit-identical to `loglik_trace`, and it is named apart from `FitResult.objective_trace`, which uses the negative-loglik convention.
3. **Provenance.** The fitted prior (`slope_prior_mu/sd`) is recorded on the Rust, PyO3, and Python results.
4. **MAP SE definition.** With the prior, `bifactor_oakes_se` returns `information` = Oakes observed information + the diagonal analytic prior curvature `(1/sd² − 1 − z/sd)/a²` on slopes.
   - `vcov`/`se` are a **posterior-curvature (Laplace) approximation to the posterior covariance**, not a frequentist sampling covariance.
   - A likelihood-only information at MAP estimates is documented as unsupported.
   - The derivation is this crate's: the log posterior is additive and the prior Hessian is diagonal, so slope/threshold and cross-item blocks are the unchanged Oakes terms, and population parameters stay conditional as in MML.
   - Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika, 51*(2), 177–195. https://doi.org/10.1007/BF02293979 (metadata verified via Crossref). It is cited for MAP item estimation. The implemented formula is checked by tests, not by the citation.
5. The folded-lognormal support and constant Jacobian are documented, and the doc/code mismatch (`max(|a|,1e-12)` vs `+inf`) is fixed. A CHANGELOG fragment was added.

## Evidence (kept separate)

| Tier | Command | Tree | Result |
|---|---|---|---|
| Native, executed | `cargo test -j 2 -p mlsirm-core --lib bifactor_` (Air) | dirty patch sha256 `772acb59…` (= `fe51d0bc` minus the `#[ignore]` reproducer) | exit 0, 45 passed |
| Native, executed | `cargo test -p mlsirm-core --test bifactor_grm_mirt_agreement --test bifactor_multigroup_single_group --test bifactor_oakes_mirt` | run-1 patch (before the EM fix) | 4 passed; no later rerun |
| Compile-only | `cargo test -j 2 -p mlsirm-core --no-run` | `772acb59…` | exit 0 (all targets compile; **not executed**) |
| Compile-only | `cargo check --manifest-path crates/fast-mlsirm-py/Cargo.toml` | `772acb59…` | exit 0 |
| Native reproducer | `cargo test --lib reproducer_multigroup -- --ignored` | `fe51d0bc` tree | exit 101, **expected failure**, same delta as baseline |
| Forwarding only | `pytest tests/test_bifactor_slope_prior.py` + changelog contract, with a fake core (local, no compiled core) | final | 37 passed; **not native MAP evidence** |

**Tested tree vs review head (mechanical).** The tree from applying `772acb59` to `e9f5aa53` is `072248e4`; the head `db7ae8fd` tree is `7ab24f6b`. `git diff 072248e4 db7ae8fd` = **+29 / −0 lines only**: 10 docstring lines (`bifactor_grm.py`, `bifactor_multigroup.py`) and the 19-line `#[ignore]` reproducer in `tests/unit/bifactor_grm_tests.rs`. There are no code changes. Hash note: `772acb59` is the sha256 of the full patch artifact, and it equals `git diff --binary e9f5aa53 072248e4` (reproduced on Air and locally). The run-3 log's `DIFF_SHA 09c9c3bd` was a worktree `git diff HEAD --binary` sha16 (tracked files only). Recomputing the tracked-only diff from the tree gives `b33f3067`, so **`09c9c3bd` is not reconciled by hash**. The run-3 ↔ `772acb59` link rests on the procedure (that exact file was applied immediately before the run) and on the test names in the log, which exist only in that patch. The 45 executed passes belong to the tested tree, **not** to a final-head full suite; the full native suite and py parity are still pending.

Earlier RED runs are preserved: run 1 (MG prior rejected by the likelihood guard) and run 2 (the MML comparison fit on the 12-person toy fixture).

## Checkpoint compatibility (deliberate)

The new `BifactorGrmFit`/`BifactorMultigroupFit` fields (`slope_prior_mu`, `slope_prior_sd`, `em_objective_trace`) come after all existing fields and default to the immutable `None`.
- A dataclass field with an immutable default is also a class attribute, so fits pickled before this PR (e.g. the preserved research checkpoints) still unpickle, and the new attributes read as `None`.
- This is a decision, not an accident. Switching any of these to `field(default_factory=...)` or a mutable default would drop the class attribute and make old pickles raise `AttributeError` on that field.
- #2091 does not overlap: its consume path reads only `a_general`/`a_specific`/`threshold`/`specific_sd`, and merge-tree with it is clean.

## Pre-existing, out of scope (tracked gap)

On the original 12-person two-group toy fixture, **MG MML with no prior** fails its own EM guard (`anchor=None`: iteration 9, delta −6.141380e-3; `[T,T,F,F]`: iteration 20, delta −1.514901e-1).
- The released **0.11.4** binary (no AC changes; zero diff in `bifactor_grm.rs`/`bifactor_multigroup.py` from `v0.11.4` to `99c228a8`) reproduces both errors bit-for-bit.
- That fixture is kept as an `#[ignore]` reproducer. The AC behavior is validated separately on a simulated 2×200 six-item design.
- It is not repaired here; the MG EM owner decides.
- Tracked in **#2093** (actual observed errors plus the 0.11.4 baseline reproducer).

## Not yet done / gates

- [ ] Full `cargo test --workspace` and `cargo test --manifest-path crates/fast-mlsirm-py/Cargo.toml` (CI)
- [ ] Editable install + full `pytest`, including the Rust↔NumPy parity gate (CI)
- [ ] Central Security Scan (osv-scan, dependency-review, trivy-fs)
- [ ] **Independent review**. The author has not approved this PR.
- MAP SE coverage is single-group Oakes only, the same scope as MML Oakes. FIPC exposes no prior knob.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_013tDFdELn99o7js5iA4GB1h
