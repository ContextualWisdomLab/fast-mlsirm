# Evidence: multiple-group bifactor expected raw scores, from a preserved checkpoint

Date: 2026-09-21. Scope: the library path that consumes a saved
`BifactorMultigroupFit` for expected raw scores and monotonicity **without
refitting**.

## The failure this reproduces

A research run (W / avoidance-coping, concurrent multiple-group bifactor GRM,
N = 1020, 13 items, 3 age groups, `q = 121`) fitted in 121 EM iterations and
then raised while scoring:

```
File ".../fast_mlsirm/polytomous.py", line 612, in check_bifactor_expected_total_score_monotonicity
    raise ValueError("fit.a_general must be a non-empty 1-D array")
```

The fit was fine; the consume path was not. `BifactorMultigroupFit.a_general`
is `n_groups x n_items` (here `(3, 13)`), and the expected-score path accepted
only the single-group 1-D layout. The fit was preserved so the finding could
be re-verified without spending another calibration:

- `/data/orca/workspaces/late-life-w-ac-mg-s1-20260921/local/library_score_corr_checkpoints_W/W_ac_mg_q121_seed11400714819323198485.pkl`
- sha256 `d74eb91aa38a0e42cb1253feba58db0f127d8bc3b24b0edfd16d73a216c81411`, 21088 bytes

## Re-verification (no refit)

The branch's `python/fast_mlsirm` tree was staged read-only alongside the
host's existing compiled `_core` (the shared analysis virtualenv was not
modified), the checkpoint was unpickled, and the previously-failing call was
made on it directly. Raw output of the evidence script:

```json
{
  "fast_mlsirm": "0.11.4",
  "fast_mlsirm_path": "/data/orca/workspaces/late-life-w-ac-mg-s1-20260921/local/branch_verify/fast_mlsirm/__init__.py",
  "checkpoint": "/data/orca/workspaces/late-life-w-ac-mg-s1-20260921/local/library_score_corr_checkpoints_W/W_ac_mg_q121_seed11400714819323198485.pkl",
  "checkpoint_sha256": "d74eb91aa38a0e42cb1253feba58db0f127d8bc3b24b0edfd16d73a216c81411",
  "checkpoint_bytes": 21088,
  "refit": false,
  "fit_type": "fast_mlsirm.bifactor_multigroup.BifactorMultigroupFit",
  "converged": true,
  "n_iter": 121,
  "termination_reason": "tolerance_met",
  "n_groups": 3,
  "n_cat": 4,
  "n_specific": 3,
  "a_general_shape": [
    3,
    13
  ],
  "threshold_shape": [
    3,
    13,
    3
  ],
  "item_param_rows_identical_exact": {
    "a_general": true,
    "a_specific": true,
    "threshold": true
  },
  "specific_sd_all_unit": true,
  "general_mean": [
    0.0,
    0.0038839876995715584,
    0.06254299777964781
  ],
  "general_sd": [
    1.0,
    0.8906933597128449,
    0.7872380872528046
  ],
  "q_specific": 121,
  "n_persons": 1020,
  "n_unique_theta": 873,
  "score_metric": "expected_raw_E[T|G_EAP]_pointwise",
  "score_mean": 7.182414071338032,
  "score_sd": 3.1851250311568875,
  "score_min": 2.0607309822095514,
  "score_max": 22.30616853277256,
  "explicit_group_max_abs_diff": 0.0,
  "unique_grid_workaround_max_abs_diff": 0.0,
  "monotone_on_unique_grid": true,
  "total_decrease_on_unique_grid": 0.0
}
```

**Provenance of the tested code.** The host has no toolchain able to build
this branch's Rust core (cargo 1.75), so the branch's `python/fast_mlsirm` tree
was staged next to the host's already-compiled `_core` instead. The
`fast_mlsirm: 0.11.4` string in the output is that host distribution's package
metadata, **not** the code under test. The Python actually executed is commit
`54a5925c`: `sha256(python/fast_mlsirm/polytomous.py)` is
`b9de819503142bfb7ec914e89d0723d23afb9c5b857c1b907d4132797d86298d` and
`sha256(python/fast_mlsirm/_legacy_init.py)` is
`db1fc2ddfc65c01a4d958d39129c2ac1be84fe64b09fed49227a71d809102771`, identical
on the host and in `git show 54a5925c:`.

**The compiled core IS on this path** (corrected; an earlier revision of this
document claimed the opposite). `predict_bifactor_expected_total_score` calls
`predict_expected_response_polytomous` per item, which routes through
`_polytomous_predictions`, which *requires* the native extension — it raises
`RuntimeError("polytomous predictions require the compiled Rust core")` when the
module is absent and otherwise calls `core.polytomous_predictions`
(`python/fast_mlsirm/polytomous.py:218-226`). The new group-selection and
validation logic is NumPy, but every expected-score value in this evidence was
produced by native code. The run is therefore **mixed-provenance by
construction**: branch Python over a previously built native artifact, bound
here so the mixture is auditable rather than implied.

| Loaded native artifact | Value |
|---|---|
| Module resolved at run time (`fast_mlsirm._core.__file__`) | `…/late-life-w-ac-mg-s1-20260921/local/branch_verify/fast_mlsirm/_core.cpython-312-x86_64-linux-gnu.so` |
| sha256 / size | `a0453b7af30038e5f848c7a1f0735c683e41fa809341770c59e123a2d10978ed` / 14362448 bytes |
| `polytomous_predictions` exported | yes (verified on the loaded module, not inferred) |
| Source it was copied from | `/data/orca/workspaces/fmls-perf-diag-venv/lib/python3.12/site-packages/fast_mlsirm/_core.cpython-312-x86_64-linux-gnu.so`, byte-identical (same sha256) |
| Installed distribution | `fast_mlsirm-0.11.4.dist-info`; its `RECORD` lists `sha256=oEU7evMAOOX4SMeh8HNcaD5B-oCTQXcMWeEjotEJeO0`, which decodes to the same digest |
| Wheel identity | `fast_mlsirm-0.11.4-cp312-cp312-manylinux_2_35_x86_64.whl`, built by maturin 1.15.0 |
| Wheel origin (`direct_url.json`) | `file:///data/orca/workspaces/fmls-g4w-fipc-s1-wheels/32807ed0/…`, archive sha256 `664aef1a467063428423fe5c38115c197d1a7dcd4e182204d63fc5324355f5f8` |

That last row is the material caveat: the native half came from a **locally
built wheel on an internal path, not an immutable published release**. It is
adequate to demonstrate that the multiple-group consume path works on the
preserved fit without refitting; it is *not* release acceptance, and it does not
pre-empt the acceptance owed once an immutable release exists (release
provenance, version and artifact digest, the installed loaded-core path and hash
at that time, the saved fit consumed with no refit, and validation of all 1020
row keys with their group correspondence — index-wise person/row/group alignment
and per-person curve identity, not merely the score distribution).

What it establishes:

- **The failing call now succeeds on the identical object.** `refit: false`,
  same sha256, `n_iter: 121`, `converged: true` — the scores come from the
  preserved fit, not a new one.
- **The identity claim is checked, not assumed.** All three item-parameter
  blocks (`a_general`, `a_specific`, `threshold`) are *exactly* equal across
  the three groups, as an all-anchored fit (`anchor_mask=None`) makes them, so
  `group=None` is admissible here. `explicit_group_max_abs_diff: 0.0` confirms
  naming group 0, 1 or 2 gives the same curve for this fit; a fit whose rows
  differed would raise rather than silently score group 0.
- **The group's population parameters stay out of the conditional curve.**
  `general_mean`/`general_sd` differ across groups (`sd` 1.0 / 0.891 / 0.787),
  and correctly do not enter `E[T | theta_G]`: multiple-group item parameters
  and `theta_g_eap` are on the one common (reference) metric the E-step places
  every group's nodes on, so no per-group rescaling of `theta` is applied.
- **`specific_sd` is unit here** (`estimate_specific_vars=False`), which is
  what makes the per-item specific-factor marginalization computable from the
  fit alone; a non-unit `specific_sd` raises, because the fit does not carry
  the `specific_map` needed to match an item to its specific factor.
- **The old grid workaround agrees exactly.** Scoring the 873 unique EAP values
  through `check_bifactor_expected_total_score_monotonicity` and mapping back
  reproduces the pointwise result to `0.0` — the check and the new prediction
  read one kernel. The curve is monotone on that grid
  (`total_decrease_on_unique_grid: 0.0`).

## Checkpoint schema: fit and score keys separated

The outer producer (`analysis/library_score_corr_W_only_s1.py`) and the inner
fit step (`library_score_correlations.py::score_ac_multigroup`) both called
`checkpointed()` with the job id `W_ac_mg_q{Q_NODES}_seed{SEED}`. The inner
call wrote the fit under that key first, so the preserved `.pkl` holds a
`BifactorMultigroupFit`, and a re-run of the outer call would have read a fit
back where a score dict was expected. The outer key is now
`W_ac_mg_score_q{Q_NODES}_seed{SEED}`; the inner (fit) key is unchanged, so
the preserved checkpoint keeps its name and its meaning.

```diff
--- library_score_corr_W_only_s1.py (before)
+++ analysis/library_score_corr_W_only_s1.py (after)
@@ -25,8 +25,11 @@
+    # Distinct schema keys: score_ac_multigroup checkpoints the FIT under
+    # W_ac_mg_q..._seed..., so the score dict must not reuse that key -- the
+    # outer read would otherwise hand a BifactorMultigroupFit back as a score.
     out = lsc.checkpointed(
-        f"W_ac_mg_q{lsc.Q_NODES}_seed{lsc.SEED}",
+        f"W_ac_mg_score_q{lsc.Q_NODES}_seed{lsc.SEED}",
         lambda: lsc.score_ac_multigroup(x, masks),
     )
```

**Ownership and delivery path.** That file lives on the analysis host, in a
directory that is not a git repository, so the edit cannot ride in this
repository's pull request and would otherwise exist only in a shell history.
It is owned by this change's author as the expected-raw / checkpoint-schema
owner, and is delivered by this record plus an on-host backup:

| Artifact | sha256 |
|---|---|
| `analysis/library_score_corr_W_only_s1.py` (after) | `427fdad0304d53ba2156a4c0de447af0597d79da26b6b2712c562f179030e4f0` |
| backup before the edit, `local/branch_verify/library_score_corr_W_only_s1.py.bak` | `ee862685a97118b24941e7de9d3824cab7d1efa825796fdc49ac97885b1a270a` |
| evidence script, `local/branch_verify/verify_mg_expected_raw.py` | `8ef158a5d9867c345550953e34e448e3f02963c11f1cab11f6f5bafbe210e25f` |
| preserved fit checkpoint, unchanged by the edit | `d74eb91aa38a0e42cb1253feba58db0f127d8bc3b24b0edfd16d73a216c81411` |

The shared `library_score_correlations.py::expected_raw_from_bifactor_fit` was
**not** touched: that consumer has its own writer, and migrating it onto the
public API is coordinated work scheduled after release acceptance.

## Follow-ups (not in this change)

- The research helper `expected_raw_from_bifactor_fit` evaluates on unique EAPs
  and maps back only because no pointwise expected-raw API existed. Once a
  release carrying `predict_bifactor_expected_total_score` is installed on the
  analysis host, that helper can call it directly and drop the
  unique/`return_inverse` step.
- A fit with an estimated `specific_sd` cannot be scored this way at all until
  some fit object carries `specific_map`. Raising is deliberate; widening it is
  a separate change.
