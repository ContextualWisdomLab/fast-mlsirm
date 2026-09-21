# Bounded handoff inventory — 2026-09-21T03:52Z (investigator only, no impl)

Lead WT HEAD: `a3602747` on `orchestration-lead-fmls-v4` (clean; untracked evidence only). No builds/refits/sessions/polling/creds; source files untouched; run812 mailbox untouched; A4 STOPPED; shared perf venv + s1 artifacts + manuscript prose preserved.

## Owner / head / blocker (minimal)

| Area | Active owner (authority) | Head / evidence | Remaining acceptance / blocker |
|------|--------------------------|-----------------|--------------------------------|
| Scorer (Z/Y focal-prior) | `ctx_e136e59a77b7` / `term_857ad77a` (fmls-score-focal-prior) | `be6ff91a`, core sha16 `05c07681`, PASS → research relay (LEAD_STATUS 0331) | None numerical; keep venv isolated, no shared-venv swap |
| AC (lnorm slope prior) | OpenCode `term3cded168` owns AC (TASK; successor of `ctx_b9cb84086593`/`term_47c1c3b9` fmls-ac-slope-prior) | Base `99c228a8` + dirty `bifactor_grm.rs` SlopePrior; preserved `ping_preserve_20260921T0323Z/ac/diff.patch` (4453 B) | NOT ACCEPTED (HOLD) per `AC_SLOPE_PRIOR_VERIFY_0302.md`: support/Jacobian, 0-boundary, fixed-anchor exclusion, FD, surfaces |
| W MG (a_general 2D vs 1D helper) | Cursor `62f7339e` FAILED PING; prior `ctx_c311ec6ce1c1` | **No `ping_preserve/.../MG` dir — exact preserved diff ABSENT** | Handoff proposal (no duplicate impl): single successor recovers 62f7339e WT if reachable, else re-derives from `ctx_c311` spec; blocked until diff or scope re-issue lands |
| E (FIPC mean Class-A) | E-OpenCode Air `term84af046d` resumed by root (TASK); plus `ctx_6816b0402fca`/`term_750181e5` (mean) + `ctx_cda09db9ede6`/`term_71c2bea8` (harness-glob); next `ctx_d9c2b100491f` | `CONSUMER_32807ed0.json` all_pass=false, converged=false, max_iter; binary sha16 `a0453b7a` = 32807ed0 wheel (provenance correction disproves stale-binary read); `FIPC_STALL_UPDATE_DIAG_6c30712d.*` + `RUST_ACCEPT_PATH_CAPTURE_3fc6160a.*` + `e_mean_diag/MEAN_DRIFT_DIAG.*` | Fix early mean-step rejection (joint/mean-only probes +0.89 LL better yet rejected; mean frozen rows 0→95); keep FIPC target + LL guard; stall-exit≠accept; canonical s1 consumer @ tip with decision trace |
| Harness (sha+row_order) | `ctx_41d1c0d9afa4`/`term_de44534f` (fmls-fipc-harness-provenance); E-harness `ctx_cda09db9ede6` scoped to consumer script | Base `99c228a8` clean (0-byte diffs); PING recovered same ctx | Fix `sha`/`build_source_sha` stamping (still `6c30712d`), REQUIRED_GATES two-sided sd (`max>1.01` or `min<0.99`), `expected_raw` flat `a_primary` reshape |

## One concrete next action

Root/CO assigns a single W-MG successor to recover-or-reissue the `62f7339e` diff (AC owner `term3cded168` and E owner `term84af046d` held unchanged; no parallel re-implementation, no new builds until the handoff lands).
