# Ignored Rust Subprocess Deadlines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound Cargo metadata, ignored-test inventory, and long-running statistical-study subprocesses without terminating valid recovery studies or leaking child-controlled output on timeout.

**Architecture:** Add one standalone script utility that owns operation-specific deadline policy and bounded process execution, then migrate only `run_ignored_rust_shard.py` in this slice. GitHub Actions job/step timeouts remain an independent outer ceiling; other repository subprocess call sites remain follow-up work under issue #555.

**Tech Stack:** Python 3.12+ standard library, `subprocess.Popen`, POSIX process groups, pytest, GitHub Actions.

## Global Constraints

- Preserve the scientific recovery workload; do not replace it with a universal short timeout.
- Timeout evidence must not echo command arguments, stdout, stderr, source content, or provider-controlled values.
- Operator overrides are decimal integer seconds constrained by an operation-specific minimum and maximum.
- POSIX statistical CI runs start in a new session and terminate the process group on timeout.
- Existing GitHub Actions `timeout-minutes` values remain independent outer ceilings.
- No new runtime dependency.
- Added production paths require complete statement/branch tests and complete public docstrings.
- The PR stays Draft until exact-head Python, Rust/PyO3, package, GPU-no-skip, fuzz, Security Scan, and SAST evidence is green.

---

### Task 1: Define RED deadline and cleanup contracts

**Files:**
- Create: `tests/test_subprocess_deadlines.py`

**Interfaces:**
- Consumes: future `scripts/_subprocess_deadlines.py`
- Produces: exact required API and failure behavior for Task 2

- [x] **Step 1: Write failing tests** for operation defaults, bounded overrides, redacted timeout evidence, process-group isolation, SIGTERM/SIGKILL escalation, and `check=True` behavior.
- [ ] **Step 2: Run the pull-request Python job and verify RED** because `scripts/_subprocess_deadlines.py` does not yet exist.

### Task 2: Implement the minimal bounded subprocess utility

**Files:**
- Create: `scripts/_subprocess_deadlines.py`
- Test: `tests/test_subprocess_deadlines.py`

**Interfaces:**
- Produces: `SubprocessOperation`, `BoundedSubprocessTimeout`, `PROCESS_GROUP_GRACE_SECONDS`, `resolve_timeout_seconds()`, `run_bounded()`.

- [ ] **Step 1: Add three policies:** metadata 30 s in [5,120], Cargo ignored-test inventory 120 s in [30,600], statistical tests 1800 s in [60,7200].
- [ ] **Step 2: Use `Popen` with `start_new_session=True` on POSIX and captured pipes only when requested.**
- [ ] **Step 3: On timeout, terminate the POSIX process group, escalate after a bounded grace period, drain the process, and raise only the redacted structured timeout error.**
- [ ] **Step 4: Preserve ordinary `CalledProcessError` behavior only when `check=True`.**
- [ ] **Step 5: Run focused tests until GREEN.**

### Task 3: Migrate the ignored Rust shard runner

**Files:**
- Modify: `scripts/run_ignored_rust_shard.py`
- Modify: `tests/test_ignored_rust_shard.py`

**Interfaces:**
- Consumes: `run_bounded()` and the three operation classes from Task 2.

- [ ] **Step 1: Add tests proving each runner path selects the intended operation policy and a timeout returns a non-zero process result without being treated as success.**
- [ ] **Step 2: Replace the three unbounded `subprocess.run` call classes with `run_bounded`.**
- [ ] **Step 3: Preserve exact Cargo command vectors, inventory, sharding, and dedicated-test exclusion semantics.**
- [ ] **Step 4: Run `pytest tests/test_subprocess_deadlines.py tests/test_ignored_rust_shard.py -q`.**

### Task 4: Add operating evidence and documentation

**Files:**
- Create: `docs/doctoring/operation-specific-subprocess-deadlines.md`
- Create: `docs/changelog.d/555-operation-specific-subprocess-deadlines.md`
- Modify: `CHANGELOG.md` through `scripts/render_changelog_fragments.py --update`

**Interfaces:**
- Produces: APA 7 primary-source traceability and release-note evidence.

- [ ] **Step 1: Document deadline rationale, outer Actions timeouts, process-group behavior, redaction, override ranges, failure semantics, and rollback.**
- [ ] **Step 2: Cite current Python subprocess and GitHub Actions workflow-syntax primary documentation in APA 7 form.**
- [ ] **Step 3: Render and check the authoritative changelog block.**

### Task 5: Exact-head verification and review

- [ ] **Step 1: Verify focused Python tests.**
- [ ] **Step 2: Verify full Python test/coverage/docstring gates.**
- [ ] **Step 3: Verify Rust workspace and PyO3 tests.**
- [ ] **Step 4: Verify package reinstall and release acceptance.**
- [ ] **Step 5: Verify explicit GPU no-skip, fuzz, Security Scan, and SAST.**
- [ ] **Step 6: Request exact-head independent review only after the unchanged head is green.**
- [ ] **Step 7: Mark Ready and enable protected auto-merge only when every required gate is satisfied.**

## Self-review

- Issue #555 is intentionally not closed by this PR: GitHub CLI, packaging, release-evidence, and other subprocess call sites remain later operation-specific slices.
- No placeholder implementation is delegated to Python psychometric code; this slice changes CI/reliability control flow only.
- The scientific study receives a substantially longer deadline than metadata/inventory work and retains its existing Actions job ceiling.
