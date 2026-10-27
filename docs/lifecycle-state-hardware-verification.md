# Lifecycle state and hardware verification

## Ordinary canonical CI

The central pinned Rust job runs `cargo test --locked --workspace`. Default
`gpu` features compile GPU code; they do not grant a host GPU lease. The three
physical hardware cases are explicitly ignored in ordinary selection, following
the repository's existing hardware policy. Validation-only tests stay active:
`native_hardware_enrollment_is_explicit` checks the exact integration ignored
set, and `gpu_group_update::tests::hardware_enrollment_is_explicit` checks the
unit registration. Neither test performs GPU dispatch. Ignored cases are not
passes and ordinary CI success does not satisfy hardware acceptance.

## Mandatory physical GPU acceptance

The host GPU owner must run all three cases through
`tools/verify_lifecycle_hardware.py`. General-only posterior/controller,
group-moment update, and mixed bifactor controls remain required without
relaxed assertions, settings or fallback. This runner refuses an incomplete
cohort. Mixed-case failure or its unchanged 240-second bound prevents an
accepted hardware receipt; two passing cases do not substitute for the third.

Prepare the exact source's native test executables with:

```sh
cargo +1.97.1 test --locked -p mlsirm-core --lib --test native_lifecycle --no-run --message-format=json
```

Use the compiler-artifact paths from that build, not a guessed Cargo hash:

```sh
python3 tools/verify_lifecycle_hardware.py \
  --lease "$CANONICAL_EXISTING_GPU_LEASE" \
  --integration "$NATIVE_LIFECYCLE_EXECUTABLE" \
  --unit "$MLSIRM_CORE_UNIT_EXECUTABLE" \
  --output "$NEW_PRIVATE_HARDWARE_RECEIPT_DIRECTORY"
```

For the current authorized macOS host, the existing canonical lease is
`/Users/seonghobae/.local/state/cwl-verification/full-fipc-score-only-20261011-025246/GPU-precision-AB-20261013/B/hierarchical_GPU_shared.lock`.
A different host requires its actual host-owner canonical path. Never create a
replacement lock. Never export `FMLS_DS_GPU_LEASED=1` from a shell to bypass
admission. Only the runner sets the marker after taking that existing lock and
observing independent contender refusal. The actual child inherits the lease;
its exact testcase, binary SHA-256, direct exit, deadline, settled wait and
streams are retained. Missing adapter, failure, timeout, absent/filtered case,
missing lease or contention returns nonzero and no accepted receipt.

This explicitly admitted local hardware gate is separate from the ordinary
central workflow. Publication/integration needs the exact successor's reviewed
hardware receipt as well as ordinary CI; this document is not a hosted workflow
run or an executed hardware receipt.

## Trusted private fitted state

Restore validates observed categories before native likelihood/count indexing;
unobserved response placeholders are ignored. Reference moments, fixed specific
moments, common item coordinates, inactive specific slopes, positive latent
SDs, finite EAP and finite/nonnegative posterior SD must match the saved model
contract. Category counts must equal the observed-response counts; fit traces
must have their native `n_iter + 1` shape. Inconsistency is rejected, never
silently normalized. Valid state is re-exported without optimization.

Posterior scores, traces, convergence flags and dispatch metadata remain saved
trusted values. Only the likelihood is evaluated on restore/summary. Persistence
is not authenticated execution evidence, cross-source migration, or proof of
all-start/all-stage GPU completion.
