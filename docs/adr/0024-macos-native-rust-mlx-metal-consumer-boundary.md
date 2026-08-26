# ADR-0024: macOS-native Rust-owned MLX Metal consumer boundary

Status: **Accepted**  
Implementation maturity: **contract-active / service-target**  
Date: 2026-08-26

## Context

MLX Metal cannot execute inside Colima's Linux VM. A consumer must not infer
Metal from Apple hardware, accept a forged container receipt, or move
psychometric arithmetic into Python.

## Decision

fast-mlsirm consumes accelerated TEPP artifacts only with the actual execution
receipt. Apple Silicon Metal is `mlx_metal_macos_native`, produced by a
macOS-native Rust-owned MLX service reached from Compose through authenticated
Unix-socket or host-gateway transport. Linux may report only `rust_cpu`,
`mlx_cpu`, `mlx_cuda`, or `rust_opencl` when actually executed;
`mlx_opencl` is invalid. Rust remains numerical authority and CPU f64 remains
the reference and explicit portability fallback.

Receipt, objective, parameter/draw digests, environment, and method-derived
parity are persisted and fail closed. Customer UI receives an actionable
availability explanation, not package, schema, hash, transport, or backend copy.

## Invariants and acceptance evidence

Contract tests reject Metal-in-Linux, unknown backends, missing receipts,
forged parity, and Python-owned results. Hardware E2E must prove native Metal;
container E2E proves authenticated host access and cannot claim Metal.
True-parameter recovery and CPU/MLX parity are required before numerical use.

## Alternatives considered

Colima Metal, Python-owned MLX, and inferred backend labels were rejected.
Native Rust-owned MLX plus authenticated transport and exact receipts was
accepted.

## Failure, operability, security, and recovery

Authentication, timeout, OOM, driver, receipt, or parity failure leaves the
result unavailable. Explicit CPU replay creates a new CPU receipt and never
rewrites Metal provenance. Transport is local, authenticated, bounded, replay
protected, and content redacting.

## Compatibility and rollback

This tightens the proposed v2 producer schema before estimator release. Rollback
disables acceleration and uses an explicitly receipted CPU run. A different
accelerator or authority requires coordinated TEPP/fast-mlsirm supersession.
