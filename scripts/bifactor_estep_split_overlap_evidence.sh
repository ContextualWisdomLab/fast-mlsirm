#!/usr/bin/env bash
# PR #2043 manual overlap evidence: concurrent CPU+GPU bifactor E-step split (#2001 L3).
# Captures monotonic wall offsets printed by measure_concurrent_split_estep_vs_cpu_reference.
# Not a CI gate. Q=21 is a smoke grid; Q=121 is a representative grid observation only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.cargo/bin:/opt/homebrew/bin:${PATH}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"

OUT="${1:-docs/orchestration/bifactor-estep-split-overlap-2043-timestamps.txt}"
mkdir -p "$(dirname "$OUT")"

{
  echo "# bifactor E-step split overlap evidence (PR #2043)"
  echo "# captured: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "# git_head: $(git rev-parse HEAD)"
  echo "# command: cargo test --manifest-path crates/mlsirm-core/Cargo.toml --lib measure_concurrent_split_estep_vs_cpu_reference -- --ignored --nocapture"
  echo "# host: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  echo "# metal: $(system_profiler SPDisplaysDataType 2>/dev/null | awk '/Metal Support/ {print $3, $4; exit}')"
  echo
  cargo test --manifest-path crates/mlsirm-core/Cargo.toml --lib measure_concurrent_split_estep_vs_cpu_reference -- --ignored --nocapture
} 2>&1 | tee "$OUT"

echo "Wrote $OUT"
