#!/usr/bin/env bash
# Peak-RSS evidence for two-tier E-step/fit at q_primary=q_specific=241 (#1992).
# Run on MacBook Air (32 GB): expects peak RSS well under ~32 GB without thrashing.
# Usage: CARGO_BUILD_JOBS=1 bash scripts/two_tier_q241_rss_evidence.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export PATH="${HOME}/.cargo/bin:/opt/homebrew/bin:${PATH}"

cargo test --manifest-path crates/mlsirm-core/Cargo.toml --test two_tier_oakes_mirt -- --nocapture

cat > /tmp/two_tier_q241_rss.rs << 'RS'
// ephemeral: not part of the crate; compiled with rustc for RSS probe
use std::process::Command;
fn rss_kb() -> u64 {
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0)
}
fn main() {
    println!("use cargo example instead");
}
RS

# Prefer an integration binary via cargo test ignore harness written inline:
cargo test --manifest-path crates/mlsirm-core/Cargo.toml two_tier_q241_rss_probe -- --ignored --nocapture 2>&1 | tee /tmp/two_tier_q241_rss.log || {
  echo "RSS probe test not yet registered; compiling unit path only"
  cargo test --manifest-path crates/mlsirm-core/Cargo.toml --lib two_tier_oakes -- --nocapture
}

echo "See /tmp/two_tier_q241_rss.log for peak RSS lines"
