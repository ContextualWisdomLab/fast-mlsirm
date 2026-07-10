#!/bin/bash -eux

cd "$SRC/fast-mlsirm"
mkdir -p "$OUT"
id
ls -ld "$SRC/fast-mlsirm" "$OUT" "$(dirname "$LIB_FUZZING_ENGINE")"
cargo fuzz build -O neg_loglik
cp fuzz/target/x86_64-unknown-linux-gnu/release/neg_loglik "$OUT/"
