#!/bin/bash -eux

cd "$SRC/fast-mlsirm"
mkdir -p "$OUT"
id
echo "LIB_FUZZING_ENGINE=${LIB_FUZZING_ENGINE:-unset}"
ls -ld "$SRC/fast-mlsirm" "$OUT"
cargo fuzz build -O neg_loglik
cp fuzz/target/x86_64-unknown-linux-gnu/release/neg_loglik "$OUT/"
