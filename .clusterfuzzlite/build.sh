#!/bin/bash -eux

cd "$SRC/fast-mlsirm"
cargo fuzz build -O neg_loglik
cp fuzz/target/x86_64-unknown-linux-gnu/release/neg_loglik "$OUT/"
