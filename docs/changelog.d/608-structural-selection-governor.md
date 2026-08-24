# Govern structural-model pair decisions

## Added

- Add a governed structural-model selection gate that keeps factor retention separate from structure choice, requires explicit parameter-space relation evidence, refuses pairwise selection before the relation-appropriate LR/bootstrap/Vuong procedure, and gates any winner on recovery and intended-score interpretation evidence. The new Python surface performs validation and policy orchestration only; numerical comparison and psychometric arithmetic remain Rust-owned.
