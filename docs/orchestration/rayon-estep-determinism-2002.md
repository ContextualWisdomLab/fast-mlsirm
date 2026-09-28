# Deterministic rayon E-step person parallelism (#2002)

## Decision

CPU E-step person sweeps in `bifactor_grm` and `two_tier_grm` partition
respondents into a caller-chosen number of chunks
(`e_step_n_chunks`) whose boundaries depend only on `n_persons` and that
count. Each fit start owns one **local** rayon pool of size
`e_step_n_threads` (never `build_global`) and reuses it for every E-step
of that start. Non-empty chunk partials are folded in **chunk-index
order**. The first partial is the accumulator, so a one-chunk sweep
keeps a single counts tensor; later windows hold that accumulator plus
at most `e_step_n_threads` new partials. Trailing
empty chunks are not evaluated, so they do not allocate a counts tensor.
`e_step_n_chunks` is not clamped.

## Why

Floating-point addition is not associative (Higham, 2002, §§4.1–4.2). A
reduction whose association tree depends on the runtime thread count is
not reproducible. Fixing the association tree to `e_step_n_chunks` makes
same-chunk-count runs bit-identical across thread counts and across
repeated runs. Person contributions are independent in Bock–Aitkin EM
(Bock & Aitkin, 1981) and in the Gibbons–Hedeker reduced bifactor E-step
(Gibbons et al., 2007).

## Provenance (ADR-0028)

Both `e_step_n_chunks` and `e_step_n_threads` are required caller
arguments with no unsourced defaults; they are recorded on fit results.

## Tests

- `estep_parallel::tests::ordered_map_is_bit_identical_across_thread_counts`
- `bifactor_grm::tests::estep_chunked_is_bit_identical_across_thread_counts`
- `bifactor_grm::tests::fit_same_chunks_bit_identical_across_thread_counts`
- `bifactor_grm::tests::fit_records_e_step_chunk_provenance`
- `two_tier_grm::tests::fit_records_e_step_chunk_provenance`
- `estep_parallel::tests::fold_from_first_matches_zero_seed_and_skips_empty_input`
