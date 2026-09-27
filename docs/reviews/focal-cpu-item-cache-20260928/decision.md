# Optional CPU item-table reuse

Keep the explicit opt-in candidate; CPU streaming remains the default. Reuse the existing E-step table argument and GPU table producer. Cai (2010), p.588 eq.9 and Appendices A/B pp.608–609 define the fixed item/node probabilities; DOI 10.1007/s11336-010-9178-0. Rust Vec::try_reserve_exact documentation explicitly returns errors for capacity overflow or allocator failure; https://doc.rust-lang.org/std/vec/struct.Vec.html#method.try_reserve_exact (operative body opened). Python 3.12 perf_counter measures elapsed durations by differences; https://docs.python.org/3.12/library/time.html#time.perf_counter (operative body opened).

Baseline source 4fb92ba; experiment source 2d26508 committed before measurement. Apple M1, Python 3.12.14; 384 synthetic persons, same fixture, 16 items, P2/S4, four categories, five nodes, three timings. Baseline median 0.104023000 s; candidate 0.014406958 s. Native means, second moments, SDs and loglik hashes are byte-identical. This bounded benchmark does not estimate study runtime or scientific accuracy.

Two native full-product fixture checks and one actual Apple M1 GPU score/update parity check pass, zero skips/errors/failures. Initial default test command skips two GPU tests; actual hardware invocation explicitly enables FOCAL_GPU_NATIVE=1 and selects only the short parity node. Full candidate continuous population recovery remains pending. CPU cache memory grows with items × primary grid × specific nodes × categories; tables rebuild after every density update.

Reproduce from the repository root: `.venv/bin/python docs/reviews/focal-cpu-item-cache-20260928/benchmark.py --cache`.

API validation/native bridge/mean-rank checks: 21 passed, zero skips/errors/failures (`api-native.xml`). Full continuous recovery is live on the same isolated installed core, recorded in `continuous-live.json`; no terminal result is inferred.
