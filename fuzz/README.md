# Fuzzing fast-mlsirm

This directory holds the coverage-guided fuzz harnesses for the untrusted-input
surfaces of fast-mlsirm. Property-based tests that run inside the normal
`pytest` / `cargo test` suites live alongside the source
(`tests/test_fuzz_properties.py`, `crates/mlsirm-core/tests/proptest_neg_loglik.rs`).

## Why these targets

The surfaces were selected with **CodeGraph** by ranking blast radius on the
parser / loader / validator nodes:

```
codegraph explore "parse load config CSV JSON input file deserialization untrusted input"
codegraph explore "neg_loglik_and_grad config Params ModelConfig"
```

| Surface | File | Why it is untrusted |
| --- | --- | --- |
| `load_factor_csv` | `python/fast_mlsirm/io.py` | Reads an on-disk item→factor CSV via `numpy.loadtxt`; reached from the CLI. |
| `render_diagnostics_report` | `python/fast_mlsirm/report.py` | Parses an **arbitrary JSON** diagnostics file and renders it to HTML. |
| `MLS2PLMConfig` / `FitConfig` `.validate()` | `python/fast_mlsirm/config.py` | Every CLI / API call funnels user numeric parameters through these validators. |
| `neg_loglik_and_grad` | `crates/mlsirm-core/src/lib.rs` | The core numeric kernel — widest Rust blast radius; consumes response data + parameter vectors. |

## Tools & Licenses

| Tool | License | Where |
| --- | --- | --- |
| [Atheris](https://github.com/google/atheris) (coverage-guided) | Apache-2.0 | `fuzz/atheris/*.py` |
| [Hypothesis](https://hypothesis.readthedocs.io/) (property-based) | MPL-2.0, test-only dev dependency | `tests/test_fuzz_properties.py` |
| [proptest](https://github.com/proptest-rs/proptest) (property-based) | MIT / Apache-2.0 | `crates/mlsirm-core/tests/proptest_neg_loglik.rs` |

## The contract

Every harness enforces the same invariant: on **arbitrary input** the code
under test either succeeds or fails with a *documented, benign* exception
(`ValueError` / `json.JSONDecodeError` / `UnicodeDecodeError` / `OSError`). An
`AssertionError`, `KeyError`, `IndexError`, `TypeError`, `RecursionError`, panic
or hang is a bug and produces a reproducer. Additional invariants:

- `render_diagnostics_report`: a planted `<script>` sentinel must never appear
  unescaped in the rendered HTML (HTML-injection guard).
- `neg_loglik_and_grad`: gradient vector lengths match the config, and finite
  inputs yield finite objective / log-likelihood / gradients.

## Running

Property tests (fast, run in CI automatically):

```bash
pip install -e .[dev]
pytest tests/test_fuzz_properties.py
cargo test --workspace          # includes proptest_neg_loglik
```

Coverage-guided Atheris harnesses (bounded budget, matches the CI `fuzz` job):

```bash
pip install -e .[fuzz]          # Atheris ships wheels for CPython 3.8-3.12
python fuzz/atheris/fuzz_load_factor_csv.py -max_total_time=60 fuzz/corpus/load_factor_csv
python fuzz/atheris/fuzz_render_report.py   -max_total_time=60 fuzz/corpus/render_report
python fuzz/atheris/fuzz_config.py          -max_total_time=60 fuzz/corpus/config
```

Longer local campaign: raise `-max_total_time` (seconds) or add `-runs=N`. New
crash reproducers are written to the working directory as `crash-<hash>`; drop
minimized ones into the matching `fuzz/corpus/<target>/` directory to grow the
seed corpus.

## Further reading

Manès et al., *The Art, Science, and Engineering of Fuzzing: A Survey*
(IEEE TSE 2019 / arXiv:1812.00140), https://arxiv.org/abs/1812.00140,
background on coverage-guided fuzzing as implemented by AFL / libFuzzer /
Atheris.
