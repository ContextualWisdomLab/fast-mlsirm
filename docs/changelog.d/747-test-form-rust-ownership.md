# Fixed-form greedy assembly Rust ownership

## Fixed

- Public `assemble_test_form` delegates ordering, exclusion, and content-feasibility
  decisions to the compiled Rust core (`assemble_test_form_greedy`), keeping Python
  for validation and marshalling only.
