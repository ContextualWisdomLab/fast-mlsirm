# Customer copy actionability

## Changed

- Public-API error paths now state the invalid input and the concrete next
  action instead of internal validation vocabulary: judge-scoring projection
  errors say to rebuild criterion mappings as plain dicts keyed by criterion id
  (replacing "exact built-in dict" jargon), Rust backend unavailability errors
  name the install/reference-path next steps, and unknown serving item codes
  point at the bundle's items list.
- CLI workspace-boundary and candidate-input errors append actionable next
  steps (move the file under the working directory, use unique
  `label=path.npy` candidate flags, reduce oversized candidate sets) without
  changing exit codes, validation order, or fail-closed behavior.
- Diagnostics report renderer errors tell the customer how to recover:
  choose a `.html` output name and regenerate unsupported JSON via
  `fast-mlsirm diagnose-fit` / `fast-mlsirm diagnose-dimensions`.
- README quickstart guidance no longer instructs an unexecutable command
  (`fit --backend numpy`; production backend choices are `{rust, auto}`) and
  now points to the explicit NumPy reference path (`--reference` /
  `fast_mlsirm.fit_reference`); feature copy describes fixed-item calibration
  by its method instead of legacy package names.
