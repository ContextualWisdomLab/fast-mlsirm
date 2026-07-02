# Support Policy

## Commercial Beta Support Scope

`fast-mlsirm` 0.1.x is supportable as a commercial beta for teams that need:

- MLS2PLM binary-response simulation;
- regularized point-estimate fitting through the NumPy or Rust backend;
- fit, dimensionality, and response-process diagnostics from supported inputs;
- CLI and Python API workflows on supported Python versions.

Support covers installation, reproducible crashes, backend parity regressions,
documented CLI/API behavior, and diagnostics output mismatches against the
documented formula contract.

## Out of Scope

The current commercial beta does not include:

- clinical, educational-placement, hiring, or high-stakes decision guarantees;
- Bayesian posterior inference;
- ordinal model estimation;
- hosted dashboards or multi-user administration;
- performance guarantees for sparse or very large response matrices;
- custom psychological construct interpretation.

## Requesting Support

Open a GitHub issue with:

- package version and commit SHA;
- operating system, CPU architecture, Python version, and Rust version;
- install command and backend selection;
- minimal reproduction data or a synthetic reproduction script;
- expected and observed behavior.

For private datasets, replace the data with a synthetic reproduction before
opening an issue.
