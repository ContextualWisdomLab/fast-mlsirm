# Bifactor FIPC source scope (2026-09-27)

Source: Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement, 43*(4), 355–381. https://doi.org/10.1111/j.1745-3984.2006.00021.x

The actual local PDF is read with pdftotext -layout. PDF pages 7, 8, 9 have printed footers 361, 362, 363, establishing offset +354 for these passages. Printed p. 362 equations 14–15 describe updating discrete weights at ability points q_k. Printed p. 363 Table 1 and the following paragraph state that all five methods use fixed ability points across EM cycles. Operative phrase: “The ability points should not be rescaled after each EM cycle.” Text extraction is inspected; page images are not independently compared this turn. The prior PR #2112 source audit records group item BPIRS7P8 / attachment FXIWS8GY; those catalogue identifiers are historical information and are not revalidated by this inspection.

Local PDF SHA-256: `4b8fba4bb92a998c7198d9aa41430c1f44e8a11e9892cfbe178c009d643e3b9b`.

Inspected implementation base: 6dd48140c1a267315c7ad1e63a55a47661449c4a. In crates/mlsirm-core/src/bifactor_grm.rs, the focal fitter updates general mean/variance from posterior moments and optionally specific variances, keeps specific means zero, and moves general/specific nodes with their fitted Gaussian SDs. This differs from the fixed-node weight update cited above. The previous attribution of this exact algorithm to Kim's MWU-MEM and the no-rescaling claim are withdrawn from the Python docstring and corresponding Rust documentation. Unverified Bock–Zimowski attribution in this FIPC section is removed. Broader multigroup claims elsewhere are outside this correction's scope.

The source comparison does not establish the Gaussian fitter's scientific validity, recovery, interval coverage, or numerical adequacy. These require their own opened source passages and empirical checks. No estimation algorithm, settings, or returned values change here. Python executable AST excluding docstrings and Rust non-comment lines are compared to the base and match exactly. No native build or model fit is claimed.

Research consequence: the E G+4+W path needs multiple primary factors and focal latent means as well as variances. Reusing the current bifactor FIPC API cannot silently preserve that contract: its specifics have zero means and its model has one general factor. Keep numerical work inside the library and resolve this contract before accepting study outputs.
