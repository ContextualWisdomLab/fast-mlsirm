# Fixed-item focal free-mean rank guard

Parent candidate: 3ec4ed31c111f6fceae9333d5f4af868c7b4bd52 (PR #2214).

## Source and scope

Cai (2010), p.589 equation 11 defines the fixed linear predictors. A loading nullspace shift A*v=0 leaves them unchanged under mu+v. This is a derived necessary identification condition, not a claim that Cai prescribes this implementation.

LAPACK DGETF2 Purpose, INFO and pivot loop were opened at https://www.netlib.org/lapack/double/dgetf2.f. Reuse the existing partial-row-pivot helper on column-normalized A. max(rows, columns)*machine epsilon is an implementation guard, not a published scientific cutoff. DGELSY defines effective rank using pivoted QR and a condition bound; this helper does not implement that definition.

Passing this guard does not establish variance or joint identification, conditioning, population recovery, actual-source convergence or quadrature sensitivity. Caller-declared-prior scoring remains valid without fitting free means.

## Evidence

The installed source 9b00cf2dcb0cfd8ada2d68be29d721e696e8ef9f candidate accepts equal primary/specific loading columns. Distinct means [0,0] and [0.5,-0.5] give the same native log likelihood and both loose-tolerance branch probes claim convergence. These are synthetic defect diagnostics, not study estimates.

The actual native rejection test fails against that old installed extension with `DID NOT RAISE ValueError`: 1 failed in 1.06s. Receipt: s1 `/data/orca/workspaces/fmls-focal-native-9b00cf2-20260927/native-mean-rank-old-core.log`, with pytest status recorded separately in `.exit`.

An extracted-current-function Rust harness passes 5 tests (0 failures), including duplicate columns, a three-column dependency, insufficient observed rows, a valid six-latent bank, missing priors and existing EM termination paths. Local receipt: `/private/tmp/focal-mean-rank-result-20260927.log`. This is bounded function evidence, not full Cargo or rebuilt native acceptance.

## Remaining acceptance

Rebuild this exact repair head, run the new native rejection test and prior native/routing/six-latent cases, and run LLTM/DIF regression checks because the existing rank helper is shared. Required current-head hosted checks and substantive independent review remain necessary. No immutable release or study estimate is accepted by this report.
