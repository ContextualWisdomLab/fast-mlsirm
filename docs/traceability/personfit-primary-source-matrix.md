# Nonparametric person-fit primary-source traceability

Status: **Proposed / incomplete scientific doctoring**  
Issue: #1790  
Protected source baseline when this artifact was started: `main@493326f2de49ea1704da0ded19868ed05d2fe00f`  
Rust numerical owner: `crates/mlsirm-core/src/personfit_np.rs`

This artifact separates three claims that must not be collapsed:

1. **computation** — what the shipped complete-data dichotomous statistics calculate;
2. **interpretation / flagging policy** — how a statistic or cutoff is used to identify an unusual response pattern;
3. **validity for an operational use** — whether that interpretation supports a particular decision for a particular population and purpose.

Only the first claim is in scope here. A source equation, package comparator, or code-level conformance result does not by itself establish a universal cutoff or validity for a high-stakes use.

## Evidence-state vocabulary

| State | Meaning | Satisfies primary-equation gate? |
|---|---|---:|
| `PRIMARY_READ_EQUATION` | The original/source publication was lawfully read at the exact equation/definition needed for the implementation. | Yes |
| `PRIMARY_READ_DEFINITION` | The original/source publication was lawfully read for the construct or ordering definition, but the exact equation used by this implementation was not established. | No |
| `PRIMARY_METADATA_ONLY` | Publisher or authoritative bibliographic metadata identifies the original/source publication, but the relevant full-text equation was not read. | No |
| `PEER_REVIEWED_BRIDGE` | A later peer-reviewed source states the definition/equation and cites the original source family. | No |
| `PACKAGE_COMPARATOR` | An immutable package implementation is used as a software comparator. It is not scientific authority by itself. | No |
| `UNAVAILABLE` | The relevant original text/equation has not been lawfully obtained. | No |

One statistic can legitimately carry several evidence states at once. `PRIMARY_METADATA_ONLY`, `PEER_REVIEWED_BRIDGE`, or `PACKAGE_COMPARATOR` must never be promoted silently to `PRIMARY_READ_EQUATION`.

## Shipped-scope contract

Protected main exposes seven statistics through `PersonFitNp`: `g`, `gnormed`, `nci`, `u3`, `zu3`, `c_sato`, and `cstar`. The current Rust module explicitly identifies itself as a computational port of CRAN PerFit source at commit `c9df433cba3d7b03d16284e832d55785cb90464c`; it also explicitly states that the named original statistical sources were not read for the implementation. That package-source provenance is retained as an immutable comparator, not upgraded into primary-source authority.

The shipped scope is complete dichotomous `{0,1}` response data. Missing-value imputation and polytomous variants are outside this function. CPU `f64` remains the numerical reference for this scalar kernel.

## Equation-to-code matrix

The bridge equations below are from Tendeiro, Meijer, and Niessen (2016), a peer-reviewed Journal of Statistical Software article. They are recorded as `PEER_REVIEWED_BRIDGE`, not as evidence that the original 1970s/1980s source equation was directly read.

| Statistic / Rust field | Original/source family | Current evidence state | Peer-reviewed bridge / declared computation | Rust implementation identity | Primary-source gap |
|---|---|---|---|---|---|
| G / `g` | van der Flier (1977/1980); Meijer (1994) | `PRIMARY_METADATA_ONLY` + `PEER_REVIEWED_BRIDGE` + `PACKAGE_COMPARATOR` | Number of ordered `(0,1)` Guttman-error pairs after items are ordered from easiest to hardest. Tendeiro et al. (2016, p. 8) describes this definition. | `person_fit_np`: stable proportion-correct ordering and nested pair count | Read and pin the relevant original definition/equation and its ordering convention. |
| Gnormed / `gnormed` | van der Flier; Meijer (1994) | `PRIMARY_METADATA_ONLY` + `PEER_REVIEWED_BRIDGE` + `PACKAGE_COMPARATOR` | For dichotomous data, divide G by the maximum possible Guttman errors at total score `s_n`, namely `s_n (I - s_n)`; Tendeiro et al. (2016, p. 8). | `g[p] / (nc[p] * (I - nc[p]))`, with declared perfect-row convention | Establish the original normalization definition and any boundary convention from primary text. |
| NCI / `nci` | Tatsuoka & Tatsuoka (1982, 1983) | `PRIMARY_READ_DEFINITION` + `PEER_REVIEWED_BRIDGE` + `PACKAGE_COMPARATOR` | The 1982 publisher abstract defines NCI as proximity to a baseline response pattern with all 0s before all 1s under a prescribed item order. Tendeiro et al. (2016, p. 8) gives `NCI = 1 - 2 G_n`. | `1.0 - 2.0 * gnormed[p]` for non-perfect rows; explicit package-faithful perfect-row convention | Obtain the exact primary equation/page and determine whether the implemented perfect-row convention is part of the statistic definition or package post-processing. |
| U3 / `u3` | van der Flier (1982) | `PRIMARY_METADATA_ONLY` + `PEER_REVIEWED_BRIDGE` + `PACKAGE_COMPARATOR` | Tendeiro et al. (2016, Eq. 5) gives `U3 = [f(x*) - f(x)] / [f(x*) - f(x')]`, where `f(x) = Σ x_i log[p_i/(1-p_i)]`, `x*` is the Guttman vector and `x'` the reversed Guttman vector at the same total score. | `lo`, cumulative easiest/hardest sums, `xdot_lo`, then normalized difference | Read the exact van der Flier equation and boundary conventions directly. |
| ZU3 / `zu3` | van der Flier (1982) | `PRIMARY_METADATA_ONLY` + `PEER_REVIEWED_BRIDGE` + `PACKAGE_COMPARATOR` | Tendeiro et al. (2016) states that ZU3 is a standardized-normal version of U3 and notes published concerns about the adequacy of its asymptotic approximation. | `alpha`, `expv`, `beta`, `varv`, standardized `(u-expv)/sqrt(varv)` | Obtain the primary standardization equations and assumptions; do not turn asymptotic-normal language into a universal cutoff claim. |
| Sato C / `c_sato` | Sato (1975) | `PRIMARY_METADATA_ONLY` + `PEER_REVIEWED_BRIDGE` + `PACKAGE_COMPARATOR` | Tendeiro et al. (2016, Eq. 3) gives `C = 1 - Cov(x,p) / Cov(x*,p)`, with `x*` correct on the `s_n` easiest items. | `1.0 - cov1(row_ord,pio) / cov1(easiest,pio)` | Obtain the 1975 source text/equation and verify the exact covariance/boundary convention; do not substitute the separate 1980 S-P report as though it were the 1975 source. |
| C* / `cstar` | Harnisch & Linn (1981) | `PRIMARY_METADATA_ONLY` + `PEER_REVIEWED_BRIDGE` + `PACKAGE_COMPARATOR` | Tendeiro et al. (2016, Eq. 4) gives `[Cov(x*,p)-Cov(x,p)] / [Cov(x*,p)-Cov(x',p)]`, where `x'` is the reversed Guttman vector. | algebraically equivalent cumulative proportion-correct normalization | Obtain and read the 1981 primary equation and conventions. Preserve the verified title “Analysis of item response patterns: Questionable test data and dissimilar curriculum practices.” |

## Code-current boundary observations

The protected implementation already carries several package-faithful conventions that require separate source disposition instead of being assumed to follow from the equation alone:

- stable descending item proportion-correct ordering with original-column-index tie break;
- exact rejection of non-`{0,1}` input and ragged rows;
- G/Gnormed/NCI special handling for perfect rows;
- NaN outputs for U3/ZU3/C/C* on perfect or declared degenerate arithmetic;
- non-finite log-odds from item proportions of 0 or 1 normalized to zero before U3/ZU3 arithmetic;
- sample covariance (`n-1`) in the C computation, whose common denominator cancels in the ratio.

Each convention needs one of: direct primary-source support, a clearly named package-compatibility decision, or a separately reviewed scientific discrepancy decision. “It matches PerFit” is acceptable comparator evidence but is not enough to relabel the convention as primary-source-defined.

## Required conformance RED → GREEN after source ownership clears

PR #1764 is the current production writer for `crates/mlsirm-core/src/personfit_np.rs`. This doctoring lane must not compete with it. When that writer clears, the executable conformance slice should start with fail-first evidence against the then-current protected implementation and keep all seven statistics in one bounded contract:

- complete heterogeneous `{0,1}` matrices;
- tied item proportions and deterministic ordering;
- perfect rows and all-equal/degenerate item proportions;
- minimum admitted person/item dimensions;
- source-derived hand-computable fixtures wherever a primary equation is lawfully available;
- immutable PerFit comparator evidence for the declared complete-data scope;
- exact/ULP/tolerance policy justified statistic by statistic rather than one blanket correlation threshold;
- unchanged error precedence for ragged, non-binary, NaN/missing, and too-small inputs;
- explicit distinction between source-faithful NaN/zero rules and incidental IEEE behavior.

If a primary source disagrees with the PerFit comparator or current Rust arithmetic, record the discrepancy first and route the formula change through a separate scientific-decision repair. This artifact does not authorize a silent formula rewrite.

## Standards boundary

The 2014 *Standards for Educational and Psychological Testing* remains the current published AERA/APA/NCME standards baseline while the sponsoring organizations revise that edition. Revision-session and draft material is watch evidence, not a replacement normative standard until a new edition is published. For this numerical traceability work, that means evidence supporting score computation must remain distinguishable from evidence supporting score interpretation and use.

## References (APA 7)

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Harnisch, D. L., & Linn, R. L. (1981). Analysis of item response patterns: Questionable test data and dissimilar curriculum practices. *Journal of Educational Measurement, 18*(3), 133–146. https://doi.org/10.1111/j.1745-3984.1981.tb00848.x

Meijer, R. R. (1994). The number of Guttman errors as a simple and powerful person-fit statistic. *Applied Psychological Measurement, 18*(4), 311–314. https://doi.org/10.1177/014662169401800402

Sato, T. (1975). *The construction and interpretation of S-P tables*. Meiji Tosho. [Primary equation not yet read in this doctoring lane.]

Tatsuoka, K. K., & Tatsuoka, M. M. (1982). Detection of aberrant response patterns and their effect on dimensionality. *Journal of Educational Statistics, 7*(3), 215–231. https://doi.org/10.3102/10769986007003215

Tendeiro, J. N., & Meijer, R. R. (2014). Detection of invalid test scores: The usefulness of simple nonparametric statistics. *Journal of Educational Measurement, 51*(3), 239–259. https://doi.org/10.1111/jedm.12046

Tendeiro, J. N., Meijer, R. R., & Niessen, A. S. M. (2016). PerFit: An R package for person-fit analysis in IRT. *Journal of Statistical Software, 74*(5), 1–27. https://doi.org/10.18637/jss.v074.i05

van der Flier, H. (1982). Deviant response patterns and comparability of test scores. *Journal of Cross-Cultural Psychology, 13*(3), 267–298. https://doi.org/10.1177/0022002182013003001

## Unresolved source work

The artifact remains deliberately incomplete until the relevant original texts are lawfully obtained and the exact equation/page/convention evidence is recorded for every shipped statistic. In particular, bridge-equation coverage for C, C*, U3, G/Gnormed, and NCI does not close the primary-equation gate; ZU3 still needs its primary standardization equations and assumptions. Closure of #1790 additionally requires executable conformance evidence for all seven statistics and explicit disposition of every discovered source discrepancy.