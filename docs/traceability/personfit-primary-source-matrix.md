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
| `PRIMARY_READ_EQUATION` | The original/source publication was lawfully read at the exact equation **and the implementation-relevant conventions needed to reproduce it**. | Yes |
| `PRIMARY_READ_DEFINITION` | The original/source publication was lawfully read for the construct or ordering definition, but the exact equation and implementation-relevant conventions used by this implementation were not established. | No |
| `PRIMARY_METADATA_ONLY` | Publisher or authoritative bibliographic metadata identifies the original/source publication, but the relevant full-text equation was not read. | No |
| `PEER_REVIEWED_BRIDGE` | A later peer-reviewed source states the definition/equation and cites the original source family. | No |
| `PACKAGE_COMPARATOR` | A versioned immutable package implementation is used as a software comparator. It is not scientific authority by itself. | No |
| `UNAVAILABLE` | The relevant original text/equation has not been lawfully obtained. | No |

One statistic can legitimately carry several evidence states at once. `PRIMARY_METADATA_ONLY`, `PRIMARY_READ_DEFINITION`, `PEER_REVIEWED_BRIDGE`, or `PACKAGE_COMPARATOR` must never be promoted silently to `PRIMARY_READ_EQUATION`.

## Shipped-scope contract and comparator identity

Protected main exposes seven statistics through `PersonFitNp`: `g`, `gnormed`, `nci`, `u3`, `zu3`, `c_sato`, and `cstar`. The current Rust module identifies itself as a computational port of the CRAN PerFit R package. The package comparator used by that implementation is now recorded without repository ambiguity:

- canonical CRAN mirror repository: `cran/PerFit`;
- package release: `PerFit 1.4.7`, published by CRAN on 2025-04-02;
- immutable source commit: `c9df433cba3d7b03d16284e832d55785cb90464c` (`version 1.4.7`);
- immutable Git tree: `faed8f05c92c08f63bb34781666a17c0f2f814cd`;
- source files read for this port: `R/G.R`, `R/Gnormed.R`, `R/NCI.R`, `R/U3.R`, `R/ZU3.R`, `R/C.Sato.R`, `R/Cstar.R`, `R/Accessory.R` (`final.PFS`), and `R/SanityChecks.R` (`Sanity.prv`).

The commit/tree pair is the immutable comparator identity used here. No downloaded CRAN tarball SHA-256 has been recorded, so this artifact does **not** claim byte identity to a separately downloaded `PerFit_1.4.7.tar.gz`. The package implementation is software evidence, not primary-source equation authority.

The protected Rust unit-test header also refers to `files/perfit_spec.md` as oracle documentation, but that path is not present on protected main. Until that reference is restored or replaced by a repository-tracked, versioned artifact, it must not be treated as independent comparator provenance.

The shipped scope is complete dichotomous `{0,1}` response data. Missing-value imputation and polytomous variants are outside this function. CPU `f64` remains the numerical reference for this scalar kernel.

## Equation-to-code matrix

The bridge equations below are from Tendeiro, Meijer, and Niessen (2016), a peer-reviewed *Journal of Statistical Software* article. They are recorded as `PEER_REVIEWED_BRIDGE`, not as evidence that the original 1970s/1980s source equation was directly read.

| Statistic / Rust field | Original/source family | Current evidence state | Peer-reviewed bridge / declared computation | Rust implementation identity | Primary-source gap |
|---|---|---|---|---|---|
| G / `g` | Loevinger (1947, 1948; historical all-error-pair definition); van der Flier (1977); Meijer (1994, 1995) | `PRIMARY_READ_DEFINITION` (Meijer, 1994) + `PRIMARY_METADATA_ONLY` (Loevinger, 1947, 1948; van der Flier, 1977) + `PEER_REVIEWED_BRIDGE` + `PACKAGE_COMPARATOR` | Meijer (1994, pp. 311–312) directly defines a Guttman error as a `0` on the easier item paired with a `1` on the more difficult item and orders items by nonincreasing proportion-correctness. Meijer (1995, p. 166) corrects the historical attribution: the all-error-pair count used by G is Loevinger's 1947/1948 definition, not Guttman's own 1950 error count. Tendeiro et al. (2016, p. 8) provides the later computational bridge. | Stable descending proportion-correct order; count an earlier `0` paired with a later `1`. | Read and pin Loevinger (1947, 1948) and van der Flier (1977) directly; establish the original equation lineage and tie convention. Meijer (1994) does not specify the current original-column-index tie break in the pages read here. |
| Gnormed / `gnormed` | Loevinger (1947, 1948; historical all-error-pair definition); van der Flier (1977); Meijer (1994, 1995) | `PRIMARY_READ_DEFINITION` (Meijer, 1994) + `PRIMARY_METADATA_ONLY` (Loevinger, 1947, 1948; van der Flier, 1977) + `PEER_REVIEWED_BRIDGE` + `PACKAGE_COMPARATOR` | Meijer (1994, p. 312) defines `G*` by dividing G by the maximum number of Guttman errors at a fixed number-correct score and states that the maximum for score `r` over `k` items is `r(k-r)`. Meijer (1995, p. 166) separately corrects the historical source of the underlying all-error-pair count to Loevinger (1947, 1948). Tendeiro et al. (2016, p. 8) gives the same normalization as a later bridge. | `g[p] / (nc[p] * (I - nc[p]))`, with declared perfect-row convention. | Read Loevinger (1947, 1948) and van der Flier (1977) directly and establish the original count/normalization lineage plus perfect-row convention; the exact displayed Meijer equation is not promoted beyond definition-level evidence here. |
| NCI / `nci` | Tatsuoka & Tatsuoka (1982; PerFit also cites the 1983 ICI article) | `PRIMARY_READ_DEFINITION` + `PEER_REVIEWED_BRIDGE` + `PACKAGE_COMPARATOR` | The 1982 publisher abstract defines NCI as proximity to a baseline with all `0`s before all `1`s when items are in a prescribed order, but the abstract does **not** state that order's direction. PerFit 1.4.7 operationalizes the equivalent Gnormed relation as `NCI = 1 - 2 Gnormed`. | PerFit/Rust order by descending proportion-correctness (easy→hard). In that order, a conforming fixed-score pattern has earlier `1`s then later `0`s; each earlier `0` paired with a later `1` is one Guttman error. This is the order used by conformance fixtures unless a directly read primary equation establishes a different convention. | Obtain the exact primary equation/page and source-order direction; determine whether the perfect-row `NCI = 0` convention is source-defined or package post-processing. The 1983 ICI paper is not promoted to NCI equation authority merely because PerFit cites it. |
| U3 / `u3` | van der Flier (1980, 1982) | `PRIMARY_METADATA_ONLY` + `PEER_REVIEWED_BRIDGE` + `PACKAGE_COMPARATOR` | Tendeiro et al. (2016, Eq. 5) gives `U3 = [f(x*) - f(x)] / [f(x*) - f(x')]`, where `f(x) = Σ x_i log[p_i/(1-p_i)]`, `x*` is the Guttman vector and `x'` the reversed Guttman vector at the same total score. | `lo`, cumulative easiest/hardest sums, `xdot_lo`, then normalized difference. | Read the exact van der Flier equation and boundary conventions directly. |
| ZU3 / `zu3` | van der Flier (1980, 1982) | `PRIMARY_METADATA_ONLY` + `PEER_REVIEWED_BRIDGE` + `PACKAGE_COMPARATOR` | Tendeiro et al. (2016) states that ZU3 is a standardized-normal version of U3 and notes published concerns about the adequacy of its asymptotic approximation. | `alpha`, `expv`, `beta`, `varv`, standardized `(u-expv)/sqrt(varv)`. | Obtain the primary standardization equations and assumptions; do not turn asymptotic-normal language into a universal cutoff claim. |
| Sato C / `c_sato` | Sato (1975) | `PRIMARY_METADATA_ONLY` + `PEER_REVIEWED_BRIDGE` + `PACKAGE_COMPARATOR` | Tendeiro et al. (2016, Eq. 3) gives `C = 1 - Cov(x,p) / Cov(x*,p)`, with `x*` correct on the `s_n` easiest items. | `1.0 - cov1(row_ord,pio) / cov1(easiest,pio)`. | Obtain the 1975 source text/equation and verify the exact covariance/boundary convention; do not substitute the separate 1980 S-P report as though it were the 1975 source. |
| C* / `cstar` | Harnisch & Linn (1981 journal article); Harnisch & Linn (1981 final report, ERIC ED222557) | `PRIMARY_READ_DEFINITION` + `PEER_REVIEWED_BRIDGE` + `PACKAGE_COMPARATOR` | Tendeiro et al. (2016, Eq. 4) gives `[Cov(x*,p)-Cov(x,p)] / [Cov(x*,p)-Cov(x',p)]`, where `x'` is the reversed Guttman vector. The directly read Harnisch–Linn final report defines the S-P/Guttman ordering and the 0..1 bounded modified caution-index intent, but its scanned displayed equation has not yet been reliably transcribed. | Algebraically equivalent cumulative proportion-correct normalization. | Reconcile the journal article and same-authors final-report equation at equation/page level. Do not promote the readable definition/order/boundedness evidence to `PRIMARY_READ_EQUATION` until the displayed equation is visually/transcription-verified. |

The NCI ordering statement is deliberately asymmetric about evidence strength: the primary publisher abstract establishes the `0...01...1` baseline **under an unspecified prescribed order**; the easy→hard direction and `(0,1)` error counting are pinned package-comparator behavior. A future primary-equation read must reconcile those orientations explicitly rather than choosing whichever direction makes a fixture pass.

## Directly read Meijer source-definition evidence

The University of Twente Research Information repository exposes Meijer (1994), *The Number of Guttman Errors as a Simple and Powerful Person-Fit Statistic*, as an open-access final published version. The directly read pp. 311–312 establish the following implementation-relevant definition evidence:

- a Guttman error is an item pair with a `0` on the easier item and a `1` on the more difficult item;
- items are ordered so the proportion correct is nonincreasing, which is the easy→hard direction used by the current comparator and Rust implementation;
- `G*` is introduced to reduce the number-correct-score dependence of raw G by dividing G by the maximum possible Guttman-error count at the same score;
- for score `r` over `k` items, that maximum is stated as `r(k-r)`.

The paper contains displayed equations for G and `G*`, but this doctoring pass does not promote them to `PRIMARY_READ_EQUATION`: the implementation-relevant tie rule and perfect-row convention are not established by the pages read, and the displayed equation glyphs were not independently verified here at transcription quality. The direct read therefore upgrades G and Gnormed only to `PRIMARY_READ_DEFINITION` for the Meijer (1994) source family. It does not satisfy the still-open van der Flier (1977) primary-equation gate, and it does not promote U3 because van der Flier remains the original U3 source owner.

## Historical all-error-pair attribution correction

The University of Twente repository also exposes Meijer (1995), the one-page peer-reviewed supplement to the 1994 article, as an open-access final published version. The supplement was read directly because it corrects a scientific provenance error that matters to the source matrix:

- Meijer states that calling G a count of errors from the deterministic Guttman model can wrongly imply that the count is Guttman's own error definition;
- Guttman (1950) instead counts responses predicted incorrectly from a person's scale score;
- the all-error-pair definition used for G is attributed by Meijer to Loevinger (1947, 1948);
- Meijer's five-item illustration reports different counts for the same aberrant response pattern under the two definitions, confirming that these are not merely two names for one arithmetic rule.

This is a **historical attribution correction**, not `PRIMARY_READ_EQUATION` evidence for Loevinger. The 1947 and 1948 Loevinger texts have publisher/authoritative metadata but their implementation-relevant pages/equations have not been directly verified in this doctoring lane. The matrix therefore adds them as `PRIMARY_METADATA_ONLY` and keeps the primary-equation gate open. It also avoids relabeling the shipped `g` field: any API-name change would require a separately reviewed compatibility decision, not a bibliographic correction.

## Directly read Harnisch–Linn primary definition evidence

The same-authors 1981 technical final report, *Identification of Aberrant Response Patterns* (ERIC ED222557; NIE grant G-80-0003), is available as lawful full text and was read directly for the modified caution-index definition boundary. Its `DEFINITION AND COMPARISON OF INDICES` section (report pp. 5–8; ERIC scan pages 13–16) establishes that:

- S-P-table item columns run left-to-right in ascending difficulty, i.e. easy→hard, while examinee rows run top-to-bottom by descending number correct;
- the ideal Guttman pattern places correct responses to easier items before correct responses to harder items, so a correct response to a difficult item implies correct responses to all easier items in the ideal pattern;
- Harnisch and Linn's modified form of Sato's caution index, `Ci*`, is introduced with lower bound 0 and upper bound 1 to avoid extreme values of the unmodified caution index.

The report also contains a displayed algebraic definition for `Ci*`, but the available scan/text rendering does not preserve that expression reliably enough for an exact transcription. This source is therefore `PRIMARY_READ_DEFINITION`, not `PRIMARY_READ_EQUATION`. The scientific gate stays open until the displayed equation can be visually/transcription-verified and reconciled with the 1981 *Journal of Educational Measurement* article and the PerFit/Rust `C*` expression. Shared authorship and year are not treated as proof of equation identity.

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
- immutable PerFit comparator evidence pinned to the version/commit/tree above;
- exact/ULP/tolerance policy justified statistic by statistic rather than one blanket correlation threshold;
- unchanged error precedence for ragged, non-binary, NaN/missing, and too-small inputs;
- explicit distinction between source-faithful NaN/zero rules and incidental IEEE behavior.

If a primary source disagrees with the PerFit comparator or current Rust arithmetic, record the discrepancy first and route the formula change through a separate scientific-decision repair. This artifact does not authorize a silent formula rewrite.

## Standards boundary

The 2014 *Standards for Educational and Psychological Testing* remains the current published AERA/APA/NCME standards baseline while the sponsoring organizations revise that edition. Revision-session and draft material is watch evidence, not a replacement normative standard until a new edition is published. For this numerical traceability work, that means evidence supporting score computation must remain distinguishable from evidence supporting score interpretation and use.

## References (APA 7)

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Harnisch, D. L., & Linn, R. L. (1981). Analysis of item response patterns: Questionable test data and dissimilar curriculum practices. *Journal of Educational Measurement, 18*(3), 133–146. https://doi.org/10.1111/j.1745-3984.1981.tb00848.x

Harnisch, D. L., & Linn, R. L. (1981). *Identification of aberrant response patterns: Final report* (NIE Grant G-80-0003; ERIC ED222557). University of Illinois, Urbana-Champaign / Education Commission of the States / National Institute of Education. https://files.eric.ed.gov/fulltext/ED222557.pdf [Full text directly read for S-P ordering, Guttman interpretation, and modified-index boundedness; exact displayed equation not yet reliably transcribed.]

Loevinger, J. (1947). A systematic approach to the construction and evaluation of tests of ability. *Psychological Monographs, 61*(4), i–49. https://doi.org/10.1037/h0093565 [Primary bibliographic metadata verified; implementation-relevant error-pair pages/equation not yet directly verified in this lane.]

Loevinger, J. (1948). The technic of homogeneous tests compared with some aspects of “scale analysis” and factor analysis. *Psychological Bulletin, 45*(6), 507–529. https://doi.org/10.1037/h0055827 [Publisher/PubMed metadata verified; implementation-relevant error-pair pages/equation not yet directly verified in this lane.]

Meijer, R. R. (1994). The number of Guttman errors as a simple and powerful person-fit statistic. *Applied Psychological Measurement, 18*(4), 311–314. https://doi.org/10.1177/014662169401800402 [University of Twente Research Information exposes the final published version at https://research.utwente.nl/en/publications/the-number-of-guttman-errors-as-a-simple-and-powerful-person-fit-/; pp. 311–312 directly read for G/G* definition, item ordering, and fixed-score normalization.]

Meijer, R. R. (1995). A supplement to “The number of Guttman errors as a simple and powerful person-fit statistic.” *Applied Psychological Measurement, 19*(2), 166. https://research.utwente.nl/en/publications/a-supplement-to-the-number-of-guttman-errors-as-a-simple-and-powe/ [Open-access final published version directly read for the Guttman-versus-Loevinger error-definition attribution correction.]

Sato, T. (1975). *The construction and interpretation of S-P tables*. Meiji Tosho. [Primary equation not yet read in this doctoring lane.]

Tatsuoka, K. K., & Tatsuoka, M. M. (1982). Detection of aberrant response patterns and their effect on dimensionality. *Journal of Educational Statistics, 7*(3), 215–231. https://doi.org/10.3102/10769986007003215

Tatsuoka, K. K., & Tatsuoka, M. M. (1983). Spotting erroneous rules of operation by the individual consistency index. *Journal of Educational Measurement, 20*(3), 221–230. https://doi.org/10.1111/j.1745-3984.1983.tb00201.x [PerFit cites this article alongside the 1982 NCI paper; its NCI-equation authority has not yet been established by a direct primary read.]

Tendeiro, J. N., & Meijer, R. R. (2014). Detection of invalid test scores: The usefulness of simple nonparametric statistics. *Journal of Educational Measurement, 51*(3), 239–259. https://doi.org/10.1111/jedm.12046

Tendeiro, J. N., Meijer, R. R., & Niessen, A. S. M. (2016). PerFit: An R package for person-fit analysis in IRT. *Journal of Statistical Software, 74*(5), 1–27. https://doi.org/10.18637/jss.v074.i05

van der Flier, H. (1977). Environmental factors and deviant response patterns. In Y. H. Poortinga (Ed.), *Basic problems in cross-cultural psychology* (pp. 30–35). Swets & Zeitlinger. [Primary text not yet read; bibliographic metadata is corroborated by the pinned PerFit 1.4.7 `G` manual and Meijer & Sijtsma (2001).]

van der Flier, H. (1980). *Vergelijkbaarheid van individuele testprestaties* [Comparability of individual test performance]. Swets & Zeitlinger. [Primary text not yet read; bibliographic metadata is corroborated by later peer-reviewed sources and PerFit documentation.]

van der Flier, H. (1982). Deviant response patterns and comparability of test scores. *Journal of Cross-Cultural Psychology, 13*(3), 267–298. https://doi.org/10.1177/0022002182013003001

## Reproducible metadata links for currently unread primary texts

The entries above without a direct primary-text URL remain `PRIMARY_METADATA_ONLY`; these links are corroborating metadata, not substitutes for reading the primary equation:

- Loevinger (1947): APA DOI metadata, `https://doi.org/10.1037/h0093565`;
- Loevinger (1948): PubMed/publisher DOI metadata, `https://pubmed.ncbi.nlm.nih.gov/18893224/` and `https://doi.org/10.1037/h0055827`;
- van der Flier (1977): pinned PerFit 1.4.7 `G` manual, `https://github.com/cran/PerFit/blob/c9df433cba3d7b03d16284e832d55785cb90464c/man/G.Rd`;
- van der Flier (1980): peer-reviewed bibliographic corroboration in Mellenbergh (2000), `https://doi.org/10.1177/075910630006800116`;
- Tatsuoka & Tatsuoka (1983): publisher DOI, `https://doi.org/10.1111/j.1745-3984.1983.tb00201.x`;
- PerFit comparator: `https://github.com/cran/PerFit/tree/c9df433cba3d7b03d16284e832d55785cb90464c`.

## Unresolved source work

The artifact remains deliberately incomplete until the relevant original texts are lawfully obtained and the exact equation/page/convention evidence is recorded for every shipped statistic. In particular, bridge-equation coverage for C, C*, U3, G/Gnormed, and NCI does not close the primary-equation gate; ZU3 still needs its primary standardization equations and assumptions. The direct Meijer (1994) read establishes definition/order/fixed-score-normalization evidence for G/Gnormed, and the directly read Meijer (1995) supplement corrects the all-error-pair source lineage to Loevinger (1947, 1948), but neither substitutes for a direct read of the Loevinger or van der Flier primary texts. Exact equation-level evidence, tie handling, and perfect-row disposition therefore remain open. The Harnisch–Linn final report narrows the C* convention gap to the exact displayed equation and journal/report reconciliation, but it does not close that equation gate. Closure of #1790 additionally requires executable conformance evidence for all seven statistics, restoration or explicit replacement of the missing `files/perfit_spec.md` oracle-provenance reference, and explicit disposition of every discovered source discrepancy.