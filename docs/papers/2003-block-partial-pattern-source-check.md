# Source-check: block partial-pattern collapse (#2003)

Verified 2026-09-18 via Zotero local API (no `/api/local/authorize`). OCR
extracts were read against form-feed page splits; symbol glyphs in the Bock
and Gibbons (1992) scans are OCR-damaged — prose claims below were checked
against page context; mark **OCR: 자구 미대조 for math glyphs**.

## Claim A — EM depends on distinct response-pattern frequencies

| Item | Content |
|---|---|
| 주장 | Marginal ML EM uses observed pattern counts; the computational unit is the number of distinct patterns `s`, not `N` persons. |
| 출처 | Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika, 46*(4), 443–459. |
| 위치 | Group 6347780 item `RYD8F6DD`, attachment `G3IGFSIZ` (`~/Documents/Zotero/storage/G3IGFSIZ/`). Printed page = PDF page + 442 (PDF 3 → 445; PDF 6 → 448). |
| 인쇄 쪽 | p. 445; p. 448 |
| 인용문 (p. 445) | "Since the counts of observed patterns effectively assign each subject to one and only one of 2^n categories, the frequencies r_l are multinomially distributed with parameters N and P_l = P(x = x_l). The log likelihood is, therefore, log L = C + Σ_l r_l log P_l" (OCR; `r~`/`2"` glyphs damaged). |
| 인용문 (p. 448) | "recoding the i-th subject to the l-th score pattern … There are s distinct score patterns, and hence s values of E(θ\|x_l)." |
| 제한 문장 | Bock & Aitkin collapse **whole** response patterns, not bifactor blocks. Block-level application needs Claim B. |
| 판정 | 뒷받침함 (whole-pattern frequency EM) |
| OCR | 미대조 (math glyphs); prose intent confirmed on page |

## Claim B — Bifactor structure justifies *per-block* factorization

| Item | Content |
|---|---|
| 주장 | Under the bifactor restriction, person likelihood factors over specific-factor blocks given the general node; items are conditionally independent across blocks (paragraphs/domains) while dependence is allowed within a block. Therefore a block's reduced integral `I_psg` depends only on that block's response partial pattern (including missingness), so identical within-block patterns share the same `block_acc` / `log_i`. |
| 출처 | Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor analysis. *Psychometrika, 57*(3), 423–436. |
| 위치 | Group 6347780 item `BSFU3HV3`, attachment `GQULCJQC`. Printed = PDF + 422 (PDF 1 → 423; PDF 3 → 425). |
| 인쇄 쪽 | p. 423; p. 425 |
| 인용문 (p. 423) | "items would be conditionally independent between paragraphs, but conditionally dependent within paragraphs." Also: each item loads the primary dimension and "at most one of the s − 1 group factors"; the restriction "permits conditional dependence within identified subsets of items." |
| 인용문 (p. 425) | "The bi-factor restriction reduces the s-dimensional integral in (4) to a two-dimensional integral, one for θ1 and one for θ2, …, θs." And: "if each variate is related to a single dimension only, then the s dimensions are unconditionally independent, and the joint probability is the product of s unidimensional probabilities" (applies to the nuisance/specific dimensions; primary shares loadings). |
| 출처 (graded) | Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007). Full-information item bifactor analysis of graded response data. *Applied Psychological Measurement, 31*(1), 4–19. |
| 위치 | Group 6347780 item `ELG7Q32J`, attachment `BZ6HNTLQ`. PDF 2 → printed p. 5; PDF 4 → p. 7; PDF 5 → p. 8. |
| 인용문 (p. 5) | "items were conditionally independent between paragraphs but conditionally dependent within paragraphs" (citing the binary bifactor motivation). |
| 인용문 (p. 7) | "Assuming conditional independence of the n items, the probability of person i responding with response pattern … conditional on θ is … L_i(θ)." |
| 인용문 (p. 8) | "the bifactor restriction always results in a two-dimensional integral regardless of the number of dimensions"; product-of-unidimensional-probabilities reduction for specific dimensions (Stuart, 1958). |
| 제한 문장 | Neither paper writes "collapse partial patterns inside the E-step" as an algorithm tip. The CI/product factorization is what makes Bock–Aitkin pattern collapsing *valid when applied per block*. Whole-instrument pattern collapse remains valid but yields little reduction on the motivating CP3 design (~1.29× vs ~13× per-block). |
| 판정 | 조건부로 뒷받침함 — CI + dimension reduction support per-block identity of `I_psg`; the engineering reduction is an implementation of that identity, not a separate theorem. |
| OCR | Gibbons 1992: 미대조 for math; Gibbons 2007: digital text, clearer |

## Design implication (no unsourced defaults — ADR-0028)

- Pattern keys include **missingness**: a missing slot is a distinct symbol from every observed category (MAR cells are not imputed into a category).
- Provenance fields record **measured** `n_persons` (before) and unique partial-pattern counts (after) per block; no default unique-count or reduction factor is hardcoded.
- Numerical acceptance: collapsed vs person-wise E-step loglik and expected counts agree within a **measured** tolerance (tests pin the measured bound; not an unsourced `1e-12` guess without a fixture).

## PDFs

Paywalled Psychometrika / APM texts are **not** redistributed into `docs/papers/`. Locators above point at the maintainer Zotero library.
