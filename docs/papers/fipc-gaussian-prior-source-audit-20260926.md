# FIPC Gaussian prior and EAP source audit (2026-09-26)

This audit covers the method claims in `fit_poly_fipc` and
`score_poly_fipc_group_persons`. It does not establish empirical recovery of the
Gaussian moment-update fitter. The Zotero group is `elderly-gad` (6347780).

| Claim checked | Source and Zotero attachment | PDF → printed page | Operative passage | Limitation and judgment | Image check |
|---|---|---|---|---|---|
| EAP and posterior SD use likelihood-weighted prior quadrature | Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement, 6*(4), 431–444. DOI `10.1177/014662168200600405`; item `L9D5SMS8`, attachment `IMDTIXH6` | 4 → 433 (also 3 → 432) | “the weights are the probabilities at the corresponding points of a discrete prior distribution” (p. 433, with EAP/PSD equations 4–6) | Supports posterior weighting. The authors note that Gauss–Hermite exactness does not cover the likelihood functions considered there; neither 121 nor 481 points is licensed by this passage. | Printed p. 433 image compared with OCR. |
| Updating the prior in fixed-parameter calibration | Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement, 43*(4), 355–381. DOI `10.1111/j.1745-3984.2006.00021.x`; item `BPIRS7P8`, attachment `FXIWS8GY` | 8 → 362; 9 → 363; 11 → 365 | “both the item parameters and the latent weights are concurrently estimated” (p. 362); “The ability points should not be rescaled after each EM cycle.” (p. 363) | Kim's MWU-MEM updates discrete weights on fixed ability points. This library instead updates Gaussian moments and moves trait-scale nodes each sweep. Kim's recovery result does not prove recovery for this implementation. The N(0,1) initialization is consistent with Kim p. 365. | Printed pp. 362–363 images compared with OCR; p. 365 prose read from PDF text. |

The fitted Gaussian prior is therefore an explicit library model choice. Its
validity and sensitivity require evidence for that model, including comparisons
to a fixed-node weight-update fit where available. The former Bock–Zimowski
node-shift attribution is not used here because its cited passage has not been
checked in this audit.

The library's synthetic `test_poly_fipc_recovers_shift_and_preserves_anchors`
and reverse-keyed-anchor test exercise Gaussian shift recovery and fixed-item
invariance; the five focused FIPC recovery tests pass on this branch. Their
fixtures do not compare the Gaussian update with Kim's fixed-node MWU-MEM and
do not establish fit validity for a particular research dataset.
