# Codebook-anchored category keying

Status: Proposed. Scope: Python Public Binding / response marshalling. No
likelihood, optimizer, statistical diagnostic, Rust kernel, or score formula
changes. The numerical owner remains Rust. This is a new admission workflow,
not a claim to have invented a new IRT model.

## Failure being prevented

A numeric response matrix does not establish an item's substantive direction.
Implicit column positions, missing codes transformed by arithmetic reversal,
or double application of reverse keys can detach fitted scores from the
construct named in a report. Reversed wording can also create substantive
method effects; a negative loading alone does not establish what a respondent
understood (Weijters & Baumgartner, 2012).

`ScaleCodebook` requires explicit item identities, increasing-construct raw-code
orders, a high-score meaning, and a caller-supplied source reference. Preparation
performs lookup only, preserves every row and missing value, rejects undeclared
codes, emits immutable responses in codebook order, and carries a versioned key
fingerprint. It never chooses keys, deletes items/respondents, flips estimates,
or selects a model to obtain a desired correlation, coefficient, or p-value.
The source reference is recorded, not independently verified by this library.

## Mathematical interpretation and limits

For the GRM (Samejima, 1969), `P(Y >= k | theta) = logistic(a*theta + beta_k)`.
With decreasing cumulative intercepts and `a > 0`, the derivative of each
cumulative probability is `a*P*(1-P) >= 0`. Since `E[Y|theta]` is the sum of
these cumulative probabilities for integer categories, its derivative is
nonnegative. An intermediate category probability itself need not increase.

For the GPCM (Muraki, 1992), with ordered scores `k` and category predictor
`k*a*theta + c_k`, `d E[Y|theta]/d theta = a*Var(Y|theta) >= 0` when `a > 0`.
Unconstrained nominal category slopes do not provide that guarantee. These
identities explain model choice; this adapter does not reimplement either cell.

For an interaction-map model, a positive trait coefficient gives the analogous
conditional direction only while person/item positions are held fixed.
Rotation or reflection of the interaction map leaves distances unchanged and
cannot establish a substantive trait sign. Conditional monotonicity does not
prove ordering between arbitrary people with different interaction positions.
Nor does it guarantee cross-construct correlations or validate a clinical cutoff.

The adapter does not prove that a source codebook is correct, establish whether
an unlabelled array was already keyed, repair wording effects, validate
unidimensionality, infer a missing-data mechanism, or certify convergence.
A `PreparedScale` record cannot accidentally re-enter its raw-ndarray boundary,
but an explicitly extracted or copied array still requires truthful provenance.
Signed/weak item evidence must be investigated; indiscriminate positive-slope
constraints are not a replacement for valid measurement.

## Use and evidence

```python
from fast_mlsirm.scale_codebook import OrderedItem, ScaleCodebook, prepare_scale_responses
from fast_mlsirm.polytomous import fit_polytomous, score_polytomous

book = ScaleCodebook(
    scale_id="example", high_score_meaning="more target propensity",
    source_ref="approved-codebook:example/v1",
    items=(OrderedItem("I1", (0, 1, 2, 3)), OrderedItem("I2", (3, 2, 1, 0))),
)
prepared = prepare_scale_responses(raw_array, ("I1", "I2"), book)
fit = fit_polytomous(prepared.responses, n_cat=prepared.n_categories, model="grm")
if not fit.converged:
    raise RuntimeError(fit.termination_reason)
scores = score_polytomous(prepared.responses, fit)
```

The declared tests use artificial item identities and responses only. Unit
contracts cover exact keying, missing-code preservation, column and row order,
immutability, key fingerprint changes, invalid storage/codes, callback rejection,
and the logical-cell budget. Rust acceptance uses three fixed simulated GRM
fixtures with exact reversal recovery, known-trait bias/MAE/RMSE checks, and
cumulative/expected-score direction checks. These are bounded regression
fixtures, not a Monte Carlo coverage claim or validation on any external study.
The additional workflow tests an installed native wheel and preserves its exact
source/wheel identities. It neither replaces nor weakens existing required CI.

## References

Muraki, E. (1992). A generalized partial credit model: Application of an EM
algorithm. *Applied Psychological Measurement, 16*(2), 159–176.
https://doi.org/10.1177/014662169201600206

Samejima, F. (1969). Estimation of latent ability using a response pattern of
graded scores. *Psychometrika, 34*(S1), 1–97.
https://doi.org/10.1007/BF03372160

Weijters, B., & Baumgartner, H. (2012). Misresponse to reversed and negated items
in surveys: A review. *Journal of Marketing Research, 49*(5), 737–747.
https://doi.org/10.1509/jmr.11.0368

Primary-source retrieval for this bounded change verified the Samejima original
monograph record/extract, Muraki's original article abstract and ETS research
record, and the Weijters–Baumgartner original review abstract. No unaccessed
full text is represented as read. The derivatives above are explicit algebraic
consequences of the stated response-cell definitions, not new empirical claims.
