# Two-tier threshold order validation

## Fixed

- Two-tier expected-raw scoring now rejects GRM boundary intercepts that are
  not finite and strictly decreasing within each item. Before this change, an
  out-of-order row produced category probabilities that were clamped at zero
  and renormalized. That is not a graded-response distribution, so the
  returned expected raw total had no meaning. Fitted two-tier models always
  satisfy this ordering, so no fitted result is affected.
