# Primary-only reduction source verification

Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika, 75*(4), 581–612. https://doi.org/10.1007/s11336-010-9178-0

The actual PDF `/Users/seonghobae/papers/Cai_2010_two_tier.pdf` maps PDF pages 7/8/9 to printed pages 587/588/589. Equation 7 is on **587**, not 588; current docstrings/comments correct the first candidate locator. The p.588 paragraph makes the specific factor optional for an item; p.589 equations 11–12 define its linear predictor and adjacent cumulative differences. With all specific slopes absent, equation 7 gives the product across the primary-only item probabilities. This is our algebraic specialization, not an explicit S=0 empirical validation claim from the paper. Equation 8's regrouping by S is not used as an empty-product likelihood.

The paper states only single-sample models are discussed (p.587). Multi-group composition and recovery therefore require independent implementation evidence. The actual synthetic group pipeline and independent standard-normal unidimensional scoring comparisons provide bounded execution evidence, not actual study validity.

wgpu-core 30.0.0 primary source `src/binding_model.rs`, lines 166–167, declares `BindingZeroSize`. Empty specific arrays keep an unread 4-byte binding while shader dimension ns stays 0. No latent factor or probability term is introduced. The actual hardware tests and actual CI consumer negative test verify the route and admission contract.

Python executable ASTs remain identical after the locator correction. Rust edits correct a comment only. Final native core and tests are bound in final-provenance.json; no new statistical computations occur in this documentation change.
