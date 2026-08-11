# CAT item information and next-item selection ownership

## Status

Implemented for public `fast_mlsirm.test_design.item_information` and `select_cat_item`. Ability estimation was already Rust-owned; this note closes the remaining information/selection numerical ownership gap for adaptive testing and ATA information reuse.

## Problem

CAT item information `I_j(theta) = a_j^2 P_j Q_j` and the maximum-information argmax/exclusion policy were still evaluated in pure Python. That diverged from the repository ownership rule that psychometric numerical work is Rust-primary with optional GPU/CPU device selection (van der Linden & Pashley, 2010).

## Decision

- Add Rust `cat_item_information` / `cat_select_item` on the existing `bank_information_device` kernel.
- Python validates factor identities, marshals bank parameters, and returns the Rust result without recomputing Fisher information or selection ranks.
- Device policy remains `auto` (GPU preference with CPU fallback), matching ability SE.

## Evidence contract

GREEN ownership tests replace the Rust call with sentinel results and prove the public entrypoints return those sentinels. Ordinary CAT and ATA suites remain green under the product CI matrix.

## References

van der Linden, W. J., & Pashley, P. J. (2010). Item selection and ability estimation in adaptive testing. In W. J. van der Linden & C. A. W. Glas (Eds.), *Elements of adaptive testing* (pp. 3–30). Springer. https://doi.org/10.1007/978-0-387-85461-8_1
