# Rust-owned CAT item information

## Changed

- Routed public CAT `item_information()` probability and Fisher-information arithmetic through the existing compiled Rust bank-information kernel while keeping Python limited to bounded validation, immutable marshalling, and result transport.
- Preserved the existing simple-structure MIRT/MLS2PLM-family semantics and population-mean latent-position convention; the final global CAT item-selection policy remains a separate #629 ownership slice rather than being silently replaced by a different Rust adaptive policy.
