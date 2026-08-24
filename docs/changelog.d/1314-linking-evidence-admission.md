# Linking evidence admission

## Fixed

- Seal fixed-anchor and common-item IRT linking numeric evidence before NumPy materialization so arbitrary array/container/numeric protocols cannot execute while scientific inputs are being admitted.
- Preserve exact NumPy real-numeric arrays and ordinary built-in list/tuple evidence containing package-trusted Python/NumPy real scalars while keeping linking arithmetic Rust-owned.
