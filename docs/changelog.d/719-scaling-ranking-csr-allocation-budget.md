# LSR ranking CSR live allocation budget

## Fixed

- Cap geometric growth of LSR/I-LSR ranking CSR `uint64` buffers so intermediate capacities never exceed the declared live CSR byte budget, and stream validated item indices without a list→`uint64` temporary beside the live payload.
