# Reject executable ICC semantic controls

## Fixed

- Validate ICC model/type/unit choices and r0/confidence controls before native-core discovery or ratings materialization, reject caller-defined conversion/comparison protocol providers and scalar subclasses without executing their callbacks, preserve the Rust parameter ranges, and normalize trusted NumPy real scalars to exact Python floats before Rust dispatch.
