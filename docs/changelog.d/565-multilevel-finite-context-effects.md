# Fail closed on unsafe multilevel contextual effects

## Fixed

- Multilevel contextual-effect evaluation now fails closed when any referenced context random-effect value is NaN or infinite and when finite inputs overflow the weighted sum, preventing non-finite predictor results from escaping the Rust boundary while leaving unreferenced table capacity outside sparse validation work.
- Python context-effect marshalling snapshots each required mapping value once without caller-defined membership probes and normalizes hostile lookup callbacks to non-reflective package errors before native dispatch.
