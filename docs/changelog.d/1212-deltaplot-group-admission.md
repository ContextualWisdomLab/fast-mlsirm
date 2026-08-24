# Delta-plot group evidence admission

## Fixed

- Reject non-real-numeric Delta-plot group storage before real-valued coercion, preventing textual reference/focal labels from being silently reinterpreted and object-dtype cells from executing caller numeric callbacks during Python-to-Rust admission.
- Preserve ordinary numeric and Boolean 0/1 group arrays while keeping Angoff Delta-plot psychometric arithmetic unchanged in the Rust core.
