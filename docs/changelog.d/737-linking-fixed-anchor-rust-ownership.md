# Fixed-anchor linking Rust ownership

## Fixed

- Public `link_fixed_item_parameters` delegates affine scale/shift estimation and
  parameter transformation to the compiled Rust core, retaining Python for
  validation and evidence packaging only.
