# CAT item information and selection Rust ownership

## Fixed

- Public `item_information` and `select_cat_item` now delegate Fisher information and maximum-information ranking to the compiled Rust core (`cat_item_information` / `cat_select_item`), reusing the frozen-bank information kernel already used by ability SE.
