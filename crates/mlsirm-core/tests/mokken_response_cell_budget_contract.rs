//! Direct-Rust resource parity for the public Mokken response-cell envelope.
//!
//! The Python binding rejects analyses above 20,000,000 logical response
//! cells before dense marshalling. Direct Rust callers must not bypass that
//! package resource contract and enter `pairwise`, which owns response-sized
//! centered and sorted workspaces.

use mlsirm_core::mokken::{aisp, coef_h};

const MAX_RESPONSE_CELLS: usize = 20_000_000;

#[test]
fn direct_rust_rejects_response_work_above_the_public_cell_budget_before_payload_replay() {
    let n_items = 3usize;
    let n_persons = MAX_RESPONSE_CELLS / n_items + 1;
    let response_cells = n_persons * n_items;
    assert!(response_cells > MAX_RESPONSE_CELLS);

    let expected = format!(
        "n_persons * n_items = {response_cells} exceeds the {MAX_RESPONSE_CELLS}-cell response budget"
    );

    assert_eq!(coef_h(&[], n_persons, n_items).unwrap_err(), expected);
    assert_eq!(
        aisp(&[], n_persons, n_items, 0.3, 0.05).unwrap_err(),
        expected
    );
}

#[test]
fn exact_response_cell_budget_remains_admissible_to_payload_validation() {
    let n_items = 2usize;
    let n_persons = MAX_RESPONSE_CELLS / n_items;

    let error = coef_h(&[], n_persons, n_items).unwrap_err();
    assert_eq!(
        error,
        format!(
            "responses length 0 != n_persons*n_items {}",
            MAX_RESPONSE_CELLS
        )
    );
}
