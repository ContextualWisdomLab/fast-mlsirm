"""Synthetic-only response-keying contract; no external questionnaire data."""
from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec
import numpy as np
import pytest


def test_public_codebook_contract_exists():
    """The public package must expose the response-keying boundary."""
    assert find_spec("fast_mlsirm.scale_codebook") is not None


@pytest.fixture
def api():
    """Load the public metadata adapter, independently of numerical fitting."""
    try:
        return import_module("fast_mlsirm.scale_codebook")
    except ModuleNotFoundError as exc:
        pytest.fail(f"the explicit scale-codebook contract is unavailable: {exc}")


@pytest.fixture
def book(api):
    """Declare target-construct order rather than infer keys from responses."""
    return api.ScaleCodebook(
        scale_id="synthetic-scale", high_score_meaning="more of the target trait",
        source_ref="synthetic://ordered-keys/v1",
        items=(api.OrderedItem("I1", (0, 1, 2, 3)),
               api.OrderedItem("I2", (3, 2, 1, 0))),
        missing_codes=(-1, 99),
    )


def test_explicit_reverse_key_preserves_missing_and_input(api, book):
    raw = np.array([[0., 3.], [2., 1.], [-1., 99.], [np.nan, 0.]])
    before = raw.copy()
    result = api.prepare_scale_responses(raw, ("I1", "I2"), book)
    np.testing.assert_equal(result.responses, [[0., 0.], [2., 2.], [np.nan, np.nan], [np.nan, 3.]])
    np.testing.assert_equal(raw, before)
    assert result.item_ids == ("I1", "I2")
    assert result.n_categories == 4
    assert result.high_score_meaning == book.high_score_meaning
    assert result.codebook_sha256 == api.codebook_sha256(book)
    assert not result.responses.flags.writeable
    with pytest.raises(ValueError):
        result.responses.setflags(write=True)
    with pytest.raises(ValueError, match="ndarray"):
        api.prepare_scale_responses(result, ("I1", "I2"), book)


def test_input_column_order_cannot_relabel_items(api, book):
    raw = np.array([[3, 0], [1, 2]])
    result = api.prepare_scale_responses(raw, ("I2", "I1"), book)
    np.testing.assert_equal(result.responses, [[0, 0], [2, 2]])
    assert result.item_ids == ("I1", "I2")


def test_row_order_is_preserved(api, book):
    raw = np.array([[0, 3], [2, 1], [1, 0]])
    order = [2, 0, 1]
    first = api.prepare_scale_responses(raw, ("I1", "I2"), book)
    second = api.prepare_scale_responses(raw[order], ("I1", "I2"), book)
    np.testing.assert_equal(second.responses, first.responses[order])


def test_mapping_is_lookup_not_range_inference(api, book):
    changed = replace(book, items=(api.OrderedItem("I1", (10, 30, 20, 40)),
                                  api.OrderedItem("I2", (40, 20, 30, 10))))
    raw = np.array([[30, 30], [40, 10]])
    result = api.prepare_scale_responses(raw, ("I1", "I2"), changed)
    np.testing.assert_equal(result.responses, [[1, 2], [3, 3]])
    assert api.codebook_sha256(book) != api.codebook_sha256(changed)
    assert api.codebook_sha256(book) == api.codebook_sha256(replace(book))


@pytest.mark.parametrize("columns", [("I1", "I1"), ("I1", "other"), ("I1",), ["I1", "I2"], ("I1", 2)])
def test_bad_column_identities_fail_closed(api, book, columns):
    with pytest.raises(ValueError):
        api.prepare_scale_responses(np.zeros((2, 2)), columns, book)


@pytest.mark.parametrize("raw", [np.array([[0., np.inf]]), np.array([[0., -np.inf]]),
    np.array([[0., 1.5]]), np.array([[0., 4.]]), np.array([[0., -2.]]),
    np.array([[0., 2147483648.]], dtype=np.float32),
    np.array([[0, 2**64 - 1]], dtype=np.uint64),
    np.zeros((1, 2), dtype=complex), np.array([["0", "1"]]),
    np.array([[0, 1]], dtype=object), np.zeros(2), np.zeros((0, 2)),
    np.zeros((2, 3)), np.zeros((1, 2), dtype=np.longdouble)])
def test_bad_response_storage_or_codes_fail_closed(api, book, raw):
    with pytest.raises(ValueError):
        api.prepare_scale_responses(raw, ("I1", "I2"), book)


def test_no_user_conversion_callbacks(api, book):
    class Hostile:
        def __array__(self, *args, **kwargs):
            raise AssertionError("caller conversion executed")
    class HostileArray(np.ndarray):
        def __array_finalize__(self, obj):
            pass
    for raw in (Hostile(), np.zeros((2, 2)).view(HostileArray)):
        with pytest.raises(ValueError, match="ndarray"):
            api.prepare_scale_responses(raw, ("I1", "I2"), book)


def test_resource_budget_precedes_materialization(api, book):
    raw = np.broadcast_to(np.zeros((1, 2)), (10_000_001, 2))
    with pytest.raises(ValueError, match="20,000,000"):
        api.prepare_scale_responses(raw, ("I1", "I2"), book)


@pytest.mark.parametrize("field,value", [
    ("scale_id", ""), ("high_score_meaning", " "), ("source_ref", ""),
    ("scale_id", 42), ("source_ref", "s" * 2049),
    ("items", ()), ("items", []), ("missing_codes", [99]),
    ("missing_codes", (99, 99)), ("missing_codes", (True,)),
    ("missing_codes", (2**40,)), ("missing_codes", (0,)),
])
def test_invalid_codebook_fails_before_response_access(api, book, field, value):
    with pytest.raises(ValueError):
        api.prepare_scale_responses(object(), ("I1", "I2"), replace(book, **{field: value}))


def test_item_contract_replayed_after_frozen_record_mutation(api, book):
    object.__setattr__(book.items[0], "ordered_codes", (0, 0, 2, 3))
    with pytest.raises(ValueError):
        api.codebook_sha256(book)


@pytest.mark.parametrize("codes", [(0,), tuple(range(65)), (0, True, 2, 3), (0, 1.0, 2, 3),
    (0, 1, 2, 2**40), [0, 1, 2, 3], (0, 1, 2)])
def test_invalid_item_orders_or_heterogeneous_category_counts(api, book, codes):
    changed = replace(book, items=(api.OrderedItem("I1", codes), book.items[1]))
    with pytest.raises(ValueError):
        api.codebook_sha256(changed)


def test_duplicate_item_identity_and_untrusted_record_fail(api, book):
    for changed in (replace(book, items=(book.items[0], book.items[0])),
                    replace(book, items=(object(), book.items[1])), object()):
        with pytest.raises(ValueError):
            api.codebook_sha256(changed)


def test_identical_response_distribution_does_not_change_declared_keys(api, book):
    raw = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    result = api.prepare_scale_responses(raw, ("I1", "I2"), book)
    np.testing.assert_equal(result.responses[:, 1], [3, 2, 1, 0])


def test_boolean_binary_evidence_is_admitted_when_declared(api, book):
    binary = replace(book, items=(api.OrderedItem("I1", (0, 1)), api.OrderedItem("I2", (1, 0))))
    result = api.prepare_scale_responses(np.array([[False, True], [True, False]]), ("I1", "I2"), binary)
    np.testing.assert_equal(result.responses, [[0, 0], [1, 1]])
