"""Immutable schema-v1 fingerprint fixtures; changes require an explicit version decision."""
from dataclasses import replace
import hashlib
from fast_mlsirm.scale_codebook import OrderedItem, ScaleCodebook, codebook_sha256

ASCII_WIRE = (
    b'{"high_score_meaning":"more of the target trait",'
    b'"items":[["I1",[0,1,2,3]],["I2",[3,2,1,0]]],'
    b'"missing_codes":[-1,99],"scale_id":"synthetic-scale",'
    b'"schema":"fast_mlsirm.scale_codebook/1",'
    b'"source_ref":"synthetic://ordered-keys/v1"}'
)
ASCII_SHA256 = "fb4aaea9b37f3fa9304c38c730138aacf4691a45e3d779798457765d859e37e2"
UNICODE_SHA256 = "9b6681aed804fd615f0f94fc95ba6af985a26ce248e5f9197ea5cc0523c074f4"


def canonical_book():
    """Return the independent canonical record corresponding to ASCII_WIRE."""
    return ScaleCodebook(
        "synthetic-scale", "more of the target trait", "synthetic://ordered-keys/v1",
        (OrderedItem("I1", (0, 1, 2, 3)), OrderedItem("I2", (3, 2, 1, 0))), (-1, 99),
    )


def test_schema_v1_ascii_fingerprint_is_golden():
    """Lock field names, field order, separators, schema identity and category order."""
    assert hashlib.sha256(ASCII_WIRE).hexdigest() == ASCII_SHA256
    assert codebook_sha256(canonical_book()) == ASCII_SHA256


def test_schema_v1_unicode_escape_is_golden():
    """Lock the ASCII JSON escaping contract rather than only ASCII examples."""
    book = replace(canonical_book(), high_score_meaning="more θ")
    wire = ASCII_WIRE.replace(b"more of the target trait", b"more \\u03b8")
    assert hashlib.sha256(wire).hexdigest() == UNICODE_SHA256
    assert codebook_sha256(book) == UNICODE_SHA256


def test_schema_v1_item_order_and_key_changes_are_not_aliases():
    """Changing semantic column/key authority must invalidate a retained receipt."""
    book = canonical_book()
    reversed_columns = replace(book, items=book.items[::-1])
    changed_key = replace(book, items=(book.items[0], OrderedItem("I2", (0, 1, 2, 3))))
    assert codebook_sha256(reversed_columns) != ASCII_SHA256
    assert codebook_sha256(changed_key) != ASCII_SHA256
