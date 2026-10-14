"""WGSL source guard for repeated person lookup, not hardware acceptance."""
from pathlib import Path


SHADER = Path(__file__).resolve().parents[1] / "crates/mlsirm-core/src/gpu_specific_tree.wgsl"


def test_specific_chunks_initial_lookup_precedes_leaf_loop():
    """No leaf may restart the compact observed-person lookup at person zero."""
    source = SHADER.read_text(encoding="utf-8")
    chunks = source.split("fn specific_chunks(", 1)[1].split("fn specific_merge(", 1)[0]
    before, leaves = chunks.split("for (var k = 0u; k < count; k = k + 1u)", 1)
    assert "var p = 0u;" in before
    assert "var rank = 0u;" in before
    assert "var p = 0u;" not in leaves
    assert "var rank = 0u;" not in leaves
    assert "if (person_rank != current_rank)" in leaves
    assert "current_rank = person_rank;" in leaves


def test_specific_chunks_keeps_retained_leaf_products_and_tree_order():
    """Lookup edits cannot replace retained arithmetic or reverse carry merges."""
    source = SHADER.read_text(encoding="utf-8")
    chunks = source.split("fn specific_chunks(", 1)[1].split("fn specific_merge(", 1)[0]
    arithmetic = chunks.split("        let post = joint[", 1)[1]
    assert arithmetic.startswith("((p * dims.ns + s) * dims.qg + t) * dims.qs + h];")
    assert "var value = vec2<f32>(post, post * node * node);" in arithmetic
    assert "var occupied = k;" in arithmetic
    assert "value = bins[level] + value;" in arithmetic
    assert "occupied = occupied >> 1u;" in arithmetic
    assert "result = bins[level] + result;" in arithmetic
    assert "counts[offset] = result.x;" in arithmetic
    assert "counts[offset + 1u] = result.y;" in arithmetic
