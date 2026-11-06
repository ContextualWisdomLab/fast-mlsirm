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
    # Specific-mean fitting consumes mass, first moment, and second moment.
    # Preserve left-associated retained products in all three channels.
    assert "let node = ts[(g * dims.ns + s) * dims.qs + h];" in arithmetic
    assert "var value = vec3<f32>(post, post * node, post * node * node);" in arithmetic
    assert "var bins: array<vec3<f32>, 32>;" in chunks
    assert "var result = vec3<f32>(0.0);" in arithmetic
    assert "let offset = dims.ng * dims.ni * dims.stride * dims.nc + 3u * idx;" in arithmetic
    assert "let ordinal = begin + k;" in chunks
    assert "let person_rank = ordinal / per_person;" in chunks
    assert "let t = (ordinal / dims.qs) % dims.qg;" in chunks
    assert "let h = ordinal % dims.qs;" in chunks
    assert "var occupied = k;" in arithmetic
    assert "if ((occupied & 1u) == 0u) { break; }" in arithmetic
    assert "value = bins[level] + value;" in arithmetic
    assert "occupied = occupied >> 1u;" in arithmetic
    assert "level = level + 1u;" in arithmetic
    assert "bins[level] = value;" in arithmetic
    assert "if ((count & (1u << level)) != 0u) { result = bins[level] + result; }" in arithmetic
    assert "counts[offset] = result.x;" in arithmetic
    assert "counts[offset + 1u] = result.y;" in arithmetic
    assert "counts[offset + 2u] = result.z;" in arithmetic


def test_specific_merge_preserves_three_moment_channels_and_tail_order():
    """Mass, first, and second moments retain the full-chunk then tail tree."""
    source = SHADER.read_text(encoding="utf-8")
    merge = source.split("fn specific_merge(", 1)[1]
    assert "let full = count_total / 256u;" in merge
    assert "let tail = count_total % 256u;" in merge
    assert "let offset = dims.ng * dims.ni * dims.stride * dims.nc + 3u * idx * capacity;" in merge
    assert "var bins: array<vec3<f32>, 32>;" in merge
    assert "var value = vec3<f32>(counts[offset + 3u * k], counts[offset + 3u * k + 1u], counts[offset + 3u * k + 2u]);" in merge
    assert "var occupied = k;" in merge
    assert "value = bins[level] + value;" in merge
    assert "occupied = occupied >> 1u;" in merge
    assert "bins[level] = value;" in merge
    result = merge.split("var result = vec3<f32>(0.0);", 1)[1]
    tail, finish = result.split("for (var level = 0u; level < 32u; level = level + 1u)", 1)
    assert "if (tail != 0u)" in tail
    assert "result = vec3<f32>(counts[offset + 3u * full], counts[offset + 3u * full + 1u], counts[offset + 3u * full + 2u]);" in tail
    assert "if ((full & (1u << level)) != 0u) { result = bins[level] + result; }" in finish
    assert "let row = g * (3u + 3u * dims.ns);" in finish
    assert "moments[row + 3u + s] = result.x;" in finish
    assert "moments[row + 3u + dims.ns + s] = result.y;" in finish
    assert "moments[row + 3u + 2u * dims.ns + s] = result.z;" in finish

    # Bind the shader layout to the real Rust allocation/readback consumer.
    host = SHADER.with_name("gpu_bifactor.rs").read_text(encoding="utf-8")
    layout = host.split("fn specific_tree_layout(", 1)[1].split("pub(crate) fn e_step_reduced_gpu(", 1)[0]
    assert "let len = counts.checked_add(tiles.checked_mul(3)?)?;" in layout
    readback = host.split("let row = 3 + 3 * ns;", 1)[1]
    assert "w_spec[g * ns + s] = f64::from(moments_vec[g * row + 3 + s]);" in readback
    assert "s1_spec[g * ns + s] = f64::from(moments_vec[g * row + 3 + ns + s]);" in readback
    assert "s2_spec[g * ns + s] = f64::from(moments_vec[g * row + 3 + 2 * ns + s]);" in readback
