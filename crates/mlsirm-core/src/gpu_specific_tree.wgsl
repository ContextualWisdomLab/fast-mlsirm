// Power-of-two partition of the retained compact binary-carry tree.
// Basis: Higham (1993, pp. 783, 797).
// The chunk/tail layout is an implementation derivation; products and
// addition orientation remain the predecessor operations.
// Reference: Higham, N. J. (1993). The accuracy of floating point summation.
// SIAM Journal on Scientific Computing, 14(4), 783–799. doi:10.1137/0914050.
// The leaf sequence is compact observed-person, t, h in predecessor order.
@compute @workgroup_size(64)
fn specific_chunks(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = flat_idx(wid, lid, nwg);
    let chunk_size = 256u;
    let capacity = (dims.np * dims.qg * dims.qs + chunk_size - 1u) / chunk_size;
    if (idx >= dims.ng * dims.ns * capacity) { return; }
    let chunk = idx % capacity;
    let gs = idx / capacity;
    let g = gs / dims.ns;
    let s = gs % dims.ns;
    var persons = 0u;
    for (var p = 0u; p < dims.np; p = p + 1u) {
        if (gid[p] == g && anyobs[p * dims.ns + s] != 0u) { persons = persons + 1u; }
    }
    let count_total = persons * dims.qg * dims.qs;
    let begin = chunk * chunk_size;
    if (begin >= count_total) { return; }
    let count = min(chunk_size, count_total - begin);
    var bins: array<vec2<f32>, 32>;
    for (var k = 0u; k < count; k = k + 1u) {
        let ordinal = begin + k;
        let person_rank = ordinal / (dims.qg * dims.qs);
        let t = (ordinal / dims.qs) % dims.qg;
        let h = ordinal % dims.qs;
        var p = 0u;
        var rank = 0u;
        loop {
            if (gid[p] == g && anyobs[p * dims.ns + s] != 0u) {
                if (rank == person_rank) { break; }
                rank = rank + 1u;
            }
            p = p + 1u;
        }
        let post = joint[((p * dims.ns + s) * dims.qg + t) * dims.qs + h];
        let node = ts[(g * dims.ns + s) * dims.qs + h];
        var value = vec2<f32>(post, post * node * node);
        var occupied = k;
        var level = 0u;
        loop {
            if ((occupied & 1u) == 0u) { break; }
            value = bins[level] + value;
            occupied = occupied >> 1u;
            level = level + 1u;
        }
        bins[level] = value;
    }
    var result = vec2<f32>(0.0);
    for (var level = 0u; level < 32u; level = level + 1u) {
        if ((count & (1u << level)) != 0u) { result = bins[level] + result; }
    }
    let offset = dims.ng * dims.ni * dims.stride * dims.nc + 2u * idx;
    counts[offset] = result.x;
    counts[offset + 1u] = result.y;
}

@compute @workgroup_size(64)
fn specific_merge(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = flat_idx(wid, lid, nwg);
    if (idx >= dims.ng * dims.ns) { return; }
    let g = idx / dims.ns;
    let s = idx % dims.ns;
    let capacity = (dims.np * dims.qg * dims.qs + 255u) / 256u;
    var persons = 0u;
    for (var p = 0u; p < dims.np; p = p + 1u) {
        if (gid[p] == g && anyobs[p * dims.ns + s] != 0u) { persons = persons + 1u; }
    }
    let count_total = persons * dims.qg * dims.qs;
    let full = count_total / 256u;
    let tail = count_total % 256u;
    let offset = dims.ng * dims.ni * dims.stride * dims.nc + 2u * idx * capacity;
    var bins: array<vec2<f32>, 32>;
    for (var k = 0u; k < full; k = k + 1u) {
        var value = vec2<f32>(counts[offset + 2u * k], counts[offset + 2u * k + 1u]);
        var occupied = k;
        var level = 0u;
        loop {
            if ((occupied & 1u) == 0u) { break; }
            value = bins[level] + value;
            occupied = occupied >> 1u;
            level = level + 1u;
        }
        bins[level] = value;
    }
    var result = vec2<f32>(0.0);
    if (tail != 0u) {
        result = vec2<f32>(counts[offset + 2u * full], counts[offset + 2u * full + 1u]);
    }
    for (var level = 0u; level < 32u; level = level + 1u) {
        if ((full & (1u << level)) != 0u) { result = bins[level] + result; }
    }
    let row = g * (3u + 2u * dims.ns);
    moments[row + 3u + s] = result.x;
    moments[row + 3u + dims.ns + s] = result.y;
}
