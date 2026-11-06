// Cumulative indicator moments and Xi: Maydeu-Olivares (2013), equations 4, 10.
struct Shape { groups:u32, items:u32, z:u32, q:u32, moments:u32, stride:u32, total:u32, pad:u32 }
@group(0) @binding(0) var<uniform> shape:Shape;
@group(0) @binding(1) var<storage,read> params:array<f32>;
@group(0) @binding(2) var<storage,read> nodes:array<f32>;
@group(0) @binding(3) var<storage,read> weights:array<f32>;
@group(0) @binding(4) var<storage,read> constraints:array<u32>;
@group(0) @binding(5) var<storage,read_write> result:array<f32>;
fn cum(g:u32,i:u32,k:u32,t:u32)->f32 {
    if k==0u { return 1.0; }
    let base=g*shape.stride;
    let theta=params[base+shape.items*(1u+shape.z)]+params[base+shape.items*(1u+shape.z)+1u]*nodes[t];
    let eta=params[base+i]*theta+params[base+shape.items+i*shape.z+k-1u];
    return 1.0/(1.0+exp(-eta));
}
@compute @workgroup_size(64)
fn moments_xi(@builtin(global_invocation_id) id:vec3<u32>) {
    let idx=id.x; if idx>=shape.total {return;}
    let s=shape.moments; let per=s+s*s; let g=idx/per;let local=idx%per;
    let is_cross=local>=s;
    var a=local;var b=local;
    if is_cross {a=(local-s)/s;b=(local-s)%s;}
    var ma=0.0;var mb=0.0;var joint=0.0;
    for(var t=0u;t<shape.q;t++) {
        var pa=1.0;var pb=1.0;var pab=1.0;
        for(var i=0u;i<shape.items;i++) {
            let ka=constraints[a*shape.items+i];let kb=constraints[b*shape.items+i];
            pa*=cum(g,i,ka,t);pb*=cum(g,i,kb,t);pab*=cum(g,i,max(ka,kb),t);
        }
        ma+=weights[t]*pa;mb+=weights[t]*pb;joint+=weights[t]*pab;
    }
    if is_cross {result[idx]=joint-ma*mb;} else {result[idx]=ma;}
}
