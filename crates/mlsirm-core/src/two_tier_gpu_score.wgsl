// E[T|G] = sum_i sum_k P(Y_i>=k|G), integrating independent W and S_i.
// Node counts and priors are explicit inputs; no person W plug-in is accepted.
@group(0) @binding(0) var<storage,read> data: array<f32>;
@group(0) @binding(1) var<storage,read_write> result: array<f32>;
fn sigmoid(x:f32)->f32 {
    let z=exp(-abs(x));
    if(x>=0.0){return 1.0/(1.0+z);}
    return z/(1.0+z);
}
// Binary-tree reduction has explicit array/loop dependencies. Unlike the old
// algebraic compensation expression it does not rely on a compiler preserving
// a subtraction's rounding residual under Metal fast math. Each populated bin
// contains 2^level terms; the host buffer/index bounds keep counts below 2^32.
struct Pairwise {
    bins: array<f32,32>,
    count: u32,
}
fn add(sum:ptr<function,Pairwise>, value:f32) {
    var carry=value;
    var occupied=(*sum).count;
    var level=0u;
    while((occupied & 1u)!=0u) {
        carry=(*sum).bins[level]+carry;
        occupied=occupied>>1u;
        level=level+1u;
    }
    (*sum).bins[level]=carry;
    (*sum).count=(*sum).count+1u;
}
fn finish(sum:ptr<function,Pairwise>)->f32 {
    var value=0.0;
    var occupied=(*sum).count;
    var level=0u;
    while(occupied!=0u) {
        if((occupied & 1u)!=0u){value=value+(*sum).bins[level];}
        occupied=occupied>>1u;
        level=level+1u;
    }
    return value;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let person=id.x; let ni=u32(data[0]); let nc=u32(data[1]);
    let persons=u32(data[2]); let qw=u32(data[3]);let qs=u32(data[4]);
    if(person>=persons){return;}
    let g=data[7u+person]; let item_start=7u+persons;
    let w_start=item_start+ni*(nc+3u); let s_start=w_start+2u*qw;
    // GH rules are probability weights. Renormalize only their uploaded f32
    // masses on device, with the same reduction used for the expectations.
    var w_mass:Pairwise;
    var s_mass:Pairwise;
    for(var w=0u;w<qw;w++){add(&w_mass,data[w_start+2u*w+1u]);}
    for(var s=0u;s<qs;s++){add(&s_mass,data[s_start+2u*s+1u]);}
    let mass=finish(&w_mass)*finish(&s_mass);
    var total:Pairwise;
    for(var i=0u;i<ni;i++){
        let base=item_start+i*(nc+3u);
        for(var k=0u;k<nc-1u;k++){
            var wording_sum:Pairwise;
            for(var w=0u;w<qw;w++){
                let wording=data[5]+data[6]*data[w_start+2u*w];
                var specific_sum:Pairwise;
                for(var s=0u;s<qs;s++){
                    let specific=data[base+3u]*data[s_start+2u*s];
                    let eta=data[base]*g+data[base+1u]*wording+data[base+2u]*specific;
                    add(&specific_sum,data[s_start+2u*s+1u]*sigmoid(eta+data[base+4u+k]));
                }
                add(&wording_sum,data[w_start+2u*w+1u]*finish(&specific_sum));
            }
            add(&total,finish(&wording_sum)/mass);
        }
    }
    result[person]=finish(&total);
}
