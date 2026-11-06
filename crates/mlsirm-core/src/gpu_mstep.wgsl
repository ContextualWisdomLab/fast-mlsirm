@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
 if(gid.x<width()){result[gid.x]=criterion(gid.x)+prior(gid.x);}
}
