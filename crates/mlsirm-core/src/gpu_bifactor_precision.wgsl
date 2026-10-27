// Finite-quadrature E-step in two-f32 arithmetic. One invocation/workgroup
// owns rounding storage; likelihood, posteriors, counts and moments share it.
@group(0) @binding(0) var<storage, read> data: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> result: array<vec2<f32>>;
var<private> trial: bool;
// A raw pair cannot preserve a low word once scaling reaches f32 subnormals.
// Propagate a typed refusal to the host instead of admitting false stationarity.
var<private> arithmetic_range_failure: bool;
fn ds(x:f32)->vec2<f32>{return vec2<f32>(x,0.0);}
// Observable round-to-f32 boundaries prevent Metal fast-math reassociation
// from deleting the low word. Exactly one invocation owns this workgroup.
var<workgroup> rounding: atomic<u32>;
fn rn(x:f32)->f32 {atomicStore(&rounding,bitcast<u32>(x));return bitcast<f32>(atomicLoad(&rounding));}
fn add(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{
 // TwoSum both word pairs before combining them. Adding low words to the
 // high residual one at a time loses their rounding error on cancellation
 // (e.g. adjacent thresholds); that error becomes relative error in log(gap).
 let s=rn(a.x+b.x);let v=rn(s-a.x);
 let e=rn(rn(a.x-rn(s-v))+rn(b.x-v));
 let t=rn(a.y+b.y);let z=rn(t-a.y);
 let f=rn(rn(a.y-rn(t-z))+rn(b.y-z));
 let c=rn(e+t);let h=rn(s+c);let l=rn(rn(c-rn(h-s))+f);
 let hi=rn(h+l);return vec2<f32>(hi,rn(l-rn(hi-h)));
}
fn neg(a:vec2<f32>)->vec2<f32>{return -a;}
fn sub(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{return add(a,-b);}
fn mul(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{
 let p=rn(a.x*b.x);let e=rn(rn(rn(fma(a.x,b.x,-p)+rn(a.x*b.y))+rn(a.y*b.x))+rn(a.y*b.y));
 let h=rn(p+e);return vec2<f32>(h,rn(e-rn(h-p)));
}
fn divd(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{let q=a.x/b.x;let r=sub(a,mul(b,ds(q)));return add(ds(q),ds((r.x+r.y)/b.x));}
fn lt(a:vec2<f32>,b:vec2<f32>)->bool{return a.x<b.x || (a.x==b.x && a.y<b.y);}
fn le(a:vec2<f32>,b:vec2<f32>)->bool{return !lt(b,a);}
fn ab(a:vec2<f32>)->vec2<f32>{if(lt(a,ds(0.0))){return -a;}return a;}
fn finite(a:vec2<f32>)->bool{return abs(a.x)<=3.402823466e38 && abs(a.y)<=3.402823466e38;}
fn ex(a:vec2<f32>)->vec2<f32>{
 // E-step tails may lose low-word relative precision below exp(-66).
 // Keep IEEE arithmetic (including underflow), rather than the optimizer's
 // relative-gradient range refusal. Strict native parity is tested separately;
 // this is not a universal two-word relative-error guarantee for tiny weights.
 // Range reduction and Taylor evaluation avoid f32 transcendental rounding.
 let ln2=vec2<f32>(0.6931471824645996,-1.904654323148236e-9);
 let k=round(a.x*1.4426950408889634);let r=sub(a,mul(ds(k),ln2));
 var term=ds(1.0);var sum=term;
 for(var j=1u;j<=18u;j++){term=divd(mul(term,r),ds(f32(j)));sum=add(sum,term);}
 // Split scaling so a finite result near MAX_F32 never materializes 2^128.
 let high=i32(clamp(k,-126.0,127.0));
 return mul(mul(sum,ds(ldexp(1.0,high))),ds(ldexp(1.0,i32(k)-high)));
}
fn lg(a:vec2<f32>)->vec2<f32>{
 // Read the exponent from the representation; log2(MAX_F32) can round to 128.
 let parts=frexp(a.x);let k=parts.exp-1;
 let m=vec2<f32>(ldexp(a.x,-k),ldexp(a.y,-k));
 let z=divd(sub(m,ds(1.0)),add(m,ds(1.0)));let z2=mul(z,z);
 var term=z;var sum=z;
 for(var j=3u;j<=39u;j+=2u){term=mul(term,z2);sum=add(sum,divd(term,ds(f32(j))));}
 return add(mul(ds(2.0),sum),mul(ds(f32(k)),vec2<f32>(0.6931471824645996,-1.904654323148236e-9)));
}
fn ls(x:vec2<f32>)->vec2<f32>{
 let z=ex(-ab(x));var l=lg(add(ds(1.0),z));
 if(z.x<0.001){
  var term=z;l=z;
  for(var j=2u;j<=8u;j++){term=mul(term,-z);l=add(l,divd(term,ds(f32(j))));}
 }
 if(lt(x,ds(0.0))){return sub(x,l);}return -l;
}
// Logistic directly from one range-reduced exponential. Avoid composing
// log-sigmoid with another exponential for every gradient/Hessian entry.
fn sig(x:vec2<f32>)->vec2<f32>{
 let z=ex(-ab(x));let denominator=add(ds(1.0),z);
 if(lt(x,ds(0.0))){return divd(z,denominator);}
 return divd(ds(1.0),denominator);
}
fn gap_mass(x:vec2<f32>)->vec2<f32>{
 if(x.x<0.1){var term=x;var sum=x;for(var j=2u;j<=14u;j++){term=divd(mul(term,-x),ds(f32(j)));sum=add(sum,term);}return sum;}
 return sub(ds(1.0),ex(-x));
}

fn dim_at(i:u32)->u32{return u32(data[i].x);}
fn at(base:u32,i:u32)->vec2<f32>{return data[base+i];}
fn y(p:u32,i:u32)->i32{return i32(data[dim_at(7u)+p*dim_at(1u)+i].x);}
fn owner(p:u32)->u32{return u32(data[dim_at(8u)+p].x);}
fn block(i:u32)->i32{return i32(data[dim_at(9u)+i].x);}
fn lp(g:u32,i:u32,t:u32,h:u32,k:u32)->vec2<f32>{
 let off=u32(data[dim_at(10u)+g*dim_at(1u)+i].x);
 let n=select(t,t*dim_at(5u)+h,block(i)>=0);
 return data[dim_at(11u)+off+n*dim_at(3u)+k];
}
fn pg(p:u32,t:u32)->u32{return dim_at(18u)+p*dim_at(4u)+t;}
fn ba(p:u32,s:u32,t:u32,h:u32)->u32{return dim_at(19u)+((p*dim_at(2u)+s)*dim_at(4u)+t)*dim_at(5u)+h;}
fn ji(p:u32,s:u32,t:u32,h:u32)->u32{return dim_at(20u)+((p*dim_at(2u)+s)*dim_at(4u)+t)*dim_at(5u)+h;}
fn li(p:u32,s:u32,t:u32)->u32{return dim_at(21u)+(p*dim_at(2u)+s)*dim_at(4u)+t;}
fn marginal(p:u32)->u32{return dim_at(22u)+p;}
fn zero_mass(x:vec2<f32>)->bool{return !finite(x);}
fn flat_group(id:vec3<u32>,groups:vec3<u32>)->u32{return id.x+groups.x*(id.y+groups.y*id.z);}
// E-step tails use IEEE underflow, including scalar-f32 low-word loss at
// extreme weights. This is not a universal relative-precision guarantee;
// final finite outputs and native same-grid controls are checked separately.
@compute @workgroup_size(1)
fn posterior(@builtin(workgroup_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>){
 let p=flat_group(id,groups);if(p>=dim_at(0u)){return;}arithmetic_range_failure=false;
 let g=owner(p);let ni=dim_at(1u);let ns=dim_at(2u);let qg=dim_at(4u);let qs=dim_at(5u);
 for(var t=0u;t<qg;t++){
  let prior_g=data[dim_at(12u)+t];
  var gen=prior_g;
  if(!zero_mass(prior_g)){for(var i=0u;i<ni;i++){if(block(i)<0 && y(p,i)>=0){gen=add(gen,lp(g,i,t,0u,u32(y(p,i))));}}}
  result[pg(p,t)]=gen;
  for(var s=0u;s<ns;s++){
   var mx=ds(-3.402823466e38);
   for(var h=0u;h<qs;h++){
    var value=data[dim_at(13u)+h];
    if(!zero_mass(value)){for(var i=0u;i<ni;i++){if(block(i)==i32(s) && y(p,i)>=0){value=add(value,lp(g,i,t,h,u32(y(p,i))));}}}
    result[ba(p,s,t,h)]=value;if(lt(mx,value)){mx=value;}
   }
   var denom=ds(0.0);
   for(var h=0u;h<qs;h++){let value=result[ba(p,s,t,h)];if(!zero_mass(value)){denom=add(denom,ex(sub(value,mx)));}}
   result[li(p,s,t)]=add(mx,lg(denom));
  }
 }
 var mx=ds(-3.402823466e38);
 for(var t=0u;t<qg;t++){
  var value=result[pg(p,t)];if(!zero_mass(value)){for(var s=0u;s<ns;s++){value=add(value,result[li(p,s,t)]);}}
  result[pg(p,t)]=value;if(lt(mx,value)){mx=value;}
 }
 var denom=ds(0.0);
 for(var t=0u;t<qg;t++){let value=result[pg(p,t)];if(!zero_mass(value)){denom=add(denom,ex(sub(value,mx)));}}
 let logp=add(mx,lg(denom));result[marginal(p)]=logp;
 for(var t=0u;t<qg;t++){
  let value=result[pg(p,t)];var post=ds(0.0);if(!zero_mass(value)){post=divd(ex(sub(value,mx)),denom);}result[pg(p,t)]=post;
  for(var s=0u;s<ns;s++){
   // Conditional specific posterior times normalized general posterior.
   var seen=false;for(var i=0u;i<ni;i++){seen=seen||(block(i)==i32(s)&&y(p,i)>=0);}
   for(var h=0u;h<qs;h++){
    var joint=ds(0.0);let value=result[ba(p,s,t,h)];
    if(seen && !zero_mass(value) && lt(ds(0.0),post)){joint=mul(post,ex(sub(value,result[li(p,s,t)])));}
    result[ji(p,s,t,h)]=joint;
   }
  }
 }
 result[dim_at(23u)+p]=ds(select(0.0,1.0,arithmetic_range_failure));
}
@compute @workgroup_size(1)
fn counts(@builtin(workgroup_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>){
 let idx=flat_group(id,groups);let nc=dim_at(3u);let qg=dim_at(4u);let qs=dim_at(5u);let ni=dim_at(1u);
 let k=idx%nc;let n=(idx/nc)%(qg*qs);let i=(idx/(nc*qg*qs))%ni;let g=idx/(nc*qg*qs*ni);
 if(g>=dim_at(6u)){return;}var total=ds(0.0);let s=block(i);
 if(s>=0 || n<qg){for(var p=0u;p<dim_at(0u);p++){if(owner(p)==g && y(p,i)==i32(k)){
  var value=ds(0.0);if(s>=0){value=result[ji(p,u32(s),n/qs,n%qs)];}else{value=result[pg(p,n)];}total=add(total,value);
 }}}
 result[1u+idx]=total;
}
@compute @workgroup_size(1)
fn moments(@builtin(workgroup_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>){
 let idx=flat_group(id,groups);let ns=dim_at(2u);let s=idx%(ns+1u);let g=idx/(ns+1u);if(g>=dim_at(6u)){return;}
 var w=ds(0.0);var s1=ds(0.0);var s2=ds(0.0);
 for(var p=0u;p<dim_at(0u);p++){if(owner(p)!=g){continue;}
  for(var t=0u;t<dim_at(4u);t++){
   if(s==0u){let post=result[pg(p,t)];let node=data[dim_at(14u)+g*dim_at(4u)+t];w=add(w,post);s1=add(s1,mul(post,node));s2=add(s2,mul(mul(post,node),node));}
   else{for(var h=0u;h<dim_at(5u);h++){
    let post=result[ji(p,s-1u,t,h)];let node=data[dim_at(15u)+(g*ns+s-1u)*dim_at(5u)+h];
    w=add(w,post);s1=add(s1,mul(post,node));s2=add(s2,mul(mul(post,node),node));
   }}
  }
 }
 let row=dim_at(16u)+g*(3u+3u*ns);
 if(s==0u){result[row]=w;result[row+1u]=s1;result[row+2u]=s2;}
 else{result[row+3u+s-1u]=w;result[row+3u+ns+s-1u]=s1;result[row+3u+2u*ns+s-1u]=s2;}
}
@compute @workgroup_size(1)
fn likelihood(){var total=ds(0.0);for(var p=0u;p<dim_at(0u);p++){total=add(total,result[marginal(p)]);}result[0]=total;}
