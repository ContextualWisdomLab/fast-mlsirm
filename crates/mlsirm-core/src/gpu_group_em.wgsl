// Two-f32 Newton arithmetic. No host numerical evaluation or solve.
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
 // Preserve at least 32 normal significand bits before raw-pair scaling.
 // This is a representation refusal, not a clamp or relaxed Newton tolerance.
 if(lt(a,ds(-66.0))){arithmetic_range_failure=true;}
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
fn np()->u32{return u32(data[0].x);}
fn width()->u32{let p=np();return 1u+p+p*p;}
fn row_width()->u32{return np()+2u*u32(data[3].x)-1u;}
fn control()->u32{return 5u+np()+u32(data[2].x)*row_width();}
fn par(j:u32)->vec2<f32>{return result[width()+select(0u,np(),trial)+j];}
fn prior(slot:u32)->vec2<f32>{return ds(0.0);}
fn criterion(slot:u32)->vec2<f32>{
 let p=np();let nc=u32(data[3].x);let nn=u32(data[2].x);
 var j=slot-1u;var k=0u;let h=slot>p;
 if(h){j=(slot-1u-p)/p;k=(slot-1u-p)%p;}
 var total=ds(0.0);
 for(var n=0u;n<nn;n++){
  let start=5u+p+n*row_width();var eta=ds(0.0);
  for(var d=0u;d<p;d++){eta=add(eta,mul(par(d),data[start+d]));}
  for(var c=0u;c<nc;c++){
   let count=data[start+p+nc-1u+c];if(count.x==0.0){continue;}
   var lp=ds(0.0);var grad=ds(0.0);var hes=ds(0.0);
   if(c==0u||c==nc-1u){
    let b=select(0u,nc-2u,c==nc-1u);let e=add(eta,data[start+p+b]);
    lp=ls(-e);grad=sig(e);if(c==nc-1u){lp=ls(e);grad=-sig(-e);}
    hes=mul(sig(e),sig(-e));
   }else{
    let u=add(eta,data[start+p+c-1u]);let v=add(eta,data[start+p+c]);
    let gap=sub(data[start+p+c-1u],data[start+p+c]);
    lp=add(add(ls(u),ls(-v)),lg(gap_mass(gap)));
    grad=sub(sig(v),sig(-u));hes=add(mul(sig(u),sig(-u)),mul(sig(v),sig(-v)));
   }
   if(slot==0u){total=sub(total,mul(count,lp));}
   else{var deriv=mul(grad,data[start+j]);if(h){deriv=mul(hes,mul(data[start+j],data[start+k]));}total=add(total,mul(count,deriv));}
  }
 }
 return total;
}
fn em1(x:vec2<f32>)->vec2<f32>{
 if(lt(ab(x),ds(0.1))){
  var term=x;var sum=x;
  for(var j=2u;j<=18u;j++){term=divd(mul(term,x),ds(f32(j)));sum=add(sum,term);}
  return sum;
 }
 return sub(ex(x),ds(1.0));
}
fn l1p(x:vec2<f32>)->vec2<f32>{
 if(lt(ab(x),ds(0.1))){
  var term=x;var sum=x;
  for(var j=2u;j<=18u;j++){term=mul(term,-x);sum=add(sum,divd(term,ds(f32(j))));}
  return sum;
 }
 return lg(add(ds(1.0),x));
}
fn ls_delta(x:vec2<f32>,dx:vec2<f32>)->vec2<f32>{
 if(lt(ds(0.1),ab(dx))){return sub(ls(add(x,dx)),ls(x));}
 return -l1p(mul(sig(-x),em1(-dx)));
}
fn objective_delta()->vec2<f32>{
 let p=np();let nc=u32(data[3].x);let w=width();var total=ds(0.0);
 for(var n=0u;n<u32(data[2].x);n++){
  let start=5u+p+n*row_width();var eta=ds(0.0);var change=ds(0.0);
  for(var d=0u;d<p;d++){eta=add(eta,mul(result[w+d],data[start+d]));change=add(change,mul(sub(result[w+p+d],result[w+d]),data[start+d]));}
  for(var c=0u;c<nc;c++){
   let count=data[start+p+nc-1u+c];if(count.x==0.0){continue;}var delta=ds(0.0);
   if(c==0u){delta=ls_delta(-add(eta,data[start+p]),-change);}
   else if(c==nc-1u){delta=ls_delta(add(eta,data[start+p+nc-2u]),change);}
   else{delta=add(ls_delta(add(eta,data[start+p+c-1u]),change),ls_delta(-add(eta,data[start+p+c]),-change));}
   total=sub(total,mul(count,delta));
  }
 }
 return total;
}
@compute @workgroup_size(1)
fn main(){
 trial=false;arithmetic_range_failure=false;let p=np();let off=u32(data[1].x);let w=width();let ctl=control();let step=w+2u*p;let rb=step+p;
 for(var j=0u;j<9u;j++){result[rb+j]=ds(0.0);}
 for(var j=0u;j<p;j++){result[w+j]=data[5u+j];}
 for(var it=0u;it<u32(data[ctl+4u].x);it++){
  var norm=ds(0.0);
  for(var slot=0u;slot<w;slot++){
   result[slot]=add(criterion(slot),prior(slot));
   if(arithmetic_range_failure){result[rb]=ds(6.0);return;}
   if(!finite(result[slot])){result[rb]=ds(2.0);return;}
   if(slot>0u && slot<=p && lt(norm,ab(result[slot]))){norm=ab(result[slot]);}
  }
  for(var j=1u;j<=4u;j++){result[rb+j]=add(result[rb+j],ds(1.0));}
  if(le(norm,data[ctl+6u])){result[rb]=ds(1.0);return;}
  let f0=result[0];
  for(var j=0u;j<p;j++){result[step+j]=result[1u+j];let a=1u+p+j*p+j;result[a]=add(result[a],data[ctl+3u]);}
  for(var j=0u;j<p;j++){
   var pivot=j;var best=ab(result[1u+p+j*p+j]);
   for(var k=j+1u;k<p;k++){let a=ab(result[1u+p+k*p+j]);if(lt(best,a)){best=a;pivot=k;}}
   if(!finite(best)||le(best,data[ctl+8u])){result[rb]=ds(3.0);return;}
   if(pivot!=j){for(var k=0u;k<p;k++){let a=1u+p+j*p+k;let b=1u+p+pivot*p+k;let t=result[a];result[a]=result[b];result[b]=t;}let t=result[step+j];result[step+j]=result[step+pivot];result[step+pivot]=t;}
   for(var k=j+1u;k<p;k++){
    let factor=divd(result[1u+p+k*p+j],result[1u+p+j*p+j]);
    for(var l=j;l<p;l++){let a=1u+p+k*p+l;result[a]=sub(result[a],mul(factor,result[1u+p+j*p+l]));}
    result[step+k]=sub(result[step+k],mul(factor,result[step+j]));
   }
  }
  for(var jj=p;jj>0u;jj--){let j=jj-1u;var value=result[step+j];for(var k=j+1u;k<p;k++){value=sub(value,mul(result[1u+p+j*p+k],result[step+k]));}result[step+j]=divd(value,result[1u+p+j*p+j]);}
  var directional=ds(0.0);
  for(var j=0u;j<p;j++){if(!finite(result[step+j])){result[rb]=ds(3.0);return;}directional=add(directional,mul(result[1u+j],result[step+j]));}
  if(!finite(directional)||le(directional,ds(0.0))){result[rb]=ds(3.0);return;}
  result[rb+5u]=add(result[rb+5u],ds(1.0));
  var alpha=ds(1.0);var accepted=false;var actual_change=ds(0.0);
  for(var bt=0u;bt<u32(data[ctl+5u].x);bt++){
   trial=true;var valid=true;actual_change=ds(0.0);
   for(var j=0u;j<p;j++){result[w+p+j]=sub(result[w+j],mul(alpha,result[step+j]));valid=valid&&finite(result[w+p+j]);let change=ab(sub(result[w+p+j],result[w+j]));if(lt(actual_change,change)){actual_change=change;}}
   for(var j=0u;j<p;j++){if(data[u32(data[4].x)+j].x==1.0){valid=valid&&lt(ds(0.0),par(j));}}
   if(data[ctl].x!=0.0){for(var j=0u;j<off;j++){valid=valid&&(par(j).x!=0.0 || par(j).y!=0.0);}}
   if(valid){
    let f=add(criterion(0u),prior(0u));result[rb+1u]=add(result[rb+1u],ds(1.0));result[rb+4u]=add(result[rb+4u],ds(1.0));
    if(arithmetic_range_failure){result[rb]=ds(6.0);return;}
    var sufficient=finite(f)&&le(f,sub(f0,mul(data[ctl+9u],mul(alpha,directional))));
    if(finite(f) && !sufficient){
     let delta=objective_delta();
     if(arithmetic_range_failure){result[rb]=ds(6.0);return;}
     sufficient=finite(delta)&&le(delta,-mul(data[ctl+9u],mul(alpha,directional)));
    }
    if(sufficient&&lt(ds(0.0),actual_change)){accepted=true;}
    // Below the caller's step tolerance the objective decrement is quadratic
    // and can be unresolvable even though the gradient is still nonstationary.
    // Admit a full local Newton correction only on strict gradient improvement
    // that already meets the unchanged gradient tolerance.
    // This is not convergence: the next iteration must evaluate the gradient.
    if(bt==0u && le(actual_change,data[ctl+7u]) && lt(ds(0.0),actual_change)){
     var trial_norm=ds(0.0);var trial_finite=true;
     for(var j=1u;j<=p;j++){
      let g=add(criterion(j),prior(j));trial_finite=trial_finite&&finite(g);
      if(lt(trial_norm,ab(g))){trial_norm=ab(g);}
     }
     if(arithmetic_range_failure){result[rb]=ds(6.0);return;}
     result[rb+2u]=add(result[rb+2u],ds(1.0));
     if(finite(f) && trial_finite && lt(trial_norm,norm) && le(trial_norm,data[ctl+6u])){accepted=true;}
    }
   }
   result[rb+6u]=add(result[rb+6u],ds(1.0));trial=false;
   if(accepted){break;}alpha=mul(alpha,ds(0.5));
  }
  if(!accepted){result[rb]=ds(4.0);return;}
  for(var j=0u;j<p;j++){result[w+j]=result[w+p+j];}
  result[rb+7u]=add(result[rb+7u],ds(1.0));result[rb+8u]=add(result[rb+8u],ds(1.0));
  // A small step alone cannot certify stationarity at the caller's gradient
  // tolerance. Re-evaluate on device at the updated parameters next iteration.
 }
 result[rb]=ds(5.0);
}
