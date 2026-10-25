// Chalmers (2012), pp. 3–6, Eqs. (2)–(11). Serial correctness-first device E sweep.
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
fn ls(x:f32)->f32 {let z=exp(-abs(x));var l=log(1.0+z);if(z<0.001){l=z*(1.0-z*0.5+z*z/3.0);}return -max(-x,0.0)-l;}
fn lg(d:f32)->f32 {if(d<0.01){return log(d)+log(1.0-d*0.5+d*d/6.0-d*d*d/24.0);}return log(1.0-exp(-d));}
@compute @workgroup_size(1)
fn main(){
 let n=u32(data[0]);let items=u32(data[1]);let dims=u32(data[2]);let cat=u32(data[3]);let q=u32(data[4]);
 let nodes=5u;let weights=nodes+q*dims;let params=weights+q;let responses=params+items*(dims+cat-1u);
 let counts=1u;let theta=counts+items*q*cat;let scratch=theta+n*dims;let lp=scratch+q;
 for(var cell=0u;cell<theta+n*dims;cell++){result[cell]=0.0;}
 for(var i=0u;i<items;i++){
  let par=params+i*(dims+cat-1u);
  for(var nd=0u;nd<q;nd++){
   var eta=0.0;for(var d=0u;d<dims;d++){eta+=data[par+d]*data[nodes+nd*dims+d];}
   for(var c=0u;c<cat;c++){
    var value=0.0;
    if(c==0u){value=ls(-eta-data[par+dims]);}
    else if(c==cat-1u){value=ls(eta+data[par+dims+cat-2u]);}
    else{let a=data[par+dims+c-1u];let b=data[par+dims+c];value=ls(eta+a)+ls(-eta-b)+lg(a-b);}
    result[lp+(i*q+nd)*cat+c]=value;
   }
  }
 }
 for(var p=0u;p<n;p++){
  var maximum=-3.402823466e38;
  for(var nd=0u;nd<q;nd++){
   var v=data[weights+nd];
   for(var i=0u;i<items;i++){let y=data[responses+p*items+i];if(y>=0.0){v+=result[lp+(i*q+nd)*cat+u32(y)];}}
   result[scratch+nd]=v;maximum=max(maximum,v);
  }
  var denom=0.0;for(var nd=0u;nd<q;nd++){denom+=exp(result[scratch+nd]-maximum);}
  result[0]+=maximum+log(denom);
  for(var nd=0u;nd<q;nd++){
   let post=exp(result[scratch+nd]-maximum)/denom;
   for(var i=0u;i<items;i++){let y=data[responses+p*items+i];if(y>=0.0){result[counts+(i*q+nd)*cat+u32(y)]+=post;}}
   for(var d=0u;d<dims;d++){result[theta+p*dims+d]+=post*data[nodes+nd*dims+d];}
  }
 }
}
