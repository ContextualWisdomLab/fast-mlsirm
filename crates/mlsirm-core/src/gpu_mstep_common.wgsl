// Chalmers (2012), p. 3, Eq. (2); algebraically differentiated log category probabilities.
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
var<private> from_output: bool;
var<private> trial: bool;
fn width()->u32 {let p=u32(data[0]);return 1u+p+p*p;}
fn par(j:u32)->f32 {
 if(from_output){return result[width()+select(0u,u32(data[0]),trial)+j];}
 return data[3u+j];
}
fn ls(x:f32)->f32 {
 let z=exp(-abs(x));var l=log(1.0+z);
 if(z<0.001){l=z*(1.0-z*0.5+z*z/3.0);}
 return -max(-x,0.0)-l;
}
fn sig(x:f32)->f32 {return exp(ls(x));}
fn gap_mass(d:f32)->f32 {
 if(d<0.01){return d*(1.0-d*0.5+d*d/6.0-d*d*d/24.0);}
 return 1.0-exp(-d);
}
fn log_gap(d:f32)->f32 {return log(gap_mass(d));}
fn coordinate(j:u32,off:u32,start:u32)->f32 {
 if(j<off){return data[start+j];}return 1.0;
}
fn criterion(slot:u32)->f32 {
 let np=u32(data[0]);let off=u32(data[1]);let nn=u32(data[2]);let m=np-off;
 var j=slot-1u;var k=0u;let h=slot>np;
 if(h){j=(slot-1u-np)/np;k=(slot-1u-np)%np;}
 var total=0.0;
 for(var n=0u;n<nn;n++){
  let start=3u+np+n*(np+1u);var eta=0.0;
  for(var d=0u;d<off;d++){eta+=par(d)*data[start+d];}
  for(var c=0u;c<=m;c++){
   let count=data[start+off+c];if(count==0.0){continue;}
   var lp=0.0;var deriv=0.0;
   let xj=coordinate(j,off,start);let xk=coordinate(k,off,start);
   if(c==0u || c==m){
    let b=select(off, np-1u,c==m);let e=eta+par(b);
    lp=select(ls(-e),ls(e),c==m);
    if(j<off || j==b){
     deriv=select(sig(e),-sig(-e),c==m)*xj;
     if(h){deriv=0.0;if(k<off || k==b){deriv=sig(e)*sig(-e)*xj*xk;}}
    }
   }else{
    let bj=off+c-1u;let bk=bj+1u;
    let u=eta+par(bj);let v=eta+par(bk);let gap=par(bj)-par(bk);
    lp=ls(u)+ls(-v)+log_gap(gap);
    let r=exp(-gap)/gap_mass(gap);let t=r*(1.0+r);
    let a=sig(u)*sig(-u);let b=sig(v)*sig(-v);
    if(!h){
     if(j<off){deriv=(sig(v)-sig(-u))*xj;}
     if(j==bj){deriv=-sig(-u)-r;}
     if(j==bk){deriv=sig(v)+r;}
    }else{
     if(j<off && k<off){deriv=(a+b)*xj*xk;}
     if(j<off && k==bj){deriv=a*xj;}
     if(j<off && k==bk){deriv=b*xj;}
     if(k<off && j==bj){deriv=a*xk;}
     if(k<off && j==bk){deriv=b*xk;}
     if(j==bj && k==bj){deriv=a+t;}
     if(j==bk && k==bk){deriv=b+t;}
     if((j==bj && k==bk)||(j==bk && k==bj)){deriv=-t;}
    }
   }
   if(slot==0u){total-=count*lp;}else{total+=count*deriv;}
  }
 }
 return total;
}
fn finite(x:f32)->bool {return abs(x)<=3.402823466e38;}
