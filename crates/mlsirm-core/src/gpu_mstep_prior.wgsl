fn control()->u32 {let p=u32(data[0]);return 3u+p+u32(data[2])*(p+1u);}
fn prior(slot:u32)->f32 {
 let ctl=control();if(data[ctl]==0.0){return 0.0;}
 let np=u32(data[0]);let off=u32(data[1]);var value=0.0;
 for(var d=0u;d<off;d++){
  let a=par(d);let l=log(abs(a));let z=(l-data[ctl+1u])/data[ctl+2u];
  if(slot==0u){value+=l+0.5*z*z;}
  if(slot==1u+d){value+=(1.0+z/data[ctl+2u])/a;}
  if(slot==1u+np+d*np+d){value+=(1.0/(data[ctl+2u]*data[ctl+2u])-1.0-z/data[ctl+2u])/(a*a);}
 }
 return value;
}
