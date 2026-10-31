// Device-only Newton solve, ordered Armijo backtracking and accepted parameter copy.
// Control tail: prior,mu,sd,ridge,iterations,backtracks,gradient_tol,step_tol,pivot_tol,armijo.
@compute @workgroup_size(1)
fn main(){
 from_output=true;trial=false;
 let np=u32(data[0]);let off=u32(data[1]);let w=width();let ctl=control();
 let step=w+2u*np;let receipt_base=step+np;
 // receipt_base: status, objective, gradient, hessian, prior, solve, line_search, update, iterations.
 for(var j=0u;j<9u;j++){result[receipt_base+j]=0.0;}
 for(var j=0u;j<np;j++){result[w+j]=data[3u+j];}
 for(var it=0u;it<u32(data[ctl+4u]);it++){
  var norm=0.0;
  for(var slot=0u;slot<w;slot++){
   result[slot]=criterion(slot)+prior(slot);
   if(!finite(result[slot])){result[receipt_base]=2.0;return;}
   if(slot>0u && slot<=np){norm=max(norm,abs(result[slot]));}
  }
  result[receipt_base+1u]+=1.0;result[receipt_base+2u]+=1.0;result[receipt_base+3u]+=1.0;result[receipt_base+4u]+=1.0;
  if(norm<=data[ctl+6u]){result[receipt_base]=1.0;return;}
  let f0=result[0];
  for(var j=0u;j<np;j++){result[step+j]=result[1u+j];result[1u+np+j*np+j]+=data[ctl+3u];}
  // Partial-pivot Gaussian elimination, dynamic storage scratch. No host solve.
  for(var j=0u;j<np;j++){
   var pivot=j;var best=abs(result[1u+np+j*np+j]);
   for(var k=j+1u;k<np;k++){let a=abs(result[1u+np+k*np+j]);if(a>best){best=a;pivot=k;}}
   if(!finite(best)||best<=data[ctl+8u]){result[receipt_base]=3.0;return;}
   if(pivot!=j){
    for(var k=0u;k<np;k++){let a=1u+np+j*np+k;let b=1u+np+pivot*np+k;let t=result[a];result[a]=result[b];result[b]=t;}
    let t=result[step+j];result[step+j]=result[step+pivot];result[step+pivot]=t;
   }
   for(var k=j+1u;k<np;k++){
    let factor=result[1u+np+k*np+j]/result[1u+np+j*np+j];
    for(var l=j;l<np;l++){result[1u+np+k*np+l]-=factor*result[1u+np+j*np+l];}
    result[step+k]-=factor*result[step+j];
   }
  }
  for(var jj=np;jj>0u;jj--){let j=jj-1u;var value=result[step+j];for(var k=j+1u;k<np;k++){value-=result[1u+np+j*np+k]*result[step+k];}result[step+j]=value/result[1u+np+j*np+j];}
  var directional=0.0;var largest=0.0;
  for(var j=0u;j<np;j++){if(!finite(result[step+j])){result[receipt_base]=3.0;return;}directional+=result[1u+j]*result[step+j];largest=max(largest,abs(result[step+j]));}
  if(!finite(directional)||directional<=0.0){result[receipt_base]=3.0;return;}
  result[receipt_base+5u]+=1.0;
  var alpha=1.0;var accepted=false;var actual_change=0.0;
  for(var bt=0u;bt<u32(data[ctl+5u]);bt++){
   trial=true;var valid=true;actual_change=0.0;
   for(var j=0u;j<np;j++){result[w+np+j]=result[w+j]-alpha*result[step+j];valid=valid&&finite(result[w+np+j]);actual_change=max(actual_change,abs(result[w+np+j]-result[w+j]));}
   for(var j=off+1u;j<np;j++){valid=valid&&(par(j-1u)>par(j));}
   if(data[ctl]!=0.0){for(var j=0u;j<off;j++){valid=valid&&(par(j)!=0.0);}}
   if(valid){
    let f=criterion(0u)+prior(0u);result[receipt_base+1u]+=1.0;result[receipt_base+4u]+=1.0;
    if(finite(f)&&f<=f0-data[ctl+9u]*alpha*directional&&actual_change>0.0){accepted=true;}
    // Hager & Zhang (2005), SIAM J. Optim. 16(1), 170-192, Sec. 4, Eqs. (4.1)-(4.2):
    // near a minimizer an f32 objective difference cannot resolve Armijo decrease, so also
    // accept the derivative-based approximate Wolfe conditions with phi(alpha)<=phi(0)+eps_f.
    // eps_f is the existing strict-EM f32 roundoff bound 32*EPS*(1+|f|) (grm.rs).
    if(!accepted&&finite(f)&&actual_change>0.0&&f<=f0+32.0*1.1920929e-7*(1.0+abs(f0))){
     var dphi=0.0;
     for(var j=0u;j<np;j++){dphi-=(criterion(1u+j)+prior(1u+j))*result[step+j];}
     let dphi0=-directional;
     if(finite(dphi)&&(2.0*data[ctl+9u]-1.0)*dphi0>=dphi&&dphi>=0.9*dphi0){accepted=true;}
    }
   }
   result[receipt_base+6u]+=1.0;trial=false;
   if(accepted){break;}alpha*=0.5;
  }
  if(!accepted){result[receipt_base]=4.0;return;}
  for(var j=0u;j<np;j++){result[w+j]=result[w+np+j];}
  result[receipt_base+7u]+=1.0;result[receipt_base+8u]+=1.0;
  if(actual_change<=data[ctl+7u]){result[receipt_base]=1.0;return;}
 }
 // A bounded generalized-EM M step may complete without inner convergence.
 result[receipt_base]=5.0;
}
