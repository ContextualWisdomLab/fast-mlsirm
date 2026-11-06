struct Shape {p:u32,n:u32,j:u32,z:u32,g:u32,pad0:u32,pad1:u32,pad2:u32}
@group(0) @binding(0) var<uniform> s:Shape;
@group(0) @binding(1) var<storage,read> x:array<f32>;
@group(0) @binding(2) var<storage,read> ix:array<u32>;
@group(0) @binding(3) var<storage,read> cov:array<f32>;
@group(0) @binding(4) var<storage,read_write> o:array<f32>;
// ix: group-major item slope+intercept+gaps indexes, then mean/logSD per group.
fn par(k:u32,default_value:f32)->f32 {if(k==0xffffffffu){return default_value;}return x[k];}
@compute @workgroup_size(64)
fn transform(@builtin(global_invocation_id) id:vec3<u32>){
 let cell=id.x;if(cell>=s.n*(s.p+1u)){return;}
 let r=cell/(s.p+1u);let c=cell%(s.p+1u);let estimate=c==s.p;
 let per=2u+s.j*(2u+3u*s.z);let group=r/per;let rr=r%per;
 let mi=ix[s.g*s.j*(s.z+1u)+group*2u];let si=ix[s.g*s.j*(s.z+1u)+group*2u+1u];
 let mu=par(mi,0.);let sd=exp(par(si,0.));let dm=select(0.,1.,c==mi);let ds=select(0.,sd,c==si);
 var value=0.;var deriv=0.;
 if(rr<2u){if(rr==0u){value=mu;deriv=dm;}else{value=sd;deriv=ds;}}
 else {
 let item=(rr-2u)/(2u+3u*s.z);let typ=(rr-2u)%(2u+3u*s.z);let off=(group*s.j+item)*(s.z+1u);
 let ai=ix[off];let a=x[ai];let da=select(0.,1.,c==ai);let e=a*sd;let de=da*sd+a*ds;let v=3.289868133696453;let den=sqrt(e*e+v);
 if(typ==0u){value=a;deriv=da;}else if(typ==1u){value=e/den;deriv=de*v/(den*den*den);}
 else {
 let cut=(typ-2u)/3u;let kind=(typ-2u)%3u;let di=ix[off+1u];var d=x[di];var dd=select(0.,1.,c==di);
 for(var k=1u;k<=cut;k++){let qi=ix[off+1u+k];let gap=exp(x[qi]);d-=gap;if(c==qi){dd-=gap;}}
 if(kind==0u){value=d;deriv=dd;}
 else if(kind==1u){value=-d/a;deriv=-dd/a+d*da/(a*a);}
 else {let centered=d+a*mu;value=-centered/den;deriv=-(dd+mu*da+a*dm)/den+centered*e*de/(den*den*den);}
 }
 }
 if(estimate){o[r]=value;}else{o[s.n+r*s.p+c]=deriv;}
}
@compute @workgroup_size(64)
fn intermediate(@builtin(global_invocation_id) id:vec3<u32>){let cell=id.x;if(cell>=s.n*s.p){return;}let r=cell/s.p;let c=cell%s.p;var v=0.;for(var k=0u;k<s.p;k++){v+=o[s.n+r*s.p+k]*cov[k*s.p+c];}o[s.n+s.n*s.p+cell]=v;}
@compute @workgroup_size(64)
fn propagate(@builtin(global_invocation_id) id:vec3<u32>){let cell=id.x;if(cell>=s.n*s.n){return;}let r=cell/s.n;let c=cell%s.n;var v=0.;for(var k=0u;k<s.p;k++){v+=o[s.n+s.n*s.p+r*s.p+k]*o[s.n+c*s.p+k];}o[s.n+2u*s.n*s.p+cell]=v;}
