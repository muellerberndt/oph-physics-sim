// Two alternating half-step routing ticks versus the declared free Dirac clock.
import { symbol } from './cf1_engine.js';
import { ContinuumField } from './readout_field.js';

function multiply(A,B) {
  const re=new Float64Array(16),im=new Float64Array(16);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++)for(let k=0;k<4;k++) {
    re[4*i+j]+=A.re[4*i+k]*B.re[4*k+j]-A.im[4*i+k]*B.im[4*k+j];
    im[4*i+j]+=A.re[4*i+k]*B.im[4*k+j]+A.im[4*i+k]*B.re[4*k+j];
  }
  return {re,im};
}
export function twoTickComparison(p,theta) {
  if(p.length!==3||![...p,theta].every(Number.isFinite))throw Error('Finite momentum and mass angle required');
  // Conjugating M(theta) S by M(theta/2) gives the actual half-step frame.
  const pair=multiply(symbol(p,theta,'Y',true),symbol(p,theta,'Y',false));
  const W=multiply(multiply(symbol([0,0,0],-theta/2),pair),symbol([0,0,0],theta/2));
  const E=Math.hypot(...p,theta),c=Math.cos(2*E),f=E?Math.sin(2*E)/E:2;
  let squared=0;
  for(let j=0;j<4;j++) {
    const v=new Float64Array(8),h=new Float64Array(8);v[2*j]=1;
    ContinuumField.applyH(...p,theta,v,h);
    for(let i=0;i<4;i++) {
      const re=(i===j?c:0)+f*h[2*i+1],im=-f*h[2*i];
      squared+=(W.re[4*i+j]-re)**2+(W.im[4*i+j]-im)**2;
    }
  }
  const S=2*(Math.abs(theta)+p.reduce((a,x)=>a+Math.abs(x),0));
  // Taylor coefficients through degree two agree. Each remaining product
  // coefficient has norm <= S^n/n!, as does the single exponential.
  const bound=Math.min(2,Math.exp(S)*S**3/3);
  return {frobeniusError:Math.sqrt(squared),operatorErrorUpper:bound,totalGeneratorNorm:S};
}
