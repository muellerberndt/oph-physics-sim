import assert from 'node:assert/strict';
import { twoTickComparison } from './readout_limit.js';
for(const p of [[0,0,0],[.1,0,0],[.2,-.3,.1],[-.4,.05,.17]])for(const theta of [0,.03,.4]) {
  const r=twoTickComparison(p,theta);
  // A 4x4 matrix has Frobenius norm at most twice its operator norm.
  assert.ok(r.frobeniusError<=2*r.operatorErrorUpper+1e-13);
}
assert.ok(twoTickComparison([0,0,0],.3).frobeniusError<1e-14);
const errors=[1,.5,.25,.125].map(a=>twoTickComparison([.17*a,.11*a,.07*a],.2*a).frobeniusError);
for(let j=1;j<errors.length;j++)assert.ok(errors[j-1]/errors[j]>7.6&&errors[j-1]/errors[j]<8.4);
const fixed=[.01,.005].map(a=>twoTickComparison([.17*a,.11*a,.07*a],.2).frobeniusError);
assert.ok(fixed[0]/fixed[1]>1.9&&fixed[0]/fixed[1]<2.1);
console.log('PASS exact two-tick matrices, complete Taylor bound, cubic joint scaling and distinct fixed-angle limit');
console.log(JSON.stringify({jointScalingFrobeniusErrors:errors,fixedMassAngleErrors:fixed}));
