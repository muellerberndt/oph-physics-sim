import assert from 'node:assert/strict';
import fs from 'node:fs';
import { validateCollisionEvidence } from './collision_evidence.js';

const data=JSON.parse(fs.readFileSync(new URL('./collision_evidence.json',import.meta.url),'utf8'));
assert.equal(validateCollisionEvidence(data),data);
assert(data.frames[4].hit.display[0]>0.0029);
assert(data.frames[0].idealHit.display[0]===0);
for(const [name,change] of [
  ['named particle promotion', d=>d.scope.standardModelIdentification=true],
  ['trajectory promotion', d=>d.scope.classicalTrajectories=true],
  ['missing complement', d=>d.frames[4].complement.display=[0,0]],
  ['negative probability', d=>d.frames[0].hit.display[0]=-.001],
  ['omitted native sector', d=>d.frames[0].sites[0].components.pop()],
  ['omitted preparation flags', d=>d.preparation.allFailureFlagsRetained=false],
  ['changed source dimension', d=>d.source.reducingDimension=16],
]) {
  const altered=structuredClone(data);change(altered);
  assert.throws(()=>validateCollisionEvidence(altered),undefined,name);
}
console.log('PASS: five complete finite collision records and seven rejected scope/outcome mutations');
