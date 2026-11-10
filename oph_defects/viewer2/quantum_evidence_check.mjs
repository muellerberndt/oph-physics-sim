import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validateQuantumEvidence} from './quantum_evidence.js';
const data=JSON.parse(readFileSync(new URL('./quantum_evidence.json',import.meta.url)));
assert.equal(validateQuantumEvidence(data),data);
for(const mutate of [x=>x.scope.sourceDerivedGeV=true,x=>x.channels[0].referenceEnergy='0',x=>x.channels[0].probabilities.outside.display=[0,0],x=>x.channels[0].massGeV=.000511,x=>x.channels[0].sourceEnergy.display=[1,0]]) {
 const changed=structuredClone(data);mutate(changed);assert.throws(()=>validateQuantumEvidence(changed));
}
assert.ok(data.channels[0].sourceEnergy.display[0]>.69);
assert.ok(data.channels[1].sourceEnergy.display[1]<.1);
console.log('PASS: complete interacting energies; rejected free reference, missing outcome and mass promotion');
