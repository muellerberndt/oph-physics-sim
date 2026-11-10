import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateProcaEvidence,procaCardMarkup} from './proca_evidence.js';

const data=JSON.parse(fs.readFileSync(new URL('./proca_evidence.json',import.meta.url),'utf8'));
assert.equal(validateProcaEvidence(data),data);
assert(data.members[0].signals[0].probability.display[0]>3.8e-8);
assert(data.members[0].capacity.nats.display[0]>2.69e100);
assert(data.members[1].capacity.nats.display[1]<2.69e108);
assert(procaCardMarkup(data).includes('All 16 outcomes'));
assert(procaCardMarkup(data).includes('Δ/c²'));
for(const [name,change] of [
  ['mass or species promotion',d=>d.scope.standardModelIdentification=true],
  ['finite boost promotion',d=>d.scope.finiteSourceExactLorentzSymmetry=true],
  ['interacting limit promotion',d=>d.scope.interactingRelativisticLimit=true],
  ['implicit triplet reinterpretation',d=>d.scope.unfilteredTripletAutomaticallyVector=true],
  ['missing time momentum',d=>d.filters.timeComponentUsesCanonicalMomentum=false],
  ['missing complete record',d=>d.members[0].records.pop()],
  ['duplicate complete record',d=>d.members[0].records[1]=d.members[0].records[0]],
  ['wrong native cell count',d=>d.members[0].nativeCells='40'],
  ['omitted reference',d=>d.members[0].referenceDimension=1],
  ['overwhelmed temporal signal',d=>d.members[0].signals[0].probability.display[0]=0],
  ['memory-derived N',d=>d.scope.replicasDerivedFromMemoryBudget=true],
  ['universe-capacity promotion',d=>d.members[0].capacity.universeCapacityIdentification=true],
]) {
  const altered=structuredClone(data);change(altered);
  assert.throws(()=>validateProcaEvidence(altered),undefined,name);
}
console.log('PASS: complete native vector evidence and 12 rejected source/scope/outcome mutations');
