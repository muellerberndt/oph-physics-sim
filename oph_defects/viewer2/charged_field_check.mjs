import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateChargedField,chargedFieldMarkup} from './charged_field_evidence.js';
const d=JSON.parse(fs.readFileSync(new URL('./charged_field_evidence.json',import.meta.url),'utf8'));
assert.strictEqual(validateChargedField(d),d);
assert(d.fieldExcitation.afterClock.display[0]>7.0979e-5);
assert(d.fieldExcitation.original.display[0]>1.7739e-5);
assert.equal(new Set(d.records.map(r=>`${r.vertex}:${r.fieldExcited}`)).size,26);
assert.equal(d.energy.totalOutward[0],'1.8738097549651');
assert.equal(d.energy.totalOutward[1],'1.8738097549672');
assert.equal(d.clock.groundFidelityFloor,'0.99999999439512');
assert(chargedFieldMarkup(d).includes('Every')||chargedFieldMarkup(d).includes('every'));
for(const [name,change] of [
 ['ground replaced prepared state',x=>x.scope.groundProbabilitiesEqualPreparedProbabilities=true],
 ['mass promotion',x=>x.scope.sourceDerivedGeV=true],
 ['species promotion',x=>x.scope.standardModelIdentification=true],
 ['readout backaction omitted',x=>x.scope.postMeasurementStateUnchanged=true],
 ['source clock paused',x=>x.clock.commonAutonomousSourceEvolution=false],
 ['ideal ground projection substituted',x=>x.clock.exactGroundProjection=true],
 ['native join promoted',x=>x.nativeAttachment.fullChargedGroundIdentified=true],
 ['removed ground admission',x=>x.nativeAttachment.additionalGroundAdmissionService=false],
 ['missing position-field outcome',x=>x.records.pop()],
 ['duplicated outcome',x=>x.records[1]=structuredClone(x.records[0])],
 ['lost rare original signal',x=>x.records[1].original.display[0]=0],
 ['overweight probability',x=>x.records[0].afterClock.display=[1.1,1.2]],
 ['missing energy failure',x=>x.clock.pumpFailure=0],
 ['wrong phase menu',x=>x.clock.phaseLabels=32],
 ['reference erased',x=>x.source.referenceEnergy=0],
 ['cutoff compressed',x=>x.source.fieldStates=30],
 ['entropy confused with capacity',x=>x.capacity.stateEntropy=true],
 ['invalid cone',x=>x.geometry.edges.pop()],
]){const q=structuredClone(d);change(q);assert.throws(()=>validateChargedField(q),undefined,name);}
console.log('PASS:26 positive original detector outcomes, actual-clock/ground distinction and18 rejected mutations');
