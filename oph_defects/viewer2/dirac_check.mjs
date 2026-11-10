import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateDirac,diracFrequency,diracMarkup} from './dirac_evidence.js';
const d=JSON.parse(fs.readFileSync(new URL('./dirac_evidence.json',import.meta.url),'utf8'));
assert.equal(validateDirac(d),d);
const p=i=>d.trials[i].records.find(r=>r.word==='0010').ideal.display[0];
assert(Math.abs(p(0)-.05)<1e-15);assert(p(1)-p(0)>.08823549);
assert.equal(diracFrequency(d,0).finite,1);
for(let n=1;n<=4;n++){const x=diracFrequency(d,n);assert(x.finite<x.continuum);}
assert(diracMarkup(d).includes('full-range couplers'));
for(const [label,change] of [
 ['missing record',x=>x.trials[0].records.pop()],
 ['duplicated record',x=>x.trials[1].records[1]=structuredClone(x.trials[1].records[0])],
 ['unnormalized law',x=>x.trials[0].records[0].ideal.display=[.9,.9]],
 ['invalid probability',x=>x.trials[0].records[0].completeAttempt.display=[-1,0]],
 ['electron identity',x=>x.scope.identifiedElectron=true],
 ['GeV promotion',x=>x.scope.sourceDerivedGeV=true],
 ['chiral promotion',x=>x.scope.SMWeakChirality=true],
 ['interaction transfer',x=>x.scope.interactingRelativisticLimit=true],
 ['affine clock substituted',x=>x.scope.sameAsAffineFieldClock=true],
 ['classical run substituted',x=>x.scope.sameAsClassicalLoadRun=true],
 ['finite boost asserted',x=>x.scope.exactFiniteBoxLorentzSymmetry=true],
 ['source execution overstated',x=>x.execution.fullNativeMatrixExecuted=true],
 ['selected eigenvectors asserted',x=>x.execution.selectedNativeEigenvectorsExecuted=true],
 ['hard-core code substituted',x=>x.source.fockStatesPerSite=3],
 ['locality asserted',x=>x.source.fullSpatialRange=false],
 ['preparation failures omitted',x=>x.preparation.allOriginalFlagsRetained=false],
 ['controller premise hidden',x=>x.preparation.programmableFiniteMatrixControl=false],
 ['unaccounted workspace',x=>x.preparation.capacityIncludesControllerSynthesis=true],
 ['wrong dispersion',x=>x.dispersion.testFrequencyOverMass.display=[2,2]],
]){const q=structuredClone(d);change(q);assert.throws(()=>validateDirac(q),undefined,label);}
console.log('PASS: exact native Dirac detector contrast,16records,finite dispersion and19 rejected mutations');
