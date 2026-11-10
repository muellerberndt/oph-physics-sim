import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validateQuantumEvidence,higgsConnectionHTML,centroidHTML,higgsCurrentReadoutHTML,coupledReadoutHTML,wilsonMaxwellHTML} from './quantum_evidence.js';
const data=JSON.parse(readFileSync(new URL('./quantum_evidence.json',import.meta.url)));
assert.equal(validateQuantumEvidence(data),data);
for(const mutate of [x=>x.scope.sourceDerivedGeV=true,x=>x.channels[0].referenceEnergy='0',x=>x.channels[0].probabilities.outside.display=[0,0],x=>x.channels[0].massGeV=.000511,x=>x.channels[0].sourceEnergy.display=[1,0]]) {
 const changed=structuredClone(data);mutate(changed);assert.throws(()=>validateQuantumEvidence(changed));
}
assert.ok(data.channels[0].sourceEnergy.display[0]>.69);
assert.ok(data.channels[1].sourceEnergy.display[1]<.1);
for(const mutate of [x=>x.higgsConnection.usesClassicalScalarBackground=true,x=>x.higgsConnection.compositeCanonicalOnAllOccupations=true,x=>x.higgsConnection.sameAsMaximumElementaryResidueSelector=true,x=>x.higgsConnection.weakScatteringOrSMIdentityProved=true,x=>x.higgsConnection.QPEImplementedForThisCompositeExperiment=true,x=>x.higgsConnection.allFailureAndComplementRecordsRetained=false,x=>x.higgsConnection.detectorRecords=6,x=>x.higgsConnection.originalSettingHitContrastLower='0',x=>x.higgsConnection.crossResidueLower='-1',x=>x.higgsConnection.compositeBinWeightLower='1/0']){
 const changed=structuredClone(data);mutate(changed);assert.throws(()=>validateQuantumEvidence(changed));
}
const html=higgsConnectionHTML(data);assert.match(html,/7\.247007 percentage points/);assert.match(html,/exact finite apparatus service/);assert.match(html,/All 16/);
for(const mutate of [x=>x.centroid.extraGroundProjection=true,x=>x.centroid.exactGroundPhaseInput=true,x=>x.centroid.fullSourcePointerExecuted=true,x=>x.centroid.singleEigenvalueSelected=true,x=>x.centroid.restMassOrSMIdentity=true,x=>x.centroid.allFailuresAndPointerLabelsRetained=false,x=>x.centroid.priorExactSpectralBinService=false,x=>x.centroid.pointerLabels=2,x=>x.centroid.sourceEnergyMean.display=[.69,.7],x=>x.centroid.originalBinSuccessLower.display=[0,0],x=>x.centroid.baselineErrorUpper.display=[0,0]]){
 const changed=structuredClone(data);mutate(changed);assert.throws(()=>validateQuantumEvidence(changed));
}
const centroid=centroidHTML(data);assert.match(centroid,/41\.970618%/);assert.match(centroid,/0\.6916566792649/);assert.match(centroid,/No extra ground projection/);assert.match(centroid,/536,870,400/);
for(const mutate of [x=>x.filteredState.exactGroundProjection=true,x=>x.filteredState.fullUnconditionedLawsEquated=true,x=>x.filteredState.interactingSMTransfer=true,x=>x.filteredState.commonSubsequentPolicyRequired=false,x=>x.filteredState.allPreparationRecordsRetained=false,x=>x.filteredState.sourceTrials=1,x=>x.filteredState.historyTVUpper.display=[0,0],x=>x.filteredState.originalCompositeSuccessLower.display=[0,0]]){
 const changed=structuredClone(data);mutate(changed);assert.throws(()=>validateQuantumEvidence(changed));
}
assert.match(html,/every later detector/);assert.match(html,/31\.477352%/);assert.match(html,/3\.764603%/);assert.match(html,/does not equate the full unconditioned/);
for(const mutate of [x=>x.weakResponse.exactGroundProjection=true,x=>x.weakResponse.physicalWeakCrossSection=true,x=>x.weakResponse.WParticleIdentified=true,x=>x.weakResponse.measurementCanChangeEnergy=false,x=>x.weakResponse.allFailuresAndPointerLabelsRetained=false,x=>x.weakResponse.currentRecords.pop(),x=>x.weakResponse.originalInelasticLower.display=[0,0]]){
 const changed=structuredClone(data);mutate(changed);assert.throws(()=>validateQuantumEvidence(changed));
}
assert.match(html,/Does the quantum weak current respond/);assert.match(html,/original attempt/);assert.match(html,/can change the energy/);
console.log('PASS: complete interacting energies; rejected free reference, missing outcome and mass promotion');
for(const mutate of [x=>x.higgsCurrent.currentInvariantUnderConjugation=true,x=>x.higgsCurrent.exactGroundProjection=true,x=>x.higgsCurrent.SMParticleOrWeakBoson=true,x=>x.higgsCurrent.allFailuresRetained=false,x=>x.higgsCurrent.records.pop(),x=>x.higgsCurrent.elementaryMean.display=[.09,.1],x=>x.higgsCurrent.records[1].originalCompositeJointLower.display=[0,0]]){
 const changed=structuredClone(data);mutate(changed);assert.throws(()=>validateQuantumEvidence(changed));
}
assert.match(higgsCurrentReadoutHTML(data,'elementary'),/2\.488191%/);
assert.match(higgsCurrentReadoutHTML(data,'composite'),/0\.297229%/);
assert.throws(()=>higgsCurrentReadoutHTML(data,'opposite-charge'));
for(const mutate of [x=>x.wilsonMaxwell.inputIsCoupledGround=true,x=>x.wilsonMaxwell.exactGroundProjection=true,x=>x.wilsonMaxwell.continuumOrSMIdentification=true,x=>x.wilsonMaxwell.allComplementRecordsRetained=false,x=>x.wilsonMaxwell.wholeOperatorEvolutionErrorIncluded=false,x=>x.wilsonMaxwell.bothIndependentCurrentMethodsAgree=false,x=>x.wilsonMaxwell.completeRecords=1160,x=>x.wilsonMaxwell.targetCounts=[1,0,1],x=>x.wilsonMaxwell.totalPhotonCap=4,x=>x.wilsonMaxwell.targetLower.display=[0,0],x=>x.wilsonMaxwell.connectedLower.display=[0,0],x=>x.wilsonMaxwell.controllerWorkspaceIncludedInCapacity=true]){
 const changed=structuredClone(data);mutate(changed);assert.throws(()=>validateQuantumEvidence(changed));
}
assert.match(coupledReadoutHTML(data,true),/4\.75055 × 10<sup>-8<\/sup>/);
assert.match(coupledReadoutHTML(data,true),/4\.71131 × 10<sup>-8<\/sup>/);
assert.equal((coupledReadoutHTML(data,false).match(/0 exactly/g)||[]).length,2);
assert.match(wilsonMaxwellHTML(data),/not the coupled ground state/);
assert.match(wilsonMaxwellHTML(data),/1,161 retained labels/);
assert.match(wilsonMaxwellHTML(data),/declared choices/);
console.log('PASS: same-current preparations and coupled/free complete-record controls');
