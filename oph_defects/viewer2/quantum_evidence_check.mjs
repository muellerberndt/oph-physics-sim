import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validateQuantumEvidence,higgsConnectionHTML,centroidHTML,higgsCurrentReadoutHTML,coupledReadoutHTML,wilsonMaxwellHTML,photonCutoffHTML,quarkReadoutHTML,quarkResponseHTML,coupledGroundHTML,localCurrentHTML,localCurrentReadoutHTML} from './quantum_evidence.js';
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
for(const mutate of [x=>x.photonRefinement.overflowRetained=false,x=>x.photonRefinement.sameSourceStateReferenceClock=false,x=>x.photonRefinement.unboundedOscillatorActuallyExecuted=true,x=>x.photonRefinement.spatialContinuumOrSMIdentified=true,x=>x.photonRefinement.executedLabels=1451,x=>x.photonRefinement.cap3ErrorUpper.display=[0,0]]){
 const changed=structuredClone(data);mutate(changed);assert.throws(()=>validateQuantumEvidence(changed));
}
const cutoff=photonCutoffHTML(data);
assert.match(cutoff,/cap 3 · this run/);assert.match(cutoff,/1\.863 × 10<sup>-9<\/sup>/);
assert.match(cutoff,/1,451 labels/);assert.match(cutoff,/no infinite register is executed/);
const errors=[...cutoff.matchAll(/≤ ([\d.]+) × 10<sup>(-?\d+)<\/sup>/g)].map(x=>Number(x[1])*10**Number(x[2]));
assert.equal(errors.length,data.photonRefinement.cutoffs.length);
errors.forEach((value,i)=>assert.ok(value>=data.photonRefinement.cutoffs[i].errorUpper.display[1]));
console.log('PASS: fixed-cube complete-record cutoff, overflow and outward error displays');
for(const mutate of [x=>x.quarkResponses.staticColorReferenceRetained=false,x=>x.quarkResponses.actualCARNormalization=false,x=>x.quarkResponses.propagatingQuarkOrGeVMassIdentified=true,x=>x.quarkResponses.families[0].massGeV=.002,x=>x.quarkResponses.families[2].localLines.pop(),x=>x.quarkResponses.families[0].referenceEnergy='0',x=>x.quarkResponses.families[0].originalStrongBinLower.display=[0,0],x=>x.quarkResponses.families[0].allFullSourceBins[0].energy.display=[100,101],x=>x.quarkResponses.families[0].localLines[0].energy.display=[NaN,NaN],x=>x.quarkResponses.families[0].pumpFailure.display=[0,0],x=>x.quarkResponses.families[0].preparationSuccessLower.display=[0,0]]){
 const changed=structuredClone(data);mutate(changed);assert.throws(()=>validateQuantumEvidence(changed));
}
assert.match(quarkReadoutHTML(data,'uc'),/49\.539796%/);
assert.match(quarkReadoutHTML(data,'dc'),/49\.539796%/);
assert.match(quarkReadoutHTML(data,'Q'),/49\.570179%/);
assert.match(quarkReadoutHTML(data,'Q'),/12 have exactly zero atomic weight/);
assert.equal((quarkReadoutHTML(data,'Q').match(/<g><title>Line/g)||[]).length,25);
assert.throws(()=>quarkReadoutHTML(data,'top'));
assert.match(quarkReadoutHTML(data,'Q'),/Outside all listed bins/);assert.match(quarkReadoutHTML(data,'Q'),/conditions on preparation success/);
assert.match(quarkResponseHTML(data),/not a prediction of equal observed up\/down masses/);
console.log('PASS: complete native quark lines, zero residues and original attempt weights');
for(const mutate of [x=>x.coupledGround.originalGroundInsideCode=true,x=>x.coupledGround.originalActiveGroundMultiplicity=1,x=>x.coupledGround.changedHamiltonianExplicit=false,x=>x.coupledGround.positiveMacrocellPenalty=false,x=>x.coupledGround.allFailuresPartnersRetained=false,x=>x.coupledGround.patternPreservingOldCountLawsInvariant=false,x=>x.coupledGround.unknownGroundProjectorOrPhaseUsed=true,x=>x.coupledGround.primitiveRepairForcesCompletion=true,x=>x.coupledGround.radiationOrGeVMass=true,x=>x.coupledGround.fullNativeSpatialLocalityProved=true,x=>x.coupledGround.filters=1,x=>x.coupledGround.completeCountLabels=1160,x=>x.coupledGround.preparedStationaryCountLower.display=[0,0],x=>x.coupledGround.capacityNats.display=[Infinity,Infinity]]){
 const changed=structuredClone(data);mutate(changed);assert.throws(()=>validateQuantumEvidence(changed));
}
const ground=coupledGroundHTML(data);
assert.match(ground,/96 lowest active states outside the code/);assert.match(ground,/1\.416928 per million/);
assert.match(ground,/99\.959432% success/);assert.match(ground,/global code guard/);assert.match(ground,/not<\/strong> describe radiation/);
console.log('PASS: full-sector ground distinction, explicit completion, retained filters and occupation count');
for(const mutate of [x=>x.localCurrent.fullHamiltonianCancellation=true,x=>x.localCurrent.nativeRepairCompilationProved=true,x=>x.localCurrent.exactGroundProjection=true,x=>x.localCurrent.instantaneousFinalSourceMean=true,x=>x.localCurrent.SMParticleOrWeakBoson=true,x=>x.localCurrent.localCouplingDeclared=false,x=>x.localCurrent.priorPreparationAndBinServicesRetained=false,x=>x.localCurrent.allFailuresRetained=false,x=>x.localCurrent.sourceKeepsEvolving=false,x=>x.localCurrent.totalSourceTime='4',x=>x.localCurrent.finalPulseSourceTime='1',x=>x.localCurrent.localMatrixDimension=80,x=>x.localCurrent.records.pop(),x=>x.localCurrent.apparatusErrorUpper.display=[0,0],x=>x.localCurrent.meanErrorUpper.display=[NaN,NaN],x=>x.localCurrent.elementaryMean.display=[.09,.1],x=>x.localCurrent.records[1].originalCompositeJointLower.display=[0,0]]){
 const changed=structuredClone(data);mutate(changed);assert.throws(()=>validateQuantumEvidence(changed));
}
const local=localCurrentHTML(data);
assert.match(local,/without a control that cancels the whole Hamiltonian/);assert.match(local,/3 \+ 2⁻²⁰ source units/);
assert.match(local,/5,270,718/);assert.match(local,/10,541,436/);assert.match(local,/preparation, pump and exact energy-bin services retain/);
assert.match(localCurrentReadoutHTML(data,'elementary'),/2\.474173%/);
assert.match(localCurrentReadoutHTML(data,'composite'),/0\.295882%/);
assert.equal((localCurrentReadoutHTML(data,'composite').match(/class="qe-current-row"/g)||[]).length,5);
assert.throws(()=>localCurrentReadoutHTML(data,'opposite-charge'));
for(const name of ['native_local_current_readout','native_wilson_maxwell_refinement','native_quark_response','native_wilson_maxwell_ground']){
 for(const key of ['path','sha256','proof']){
  const changed=structuredClone(data);changed.evidence[name][key]='unverified';assert.throws(()=>validateQuantumEvidence(changed));
 }
}
console.log('PASS: local uninterrupted current pointer, complete five outcomes, source clock and apparatus boundaries');
