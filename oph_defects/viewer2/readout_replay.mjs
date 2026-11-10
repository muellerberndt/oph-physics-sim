// Reproduce an exported packet using its exact preparation, times and retained sampler inputs.
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';
import { BornReadoutField } from './readout_instrument.js';
import { packetObservables } from './readout_observables.js';

export function replayReadout(data) {
  assert.equal(data.schema, 'oph.viewer.defect_readout.v2');
  assert.equal(data.completePacketTranscript, true, 'A complete packet transcript is required');
  assert.equal(data.records.length, data.totalRecords, 'Record count differs');
  const n=data.field.gridPointsPerAxis;
  assert.ok(Number.isInteger(n) && n>=2 && n<=64 && (n&(n-1))===0, 'Finite power-of-two grid required');
  const prep=data.field.preparation;
  assert.ok(prep && prep.id===data.source.defectId, 'Prepared source identity differs');
  const field=new BornReadoutField(n); field.add(prep);
  let previous=prep.t, maxError=0;
  const close=(a,b,label)=>{
    assert.ok(Number.isFinite(a)&&Number.isFinite(b),label+' must be finite');
    const error=Math.abs(a-b);maxError=Math.max(maxError,error);
    assert.ok(error<=2e-10*Math.max(1,Math.abs(a),Math.abs(b)),label+' differs');
  };
  for (const row of data.records) {
    assert.ok(row.fieldTime>=previous, 'Detector clock goes backwards');previous=row.fieldTime;
    assert.ok(Array.isArray(row.samplerDraws) && [1,5].includes(row.samplerDraws.length), 'Recorded sampling variates required');
    assert.ok(row.samplerDraws.every(x=>Number.isFinite(x)&&x>=0&&x<1), 'Invalid sampler input');
    let result;
    if (row.samplerDraws.length===1) {
      result=field.record(field.parts.get(prep.id),row.fieldTime,row.detectorWidth,row.samplerDraws[0]);
    } else {
      let i=0;result=field.frame(row.fieldTime,new Set([prep.id]),row.detectorWidth,()=>row.samplerDraws[i++]).records[0];
      assert.equal(i,5);
    }
    assert.deepEqual(result.x,row.gridPosition,'Position outcome differs');
    assert.equal(result.band,row.band,'Band outcome differs');
    for(const [a,b] of [['positionProbability','positionProbability'],['bandProbability','bandProbability'],['probability','jointProbability'],['sourceEnergyChange','sourceEnergyChange'],['conversionWork','conversionWork']])close(result[a],row[b],a);
  }
  const final=packetObservables(field,field.parts.get(prep.id));close(final.norm,1,'Final norm');
  return {schema:'oph.viewer.defect_readout_replay.v1',defectId:prep.id,records:data.records.length,
    maxAbsoluteReplayError:maxError,finalStateExpectations:final,
    nativeParticleIdentityEstablished:false,
    scope:'Finite viewer Dirac packet and declared detector; supplied unit conversion and sampler inputs retained'};
}

if(process.argv[1]===fileURLToPath(import.meta.url)) {
  if(!process.argv[2])throw new Error('Usage: node readout_replay.mjs saved-readout.json');
  console.log(JSON.stringify(replayReadout(JSON.parse(readFileSync(process.argv[2],'utf8'))),null,2));
}
