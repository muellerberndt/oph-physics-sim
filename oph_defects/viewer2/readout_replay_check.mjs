import assert from 'node:assert/strict';
import { BornReadoutField } from './readout_instrument.js';
import { recordSnapshot } from './readout_observables.js';
import { replayReadout } from './readout_replay.mjs';
const preparation={id:73,m:.3,x0:[2,2,2],p0:[.1,.2,0],sigma:1.1,chi:[[1,0],[0,0],[0,0],[0,0]],t:0};
const field=new BornReadoutField(4);field.add(preparation);
const transcript=[];
for(let j=1;j<=22;j++) {
  const values=[.2,.31,.47,.53,j%2?.999:0],random=()=>values.shift();
  transcript.push(field.frame(j/3,new Set([73]),.9,random).records[0]);
}
const particle={id:73,cls:3,g0:preparation.x0,dir:[0,0,1],born:0,preparation,
  transcript,recs:transcript.slice(-18),nrec:transcript.length};
const data=recordSnapshot({run:'synthetic-check',observer:7,slice:0,fieldTime:8,particle,family:{name:'lepton candidate'},units:{tick:1,cell:1,hbar:1},grid:{n:4,h:1}});
assert.equal(data.records.length,22);assert.equal(data.displayedRecordLimit,18);
const replay=replayReadout(JSON.parse(JSON.stringify(data)));assert.equal(replay.records,22);
assert.ok(data.records.some(r=>r.band==='converted-negative'));
assert.ok(data.records.some(r=>r.band==='positive'));
let bad=structuredClone(data);bad.records[4].jointProbability+=.1;assert.throws(()=>replayReadout(bad),/differs/);
bad=structuredClone(data);bad.records.shift();assert.throws(()=>replayReadout(bad),/count/);
bad=structuredClone(data);bad.records[0].samplerDraws=null;assert.throws(()=>replayReadout(bad),/sampling/);
console.log('PASS complete 22-record export, both bands, exact replay, altered probability, missing history and missing sampler rejection');
