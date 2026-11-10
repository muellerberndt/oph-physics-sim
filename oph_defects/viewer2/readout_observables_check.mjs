import assert from 'node:assert/strict';
import { BornReadoutField as ContinuumField } from './readout_instrument.js';
import { packetObservables, invalidatePacketObservables, physicalReadout, recordSnapshot } from './readout_observables.js';

const field = new ContinuumField(8), mass = 0.4;
field.add({id:1,m:mass,x0:[4,4,4],p0:[0.3,0.1,0],sigma:1.3,chi:[[1,0],[0,0],[0,0],[0,0]],t:0});
const p=field.parts.get(1);
for (const a of [...p.sr,...p.si]) a.fill(0);
const index=1+2*8, k=[field.kx[index],field.ky[index],field.kz[index]], energy=Math.hypot(...k,mass);
p.sr[0][index]=Math.sqrt(field.N);
field.projectPlus(p.sr,p.si,p.m); field.normalizeSpectral(p.sr,p.si);
const plane=packetObservables(field,p), close=(a,b)=>assert.ok(Math.abs(a-b)<1e-11,`${a} != ${b}`);
plane.meanMomentum.forEach((x,a)=>close(x,k[a])); close(plane.meanEnergy,energy);
plane.meanVelocity.forEach((x,a)=>close(x,k[a]/energy)); close(plane.meanSpeed,Math.hypot(...k)/energy);
close(plane.norm,1); close(plane.restEnergy,mass);
assert.ok(plane.momentumStd.every(x=>x<1e-7));
plane.momentumMarginals.forEach((row,a)=>{close(row.reduce((x,y)=>x+y,0),1);close(row[[1,2,0][a]],1);});
close(plane.meanEnergy**2-plane.meanMomentum.reduce((a,x)=>a+x*x,0),mass**2);
console.log('PASS actual Fourier translation generator, Dirac energy, group velocity and norm');

field.evolve(p,7);field.readouts();
assert.equal(packetObservables(field,p),plane); // Free positive-band propagation conserves these quantities.
field.record(p,7,0.8,0.37);invalidatePacketObservables(p);
const updated=packetObservables(field,p); assert.notEqual(updated,plane);close(updated.norm,1);
assert.ok(updated.momentumStd.some(x=>x>0.1));assert.ok(updated.meanSpeed<=1+1e-12);
updated.momentumMarginals.forEach(row=>{close(row.reduce((x,y)=>x+y,0),1);assert.ok(row.every(x=>x>=0));});
console.log('PASS detector update invalidates conserved-state cache and changes momentum spread');

const units={hbar:6.582119569e-22,tick:1.4710399122721842e-23,cell:4.410066711161824};
const converted=physicalReadout(plane,units,16);
close(converted.restEnergyMeV,mass*units.hbar/(16*units.tick));
assert.ok(Math.abs(converted.sourceSpeedMetresPerSecond-299792458)<1);
assert.ok(converted.meanSpeedMetresPerSecond<converted.sourceSpeedMetresPerSecond);
console.log('PASS independent dimensions for momentum, energy and metres per second');

const particle={id:42,cls:3,g0:[1,2,3],dir:[0,0,1],born:0,stats:{observables:plane},recs:[{t:2,x:[3,4,5]}],nrec:1};
const snapshot=recordSnapshot({run:'L3',observer:7,slice:2,fieldTime:3,particle,family:{name:'left-handed lepton'},units,grid:{n:8,h:16}});
assert.equal(snapshot.source.defectId,42);assert.equal(snapshot.interpretation.measuredStandardModelSpeciesEstablished,false);
assert.equal(snapshot.field.model,'finite periodic free Dirac packet');assert.equal(snapshot.records.length,1);
particle.recs[0].x[0]=100;assert.equal(snapshot.records[0].gridPosition[0],3);
assert.ok(JSON.stringify(snapshot).includes('supplied viewer conversion'));
console.log('PASS exported record preserves source identity, model and unit ancestry without mutating history');
