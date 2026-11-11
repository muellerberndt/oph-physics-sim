// Check the shipped finite-record contract without needing a theory checkout.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const data=JSON.parse(readFileSync(new URL('./field_records.json',import.meta.url)));
function rational(value){
  if(value.includes('/'))return value.split('/').map(BigInt);
  const match=/^([+-]?)(\d+)(?:\.(\d+))?(?:e([+-]?\d+))?$/.exec(value);
  assert.ok(match,'invalid rational display');
  const digits=match[3]||'', power=Number(match[4]||0)-digits.length;
  let numerator=BigInt(match[2]+digits)*(match[1]==='-'?-1n:1n),denominator=1n;
  if(power>=0)numerator*=10n**BigInt(power);else denominator=10n**BigInt(-power);
  return [numerator,denominator];
}
function le(a,b){const[x,y]=rational(a),[u,v]=rational(b);return x*v<=u*y;}
let intervals=0;
function visit(value){
  if(!value||typeof value!=='object')return;
  if(value.exact!==undefined&&value.lowerText){
    const bounds=Array.isArray(value.exact)?value.exact:[value.exact,value.exact];
    assert.ok(le(value.lowerText,bounds[0])&&le(bounds[0],bounds[1])&&le(bounds[1],value.upperText),'display must enclose exact bounds');intervals++;
  }
  Object.values(value).forEach(visit);
}
visit(data);
assert.equal(data.schema,'oph.viewer.field_records.v1');
assert.deepEqual(data.scope.physicalParticleIdentities,[]);
assert.deepEqual(data.scope.sourceDerivedMassesGeV,[]);
assert.equal(data.scope.allFailuresRetained,true);
assert.equal(data.scope.sameAsClassicalLoadRun,false);
assert.equal(data.scope.interactingRelativisticLimit,false);
assert.deepEqual(data.quark.map(r=>r.records),[88,88,160]);
assert.deepEqual(data.quark[0].energy.exact,data.quark[1].energy.exact);
assert.ok(le(data.quark[0].energy.exact[1],data.quark[2].energy.exact[0]));
assert.ok(le(data.interaction.cutoffError.exact,data.interaction.completeLawSeparation.exact));
assert.equal(data.cone.kineticCompressionIsSelfEnergy,false);
assert.equal(data.interaction.matrixEnumerated,false);
assert.equal(data.revisedInteraction.sourceChanged,true);
assert.equal(data.revisedInteraction.groundSectorCompletion,'18V+1');
assert.equal(data.revisedInteraction.internalGroundGapProved,false);
assert.ok(le(data.interaction.completeLawSeparation.exact,data.revisedInteraction.completeLawSeparation.exact));
assert.equal(data.preparation[0].pointer,64);
assert.equal(data.preparation[1].pointer,4096);
assert.ok(le(data.preparation[1].groundError.exact,data.preparation[0].groundError.exact));
assert.equal(Object.keys(data.evidence).length,7);
assert.equal(data.charged.rankPerCharge,8);
assert.equal(data.charged.speciesCount,null);
assert.equal(data.charged.continuousSpinProved,false);
assert.equal(data.charged.restMassProved,false);
assert.equal(data.charged.records,(2n**53n).toString());
assert.ok(le('0.56935668',data.charged.acceptance.exact));
assert.ok(le('0.99991834',data.charged.jointFidelity.exact));
for(const record of Object.values(data.evidence)){
  assert.match(record.sha256,/^[a-f0-9]{64}$/);assert.match(record.researchSha256,/^[a-f0-9]{64}$/);
  assert.ok(record.path.startsWith('model/field_record_addenda/'));
}
console.log(`PASS: ${intervals} outward exact bounds; complete finite records, source units and physical scope`);
