// Portable checks: the reference projection never imports the external research workspace.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { MASS_REFERENCES, REFERENCE_CATALOG, REFERENCE_SOURCES, RETAINED_OM,
  referenceForRow, projectReferenceMap, escapeReferenceText } from './particle_reference_adapter.js';

const read = (name) => readFileSync(new URL(name, import.meta.url), 'utf8');
const raw = read('./research/OM/obstruction_map.json');
const map = JSON.parse(raw), before = JSON.stringify(map);
function deepFreeze(o) { if (o && typeof o === 'object') { Object.values(o).forEach(deepFreeze); Object.freeze(o); } }
deepFreeze(map);
assert.equal(createHash('sha256').update(raw).digest('hex'), RETAINED_OM.sha256);
const projection = projectReferenceMap(map);
assert.equal(projection.sourceRows, 69);
assert.equal(projection.distinctMassReferenceQuantities, 16);
assert.deepEqual(projection.categories, { charged_fermion_component: 36,
  active_neutrino_component: 6, singlet_neutrino_candidate: 6, higgs_multiplet: 2,
  higgs_resonance: 1, gauge_field: 2, vector_boson: 3, register_composite: 9, hadron: 4 });
const by = new Map(projection.rows.map((r) => [r.id, r]));
for (let i = 0; i < map.rows.length; i++) {
  const view = projection.rows[i], original = map.rows[i];
  assert.strictEqual(view.sourceRow, original);
  assert.strictEqual(view.retainedReference, original.measured);
  assert.strictEqual(view.sourceRow.mass, original.mass);
  assert.equal(view.sourceRow.status, original.status);
  assert.equal(view.combinedStatisticalScore, null);
}
assert.equal(JSON.stringify(map), before);
assert.equal(projection.sourcePredictionsOrStatusesChanged, false);
for (const [family, key] of [['u', 't'], ['d', 'b'], ['e', 'tau']]) {
  for (const prefix of ['', 'anti_']) for (const hand of ['L', 'R']) {
    assert.strictEqual(by.get(`${prefix}${family}_${hand}_g3`).reference, MASS_REFERENCES[key]);
  }
}
assert.equal(by.get('anti_u_L_g3').basis, 'conjugate of L (R component)');
assert.equal(by.get('anti_u_R_g3').basis, 'conjugate of R (L component)');
assert.throws(() => referenceForRow({ id: 'anti_u_L_g3', handedness: 'L' }), /chirality/);
assert.throws(() => projectReferenceMap({ rows: [map.rows[0], map.rows[0]] }), /Duplicate/);
assert.equal(referenceForRow({ id: 'new_unknown' }).reference.kind, 'unmapped');
assert.equal(referenceForRow({ id: 'new_unknown' }).reference.valueGeV, null);

for (const [key, mass, error] of [['s', .0935, .0008], ['c', 1.2730, .0046],
  ['b', 4.183, .007], ['t', 172.56, .31], ['h', 125.20, .11],
  ['W', 80.3692, .0133], ['Z', 91.1880, .0020], ['pi0', .1349768, .0000005]]) {
  assert.equal(MASS_REFERENCES[key].valueGeV, mass);
  assert.equal(MASS_REFERENCES[key].errorGeV, error);
}
for (const key of ['u', 'd', 's', 'c', 'b']) {
  assert.match(MASS_REFERENCES[key].uncertainty, /90%.*not a one-sigma/);
  assert.match(MASS_REFERENCES[key].convention, /MS-bar/);
}
assert.match(MASS_REFERENCES.t.convention, /Monte Carlo.*matching uncertainty/);
assert.match(MASS_REFERENCES.Z.convention, /Breit–Wigner.*34 MeV/);
assert.match(MASS_REFERENCES.rho0.convention, /not a stable-state/);
for (const id of ['H', 'H_c']) {
  assert.equal(by.get(id).reference.kind, 'field_multiplet');
  assert.equal(by.get(id).reference.valueGeV, null);
  assert.equal(by.get(id).retainedReference.value_GeV, 125.13);
}
assert.equal(by.get('h').reference.valueGeV, 125.20);
for (const row of projection.rows.filter((r) => /^(anti_)?nu_/.test(r.id))) {
  assert.equal(row.reference.valueGeV, null);
  assert.equal(row.reference.kind, row.id.includes('_L_') ? 'no_flavour_mass' : 'extension_candidate');
}
assert.match(by.get('nu_R_g1').reference.convention, /beyond the minimal SM.*retains L’s Lorentz chirality/);
assert.equal(by.get('photon').reference.kind, 'upper_limit');
assert.equal(by.get('photon').reference.upperLimitGeV, 1e-27);
assert.equal(by.get('photon').reference.valueGeV, null);
assert.equal(by.get('gluon').reference.massTermTargetGeV, 0);
assert.equal(by.get('gluon').reference.valueGeV, null);
assert.match(by.get('gluon').reference.convention, /theoretical.*not a measured/);
assert.equal(by.get('u_L_g3').retainedReference.value_GeV, 172.1);
assert.equal(by.get('d_L_g2').retainedReference.value_GeV, .0929);
assert.equal(by.get('d_L_g3').retainedReference.value_GeV, 4.186);
assert.deepEqual(JSON.parse(read('./particle_reference_catalog.json')), REFERENCE_CATALOG);
for (const source of Object.values(REFERENCE_SOURCES)) {
  assert.match(source.url, /^https:\/\/pdg\.lbl\.gov\/2025\//);
  assert.match(source.sha256, /^[0-9a-f]{64}$/);
  assert.ok(source.bytes > 0);
}
assert.equal(escapeReferenceText('<script>"&'), '&lt;script&gt;&quot;&amp;');
console.log('PASS: 69 retained rows, 16 shared references, nine row categories; source bytes, model values/statuses, limits, conjugate chirality and primary-source catalogue checked.');
