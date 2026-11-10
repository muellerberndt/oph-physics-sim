import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { realClockAmplitude, describeRegisterEvent } from './story_space_evidence.js';
import { Routing, phases } from './cf1_engine.js';
import { COSMOLOGY_LEDGER, TILT_COMPARISONS } from './cosmos_evidence.js';

// Compare the real-amplitude instrument with an independent finite controlled-
// translation Fourier sum, including the removable singularities and the seam.
let probabilityError = 0, completenessError = 0;
for (const M of [8, 32, 256]) for (const p of [-Math.PI, 0, Math.PI, .173, -.731, 2 * Math.PI / M]) {
  let total = 0;
  for (let j = 0; j < M; j++) {
    const g = realClockAmplitude(p, j, M);
    let re = 0, im = 0;
    for (let n = 0; n < M; n++) {
      const phase = n * (p - 2 * Math.PI * j / M);
      re += Math.cos(phase) / M; im += Math.sin(phase) / M;
    }
    probabilityError = Math.max(probabilityError, Math.abs(g * g - re * re - im * im));
    total += g * g;
  }
  completenessError = Math.max(completenessError, Math.abs(total - 1));
}
assert.ok(probabilityError < 4e-14);
assert.ok(completenessError < 4e-14);
assert.throws(() => realClockAmplitude(0, 3, 3));
assert.equal(describeRegisterEvent({kind:'birth',before:[0,0],after:[1,5]}),
  'birth: face classes blank faces → 1 + 5');

// The displayed rest phase is coherent. Its eigenstate has record occupation
// one half, whereas sin²(theta) is the flip probability for a blank record.
const theta = .2, q = new Routing(1, 1, 1), blank = new Routing(1, 1, 1);
q.addPlaneWave([0,0,0], [[Math.SQRT1_2,0],[0,0],[0,Math.SQRT1_2],[0,0]]); q.normalize();
blank.addPlaneWave([0,0,0], [[1,0],[0,0],[0,0],[0,0]]); blank.normalize();
blank.tickAltHalf(theta / 2);
assert.ok(Math.abs(blank.recordProbability() - Math.sin(theta) ** 2) < 1e-14);
for (let n = 1; n <= 16; n++) {
  q.tickAltHalf(theta / 2);
  assert.ok(Math.abs(q.recordProbability() - .5) < 1e-14);
  assert.ok(Math.abs(q.re[0] - Math.SQRT1_2 * Math.cos(n * theta)) < 1e-14);
  assert.ok(Math.abs(q.im[0] + Math.SQRT1_2 * Math.sin(n * theta)) < 1e-14);
}
const errors = [.2, .1, .05].map(a => Math.abs(phases([a * .7,0,0],a * 2,'Z')[0] / a - Math.hypot(.7,2)));
for (let j=1;j<errors.length;j++) assert.ok(errors[j-1]/errors[j]>3.8 && errors[j-1]/errors[j]<4.2);

// Every retained comparable tilt dataset is displayed, not only Planck.
assert.equal(TILT_COMPARISONS.length, 6);
for (const r of TILT_COMPARISONS)
  assert.ok(Math.abs((r.oph_value-r.measurement.value)/r.measurement.sigma-r.sigma_distance)<1e-13);
assert.equal(TILT_COMPARISONS.filter(r=>r.verdict==='tension').length,1);
const source = new URL('../references/cosmology_postdiction_ledger.json', import.meta.url);
{
  const raw = readFileSync(source), ledger = JSON.parse(raw);
  assert.equal(createHash('sha256').update(raw).digest('hex'),COSMOLOGY_LEDGER.sha256);
  const rows = ledger.sections.flatMap(s=>s.rows).filter(r=>r.row_id.startsWith('ns_edge_center_P_C_'));
  assert.deepEqual(TILT_COMPARISONS.map(r=>r.row_id),rows.map(r=>r.row_id));
  for (const r of TILT_COMPARISONS) {
    const original=rows.find(v=>v.row_id===r.row_id);
    for (const key of ['measurement','oph_value','sigma_distance','verdict','citation','dataset'])
      assert.deepEqual(r[key],original[key]);
  }
}
console.log('PASS: complete finite-clock effects, declared phase correction, coherent rest phase, joint axial scaling, all six pinned tilt comparisons');
console.log(JSON.stringify({probabilityError,completenessError,axialDispersionErrors:errors}));
