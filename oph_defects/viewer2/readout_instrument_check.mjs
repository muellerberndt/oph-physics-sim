// node oph_defects/viewer2/readout_instrument_check.mjs
import assert from 'node:assert/strict';
import { ContinuumField } from './readout_field.js';
import { BornReadoutField, gaussianProfile, sampleDiscrete, windowBranches, recordPositiveEnergy } from './readout_instrument.js';

let checks = 0;
function check(name, body) { body(); checks++; console.log(`pass  ${name}`); }
function rng(seed = 131) { return () => ((seed = (seed * 16807) % 2147483647) / 2147483647); }
function particle(F, m = 0.7) {
  F.add({ id: 1, m, x0: [0.3, 0.7, 1.1], p0: [0.3, 0.2, -0.1], sigma: 0.6,
    chi: [[0.5, 0], [0.5, 0], [-0.5, 0], [-0.5, 0]], t: 0 });
  return F.parts.get(1);
}
function vector(sr, si, scale = 1) {
  const result = [];
  for (let k = 0; k < 4; k++) for (let i = 0; i < sr[k].length; i++) result.push([sr[k][i] * scale, si[k][i] * scale]);
  return result;
}
function dot(a, b) {
  let re = 0, im = 0;
  for (let i = 0; i < a.length; i++) { re += a[i][0] * b[i][0] + a[i][1] * b[i][1]; im += a[i][0] * b[i][1] - a[i][1] * b[i][0]; }
  return [re, im];
}
function plusError(F, P) {
  const pr = P.sr.map(a => a.slice()), pi = P.si.map(a => a.slice()); F.projectPlus(pr, pi, P.m);
  let error = 0;
  for (let k = 0; k < 4; k++) for (let i = 0; i < F.N; i++) error += (pr[k][i] - P.sr[k][i]) ** 2 + (pi[k][i] - P.si[k][i]) ** 2;
  return error / F.N;
}

check('Gaussian translated windows form a partition of identity', () => {
  for (const n of [2, 4, 8]) for (const sigma of [0.001, 0.8, 2, 1e100]) {
    const p = gaussianProfile(n, sigma);
    for (let x = 0; x < n; x++) {
      let sum = 0; for (let j = 0; j < n; j++) sum += p.amplitude[(x - j + n) % n] ** 2;
      assert.ok(Math.abs(sum ** 3 / p.normalization - 1) < 2e-14);
    }
  }
});

check('CDF residual decoder supplies independent conditional choices and skips zero weights', () => {
  const counts = [[0, 0], [0, 0]];
  for (let i = 0; i < 1600; i++) {
    const a = sampleDiscrete([1, 3], (i + 0.5) / 1600), b = sampleDiscrete([2, 3], a.residual);
    counts[a.index][b.index]++;
  }
  assert.deepEqual(counts, [[160, 240], [480, 720]]);
  assert.equal(sampleDiscrete([0, 0, 2, 0], 0).index, 2);
  assert.throws(() => sampleDiscrete([0, 0], 0.5));
  assert.throws(() => sampleDiscrete([1, -1], 0.5));
  assert.throws(() => sampleDiscrete([1, 1], 1));
});

check('all center and band Kraus effects sum to the complete retained projector', () => {
  for (const mass of [0, 0.7]) {
    const F = new ContinuumField(2), N = F.N, dimension = 4 * N, sigma = 0.65;
    const P = particle(F, mass), inputs = [], kraus = Array.from({ length: 2 * N }, () => []);
    const C = gaussianProfile(F.n, sigma).normalization;
    for (let column = 0; column < dimension; column++) {
      for (let k = 0; k < 4; k++) { P.sr[k].fill(0); P.si[k].fill(0); }
      P.sr[Math.floor(column / N)][column % N] = Math.sqrt(N); F.projectPlus(P.sr, P.si, mass);
      inputs.push(vector(P.sr, P.si, 1 / Math.sqrt(N))); F.evolve(P, 0);
      for (let j = 0; j < N; j++) {
        const center = [j % 2, Math.floor(j / 2) % 2, Math.floor(j / 4)];
        const b = windowBranches(F, P, center, sigma);
        kraus[2 * j].push(vector(b.pr, b.pi, 1 / Math.sqrt(N * C)));
        for (let k = 2; k < 4; k++) for (let i = 0; i < N; i++) { b.nr[k][i] *= -1; b.ni[k][i] *= -1; }
        kraus[2 * j + 1].push(vector(b.nr, b.ni, 1 / Math.sqrt(N * C)));
        assert.ok(plusError(F, { ...P, sr: b.nr, si: b.ni }) < 1e-25);
      }
    }
    let error = 0;
    for (let i = 0; i < dimension; i++) for (let j = 0; j < dimension; j++) {
      const target = dot(inputs[i], inputs[j]); let real = 0, imag = 0;
      for (const columns of kraus) { const d = dot(columns[i], columns[j]); real += d[0]; imag += d[1]; }
      error = Math.max(error, Math.hypot(real - target[0], imag - target[1]));
    }
    assert.ok(error < 1e-12, `complete operator residual ${error}`);
  }
});

check('Gaussian center probabilities are normalized and differ from the prior point density', () => {
  const F = new ContinuumField(4), P = particle(F); F.evolve(P, 0); F.readouts();
  const rho = F.rho.slice(); let total = 0, difference = 0, converted = 0;
  for (let j = 0; j < F.N; j++) {
    const b = windowBranches(F, P, [j % 4, Math.floor(j / 4) % 4, Math.floor(j / 16)], 0.9);
    total += b.positionProbability; difference += Math.abs(b.positionProbability - rho[j]);
    converted += b.negativeNorm / gaussianProfile(4, 0.9).normalization;
  }
  assert.ok(Math.abs(total - 1) < 1e-12);
  assert.ok(difference > 0.2, `old sampling disagreement ${difference}`);
  assert.ok(converted > 0.01, `discarded negative-band probability ${converted}`);
});

check('both retained outcomes return normalized positive-energy states with original joint probabilities', () => {
  for (const bandDraw of [0, 1 - Number.EPSILON]) {
    const F = new BornReadoutField(8), P = particle(F), draws = [0.4, 0, 0, 0, bandDraw];
    let calls = 0; const result = F.frame(0.3, new Set([1]), 0.5, () => draws[calls++]);
    assert.equal(calls, 5); const record = result.records[0];
    assert.equal(record.convertedNegativeBand, bandDraw > 0);
    assert.equal(record.band, bandDraw > 0 ? 'converted-negative' : 'positive');
    assert.ok(record.probability > 0 && record.probability <= record.positionProbability);
    assert.equal(record.probability, record.positionProbability * record.bandProbability);
    assert.ok(Math.abs(record.positiveBandProbability + record.negativeBandProbability - 1) < 1e-14);
    assert.ok(plusError(F, P) < 1e-25);
    F.evolve(P, 0.3); assert.ok(Math.abs(F.readouts().norm - 1) < 1e-12);
    if (bandDraw > 0) assert.ok(record.conversionWork > 0 && record.conversionWork === 2 * record.sourceEnergyAfter);
    else assert.equal(record.conversionWork, 0);
    F.evolve(P, 4); assert.ok(Math.abs(F.readouts().norm - 1) < 1e-12);
  }
});

check('seeded frames reproduce every detector field and postmeasurement amplitude', () => {
  const A = new BornReadoutField(4), B = new BornReadoutField(4); particle(A); particle(B);
  const ra = A.frame(0.2, new Set([1]), 0.8, rng(19)), rb = B.frame(0.2, new Set([1]), 0.8, rng(19));
  assert.deepEqual(ra.records, rb.records); assert.deepEqual(A.parts.get(1).sr, B.parts.get(1).sr);
  assert.deepEqual(A.parts.get(1).si, B.parts.get(1).si);
});

check('seeded complete center/band frequencies follow the original Born probabilities', () => {
  const F = new BornReadoutField(2), P = particle(F), sigma = 0.5, random = rng(61);
  const originalR = P.sr.map(a => a.slice()), originalI = P.si.map(a => a.slice());
  F.evolve(P, 0); const expected = new Float64Array(2 * F.N), observed = new Uint32Array(2 * F.N);
  for (let j = 0; j < F.N; j++) {
    const b = windowBranches(F, P, [j % 2, Math.floor(j / 2) % 2, Math.floor(j / 4)], sigma);
    const C = gaussianProfile(2, sigma).normalization;
    expected[2 * j] = b.positiveNorm / (C * b.priorNorm); expected[2 * j + 1] = b.negativeNorm / (C * b.priorNorm);
  }
  const trials = 6000;
  for (let trial = 0; trial < trials; trial++) {
    for (let k = 0; k < 4; k++) { P.sr[k].set(originalR[k]); P.si[k].set(originalI[k]); }
    const r = recordPositiveEnergy(F, P, 0, sigma, random(), { random });
    const index = 2 * ((r.x[2] * 2 + r.x[1]) * 2 + r.x[0]) + Number(r.convertedNegativeBand);
    observed[index]++; assert.ok(Math.abs(r.probability - expected[index]) < 1e-12);
  }
  for (let i = 0; i < observed.length; i++) {
    const mean = trials * expected[i], sd = Math.sqrt(trials * expected[i] * (1 - expected[i]));
    assert.ok(Math.abs(observed[i] - mean) < 6 * sd + 1, `outcome ${i}: ${observed[i]} versus ${mean}`);
  }
});

check('invalid draws, widths and nonpositive source inputs fail before source mutation', () => {
  const F = new BornReadoutField(4), P = particle(F), copy = P.sr.map(a => a.slice());
  assert.throws(() => F.record(P, 0, 0, 0.5)); assert.throws(() => F.record(P, 0, 1, NaN));
  assert.deepEqual(P.sr, copy);
  for (let k = 2; k < 4; k++) for (let i = 0; i < F.N; i++) { P.sr[k][i] *= -1; P.si[k][i] *= -1; }
  assert.throws(() => recordPositiveEnergy(F, P, 0, 0.8, 0.4), /positive-energy/);
});

console.log(`${checks} complete instrument checks pass`);
