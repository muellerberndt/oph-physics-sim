// Complete Gaussian detector on the finite positive-energy Dirac carrier.
// See readout_instrument.md. The converted branch is a retained apparatus
// outcome, followed by Z_record, which returns that branch to positive energy.
import { ContinuumField } from './readout_field.js';

const BELOW_ONE = 1 - Number.EPSILON / 2;
const profiles = new Map();

function uniform(u) {
  if (!Number.isFinite(u) || u < 0 || u >= 1) throw new RangeError('detector draw must lie in [0, 1)');
  return u;
}

export function sampleDiscrete(weights, u, total = null) {
  uniform(u);
  let sum = 0, last = -1;
  for (let i = 0; i < weights.length; i++) {
    if (!Number.isFinite(weights[i]) || weights[i] < 0) throw new RangeError('invalid detector weight');
    sum += weights[i]; if (weights[i] > 0) last = i;
  }
  if (!(sum > 0) || !Number.isFinite(sum)) throw new RangeError('zero or invalid detector state');
  if (total !== null && Math.abs(total - sum) > 1e-12 * sum) throw new RangeError('inconsistent detector total');
  let threshold = u * sum;
  for (let i = 0; i < weights.length; i++) {
    const w = weights[i];
    if (w > 0 && threshold < w) return { index: i, residual: Math.max(0, threshold / w), probability: w / sum };
    threshold -= w;
  }
  // A floating CDF can miss its last interval by roundoff at u near one.
  return { index: last, residual: BELOW_ONE, probability: weights[last] / sum };
}

export function gaussianProfile(n, sigma) {
  if (!Number.isInteger(n) || n < 2 || (n & (n - 1))) throw new RangeError('detector grid must be a power of two');
  if (!(sigma > 0) || !Number.isFinite(sigma)) throw new RangeError('detector width must be finite and positive');
  const key = `${n}:${sigma}`;
  if (profiles.has(key)) return profiles.get(key);
  const amplitude = new Float64Array(n), weights = new Float64Array(n);
  let axisTotal = 0;
  for (let j = 0; j < n; j++) {
    const d = Math.min(j, n - j);
    amplitude[j] = Math.exp(-0.25 * (d / sigma) ** 2);
    weights[j] = amplitude[j] ** 2; axisTotal += weights[j];
  }
  const result = { amplitude, weights, axisTotal, normalization: axisTotal ** 3 };
  // Resolution changes do not retain an unbounded history of large arrays.
  if (profiles.size >= 16) profiles.delete(profiles.keys().next().value);
  profiles.set(key, result); return result;
}

function spectralNorm(sr, si) {
  let result = 0;
  for (let k = 0; k < 4; k++) for (let i = 0; i < sr[k].length; i++) result += sr[k][i] ** 2 + si[k][i] ** 2;
  return result;
}

function sourceCheck(field, P, t) {
  if (!Number.isFinite(t) || !Number.isFinite(P.m)) throw new RangeError('invalid detector time or mass');
  const v = new Float64Array(8), h = new Float64Array(8);
  let norm = 0, rejected = 0, energy = 0;
  for (let i = 0; i < field.N; i++) {
    const E = Math.sqrt(field.k2[i] + P.m * P.m);
    for (let k = 0; k < 4; k++) {
      v[2 * k] = P.sr[k][i]; v[2 * k + 1] = P.si[k][i];
      if (!Number.isFinite(v[2 * k]) || !Number.isFinite(v[2 * k + 1])) throw new RangeError('invalid source amplitude');
      const w = v[2 * k] ** 2 + v[2 * k + 1] ** 2; norm += w; energy += E * w;
    }
    if (E < 1e-12) continue; // Matches the source's retained zero-energy fiber.
    ContinuumField.applyH(field.kx[i], field.ky[i], field.kz[i], P.m, v, h);
    for (let a = 0; a < 8; a++) rejected += (0.5 * (v[a] - h[a] / E)) ** 2;
  }
  if (!(norm > 0) || !Number.isFinite(norm)) throw new RangeError('zero or invalid source norm');
  if (rejected > 1e-18 * norm) throw new RangeError('detector requires the retained positive-energy source');
  return { norm, energy: energy / norm, rejectedFraction: rejected / norm };
}

// Unnormalized Fourier images g_j psi and P+ g_j psi, with no source mutation.
export function windowBranches(field, P, center, sigma) {
  const { n, N, wr, wi } = field, profile = gaussianProfile(n, sigma);
  if (center.length !== 3 || center.some(x => !Number.isInteger(x) || x < 0 || x >= n)) throw new RangeError('invalid detector center');
  const sr = Array.from({ length: 4 }, () => new Float64Array(N));
  const si = Array.from({ length: 4 }, () => new Float64Array(N));
  let windowNorm = 0, priorNorm = 0;
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = (z * n + y) * n + x;
    const g = profile.amplitude[(x - center[0] + n) % n]
      * profile.amplitude[(y - center[1] + n) % n] * profile.amplitude[(z - center[2] + n) % n];
    for (let k = 0; k < 4; k++) {
      const r = wr[k][i], im = wi[k][i]; priorNorm += r * r + im * im;
      sr[k][i] = g * r; si[k][i] = g * im; windowNorm += sr[k][i] ** 2 + si[k][i] ** 2;
    }
  }
  for (let k = 0; k < 4; k++) field.fft(sr[k], si[k], -1);
  const pr = sr.map(a => a.slice()), pi = si.map(a => a.slice());
  field.projectPlus(pr, pi, P.m);
  const nr = sr.map((a, k) => a.map((v, i) => v - pr[k][i]));
  const ni = si.map((a, k) => a.map((v, i) => v - pi[k][i]));
  const positiveNorm = spectralNorm(pr, pi) / N, negativeNorm = spectralNorm(nr, ni) / N;
  if (!(windowNorm > 0) || !(priorNorm > 0)) throw new RangeError('zero-probability detector center');
  const completenessError = Math.abs(positiveNorm + negativeNorm - windowNorm) / windowNorm;
  if (completenessError > 1e-10) throw new Error('detector band decomposition failed its norm check');
  return { pr, pi, nr, ni, positiveNorm, negativeNorm, windowNorm, priorNorm,
    positionProbability: windowNorm / (profile.normalization * priorNorm), completenessError };
}

// `u` supplies the first draw. With random(), later draws are independent calls
// to that same generator. Without it, conditional CDF residuals decode one
// ideal uniform into the sequential draws; the original u is never reused.
export function recordPositiveEnergy(field, P, t, sigma, u, { random = null } = {}) {
  uniform(u); const profile = gaussianProfile(field.n, sigma);
  if (random !== null && typeof random !== 'function') throw new TypeError('random must be a function');
  const prior = sourceCheck(field, P, t);
  field.evolve(P, t);
  const density = new Float64Array(field.N);
  for (let i = 0; i < field.N; i++) for (let k = 0; k < 4; k++) density[i] += field.wr[k][i] ** 2 + field.wi[k][i] ** 2;
  let draw = sampleDiscrete(density, u), residual = draw.residual;
  const j = draw.index, n = field.n;
  const center = [j % n, Math.floor(j / n) % n, Math.floor(j / (n * n))];
  const next = () => uniform(random ? random() : Math.min(BELOW_ONE, residual));
  for (let axis = 0; axis < 3; axis++) {
    draw = sampleDiscrete(profile.weights, next()); residual = draw.residual;
    center[axis] = (center[axis] + draw.index) % n;
  }
  // The latent density point is only a classical sampler variable. It is
  // neither an apparatus record nor a quantum projection of the source.
  const branches = windowBranches(field, P, center, sigma);
  draw = sampleDiscrete([branches.positiveNorm, branches.negativeNorm], next());
  const converted = draw.index === 1;
  const sr = converted ? branches.nr : branches.pr, si = converted ? branches.ni : branches.pi;
  const selectedNorm = converted ? branches.negativeNorm : branches.positiveNorm;
  const scale = 1 / Math.sqrt(selectedNorm); let energyAfter = 0;
  for (let k = 0; k < 4; k++) for (let i = 0; i < field.N; i++) {
    // B=Z_record is +1 on components 0,1 and -1 on 2,3.
    const factor = scale * (converted && k >= 2 ? -1 : 1);
    sr[k][i] *= factor; si[k][i] *= factor;
    energyAfter += P.E[i] * (sr[k][i] ** 2 + si[k][i] ** 2) / field.N;
  }
  for (let k = 0; k < 4; k++) { P.sr[k].set(sr[k]); P.si[k].set(si[k]); }
  P.tb = t; P.ref = 0;
  const positionProbability = branches.positionProbability, bandProbability = draw.probability;
  return { x: center, weight: positionProbability, positionProbability, bandProbability,
    probability: positionProbability * bandProbability,
    band: converted ? 'converted-negative' : 'positive', convertedNegativeBand: converted,
    positiveBandProbability: branches.positiveNorm / (branches.positiveNorm + branches.negativeNorm),
    negativeBandProbability: branches.negativeNorm / (branches.positiveNorm + branches.negativeNorm),
    sourceEnergyBefore: prior.energy, sourceEnergyAfter: energyAfter,
    sourceEnergyChange: energyAfter - prior.energy, conversionWork: converted ? 2 * energyAfter : 0,
    completenessError: branches.completenessError, instrument: 'complete-gaussian-positive-energy' };
}

// Drop-in worker import: import { BornReadoutField as ContinuumField } from
// './readout_instrument.js'. All extra record fields survive the inherited frame.
export class BornReadoutField extends ContinuumField {
  record(P, t, sigma, u) {
    const draws = [u], generator = this._detectorRandom;
    const random = generator ? () => { const value = generator(); draws.push(value); return value; } : null;
    const result = recordPositiveEnergy(this, P, t, sigma, u, { random });
    return { ...result, samplerDraws: draws, detectorWidth: sigma,
      samplerMode: generator ? 'five recorded uniform variates' : 'one uniform with conditional CDF residuals' };
  }
  frame(t, recordIds = new Set(), sigma = 1, random = Math.random) {
    const previous = this._detectorRandom; this._detectorRandom = random;
    try { return super.frame(t, recordIds, sigma, random); }
    finally { this._detectorRandom = previous; }
  }
}
