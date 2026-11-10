// The committed integer rule of the big runs with its declared register, on one twelve-port carrier.
// Rule (runs/universe/L10/DATASET.md, oph_defects/kernels.py): an attempt on seam (a, b) with loads x_a, x_b and a fair
// coin c sets x_a to ceil(t/2) if c = 1 else floor(t/2), t = x_a + x_b; the current is j = old x_a - new x_a. A repair with
// j != 0 proposes the twist delta = j mod 6 of the seam's Z6 link: the face that traverses the seam a -> b changes its
// class by +delta, the face that traverses b -> a by -delta. The twist is applied when the cost change of the two faces
// is at most kappa j (d - j), d = old x_a - old x_b. Cost table [0, 5, 8, 10, 8, 5], kappa = 5 (the runs' values).
import { PV, EDGES, FACES } from './walk_engine.js';

export const COST = [0, 5, 8, 10, 8, 5], KAPPA = 5;
const sub = (u, v) => [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
export const OFACES = FACES.map(([a, b, c]) => { const n = cross(sub(PV[b], PV[a]), sub(PV[c], PV[a])); const s = [PV[a][0] + PV[b][0] + PV[c][0], PV[a][1] + PV[b][1] + PV[c][1], PV[a][2] + PV[b][2] + PV[c][2]];
  return n[0] * s[0] + n[1] * s[1] + n[2] * s[2] > 0 ? [a, b, c] : [a, c, b]; });
export const SEAMS = EDGES.map(([a, b]) => {
  let fp = -1, fm = -1;
  OFACES.forEach((f, i) => { for (let k = 0; k < 3; k++) { if (f[k] === a && f[(k + 1) % 3] === b) fp = i; if (f[k] === b && f[(k + 1) % 3] === a) fm = i; } });
  return { a, b, fp, fm };
});
export const NAMES = { 1: 'left-handed quark', 5: 'left-handed antiquark', 4: 'right-handed quark', 2: 'right-handed antiquark', 3: 'left-handed lepton' };
export function mulberry(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

export class Carrier {
  constructor(seed = 1, maxLoad = 5) {
    this.rnd = mulberry(seed); this.x = new Int32Array(12); this.cls = new Int32Array(20); this.sweep = 0;
    for (let p = 0; p < 12; p++) this.x[p] = Math.floor(this.rnd() * (maxLoad + 1));
    this.counts = { birth: 0, move: 0, annihilation: 0, fusion: 0, split: 0, exchange: 0 };
  }
  attempt() {
    const s = Math.floor(this.rnd() * SEAMS.length), coin = this.rnd() < 0.5 ? 1 : 0, { a, b, fp, fm } = SEAMS[s];
    const xa = this.x[a], xb = this.x[b], t = xa + xb, ni = coin ? Math.ceil(t / 2) : Math.floor(t / 2), cur = xa - ni, d = xa - xb;
    if (cur === 0) return null;
    this.x[a] = ni; this.x[b] = t - ni;
    const delta = ((cur % 6) + 6) % 6; if (delta === 0) return null;
    const k1 = this.cls[fp], k2 = this.cls[fm], n1 = (k1 + delta) % 6, n2 = (k2 - delta + 6) % 6;
    const de = COST[n1] + COST[n2] - COST[k1] - COST[k2], work = KAPPA * cur * (d - cur);
    if (de > work) return null;
    this.cls[fp] = n1; this.cls[fm] = n2;
    const nb = (k1 !== 0) + (k2 !== 0), na = (n1 !== 0) + (n2 !== 0);
    const kind = nb === 0 ? 'birth' : na === 0 ? 'annihilation' : nb === 1 && na === 1 ? 'move' : nb === 2 && na === 1 ? 'fusion' : nb === 1 && na === 2 ? 'split' : 'exchange';
    this.counts[kind]++;
    return { kind, seam: s, faces: [fp, fm], before: [k1, k2], after: [n1, n2], sweep: this.sweep };
  }
  runSweep() { const ev = []; for (let i = 0; i < SEAMS.length; i++) { const e = this.attempt(); if (e) ev.push(e); } this.sweep++; return ev; }
  defects() { let n = 0; for (const k of this.cls) if (k) n++; return n; }
  mobile() { let n = 0; for (const k of this.cls) if (k === 1 || k === 5) n++; return n; }
  balanced() { let lo = 99, hi = -1; for (const v of this.x) { lo = Math.min(lo, v); hi = Math.max(hi, v); } return hi - lo <= 1; }
}
export function describe(e) {
  const [k1, k2] = e.before, [n1, n2] = e.after, nm = (k) => NAMES[k];
  if (e.kind === 'birth') { const k = n1 === 3 ? 3 : (n1 === 1 || n1 === 4 ? n1 : n2); return k === 3 ? 'a left-handed lepton and its antilepton are born' : `a ${nm(k)} and its antiparticle are born`; }
  if (e.kind === 'annihilation') { const k = k1 === 3 ? 3 : (k1 === 1 || k1 === 4 ? k1 : k2); return k === 3 ? 'a lepton and its antilepton annihilate' : `a ${nm(k)} meets its antiparticle and both vanish`; }
  if (e.kind === 'move') { const k = k1 || k2; return `a ${nm(k)} moves to the next face`; }
  if (e.kind === 'fusion') return `a ${nm(k1)} and a ${nm(k2)} fuse into a ${nm(n1 || n2)}`;
  if (e.kind === 'split') return `a ${nm(k1 || k2)} splits into a ${nm(n1)} and a ${nm(n2)}`;
  return `a ${nm(k1)} and a ${nm(k2)} trade charge and become a ${nm(n1)} and a ${nm(n2)}`;
}
