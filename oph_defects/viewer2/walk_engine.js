// Engine of the Carrier tab: the public coherent programs of the theory repository on one twelve-port carrier,
// re-implemented from their proofs and checked against their published exact numbers (see carrier_check.mjs).
//   coherent-seam-repair-v1   (model/coherent_seam_repair/PROOF.md): reversible seam repair with an owned parity
//                             record, reflection of the sixty seam addresses, optional five-level fibre with link
//                             signs Z = diag(1,1,1,-1,-1)
//   coherent_defect_wave      (model/coherent_defect_wave/PROOF.md): the walk conditional on a thirty-bit link
//                             register, and wave-controlled moves of a regional flux defect
// No DOM here.
export const PHI = (1 + Math.sqrt(5)) / 2;
export const PV = [[-1, PHI, 0], [1, PHI, 0], [-1, -PHI, 0], [1, -PHI, 0], [0, -1, PHI], [0, 1, PHI],
  [0, -1, -PHI], [0, 1, -PHI], [PHI, 0, -1], [PHI, 0, 1], [-PHI, 0, -1], [-PHI, 0, 1]];
const d2 = (a, b) => (PV[a][0] - PV[b][0]) ** 2 + (PV[a][1] - PV[b][1]) ** 2 + (PV[a][2] - PV[b][2]) ** 2;
export const EDGES = [];
for (let a = 0; a < 12; a++) for (let b = a + 1; b < 12; b++) if (Math.abs(d2(a, b) - 4) < 1e-9) EDGES.push([a, b]);
const EIDX = new Map(EDGES.map(([a, b], i) => [a * 12 + b, i]));
export const edgeIndex = (a, b) => EIDX.get(Math.min(a, b) * 12 + Math.max(a, b));
export const FACES = [];
for (let a = 0; a < 12; a++) for (let b = a + 1; b < 12; b++) for (let c = b + 1; c < 12; c++)
  if (edgeIndex(a, b) !== undefined && edgeIndex(b, c) !== undefined && edgeIndex(a, c) !== undefined) FACES.push([a, b, c]);
export const FACE_EDGES = FACES.map(([a, b, c]) => [edgeIndex(a, b), edgeIndex(b, c), edgeIndex(a, c)]);
export const ADDR = EDGES.map((e) => e.slice()).concat(EDGES.map(([a, b]) => [b, a]));   // sixty oriented seams
export const addrIndex = (a, b) => ADDR.findIndex((k) => k[0] === a && k[1] === b);
export const NEIGH = Array.from({ length: 12 }, (_, p) => EDGES.filter((e) => e.includes(p)).map((e) => (e[0] === p ? e[1] : e[0])));
export const bit = (word, e) => (word >>> e) & 1;
export const faceFlux = (word) => FACE_EDGES.map((es) => bit(word, es[0]) ^ bit(word, es[1]) ^ bit(word, es[2]));

export function regionOf(port) {
  const faces = FACES.map((f, i) => (f.includes(port) ? i : -1)).filter((i) => i >= 0);
  const interior = EDGES.map((e, i) => (e.includes(port) ? i : -1)).filter((i) => i >= 0);
  const boundary = [...new Set(faces.flatMap((f) => FACE_EDGES[f]))].filter((e) => !interior.includes(e));
  return { port, faces, interior, boundary };
}
// the permitted move of the link program: flip an interior edge if exactly one of its two faces has flux
export function transport(word, e) {
  const fs = FACE_EDGES.map((es, i) => (es.includes(e) ? i : -1)).filter((i) => i >= 0);
  const fl = faceFlux(word);
  return (fl[fs[0]] ^ fl[fs[1]]) ? (word ^ (1 << e)) >>> 0 : word;
}
export function reachable(word0, interior) {
  const seen = [word0];
  for (let i = 0; i < seen.length; i++) for (const e of interior) { const w = transport(seen[i], e); if (!seen.includes(w)) seen.push(w); }
  return seen.sort((a, b) => a - b);
}

export class Walk {
  // words: link words carried by the register (one word = a static background). fibre: 1 or 5 internal levels.
  constructor({ fibre = 1, words = [0], interior = [] } = {}) {
    this.F = fibre; this.words = words.slice(); this.NG = words.length; this.interior = interior.slice();
    this.n = this.NG * 12 * fibre * 2 * 60;
    this.re = new Float64Array(this.n); this.im = new Float64Array(this.n);
    this.ticks = 0;
    this.T = interior.map((e) => this.words.map((w) => this.words.indexOf(transport(w, e))));
    if (this.T.some((row) => row.includes(-1))) throw new Error('link sector is not closed under the permitted moves');
  }
  idx(g, p, j, r, k) { return ((((g * 12 + p) * this.F + j) * 2 + r) * 60) + k; }
  clear() { this.re.fill(0); this.im.fill(0); this.ticks = 0; }
  // add amp * A|p, j> on link word g: blank parity record, uniform over the sixty addresses
  embed(g, p, j, ar = 1, ai = 0) {
    const s = 1 / Math.sqrt(60), o = this.idx(g, p, j, 0, 0);
    for (let k = 0; k < 60; k++) { this.re[o + k] += ar * s; this.im[o + k] += ai * s; }
  }
  normalize() { let t = 0; for (let i = 0; i < this.n; i++) t += this.re[i] ** 2 + this.im[i] ** 2; t = 1 / Math.sqrt(t); for (let i = 0; i < this.n; i++) { this.re[i] *= t; this.im[i] *= t; } }
  reflect() {                                            // R = 2 A A* - 1 on the address register
    for (let o = 0; o < this.n; o += 60) {
      let sr = 0, si = 0;
      for (let k = 0; k < 60; k++) { sr += this.re[o + k]; si += this.im[o + k]; }
      sr /= 30; si /= 30;
      for (let k = 0; k < 60; k++) { this.re[o + k] = sr - this.re[o + k]; this.im[o + k] = si - this.im[o + k]; }
    }
  }
  seam() {                                               // Q: the reversible repair on the addressed seam
    const h = Math.SQRT1_2, F = this.F;
    for (let g = 0; g < this.NG; g++) {
      const w = this.words[g];
      for (let k = 0; k < 60; k++) {
        const a = ADDR[k][0], b = ADDR[k][1], neg = bit(w, edgeIndex(a, b));
        for (let j = 0; j < F; j++) {
          const s = (F === 5 && j >= 3 && neg) ? -1 : 1;
          const a0 = this.idx(g, a, j, 0, k), a1 = a0 + 60, b0 = this.idx(g, b, j, 0, k), b1 = b0 + 60;
          const cr = ((this.re[a0] - s * this.re[b0]) - (this.re[a1] + s * this.re[b1])) * h;
          const ci = ((this.im[a0] - s * this.im[b0]) - (this.im[a1] + s * this.im[b1])) * h;
          this.re[a0] -= cr * h; this.im[a0] -= ci * h; this.re[b0] += s * cr * h; this.im[b0] += s * ci * h;
          this.re[a1] += cr * h; this.im[a1] += ci * h; this.re[b1] += s * cr * h; this.im[b1] += s * ci * h;
        }
      }
    }
  }
  step() { this.reflect(); this.seam(); }                // W = Q R
  unstep() { this.seam(); this.reflect(); }              // W* = R Q
  moves() {                                              // C_0 first: a wave at an endpoint of an interior edge moves the defect
    const block = this.F * 2 * 60;
    this.interior.forEach((e, i) => {
      const T = this.T[i];
      for (const p of EDGES[e]) for (let g = 0; g < this.NG; g++) {
        const g2 = T[g]; if (g2 <= g) continue;
        const o1 = this.idx(g, p, 0, 0, 0), o2 = this.idx(g2, p, 0, 0, 0);
        for (let q = 0; q < block; q++) { let t = this.re[o1 + q]; this.re[o1 + q] = this.re[o2 + q]; this.re[o2 + q] = t; t = this.im[o1 + q]; this.im[o1 + q] = this.im[o2 + q]; this.im[o2 + q] = t; }
      }
    });
  }
  unmoves() {                                            // inverse of moves(): the same involutions in reverse order
    const keep = this.interior, keepT = this.T;
    this.interior = keep.slice().reverse(); this.T = keepT.slice().reverse();
    this.moves();
    this.interior = keep; this.T = keepT;
  }
  tick() { this.step(); if (this.interior.length) this.moves(); this.ticks++; }
  clone() { const w = new Walk({ fibre: this.F, words: this.words, interior: this.interior }); w.re.set(this.re); w.im.set(this.im); w.ticks = this.ticks; return w; }
  norm2() { let t = 0; for (let i = 0; i < this.n; i++) t += this.re[i] ** 2 + this.im[i] ** 2; return t; }
  // interior Gauss operator of the public joint program: flip the links at the centre port and change the sign of
  // the spinor levels at that port. Returns the projection (1 + sign * Gamma)/2 of the state, unnormalized.
  gaussProject(centre, sign = 1) {
    const mask = this.interior.reduce((m, e) => m | (1 << e), 0) >>> 0, out = this.clone(), block = this.F * 2 * 60, lv = 120;
    for (let g = 0; g < this.NG; g++) {
      const g2 = this.words.indexOf((this.words[g] ^ mask) >>> 0);
      if (g2 < 0) throw new Error('link sector is not closed under the gauge move');
      for (let p = 0; p < 12; p++) for (let j = 0; j < this.F; j++) {
        const d = (p === centre && j >= 3) ? -1 : 1, o = this.idx(g, p, j, 0, 0), o2 = this.idx(g2, p, j, 0, 0);
        for (let q = 0; q < lv; q++) { out.re[o + q] = 0.5 * (this.re[o + q] + sign * d * this.re[o2 + q]); out.im[o + q] = 0.5 * (this.im[o + q] + sign * d * this.im[o2 + q]); }
      }
    }
    return out;
  }
  // complete readout of the face that carries the flux inside a region: one branch per face (unnormalized states)
  faceBranches(faces) {
    return faces.map((f) => {
      const b = this.clone();
      for (let g = 0; g < this.NG; g++) if (!faceFlux(this.words[g])[f]) { const o = this.idx(g, 0, 0, 0, 0), len = 12 * this.F * 120; b.re.fill(0, o, o + len); b.im.fill(0, o, o + len); }
      return { face: f, p: b.norm2(), state: b };
    });
  }
  // return instrument of the public program: on the smaller word of each gauge pair project the wave on psi, on the
  // other word on D psi. psi is given as a single-word Walk holding the wave vector. Returns the yes probability.
  returnProbability(psi, centre) {
    const mask = this.interior.reduce((m, e) => m | (1 << e), 0) >>> 0, len = 12 * this.F * 120;
    let total = 0;
    for (let g = 0; g < this.NG; g++) {
      const canonical = this.words[g] < ((this.words[g] ^ mask) >>> 0), o = this.idx(g, 0, 0, 0, 0);
      let r = 0, i = 0;
      for (let q = 0; q < len; q++) {
        const p = Math.floor(q / (this.F * 120)), j = Math.floor(q / 120) % this.F, d = (!canonical && p === centre && j >= 3) ? -1 : 1;
        r += d * (psi.re[q] * this.re[o + q] + psi.im[q] * this.im[o + q]); i += d * (psi.re[q] * this.im[o + q] - psi.im[q] * this.re[o + q]);
      }
      total += r * r + i * i;
    }
    return total;
  }
  // readouts
  populations() {
    const port = new Float64Array(12), type = new Float64Array(2), link = new Float64Array(this.NG), F = this.F, pt = new Float64Array(24);
    for (let g = 0; g < this.NG; g++) for (let p = 0; p < 12; p++) for (let j = 0; j < F; j++) {
      const o = this.idx(g, p, j, 0, 0); let t = 0;
      for (let q = 0; q < 120; q++) t += this.re[o + q] ** 2 + this.im[o + q] ** 2;
      const ty = (F === 5 && j >= 3) ? 1 : 0;
      port[p] += t; type[ty] += t; link[g] += t; pt[p * 2 + ty] += t;
    }
    return { port, type, link, portType: pt };
  }
  recordWeight() {                                       // probability that the owned record is not blank
    let t = 0;
    for (let o = 60; o < this.n; o += 120) for (let q = 0; q < 60; q++) t += this.re[o + q] ** 2 + this.im[o + q] ** 2;
    return t;
  }
  overlap(g, p, j) {                                     // <A p j, link g | psi>
    const s = 1 / Math.sqrt(60), o = this.idx(g, p, j, 0, 0); let r = 0, i = 0;
    for (let k = 0; k < 60; k++) { r += this.re[o + k]; i += this.im[o + k]; }
    return [r * s, i * s];
  }
  basisProbability(p, j, r, k) { let t = 0; for (let g = 0; g < this.NG; g++) { const o = this.idx(g, p, j, r, k); t += this.re[o] ** 2 + this.im[o] ** 2; } return t; }
  faceProbabilities(faces) {                             // probability that the flux of the link register sits on each face
    const out = new Float64Array(FACES.length), link = this.populations().link;
    for (let g = 0; g < this.NG; g++) { const fl = faceFlux(this.words[g]); for (const f of faces) if (fl[f]) out[f] += link[g]; }
    return out;
  }
}

// the committed reference rule on one load: populations under the kernel 1 - L/60
export function committedStep(pop) {
  const out = new Float64Array(12);
  for (let p = 0; p < 12; p++) { let s = 0; for (const q of NEIGH[p]) s += pop[q]; out[p] = pop[p] * (1 - 5 / 60) + s / 60; }
  return out;
}
