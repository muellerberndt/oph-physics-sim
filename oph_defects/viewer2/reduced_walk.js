// Exact reduced form of the family walk for blank-record preparations.
// The walk W = Q R (public reversible seam repair Q, reflection R of the address register carried by the quantum)
// maps the span of the embedded states A|x> and their images Q A|x> to itself:
//     W (A a + Q A b) = A (-b) + Q A (a + 2 D b),      D = A* Q A = 1 - L / NA,
// with L the (recharted) port Laplacian and NA the number of addresses of a register. So a preparation with a blank
// record is carried by two complex amplitudes per port, and its port populations and record weight follow from
// closed formulas. Checked against the full engine in reduced_check.mjs (11/12, 643/900, 1/12).
export class ReducedWalk {
  // P ports; seams as arrays sa, sb (undirected, a -> b carries the phase (zr, zi), default 1); NA addresses per register.
  constructor(P, sa, sb, NA, zr = null, zi = null) {
    this.P = P; this.sa = sa; this.sb = sb; this.M = sa.length; this.NA = NA;
    this.zr = zr || new Float64Array(this.M).fill(1); this.zi = zi || new Float64Array(this.M);
    this.deg = new Float64Array(P);
    for (let s = 0; s < this.M; s++) { this.deg[sa[s]]++; this.deg[sb[s]]++; }
    this.ar = new Float64Array(P); this.ai = new Float64Array(P); this.br = new Float64Array(P); this.bi = new Float64Array(P);
    this.tr = new Float64Array(P); this.ti = new Float64Array(P);
    this.t = 0;
  }
  clear() { this.ar.fill(0); this.ai.fill(0); this.br.fill(0); this.bi.fill(0); this.t = 0; }
  embed(x, re = 1, im = 0) { this.ar[x] += re; this.ai[x] += im; }
  normalize() { let n = 0; for (let x = 0; x < this.P; x++) n += this.ar[x] ** 2 + this.ai[x] ** 2; n = 1 / Math.sqrt(n); for (let x = 0; x < this.P; x++) { this.ar[x] *= n; this.ai[x] *= n; } }
  // out = D v = v - (deg v - sum over seams of the hopped neighbour) / NA
  applyD(vr, vi, or, oi) {
    const { sa, sb, zr, zi, deg, NA, P, M } = this;
    for (let x = 0; x < P; x++) { or[x] = vr[x] * (1 - deg[x] / NA); oi[x] = vi[x] * (1 - deg[x] / NA); }
    const f = 1 / NA;
    for (let s = 0; s < M; s++) {
      const a = sa[s], b = sb[s], cr = zr[s], ci = zi[s];
      or[b] += f * (cr * vr[a] - ci * vi[a]); oi[b] += f * (cr * vi[a] + ci * vr[a]);        // hop a -> b with z
      or[a] += f * (cr * vr[b] + ci * vi[b]); oi[a] += f * (cr * vi[b] - ci * vr[b]);        // hop b -> a with conj z
    }
  }
  step() {
    const { P } = this;
    this.applyD(this.br, this.bi, this.tr, this.ti);
    for (let x = 0; x < P; x++) {
      const nar = -this.br[x], nai = -this.bi[x], nbr = this.ar[x] + 2 * this.tr[x], nbi = this.ai[x] + 2 * this.ti[x];
      this.ar[x] = nar; this.ai[x] = nai; this.br[x] = nbr; this.bi[x] = nbi;
    }
    this.t++;
  }
  // port populations (summed over record and address) and the weight of a non-blank record
  populations(out) {
    const { sa, sb, zr, zi, deg, NA, P, M, ar, ai, br, bi } = this;
    let rec = 0;
    for (let x = 0; x < P; x++) out[x] = (NA - 2 * deg[x]) / NA * ((ar[x] + br[x]) ** 2 + (ai[x] + bi[x]) ** 2);
    const f = 2 / NA;
    for (let s = 0; s < M; s++) {
      const a = sa[s], b = sb[s], cr = zr[s], ci = zi[s];
      // at port a the partner amplitude is conj(z) b_b; at port b it is z b_a
      const par = cr * br[b] + ci * bi[b], pai = cr * bi[b] - ci * br[b];
      const pbr = cr * br[a] - ci * bi[a], pbi = cr * bi[a] + ci * br[a];
      const a0r = ar[a] + 0.5 * (br[a] + par), a0i = ai[a] + 0.5 * (bi[a] + pai), a1r = 0.5 * (br[a] - par), a1i = 0.5 * (bi[a] - pai);
      const b0r = ar[b] + 0.5 * (br[b] + pbr), b0i = ai[b] + 0.5 * (bi[b] + pbi), b1r = 0.5 * (br[b] - pbr), b1i = 0.5 * (bi[b] - pbi);
      const ra = a1r * a1r + a1i * a1i, rb = b1r * b1r + b1i * b1i;
      out[a] += f * (a0r * a0r + a0i * a0i + ra); out[b] += f * (b0r * b0r + b0i * b0i + rb);
      rec += f * (ra + rb);
    }
    return rec;
  }
}
