// node cf1_check.mjs : the executable routing tick against the numbers of repair-law/candidates/CF1.md, section 4
import { Routing, symbol, phases, branchCoin } from './cf1_engine.js';
let bad = 0;
const near = (x, y, name, tol = 1e-6) => { const ok = Math.abs(x - y) < tol; if (!ok) bad++; console.log((ok ? 'ok   ' : 'FAIL ') + name + ': ' + (+x).toPrecision(9) + ' expected ' + (+y).toPrecision(9)); };
let seed = 5; const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
const random = (w) => { for (let j = 0; j < w.re.length; j++) { w.re[j] = rnd(); w.im[j] = rnd(); } w.normalize(); };
const diff = (a, b) => { let m = 0; for (let j = 0; j < a.re.length; j++) m = Math.max(m, Math.abs(a.re[j] - b.re[j]), Math.abs(a.im[j] - b.im[j])); return m; };

// 1. identities on a 6 x 4 x 4 lattice with a random state
const w = new Routing(6, 4, 4), u = new Routing(6, 4, 4), v = new Routing(6, 4, 4); random(w);
let worst = 0, worst2 = 0;
for (let i = 0; i < 3; i++) {
  u.copyFrom(w); u.seamInvolution(i); u.seamInvolution(i); worst = Math.max(worst, diff(u, w));
  u.copyFrom(w); u.shiftPublic(i); v.copyFrom(w); v.shift(i); worst2 = Math.max(worst2, diff(u, v));
}
near(worst, 0, 'Q_i Q_i = 1 (largest defect over the three axes)', 1e-14);
near(worst2, 0, 'Q_i R = P+ T + P- T* (largest defect)', 1e-14);
u.copyFrom(w); u.tickAlt(0.2); u.tickAlt(0.2); near(u.norm2(), 1, 'two ticks keep the norm', 1e-13);
// tick against the symbol on plane waves, both orders
let ws = 0;
for (const m of [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 3, 1]]) for (let col = 0; col < 4; col++) for (const [th, kind] of [[0, 'Y'], [0.157935, 'Y'], [0.157935, 'Z']]) for (const rev of [false, true]) {
  const coin = [0, 1, 2, 3].map((k) => [k === col ? 1 : 0, 0]), p = w.momentum(m);
  u.clear(); u.addPlaneWave(p, coin); u.tick(th, kind, rev);
  const U = symbol(p, th, kind, rev), want = [0, 1, 2, 3].map((k) => [U.re[4 * k + col], U.im[4 * k + col]]);
  v.clear(); v.addPlaneWave(p, want); ws = Math.max(ws, diff(u, v));
}
near(ws, 0, 'tick against symbol on plane waves, both orders', 1e-13);

// 1b. static link signs of the lattice gauge field: public form against direct form, and gauge covariance of the tick
{ const a = new Routing(6, 4, 4), b = new Routing(6, 4, 4), c = new Routing(6, 4, 4); random(a);
  const links = [0, 1, 2].map(() => Float64Array.from({ length: a.N }, () => (rnd() > 0 ? 1 : -1)));
  a.links = links; b.links = links; c.links = links; let w1 = 0, w3 = 0;
  for (let i = 0; i < 3; i++) { b.copyFrom(a); b.seamInvolution(i); b.seamInvolution(i); w3 = Math.max(w3, diff(a, b)); b.copyFrom(a); b.shiftPublic(i); c.copyFrom(a); c.shift(i); w1 = Math.max(w1, diff(b, c)); }
  near(w3, 0, 'with link signs: Q_i Q_i = 1', 1e-14); near(w1, 0, 'with link signs: public form equals direct form', 1e-14);
  // gauge transformation g(x) = +-1: links z(x,i) -> g(x) z g(x + e_i), state psi(x) -> g(x) psi(x)
  const g = Float64Array.from({ length: a.N }, () => (rnd() > 0 ? 1 : -1));
  const lg = [0, 1, 2].map((i) => Float64Array.from({ length: a.N }, (_, x) => g[x] * links[i][x] * g[a.fwd[i][x]]));
  b.copyFrom(a); b.links = links; b.tickAlt(0.1); b.tickAlt(0.1);
  c.copyFrom(a); for (let x = 0; x < a.N; x++) for (let k = 0; k < 4; k++) { c.re[4 * x + k] *= g[x]; c.im[4 * x + k] *= g[x]; }
  c.links = lg; c.tickAlt(0.1); c.tickAlt(0.1); for (let x = 0; x < a.N; x++) for (let k = 0; k < 4; k++) { c.re[4 * x + k] *= g[x]; c.im[4 * x + k] *= g[x]; }
  near(diff(b, c), 0, 'the tick is gauge covariant (two ticks, random signs)', 1e-14); near(b.norm2(), 1, 'unitary with link signs', 1e-13); }

// 2. one axis: cos E = cos(theta) cos(p)
const th = 0.157935;
[[0, 0.157935], [0.05, 0.165598], [0.1, 0.186709], [0.2, 0.254184], [0.4, 0.428482]].forEach(([p, E]) => near(phases([p, 0, 0], th, 'Y', false)[0], E, `one axis, theta 0.157935, p ${p}`));
// 3. off axis: helicity splitting of the ordered tick, and the alternating order
const off = phases([0.1, 0.1, 0.1], th, 'Y', false); near(off[0], 0.220876, 'ordered tick, p (0.1,0.1,0.1), first helicity'); near(off[1], 0.245779, 'ordered tick, second helicity');
const off0 = phases([0.1, 0.1, 0.1], 0, 'Y', false); near(off0[0], 0.167033812, 'ordered tick, massless, first helicity', 1e-8); near(off0[1], 0.178606386, 'ordered tick, massless, second helicity', 1e-8);
const alt = phases([0.1, 0.1, 0.1], 0, 'Y', true); near(alt[0], 0.172043495, 'alternating order, massless, phase per tick', 1e-8); near(alt[1], 0.172043495, 'alternating order, both helicities equal', 1e-8);
near(phases([0.05, 0.05, 0.05], 0)[0], 0.086457987, 'alternating order at |p| 0.0866', 1e-8); near(phases([0.025, 0.025, 0.025], 0)[0], 0.043283221, 'alternating order at |p| 0.0433', 1e-8);
// executed two ticks in alternating order against the symbol
{ const a = new Routing(8, 8, 8), b = new Routing(8, 8, 8), p = a.momentum([1, 2, 3]), coin = [[1, 0], [0, 0], [0, 0], [0, 0]];
  a.addPlaneWave(p, coin); a.tickAlt(th); a.tickAlt(th);
  const U1 = symbol(p, th, 'Y', false), U2 = symbol(p, th, 'Y', true), mid = [0, 1, 2, 3].map((k) => [U1.re[4 * k], U1.im[4 * k]]);
  const want = [0, 1, 2, 3].map((k) => { let sr = 0, si = 0; for (let j = 0; j < 4; j++) { sr += U2.re[4 * k + j] * mid[j][0] - U2.im[4 * k + j] * mid[j][1]; si += U2.re[4 * k + j] * mid[j][1] + U2.im[4 * k + j] * mid[j][0]; } return [sr, si]; });
  b.addPlaneWave(p, want); near(diff(a, b), 0, 'executed two ticks in alternating order against the symbol', 1e-13); }

// 4. record after one tick from a blank record
{ const a = new Routing(64, 2, 2), blank = [[1, 0], [0, 0], [0, 0], [0, 0]];
  a.clear(); a.addPlaneWave(a.momentum([0, 0, 0]), blank); a.tick(th); near(a.recordProbability(), 0.024737, 'record at rest, theta 0.157935 (sin^2 theta)');
  a.clear(); a.addPlaneWave(a.momentum([1, 0, 0]), blank); a.tick(0); near(a.recordProbability(), 0.009607, 'record, massless, p = 2 pi / 64');
  a.clear(); a.addPlaneWave(a.momentum([1, 0, 0]), blank); a.tick(th); near(a.recordProbability(), 0.033869, 'record, both'); }

// 5. packets on a 256-cell line, p0 = 2 pi 12 / 256, sixty ticks. The coin of each plane wave is the projection of the blank
// spin-up state on the positive branch, so its phase is continuous in p and the packet is one lump. The measured velocity
// then equals dE/dp averaged over the momentum weights of the packet. (The reference run takes eigenvectors with the
// phases numpy returns; its 0.9525 and 0.8745 are specific to those phases. The velocity at the central momentum is the
// same in both: 0.956925 and 0.876719.)
for (const [theta, v0] of [[0, 1.0], [0.087847, 0.956925], [0.157935, 0.876719]]) {
  const a = new Routing(256, 1, 1); a.packetOnBranch(12, 3.0, theta); const x0 = a.meanPosition(0);
  for (let t = 0; t < 60; t++) a.tick(theta);
  const x1 = a.meanPosition(0), vmeas = (((x1 - x0 + 128) % 256 + 256) % 256 - 128) / 60;
  let sw = 0, sv = 0;
  for (let m = 0; m < 256; m++) { const d = ((m - 12 + 128) % 256 + 256) % 256 - 128, wgt = Math.exp(-d * d / 9), p = 2 * Math.PI * m / 256, E = Math.acos(Math.cos(theta) * Math.cos(p)); sw += wgt; sv += wgt * (E < 1e-12 ? 1 : Math.cos(theta) * Math.sin(p) / Math.sin(E)); }
  const p0 = 2 * Math.PI * 12 / 256, E0 = Math.acos(Math.cos(theta) * Math.cos(p0));
  near(theta === 0 ? 1 : Math.cos(theta) * Math.sin(p0) / Math.sin(E0), v0, `dE/dp at the central momentum, theta ${theta}`, 1e-6);
  near(vmeas, sv / sw, `packet velocity against dE/dp averaged over the packet, theta ${theta}`, 2e-4);
  console.log('     measured velocity ' + vmeas.toFixed(6));
}
// 6. half-step frame: same two-tick phase as the end frame on a plane wave, and no drift of a blank packet at rest
{ const a = new Routing(256, 1, 1), b = new Routing(256, 1, 1), th = 0.157935, blank = [[1, 0], [0, 0], [0, 0], [0, 0]];
  a.addPacket([128, 0, 0], 6, [0, 0, 0], blank); a.normalize(); b.copyFrom(a);
  const x0 = a.meanPosition(0); for (let t = 0; t < 40; t++) { a.tickAlt(th); b.tickAltHalf(th / 2); }
  const da = (a.meanPosition(0) - x0) / 40, db = (b.meanPosition(0) - x0) / 40;
  console.log('     drift of a blank packet at rest, cells per tick: mass step at the end ' + da.toFixed(5) + ', half-step frame ' + db.toFixed(5));
  near(Math.abs(db), 0, 'half-step frame: a blank packet at rest does not drift', 2e-3);
  const c = new Routing(64, 2, 2), p = c.momentum([3, 0, 0]); c.addPlaneWave(p, blank); c.normalize(); const c0r = Float64Array.from(c.re), c0i = Float64Array.from(c.im);
  c.tickAltHalf(th / 2); c.tickAltHalf(th / 2); const ov = c.overlap(c0r, c0i);
  const d = new Routing(64, 2, 2); d.addPlaneWave(p, blank); d.normalize(); d.tickAlt(th); d.tickAlt(th); const ov2 = d.overlap(c0r, c0i);
  console.log('     two-tick return amplitude of a blank plane wave: half-step frame ' + ov[0].toFixed(6) + ', end frame ' + ov2[0].toFixed(6) + ' (cos of twice the phase: ' + Math.cos(2 * Math.acos(Math.cos(th) * Math.cos(p[0]))).toFixed(6) + ')');
  near(ov[0], Math.cos(2 * Math.acos(Math.cos(th) * Math.cos(p[0]))), 'half-step frame: return amplitude of a blank plane wave after two ticks is cos(2 w)', 1e-9); }
console.log(bad ? `${bad} FAILED` : 'all checks pass'); process.exitCode = bad ? 1 : 0;
