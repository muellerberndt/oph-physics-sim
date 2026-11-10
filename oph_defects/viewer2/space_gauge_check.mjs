// node space_gauge_check.mjs : the force-carrier waves of gauge_wave.js against worker GL's Python run
// (repair-law/work/GL/s3_waves.json, from s3_waves.py) and the period symbols of cone_symbol.js against worker T3
// (repair-law/work/T3/cones.json, exact.json; reports/T3.md). Both JSON files are read, never written.
import { readFileSync } from 'node:fs';
import { packetRun, partnerEffect, predictedVelocity } from './gauge_wave.js';
import * as C from './cone_symbol.js';
import { symbol as cf1Symbol } from './cf1_engine.js';

let bad = 0;
const show = (ok, name, x, y) => { if (!ok) bad++; console.log((ok ? 'ok   ' : 'FAIL ') + name + ': ' + (typeof x === 'number' ? (+x).toPrecision(10) : x) + (y === undefined ? '' : ' expected ' + (typeof y === 'number' ? (+y).toPrecision(10) : y))); };
const near = (x, y, name, tol) => show(Math.abs(x - y) <= tol, name, x, y);
const below = (x, lim, name) => show(Math.abs(x) <= lim, name + ' (bound ' + lim.toExponential(0) + ')', x);
const rl = (p) => JSON.parse(readFileSync(new URL('./research/' + p, import.meta.url)));
const gl = rl('GL/s3_waves.json'), t3 = rl('T3/cones.json');

console.log('--- GL: packets of the twelve link fields (48 x 1 x 1 lattice; plane packets are uniform across)');
// Relative energy change between the start and the staggered fields after 24 ticks. GL's stored values compute the
// plaquette term as 1 - Re tr/5, which loses about five digits at amplitude 1e-3 (0.0178212 for gluon and W against
// 0.0178164 for B and lambda_8). CLEAN: GL's own code rerun on a scratch copy with the term written |U_P - 1|^2 / 10.
const PK = [['gluon lambda_1, polarization 2', 0, 1, 0.017816279575], ['W1, polarization 2', 8, 1, 0.017816279575], ['B_GUT, polarization 2', 11, 1, 0.017816279338], ['gluon lambda_8, polarization 3', 7, 2, 0.017816279575]];
let first = null;
for (const [name, a, pol, clean] of PK) {
  const r = packetRun(a, pol), ref = gl.packets_small_amplitude[name]; first = first || r;
  near(r.velocity, ref.velocity_cells_per_tick, name + ': velocity, cells per tick', 1e-9);
  below(r.gauss, 1e-16, name + ': Gauss law after 24 ticks (GL ' + ref.gauss_final.toExponential(1) + ')');
  near(r.drift, clean, name + ': relative energy change, staggered fields (GL code, cancellation-free)', 1e-10);
  near(r.drift, ref.energy_drift_relative, name + ': the same against GL\'s stored value (cancellation in GL)', 1e-5);
}
near(predictedVelocity(48), gl.packets_small_amplitude[PK[0][0]].predicted_packet_weighted_group_velocity, 'predicted packet velocity from the dispersion of the field step', 1e-12);
const full = packetRun(0, 1, { L: [48, 4, 4] });
near(full.velocity, first.velocity, 'the same packet on GL\'s 48 x 4 x 4 lattice', 1e-12);

console.log('--- GL: a partner packet in the other polarization, 16 ticks');
const PT = [['gluon lambda_1 (pol 2) with gluon lambda_2 (pol 3); commutator lambda_3', [0, 1], [1, 2], 6], ['W1 (pol 2) with W2 (pol 3); commutator W3', [8, 1], [9, 2], 10],
  ['control: gluon lambda_1 (pol 2) with B (pol 3)', [0, 1], [11, 2], null], ['control: gluon lambda_1 (pol 2) with W1 (pol 3)', [0, 1], [8, 2], null]];
for (const [name, main, partner, cdir] of PT) for (const eps of [0.05, 0.1, 0.2, 0.4]) {
  const r = partnerEffect(main, partner, eps, cdir), ref = gl.self_interaction[name]['eps=' + eps], tag = name.split(';')[0] + ', eps ' + eps;
  if (ref.relative_change_of_main_field > 1e-6) near(r.change, ref.relative_change_of_main_field, tag + ': relative change of the main packet', 1e-9 * ref.relative_change_of_main_field + 1e-12);
  else below(r.change, 2.5e-14, tag + ': relative change (GL ' + ref.relative_change_of_main_field.toExponential(1) + ')');
  near(r.fieldStrength, ref.field_strength_component_on_23_plaquettes, tag + ': commutator field strength at the start', 1e-11);
  below(r.gauss, 1e-14, tag + ': Gauss law at the end');
}

console.log('--- T3: symbols');
let seed = 7; const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return (seed / 2147483648 - 0.5) * 2 * Math.PI; };
const mul = (A, B) => { const re = new Float64Array(16), im = new Float64Array(16); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let sr = 0, si = 0; for (let k = 0; k < 4; k++) { sr += A.re[4 * i + k] * B.re[4 * k + j] - A.im[4 * i + k] * B.im[4 * k + j]; si += A.re[4 * i + k] * B.im[4 * k + j] + A.im[4 * i + k] * B.re[4 * k + j]; } re[4 * i + j] = sr; im[4 * i + j] = si; } return { re, im }; };
const dmax = (A, B) => { let m = 0; for (let k = 0; k < 16; k++) m = Math.max(m, Math.abs(A.re[k] - B.re[k]), Math.abs(A.im[k] - B.im[k])); return m; };
const lin = (c0, c1, m, c2) => { const re = new Float64Array(16), im = new Float64Array(16), P = [[3, 2, 1, 0], [3, 2, 1, 0], [2, 3, 0, 1], [2, 3, 0, 1]], V = [[[1, 0], [1, 0], [1, 0], [1, 0]], [[0, -1], [0, 1], [0, -1], [0, 1]], [[1, 0], [-1, 0], [1, 0], [-1, 0]], [[0, -1], [0, -1], [0, 1], [0, 1]]];
  for (let r = 0; r < 4; r++) { re[5 * r] += c0; const [ar, ai] = V[m][r], [br, bi] = V[3][r]; re[4 * r + P[m][r]] += c1 * ai; im[4 * r + P[m][r]] -= c1 * ar; re[4 * r + P[3][r]] += c2 * bi; im[4 * r + P[3][r]] -= c2 * br; } return { re, im }; };   // c0 - i c1 alpha_m - i c2 beta
let e1 = 0, e2 = 0, e3 = 0, e4 = 0;
for (let n = 0; n < 200; n++) {
  const p = [rnd(), rnd(), rnd()], th = 0.157935;
  e1 = Math.max(e1, dmax(C.symbol(C.periodCF1(0), p), mul(cf1Symbol(p, 0, 'Y', true), cf1Symbol(p, 0, 'Y', false))));
  const w = [C.S(0), C.S(1), C.S(2), C.M(th), C.S(2), C.S(1), C.S(0), C.M(th)];
  e2 = Math.max(e2, dmax(C.symbol(w, p), mul(cf1Symbol(p, th, 'Y', true), cf1Symbol(p, th, 'Y', false))));
  e3 = Math.max(e3, Math.abs(C.cosOmega(C.periodCF1(0), p) - Math.cos(2 * p[0]) * Math.cos(2 * p[1]) * Math.cos(2 * p[2])));
  const phi = 5 * Math.PI / 16, cp = Math.cos(phi), sp = Math.sin(phi);
  for (let i = 0; i < 3; i++) e4 = Math.max(e4, dmax(C.symbol([C.M(-phi / 2), C.L(i, phi), C.M(-phi / 2)], p), lin(cp * cp * Math.cos(p[i]) + sp * sp, cp * Math.sin(p[i]), i, sp * cp * (1 - Math.cos(p[i])))));
}
below(e1, 1e-14, 'CF-1 period from the word against two ticks of cf1_engine.js (theta = 0), 200 wave vectors');
below(e2, 1e-14, 'the same with the mass angle 0.157935 (reverse tick after forward tick)');
below(e3, 1e-14, 'CF-1 period: cos Omega = cos 2p1 cos 2p2 cos 2p3');
below(e4, 1e-14, 'member B sub-step: [cos^2 phi cos p + sin^2 phi] - i cos phi sin p alpha - i sin phi cos phi (1 - cos p) beta (T3 EXACT)');
let e5 = 0; const phiB = 5 * Math.PI / 16, thB = 0.157935;
for (let c = 0; c < 8; c++) { const cc = [c & 1, (c >> 1) & 1, (c >> 2) & 1], k = cc[0] + cc[1] + cc[2], ang = 2 * thB - 4 * k * phiB; e5 = Math.max(e5, dmax(C.symbol(C.periodB(phiB, thB), cc.map((v) => v * Math.PI)), lin(Math.cos(ang), 0, 0, Math.sin(ang)))); }
below(e5, 1e-14, 'member B corner phases P(pi c) = exp(-i (2 theta - 4 k phi) beta) at theta = 0.157935 (T3 EXACT)');

console.log('--- T3: cones over the full zone (32^3 grid, refined)');
const key = (p) => p.map((v) => Math.round(v / Math.PI * 4) / 4).map((v) => (v === 1 ? -1 : v === 0 ? 0 : v)).join(',');
const cf = C.criticalPoints(C.compile(C.periodCF1(0)), 32), ref = t3.members['W2 (CF-1)'].full;
show(cf.zeros.length === ref.count_phase0, 'CF-1 period: zero-phase points', cf.zeros.length, ref.count_phase0);
show(cf.pis.length === ref.count_phasepi, 'CF-1 period: phase-pi points', cf.pis.length, ref.count_phasepi);
const want = new Set(ref.zeros_phase0_over_pi.map((z) => z.map((v) => (v === 1 ? -1 : v)).join(','))), got = new Set(cf.zeros.map((z) => key(z.p)));
show([...want].every((k) => got.has(k)) && got.size === want.size, 'CF-1 period: the 32 positions equal T3\'s list', [...got].filter((k) => want.has(k)).length + ' of ' + want.size);
const W2 = C.periodCF1(0);
for (const chi of [1, -1]) {
  const s0 = cf.zeros.map((z) => C.blockChirality(W2, z.p, chi).chirality), spi = cf.pis.map((z) => C.blockChirality(W2, z.p, chi).chirality);
  const tally = (s) => `${s.filter((v) => v > 0).length} +, ${s.filter((v) => v < 0).length} -`;
  show(s0.reduce((a, b) => a + b, 0) === 0 && s0.every((v) => v !== 0), `arrow block ${chi > 0 ? '+1' : '-1'}: chiralities at phase 0`, tally(s0), 'net 0');
  show(spi.reduce((a, b) => a + b, 0) === 0 && spi.every((v) => v !== 0), `arrow block ${chi > 0 ? '+1' : '-1'}: chiralities at phase pi`, tally(spi), 'net 0');
}
const sub = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]].map((z) => C.blockChirality(W2, z.map((v) => v * Math.PI), 1).chirality);
show(sub.join(' ') === '1 -1 1 -1', 'block +1, one sublattice: (0,0,0) +, (0,1/2,1/2) -, (1/2,0,1/2) +, (1/2,1/2,0) - (T3 item 2)', sub.join(' '));
const B0 = C.criticalPoints(C.compile(C.periodB()), 32), rb = t3.members['B (recommended, phi = 5 pi/16)'].result_theta0;
show(B0.zeros.length === 1 && Math.hypot(...B0.zeros[0].p) < 1e-6, 'member B: zero-phase points (the origin only)', B0.zeros.length, rb.count_phase0);
show(B0.pis.length === rb.count_phasepi, 'member B: phase-pi points', B0.pis.length, rb.count_phasepi);
near(B0.lightest, rb.lightest_doubler_phase_per_period, 'member B: lightest doubler, phase per period', 1e-6);
near(B0.gapAtPi, rb.gap_at_pi_per_period, 'member B: gap at pi', 1e-9);
near(C.speedPerTick(C.periodB()), Math.cos(5 * Math.PI / 16), 'member B: massless speed per tick = cos phi (T3: ' + rb.massless_speed_per_tick_100.toFixed(6) + ')', 1e-6);
const Bm = C.criticalPoints(C.compile(C.periodB(5 * Math.PI / 16, 0.157935)), 32), rm = t3.members['B phi = 5 pi/16 with mass']['0.157935'];
show(Bm.zeros.length === 0, 'member B with theta = 0.157935: no zero-phase point', Bm.zeros.length, 0);
near(Bm.lightest, rm.lightest_doubler_phase_per_period, 'member B with mass: lightest phase (2 theta, at the origin)', 1e-6);
near(Bm.gapAtPi, rm.gap_at_pi_per_period, 'member B with mass: gap at pi', 1e-6);

console.log(bad ? `${bad} FAILED` : 'all checks pass');
process.exit(bad ? 1 : 0);
