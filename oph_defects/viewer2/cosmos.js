// The Cosmos tab: the early universe of the standard model of cosmology beside what OPH states and executes.
// Right column: the standard hot big bang, as measured. Left column: OPH, with every statement marked as executed by
// the model, stated by the cosmology papers under named premises, imported from standard physics, or not derived.
// Sources of the OPH column: reverse-engineering-reality/cosmology/*.tex and code/cosmology/postdiction_ledger
// (numbers copied from the ledger), the five-level chart of the Particles tab, and the committed level-10 run.
// The clumping picture is a two-dimensional particle-mesh calculation in the browser with the same seed on both sides.
import { $, S, fmt, marker } from './app.js';
import { loadNpz } from './npz.js';
import { hotWeights, displayedCut, COSMOLOGY_LEDGER, TILT_COMPARISONS } from './cosmos_evidence.js';

const NG = 128, NP = NG * NG, OM = 0.315, OL = 0.685, A0 = 0.02;
const ERAS = [
  { key: 'hot', name: 'Hot plasma', when: 'first fraction of a second',
    std: 'A thermalized plasma well above the electroweak scale contains approximately relativistic Standard Model fields. Its temperature and initial species depend on reheating. Per generation there are fifteen chiral matter states, before antiparticles are counted.',
    oph: 'A selected exterior-power sector of the declared five-level code has fifteen Standard Model current weights plus a neutral singlet. This representation count does not derive a thermal population, physical spin statistics, or three independently occupiable generations. The separate run stores classical centre flux.' },
  { key: 'nuclei', name: 'First nuclei', when: 'three minutes',
    std: 'Protons and neutrons fuse. By mass three quarters stay hydrogen and one quarter becomes helium, with traces of deuterium and lithium.',
    oph: 'OPH does not compute this step. Its papers import the thermal history. The same elements follow once the masses and rates are the standard ones, and those are not derived yet.' },
  { key: 'atoms', name: 'First atoms', when: '380,000 years',
    std: 'Electrons bind to nuclei, the gas turns transparent and the light of that moment still arrives today. The observed CMB temperature anisotropy is of order one part in 100,000.',
    oph: 'Imported: the binding of atoms and the transport of light. OPH supplies a candidate for the small ripples: one finite source screen, with a tilt of 0.9660 if one declared density holds.' },
  { key: 'clump', name: 'Clumping', when: 'to today',
    std: 'Gravity pulls matter into a web of filaments and clumps. Five sixths of the matter is dark and cold.',
    oph: 'The papers assume the same gravity law. One proposed dark component is unrecovered collar charge, with a matter-like dilution law under its stated premises. Its amount is not derived. The executable model has no gravity yet.' },
];
const HOT_STD = [['quark doublets', 6, '#ff7a45'], ['up-type singlets', 3, '#ff5f7a'], ['down-type singlets', 3, '#ffb199'], ['lepton doublets', 2, '#b197ff'], ['charged lepton singlet', 1, '#3fe0d0']];
const HOT_OPH = [...HOT_STD, ['neutral singlet (sixteenth state)', 1, '#9fb4e0']];
const LEDGER = [
  ['Beginning', 'no beginning assumed; a law for earlier correlations and their later records', 'an approximately 13.8-billion-year age in the reference ΛCDM fit; hot thermal history does not establish an absolute beginning', 'different', 'papers'],
  ['Expansion', 'a record density: the count per cell grows as the fourth power of the scale factor; redshift is the fourth root of a count ratio. The history itself is supplied', 'space expands by the Friedmann law', 'conditional', 'theorem on a supplied population'],
  ['Why uniform', 'not derived: uniform release and horizon coherence are premises', 'inflation is one proposed origin of near-uniform initial conditions', 'not derived', 'papers'],
  ...TILT_COMPARISONS.map((r) => ['Tilt of the ripples', '0.96602150 at P_C under declared collar density; P_fwd gives 0.96602141', `${r.measurement.value} ± ${r.measurement.sigma} (${r.dataset})`, `${r.verdict}, ${r.sigma_distance >= 0 ? '+' : ''}${r.sigma_distance.toFixed(2)} σ`, `<a href="${r.url}">primary dataset paper</a>; <a href="${COSMOLOGY_LEDGER.url}">seen-data ledger</a>`]),
  ['Running of the tilt', '0', '−0.0045 ± 0.0067', 'consistent', 'ledger'],
  ['Primordial gravity waves', '0', 'below 0.036', 'consistent, does not discriminate', 'ledger'],
  ['Curvature', '0, identified and not selected', '0.0007 ± 0.0019 (Planck + BAO)', 'consistent', 'ledger'],
  ['Particles of the hot plasma', 'a declared 16-weight chiral sector; no thermal population or family count derived', 'fifteen states per generation, three generations', 'representation comparison only', 'chosen exterior-power sector'],
  ['Number of light neutrino kinds', 'no value', '2.99 ± 0.17', 'not evaluable', 'ledger'],
  ['Hydrogen and helium', 'not computed; thermal history imported', '75 % and 25 % by mass', 'imported', 'papers'],
  ['Dark matter', 'charge on unrecovered overlap collars; thins out as the inverse volume; amount not derived', 'a cold dark component in the reference fit, about 5.4 times baryonic matter; identity unresolved', 'not derived', 'papers, theorem for the dilution'],
  ['Gravity in galaxies', 'v⁴ = G M a₀ under two premises, a₀ fitted; exponent 4', 'exponent 3.75 ± 0.10 (SPARC)', 'diagnostic +2.66 σ; unscorable', 'ledger: calibrated errors-in-variables likelihood absent'],
  ['Dark energy', 'a fixed screen capacity: w = −1 at all times', 'w₀ = −0.75 ± 0.06, w_a = −0.86 ± 0.22 (DESI DR2 + CMB + DESY5)', 'dataset-dependent preference', 'ledger: DESI DR2 with specified CMB/SN combinations; no joint score from marginal errors'],
  ['Cosmological constant', '1.0959 × 10⁻⁵² m⁻² (display value)', '1.0891 × 10⁻⁵² m⁻²', 'not evaluable', 'ledger'],
  ['Clumping', 'same gravity law assumed, plus two terms that vanish on the standard branch; no numbers yet', 'measured: the cosmic web', 'not derived', 'papers'],
];
const C = { era: 0, running: false, mode: 'branch', sims: null, seed: 7, jit: 0 };

// ---- fast Fourier transform, radix 2, in place ----------------------------------------------------------------------
const COS = new Float64Array(NG / 2), SIN = new Float64Array(NG / 2); for (let i = 0; i < NG / 2; i++) { COS[i] = Math.cos(2 * Math.PI * i / NG); SIN[i] = Math.sin(2 * Math.PI * i / NG); }
const REV = new Uint16Array(NG); for (let i = 0; i < NG; i++) { let r = 0; for (let b = 0; b < 7; b++) if (i & (1 << b)) r |= 1 << (6 - b); REV[i] = r; }
function fft1(re, im, off, stride, inv) {
  for (let i = 0; i < NG; i++) { const j = REV[i]; if (j > i) { const a = off + i * stride, b = off + j * stride; let t = re[a]; re[a] = re[b]; re[b] = t; t = im[a]; im[a] = im[b]; im[b] = t; } }
  for (let len = 2; len <= NG; len <<= 1) { const half = len >> 1, step = NG / len;
    for (let s = 0; s < NG; s += len) for (let k = 0; k < half; k++) { const wr = COS[k * step], wi = inv ? SIN[k * step] : -SIN[k * step], a = off + (s + k) * stride, b = off + (s + k + half) * stride;
      const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr; re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi; } }
}
function fft2(re, im, inv) { for (let r = 0; r < NG; r++) fft1(re, im, r * NG, 1, inv); for (let c = 0; c < NG; c++) fft1(re, im, c, NG, inv); if (inv) { const f = 1 / (NG * NG); for (let i = 0; i < NG * NG; i++) { re[i] *= f; im[i] *= f; } } }
const kOf = (i) => 2 * Math.PI * (i < NG / 2 ? i : i - NG) / NG;
const F = (a) => 1 / Math.sqrt((OM + OL * a * a * a) / a);

// ---- one universe: Zel'dovich start, then particle-mesh gravity in comoving coordinates ---------------------------------
function makeSim(ns, seed) {
  let s = seed; const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return (s + 0.5) / 2147483648; };
  const gauss = () => Math.sqrt(-2 * Math.log(rnd())) * Math.cos(2 * Math.PI * rnd());
  const re = new Float64Array(NG * NG), im = new Float64Array(NG * NG); for (let i = 0; i < NG * NG; i++) re[i] = gauss();
  fft2(re, im, false);
  // spectrum of a slice: P(k) = k^(ns + 1) T(k)^2, transfer function of Bardeen and others, one cell = 1 Mpc/h, shape 0.212
  const sxr = new Float64Array(NG * NG), sxi = new Float64Array(NG * NG), syr = new Float64Array(NG * NG), syi = new Float64Array(NG * NG);
  for (let i = 0; i < NG; i++) for (let j = 0; j < NG; j++) { const kx = kOf(j), ky = kOf(i), k = Math.hypot(kx, ky), o = i * NG + j; if (k === 0) { re[o] = im[o] = 0; continue; }
    const q = k / 0.212, T = Math.log(1 + 2.34 * q) / (2.34 * q) * Math.pow(1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4, -0.25);
    const amp = Math.sqrt(Math.pow(k, ns + 1)) * T; re[o] *= amp; im[o] *= amp;
    sxr[o] = -kx * im[o] / (k * k); sxi[o] = kx * re[o] / (k * k); syr[o] = -ky * im[o] / (k * k); syi[o] = ky * re[o] / (k * k); }       // s = i k delta / k^2
  fft2(re, im, true); fft2(sxr, sxi, true); fft2(syr, syi, true);
  let v = 0; for (let i = 0; i < NG * NG; i++) v += re[i] * re[i]; const norm = 2.6 / Math.sqrt(v / (NG * NG));                  // linear contrast 2.6 per cell today
  const sim = { ns, a: A0, x: new Float64Array(NP), y: new Float64Array(NP), px: new Float64Array(NP), py: new Float64Array(NP), x0: new Float64Array(NP), y0: new Float64Array(NP),
    rho: new Float64Array(NG * NG), fr: new Float64Array(NG * NG), fi: new Float64Array(NG * NG), gx: new Float64Array(NG * NG), gxi: new Float64Array(NG * NG), gy: new Float64Array(NG * NG), gyi: new Float64Array(NG * NG), sx: sxr, sy: syr, norm };
  for (let i = 0; i < NG; i++) for (let j = 0; j < NG; j++) { const o = i * NG + j; sim.x0[o] = j + 0.5 + A0 * norm * sxr[o]; sim.y0[o] = i + 0.5 + A0 * norm * syr[o]; }
  resetSim(sim); return sim;
}
function resetSim(sim) { sim.a = A0; sim.x.set(sim.x0); sim.y.set(sim.y0); const f = A0 * A0 / F(A0) * sim.norm; for (let o = 0; o < NP; o++) { sim.px[o] = f * sim.sx[o]; sim.py[o] = f * sim.sy[o]; } }
const wrap = (u) => { u %= NG; return u < 0 ? u + NG : u; };
function forces(sim) {
  const { rho, fr, fi, gx, gxi, gy, gyi } = sim; rho.fill(0);
  for (let p = 0; p < NP; p++) { const x = wrap(sim.x[p] - 0.5), y = wrap(sim.y[p] - 0.5), j = Math.floor(x), i = Math.floor(y), dx = x - j, dy = y - i, j1 = (j + 1) % NG, i1 = (i + 1) % NG;
    rho[i * NG + j] += (1 - dx) * (1 - dy); rho[i * NG + j1] += dx * (1 - dy); rho[i1 * NG + j] += (1 - dx) * dy; rho[i1 * NG + j1] += dx * dy; }
  for (let o = 0; o < NG * NG; o++) { fr[o] = rho[o] - 1; fi[o] = 0; }
  fft2(fr, fi, false); const coef = 1.5 * OM / sim.a;
  for (let i = 0; i < NG; i++) for (let j = 0; j < NG; j++) { const o = i * NG + j, kx = kOf(j), ky = kOf(i), k2 = 4 * (Math.sin(kx / 2) ** 2 + Math.sin(ky / 2) ** 2);
    if (k2 === 0) { gx[o] = gxi[o] = gy[o] = gyi[o] = 0; continue; }
    const pr = -coef * fr[o] / k2, pi = -coef * fi[o] / k2, sx = Math.sin(kx), sy = Math.sin(ky);            // g = -grad phi : g_k = -i k phi_k
    gx[o] = sx * pi; gxi[o] = -sx * pr; gy[o] = sy * pi; gyi[o] = -sy * pr; }
  fft2(gx, gxi, true); fft2(gy, gyi, true);
}
function stepSim(sim) {
  const da = Math.min(Math.max(0.002, 0.03 * sim.a), 1 - sim.a + 1e-9); forces(sim); const { gx, gy } = sim, fk = F(sim.a) * da, ah = sim.a + da / 2, fd = F(ah) / (ah * ah) * da;
  for (let p = 0; p < NP; p++) { const x = wrap(sim.x[p] - 0.5), y = wrap(sim.y[p] - 0.5), j = Math.floor(x), i = Math.floor(y), dx = x - j, dy = y - i, j1 = (j + 1) % NG, i1 = (i + 1) % NG;
    const w00 = (1 - dx) * (1 - dy), w01 = dx * (1 - dy), w10 = (1 - dx) * dy, w11 = dx * dy;
    sim.px[p] += fk * (w00 * gx[i * NG + j] + w01 * gx[i * NG + j1] + w10 * gx[i1 * NG + j] + w11 * gx[i1 * NG + j1]);
    sim.py[p] += fk * (w00 * gy[i * NG + j] + w01 * gy[i * NG + j1] + w10 * gy[i1 * NG + j] + w11 * gy[i1 * NG + j1]);
    sim.x[p] = wrap(sim.x[p] + fd * sim.px[p]); sim.y[p] = wrap(sim.y[p] + fd * sim.py[p]); }
  sim.a += da;
}
function contrast(sim) { let v = 0; for (let o = 0; o < NG * NG; o++) v += (sim.rho[o] - 1) ** 2; return Math.sqrt(v / (NG * NG)); }

// ---- drawing --------------------------------------------------------------------------------------------------------
function species(p, table) { const tot = table.reduce((s, r) => s + r[1], 0); let u = ((p * 2654435761) >>> 0) / 4294967296 * tot; for (const r of table) { if (u < r[1]) return r[2]; u -= r[1]; } return table[0][2]; }
function paint(id, sim, side) {
  const cv = $(id), g = cv.getContext('2d'), W = cv.width, sc = W / NG, era = ERAS[C.era].key;
  g.globalCompositeOperation = 'source-over'; g.fillStyle = '#04060d'; g.fillRect(0, 0, W, W); g.globalCompositeOperation = 'lighter';
  const still = era !== 'clump' || (side === 'oph' && C.mode === 'executed'), X = still ? sim.x0 : sim.x, Y = still ? sim.y0 : sim.y, jit = era === 'clump' ? 0 : 0.35;
  for (let p = 0; p < NP; p++) {
    let col, r = 1.6;
    if (era === 'hot') col = species(p, side === 'oph' ? HOT_OPH : HOT_STD);
    else if (era === 'nuclei') { const he = (p % 13) === 0; col = he ? '#ffe45c' : '#ff9a6b'; r = he ? 2.4 : 1.5; }
    else if (era === 'atoms') { const light = (p % 3) === 0; col = light ? '#dfe9ff' : ((p % 13) === 1 ? '#ffe45c' : '#ff9a6b'); r = light ? 1 : 1.6; }
    else { const vis = (p % 51) < 8; col = vis ? '#ffb07a' : (side === 'oph' ? '#5b6fd6' : '#4d5fa8'); r = vis ? 1.7 : 1.25; }
    const jx = jit ? jit * Math.sin(p * 12.9898 + C.jit * (1 + (p % 7) * 0.13)) : 0, jy = jit ? jit * Math.cos(p * 78.233 + C.jit * (1 + (p % 5) * 0.17)) : 0;
    g.fillStyle = col; g.globalAlpha = era === 'clump' ? 0.55 : 0.7; g.fillRect(wrap(X[p] + jx) * sc - r / 2, wrap(Y[p] + jy) * sc - r / 2, r, r);
  }
  if (era === 'hot' && side === 'oph') { g.globalAlpha = 0.9; g.strokeStyle = '#ffe45c'; g.lineWidth = 1; let s = 99; const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; };
    for (let k = 0; k < 420; k++) { const x = rnd() * W, y = rnd() * W; g.strokeRect(x - 2.5, y - 2.5, 5, 5); } }
  g.globalAlpha = 1; g.globalCompositeOperation = 'source-over';
  if (side === 'oph' && (era === 'nuclei' || era === 'atoms')) { g.fillStyle = 'rgba(255,255,255,.55)'; g.font = '600 15px system-ui'; g.fillText('imported from standard physics, not computed by OPH', 14, W - 16); }
  if (side === 'oph' && era === 'clump') { g.fillStyle = 'rgba(255,255,255,.6)'; g.font = '600 15px system-ui'; g.fillText(C.mode === 'executed' ? 'illustrative initial map: the load run supplies no gravity law' : 'assumed gravity law of the papers; amount of dark charge imported', 14, W - 16); }
}
function bars(table, note) {
  const tot = table.reduce((s, r) => s + r[1], 0);
  return `<div class="cos-bar">${table.map((r) => `<i style="width:${(100 * r[1] / tot).toFixed(2)}%;background:${r[2]}" title="${r[0]}"></i>`).join('')}</div>` +
    `<div class="cos-keys">${table.map((r) => `<span><i style="background:${r[2]}"></i>${r[0]} ${(100 * r[1] / tot).toFixed(1)} %</span>`).join('')}</div>` + (note ? `<div class="note">${note}</div>` : '');
}
function panels() {
  const era = ERAS[C.era], k = era.key; let std = '', oph = '';
  if (k === 'hot') {
    std = bars(HOT_STD, 'Share of chiral matter degrees in one generation in the ideal relativistic thermal limit. Three generations. Force carriers: eight gluons, four weak and hypercharge fields, one Higgs doublet.');
    oph = bars(HOT_OPH, 'Illustrative internal representation weights in a chosen 16-state sector. The code’s vector and spinor names label internal currents, not the declared spatial spin pointer. Yellow squares are illustrative centre-flux markers, with no inferred abundance or dark-matter mass. A neutral singlet’s production, decoupling and contribution to nucleosynthesis require rates and a thermal history.');
  } else if (k === 'nuclei') {
    const t = [['hydrogen', 75.3, '#ff9a6b'], ['helium', 24.7, '#ffe45c']];
    std = bars(t, 'By mass. Traces per hydrogen nucleus: deuterium 2.5 × 10⁻⁵, helium-3 1 × 10⁻⁵, lithium 5 × 10⁻¹⁰ predicted against 1.6 × 10⁻¹⁰ seen.');
    oph = bars(t, 'Imported. The cosmology papers compute no abundance. The neutron and proton masses, the weak rates and the baryon share are inputs that OPH has not derived.');
  } else if (k === 'atoms') {
    const t = [['hydrogen atoms', 92, '#ff9a6b'], ['helium atoms', 8, '#ffe45c']];
    std = bars(t, 'By number of atoms. Light of 2.725 K today. Ripples: tilt 0.9649 ± 0.0042, no gravity waves seen, flat within 0.2 %.');
    oph = bars(t, 'Imported atoms and light transport, with two OPH terms that vanish on the standard branch. The selected edge-centre branch gives tilt about 0.96602 under a declared collar density. The retained dataset comparisons range from +1.37 σ to −2.43 σ; they are seen-data comparisons, not a frozen prediction. Running 0 and tensor ratio 0 are branch claims; flatness is identified, not selected.');
  } else {
    const t = [['visible matter', 4.9, '#ffb07a'], ['dark matter', 26.5, '#4d5fa8'], ['dark energy', 68.6, '#39445f']];
    std = bars(t, 'Share of the energy today. Cold dark matter is the reference component; its particle identity and any nongravitational couplings are unresolved.');
    oph = bars([['visible matter', 4.9, '#ffb07a'], ['charge on unrecovered collars', 26.5, '#5b6fd6'], ['screen capacity', 68.6, '#39445f']], 'Same shares, imported. The collar charge thins out like matter (theorem). The stated capacity branch has w = −1; DESI DR2 combined fits give a dataset-dependent preference for evolving dark energy. Marginal parameter errors do not supply a joint score. In galaxies two premises give v⁴ = G M a₀ with a fitted a₀.');
  }
  $('cos-std-text').textContent = era.std; $('cos-oph-text').textContent = era.oph; $('cos-std-bars').innerHTML = std; $('cos-oph-bars').innerHTML = oph;
  $('cos-mode-row').style.display = k === 'clump' ? '' : 'none'; $('cos-run').style.display = k === 'clump' ? '' : 'none'; $('cos-reset').style.display = k === 'clump' ? '' : 'none';
  document.querySelectorAll('.cos-era').forEach((b, i) => b.classList.toggle('on', i === C.era));
}
function readout() {
  const s = C.sims, k = ERAS[C.era].key;
  if (k !== 'clump') { $('cos-read').innerHTML = `<b>${ERAS[C.era].name}</b> · ${ERAS[C.era].when} · schematic homogeneous background`; return; }
  const a = Math.min(1, s.std.a), z = Math.max(0, 1 / a - 1);
  $('cos-read').innerHTML = `<b>Clumping</b> · scale factor ${a.toFixed(3)} · redshift ${z.toFixed(2)} · density contrast per cell: standard ${contrast(s.std).toFixed(2)}, OPH ${C.mode === 'executed' ? 'frozen illustrative initial map' : contrast(s.oph).toFixed(2)} · OPH reading: the record count per cell has grown by a factor ${(Math.pow(a / A0, 4)).toExponential(2)}`;
}
function draw() { if (!C.sims) return; paint('cos-oph', C.sims.oph, 'oph'); paint('cos-std', C.sims.std, 'std'); readout(); }
function loop() {
  requestAnimationFrame(loop);
  if (!$('page-cosmos').classList.contains('on') || !C.sims) return;
  const k = ERAS[C.era].key;
  if (k !== 'clump') { C.jit += 0.05; draw(); return; }
  if (C.running) { if (C.sims.std.a >= 0.999) { run(false); return; } stepSim(C.sims.std); stepSim(C.sims.oph); draw(); }
}
function run(on) { C.running = on; $('cos-run').textContent = on ? '❚❚ pause' : '▶ let gravity act'; }
function setEra(i) { C.era = i; run(false); panels(); if (ERAS[i].key === 'clump') { forces(C.sims.std); forces(C.sims.oph); } draw(); }
let inited = false;
export function cosmosShown() {
  if (inited) { draw(); fpShown(); return; }
  inited = true;
  $('cos-eras').innerHTML = ERAS.map((e, i) => `<button class="cos-era" data-i="${i}"><b>${e.name}</b><span>${e.when}</span></button>`).join('');
  document.querySelectorAll('.cos-era').forEach((b) => b.onclick = () => setEra(parseInt(b.dataset.i)));
  $('cos-table').innerHTML = '<tr><th></th><th>OPH</th><th>standard model and measurement</th><th>verdict</th><th>source of the OPH entry</th></tr>' +
    LEDGER.map((r) => { const tag = r[3].startsWith('consistent') ? 'ok' : r[3].startsWith('tension') ? 'bad' : 'open'; return `<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td>${r[2]}</td><td><span class="cos-tag ${tag}">${r[3]}</span></td><td class="small">${r[4]}</td></tr>`; }).join('');
  C.sims = { std: makeSim(0.9649, C.seed), oph: makeSim(0.96602, C.seed) };
  $('cos-run').onclick = () => run(!C.running);
  $('cos-reset').onclick = () => { run(false); resetSim(C.sims.std); resetSim(C.sims.oph); forces(C.sims.std); forces(C.sims.oph); draw(); };
  $('cos-mode').onchange = (e) => { C.mode = e.target.value; draw(); };
  const q = new URLSearchParams(location.search);
  if (q.has('cosera')) C.era = Math.max(0, Math.min(3, parseInt(q.get('cosera'))));
  if (q.has('cosmode')) { C.mode = q.get('cosmode'); $('cos-mode').value = C.mode; }
  panels();
  if (q.has('cossteps')) { for (let i = 0; i < parseInt(q.get('cossteps')) && C.sims.std.a < 1; i++) { stepSim(C.sims.std); stepSim(C.sims.oph); } }
  if (ERAS[C.era].key === 'clump') { forces(C.sims.std); forces(C.sims.oph); }
  draw(); loop(); fpShown();
}
window.__cosmos = { C, stepSim, draw };

// ==== chapter: the first particles ====================================================================================
// OPH column, executed: the run selected at the top. Its viewer catalogue (view/catalogue.npz) holds every identity that
// appears on a stored cut (levels 9 and 10: a fixed sample) with class, birth and death depth, cause of the end, and
// birth and last position. Alive at slice T means born at depth <= T and ended after T; this reproduces the census of
// every stored cut. The screen is drawn as a Lambert equal-area disc, so an even spread on the sphere is an even spread
// on the disc. Numbers: runs/universe/L*/view/class_stats.json (screen tier), runs/universe/L*/run.json (census of the
// cuts, geometry, preparation), /particle_map.json (defect_map, level 10) and the lattice gauge study of the repair-law
// search, /rl/SIM/cf1/runs/H4_wilson_draws.json (report repair-law/reports/MA.md). Standard column: textbook facts,
// drawn as an illustration.
const FP = { on: false, seen: false, lv: [], h4: null, frames: null, R: null, T: [], k: 1, alive: [], xy: null, born: [], annXY: null,
  t0: 0, playing: false, last: 0, mr: 2.4, dirty: true, stage: 0, std: null, stdPlay: false, stdLast: 0, flash: null, stdDrawn: 0 };
const PAIRS = [     // the three kinds of pair birth, in the order of the charts (the two quark colours never touch)
  { a: 1, b: 5, ch: '0 -> 1 + 5', ann: '1 + 5 -> 0', name: 'class (1,5) pairs', one: 'class (1,5), qL candidate' },
  { a: 3, b: 3, ch: '0 -> 3 + 3', ann: '3 + 3 -> 0', name: 'class (3,3) pairs', one: 'class (3,3), lepton candidate' },
  { a: 4, b: 2, ch: '0 -> 2 + 4', ann: '2 + 4 -> 0', name: 'class (2,4) pairs', one: 'class (2,4), qR candidate' },
];
const FP_STAGES = [
  ['after inflation · model dependent', 'Reheating and a hot plasma', 'In inflationary scenarios, field energy transfers to particles and thermalizes over a model-dependent interval. This illustration assumes a later thermal plasma well above the electroweak scale, with all Standard Model degrees of freedom relativistic; it does not show the first production events.'],
  ['about 10⁻¹¹ s · 160 GeV', 'Electroweak crossover', 'The Higgs field settles to its value in the vacuum. W, Z, Higgs and top quarks become heavy. As the plasma cools further they decay away.'],
  ['about 10 µs · 155 MeV', 'Quarks bind into hadrons', 'Quarks and gluons are confined into hadrons. Hadrons and antihadrons annihilate. About one extra baryon per 1.6 billion photons is left: η is about 6.1 × 10⁻¹⁰.'],
  ['about 1 s · 1 MeV', 'Neutrinos decouple', 'Neutrinos stop scattering and stream freely from then on.'],
  ['after a few seconds', 'Electrons and positrons annihilate', 'Electron and positron pairs turn into photons. One electron per proton is left.'],
  ['about 3 minutes', 'First nuclei', 'Protons and neutrons fuse. Helium-4 takes about 0.245 of the mass, with traces of deuterium, helium-3 and lithium.'],
  ['about 380,000 years', 'Atoms and the CMB', 'Electrons bind to nuclei. The gas turns transparent, and the light of that moment is the cosmic microwave background.'],
];
const FP_KIND = {   // colour, name, marker type: f fermion (ring for the antiparticle), b boson, p photon, n nucleus, a atom
  qL: ['#ff7a45', 'left-handed quarks and antiquarks', 'f'], qR: ['#ffb347', 'right-handed quarks and antiquarks', 'f'], lep: ['#3ee6a8', 'leptons and antileptons', 'f'],
  g: ['#ff5fb0', 'gluons', 'b'], W: ['#b197ff', 'B and Wᵃ gauge fields', 'b'], H: ['#f4ead0', 'Higgs', 'b'], gam: ['#dfe9ff', 'photons', 'p'],
  nu: ['#4cc3ff', 'neutrinos, streaming freely', 'f'], e: ['#3ee6a8', 'electrons and positrons', 'f'], N: ['#ff9a6b', 'protons and neutrons', 'n'],
  p: ['#ff9a6b', 'protons and neutrons', 'n'], He: ['#ffe45c', 'helium-4 nuclei', 'n'], Ha: ['#ff9a6b', 'hydrogen atoms', 'a'], Hea: ['#ffe45c', 'helium atoms', 'a'],
};
const GROUP_LABEL = {
  '{2, 4}': 'opposite centre classes 2 and 4', '{1, 5}': 'classes 1 and 5, still reacting',
  '{3, 3}': 'two class-3 register defects', '{2, 2, 4, 4}': 'two opposite centre-class pairs', '{4, 4, 4}': 'three class-4 defects, centre neutral',
  '{2, 2, 2}': 'three class-2 defects, centre neutral', '{2, 3, 3, 4}': 'one (2,4) pair and two class-3 defects',
  '{1, 2, 3} and {3, 4, 5}': 'mixed centre-neutral triples',
};
const fpEsc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fpPct = (a, b, d = 1) => (b ? (100 * a / b).toFixed(d) : (0).toFixed(d)) + ' %';
const fpTag = (t) => `<span class="cos-tag ${{ executed: 'ok', derived: 'der', premise: 'pre', 'not derived': 'open', illustration: 'ill' }[t] || 'open'}">${t}</span>`;
const fpHash = (x) => { x = Math.imul((x ^ (x >>> 16)) >>> 0, 0x45d9f3b); x = Math.imul((x ^ (x >>> 16)) >>> 0, 0x45d9f3b); return (x ^ (x >>> 16)) >>> 0; };
const fpCol = (k) => S.pmap.class_style[String(k)].color;
const fpSum = (a) => a.reduce((s, v) => s + v, 0);
const fpPair = (P, s = 13) => marker(P.a, s) + marker(P.b, s);
const SUPER = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const fpSci = (v) => { const e = Math.floor(Math.log10(v)); return `${Math.ceil(v / 10 ** e - 1e-9)} × 10${String(e).split('').map((c) => SUPER[c]).join('')}`; };
function fpTicks(max, n = 4) {
  if (!(max > 0)) return [0, 1];
  const raw = max / n, p = 10 ** Math.floor(Math.log10(raw)), m = raw / p, step = (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p, out = [];
  for (let i = 0; i * step < max + step * 0.999; i++) out.push(+(i * step).toPrecision(10));
  return out;
}
const fpTickLab = (v, compact) => (compact && v >= 1e6 ? +(v / 1e6).toPrecision(3) + 'M' : compact && v >= 1e3 ? +(v / 1e3).toPrecision(3) + 'K' : fmt(+v.toPrecision(6)));
const fpProj = (x, y, z) => { const k = Math.sqrt(2 / Math.max(1e-9, 1 + z)) / 2; return [k * x, k * y]; };   // Lambert equal-area, whole sphere to the unit disc

async function fpLoad() {
  const get = (u) => fetch(u).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  let names = (S.runs || []).filter((r) => !r.twists && /^L\d+$/.test(r.name)).sort((a, b) => a.level - b.level).map((r) => r.name);
  if (!names.length) names = ['L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'];
  const lv = await Promise.all(names.map(async (n) => {
    const [cs, rj] = await Promise.all([get(`/runs/${n}/view/class_stats.json`), get(`/runs/${n}/run.json`)]);
    if (!cs || !rj || !cs.tiers || !cs.tiers.screen) return null;
    const ch = cs.tiers.screen.channels, cut1 = (rj.cuts || []).find((c) => c.depth === 1), f = cut1 ? cut1.counts_screen : null;
    return { name: n, level: rj.geometry.level, plaq: rj.geometry.screen_plaquettes, carriers: rj.geometry.carriers, cls: cs.tiers.screen.classes, ch,
      all: PAIRS.map((P) => ch[P.ch] || 0), ann: PAIRS.map((P) => ch[P.ann] || 0), first: f ? [f[1], f[3] / 2, f[2]] : null, firstDefects: cut1 ? cut1.screen : 0,
      prep: rj.preparation || {} };
  }));
  FP.lv = lv.filter(Boolean);
  [FP.h4, FP.tr, FP.evidence] = await Promise.all([get('/rl/SIM/cf1/runs/H4_wilson_draws.json'), get('/rl/TR/tr_exact.json'), get('run_evidence.json')]);
}
function fpWhy() {    // exact structure of the register rule on one carrier: repair-law work/TR/tr_exact.json (report TR.md)
  const el = $('cos-fp-why'), X = FP.tr; if (!el) return;
  const s = X && X.X1_superselection; if (!s || !s.lasting_33_all_proposals_raise_cost) { el.innerHTML = ''; return; }
  const rows = (X.X2_read_counts || {}).rows || [], three = rows.length > 0 && rows.every((r) => r.n_min_flow === 3 * r.d);
  const t3 = X.X3_effective_class3_hop && X.X3_effective_class3_hop.isolated_face ? X.X3_effective_class3_hop.isolated_face.sum : null, hop1 = Math.min(...(s.class1_hop_rises || [NaN]));
  el.innerHTML = `<p><b>Why</b> ${fpTag('derived')} Once the loads have settled, every nonzero repair current moves one unit of load and releases zero mismatch potential. On a carrier, each of the ${fmt(X.lasting_placements.count)} lasting placements of a class-(3,3) pair allows ${fmt(2 * X.source.seams)} unit twists, and every one of them raises the cost, by at least ${s.min_rise_over_lasting_33} units. ` +
    `So the rule applies none of them: that isolated lasting pair cannot move or annihilate under those unit proposals.${three ? ' Removing a class-(3,3) pair at separation d would take 3d unit twists.' : ''} A class-1 defect hops at ${hop1 === 0 ? 'zero cost' : 'a cost of ' + hop1}.` +
    `${t3 ? ` In a coherent version of the rule, with amplitude τ per twist and ε per unit of cost, a class-3 face hops only at third order, t3 = (${t3}) τ³/ε², and a class-1 face at first order.` : ''} ` +
    `The barrier statement is exact for the classical register rule. The third-order expression concerns a separate declared coherent extension. Both use the chosen cost table; neither identifies a physical lepton.</p>`;
}

// ---- the selected run: rows of the screen tier, their positions on the disc, and the twenty base faces ----------------
function fpBuildRun() {
  FP.frames = S.frames; FP.R = null; FP.cutCache = new Map(); FP.cutRequest = (FP.cutRequest || 0) + 1;
  if (S.meta.config && S.meta.config.twists) { fpRunHeader(); fpCensusCard(); return; }
  FP.T = S.frames.map((f) => (f.depth == null ? S.classStats.depth_at_end : f.depth));
  const sample = Math.max(1, S.meta.display_sample || 1);
  let peak = 0; for (const f of S.frames) peak = Math.max(peak, f.screen || 0);
  const stride = Math.max(1, Math.ceil(peak / sample / 16000));
  FP.R = { sample, stride, cls: new Uint8Array(0), m: 0 };
  FP.mr = Math.max(1.5, Math.min(3.4, 0.46 * Math.sqrt(Math.PI * 300 * 300 / Math.max(1, peak / sample / stride))));
  FP.k = Math.min(FP.k, FP.T.length - 1);
  fpRunHeader(); fpSetSlice(FP.k); fpCensusCard();
}
async function fpSetSlice(k) {
  const R = FP.R; if (!R) return;
  k = Math.max(0, Math.min(FP.T.length - 1, k)); FP.k = k;
  const request = ++FP.cutRequest, base = S.base, frame = FP.frames[k];
  FP.loading = true; FP.xy = null; FP.alive = []; FP.annXY = new Float32Array(0);
  const cv = $('cos-fp-oph'); fpDisc(cv.getContext('2d'), cv.width, cv.width / 2, cv.width / 2 - 16);
  const sl = $('cos-fp-slice'); sl.max = FP.T.length - 1; sl.value = k;
  $('cos-fp-read').textContent = 'Reading stored positions on this cut…';
  try {
    let cut = FP.cutCache.get(k);
    if (!cut) cut = await loadNpz(base + frame.file, ['screen_id', 'screen_cls', 'screen_xyz', 'screen_birth_depth']);
    if (request !== FP.cutRequest || R !== FP.R || base !== S.base) return;
    FP.cutCache.set(k, cut); if (FP.cutCache.size > 4) FP.cutCache.delete(FP.cutCache.keys().next().value);
    const d = displayedCut(cut, { stride: R.stride, hash: fpHash, previousDepth: k ? FP.T[k - 1] : -1 });
    R.cls = d.cls; R.m = d.ids.length; R.storedRows = d.storedRows; R.ids = d.ids;
    const xy = new Float32Array(2 * R.m);
    for (let i = 0; i < R.m; i++) { const [u, v] = fpProj(d.xyz[3 * i], d.xyz[3 * i + 1], d.xyz[3 * i + 2]); xy[2 * i] = u; xy[2 * i + 1] = v; }
    Object.assign(FP, { alive: Array.from({ length: R.m }, (_, i) => i), born: d.born, xy,
      annXY: new Float32Array(0), t0: performance.now(), dirty: true });
    fpReadout(); fpCursor();
  } catch (error) {
    if (request === FP.cutRequest) { fpPlay(false); $('cos-fp-read').textContent = 'Stored cut positions unavailable: ' + error.message; }
  } finally { if (request === FP.cutRequest) FP.loading = false; }
}
function fpRunHeader() {
  const g = S.meta.geometry, R = FP.R;
  $('cos-fp-runlab').textContent = `level ${g.level}, ${fmt(g.carriers)} carriers, the run selected at the top`;
  if (!R) {
    const cv = $('cos-fp-oph'); fpDisc(cv.getContext('2d'), cv.width, cv.width / 2, cv.width / 2 - 16);
    $('cos-fp-runtag').innerHTML = ''; $('cos-fp-read').innerHTML = 'This run is the closed-loop extension, a different law. Select one of the runs L3 to L10 at the top.'; $('cos-fp-cap').textContent = ''; return;
  }
  $('cos-fp-runtag').innerHTML = fpTag('executed');
  const audit = FP.evidence?.runs?.find((run) => run.name === S.name);
  const auditNote = audit ? (audit.data_scope.full_catalogue_local ? ' Full catalogue counts were independently recomputed.' : ' Whole-run counts are stored summaries, cross-checked against per-sweep data and local cuts; the full identity catalogue is absent locally.') : '';
  $('cos-fp-cap').innerHTML = 'Actual screen_xyz positions on each stored cut, projected with equal areas. Colours identify Z6 register classes; quark and lepton names are candidates. White halos mark surviving identities born since the previous stored cut. ' +
    `The files keep identities divisible by ${R.sample}; the display additionally selects about 1 in ${R.stride}. Counts below come from the full-run census. ` +
    'The frames are snapshots, with no invented path or annihilation location between them. A causal cut is not an equal local-clock time. ' +
    `<a href="${S.base}run.json">Run metadata</a> · <a href="run_evidence.json">audited counts and SHA-256 sources</a>.${auditNote}`;
}
function fpReadout() {
  const R = FP.R, f = FP.frames[FP.k], T = FP.T[FP.k], c = f.counts_screen, end = FP.k === FP.T.length - 1, g = S.meta.geometry;
  let h = `<b>${end ? 'End of the run' : `Slice ${FP.k + 1} of ${FP.T.length}`}</b> · causal depth ${fmt(T)} · ${fmt(f.screen)} defects on ${fmt(g.screen_plaquettes)} plaquettes`;
  h += `<div class="cos-fp-key">${[1, 5, 4, 2, 3].map((k) => `<span>${marker(k, 12)}${fmt(c[k])} class ${k}</span>`).join('')}</div>`;
  if (T === 1) h += `<div>The first slice: ${fmt(c[1] + c[2] + c[3] / 2)} pairs have appeared on this consistent cut. Local incident clocks range from ${fmt(f.ticks_min)} to ${fmt(f.ticks_max)} attempts; no cosmic time is assigned. ${fpTag('executed')}</div>`;
  $('cos-fp-read').innerHTML = h;
}

// ---- drawing ------------------------------------------------------------------------------------------------------------
function fpDisc(g, W, C, rad) {
  g.globalAlpha = 1; g.fillStyle = '#04060d'; g.fillRect(0, 0, W, W);
  const gr = g.createRadialGradient(C, C, rad * 0.05, C, C, rad); gr.addColorStop(0, 'rgba(38,56,104,.40)'); gr.addColorStop(1, 'rgba(18,28,56,.28)');
  g.fillStyle = gr; g.beginPath(); g.arc(C, C, rad, 0, 2 * Math.PI); g.fill(); g.strokeStyle = 'rgba(147,163,192,.4)'; g.lineWidth = 1; g.stroke();
}
function fpDrawOPH(now) {
  const cv = $('cos-fp-oph'); if (!cv || !FP.R || !FP.xy) return;
  const g = cv.getContext('2d'), W = cv.width, C = W / 2, rad = W / 2 - 16, r = FP.mr, xy = FP.xy, al = FP.alive, R = FP.R;
  fpDisc(g, W, C, rad);
  for (const k of [2, 4, 3, 1, 5]) {            // frozen kinds first, the moving ones on top
    const st = S.pmap.class_style[String(k)], idx = []; for (let i = 0; i < al.length; i++) if (R.cls[al[i]] === k) idx.push(i);
    if (!idx.length) continue;
    const path = (rr) => { g.beginPath(); for (const i of idx) { const x = C + rad * xy[2 * i], y = C - rad * xy[2 * i + 1]; g.moveTo(x + rr, y); g.arc(x, y, rr, 0, 2 * Math.PI); } };
    if (st.shape === 'ring') { path(r * 0.9); g.strokeStyle = st.color; g.lineWidth = Math.max(1, r * 0.42); g.stroke(); }
    else if (st.shape === 'duo') { path(r * 0.6); g.fillStyle = st.color; g.fill(); path(r); g.strokeStyle = st.rim || st.color; g.lineWidth = Math.max(0.8, r * 0.3); g.stroke(); }
    else { path(r); g.fillStyle = st.color; g.fill(); }
  }
  const age = now - FP.t0;
  if (age < 1200 && FP.born.length) {
    const f = 1 - age / 1200; g.strokeStyle = `rgba(255,255,255,${(0.55 * f).toFixed(3)})`; g.lineWidth = 1; g.beginPath();
    for (const i of FP.born) { const x = C + rad * xy[2 * i], y = C - rad * xy[2 * i + 1], rr = r + 1.6 + 3 * (1 - f); g.moveTo(x + rr, y); g.arc(x, y, rr, 0, 2 * Math.PI); }
    g.stroke();
  }
  if (age < 1600 && FP.annXY.length) {
    const f = 1 - age / 1600, a = FP.annXY; g.strokeStyle = `rgba(255,255,255,${(0.9 * f).toFixed(3)})`; g.lineWidth = 1.4; g.beginPath();
    for (let i = 0; i < a.length / 2; i++) { const x = C + rad * a[2 * i], y = C - rad * a[2 * i + 1], rr = r + 2 + 9 * (1 - f); g.moveTo(x + rr, y); g.arc(x, y, rr, 0, 2 * Math.PI); }
    g.stroke();
  }
}
function fpStdBuild() {             // fixed random places in the disc; the kind of each marker in each of the seven stages
  const N = 2200, P = [], shares = hotWeights('number'), tot = shares.reduce((sum, [, n]) => sum + n, 0);
  const h = (i, s) => fpHash(i * 7919 + s * 104729 + 17) / 4294967296;
  let nN = 0;                       // nucleons in index order: every 13th becomes a helium-4 nucleus, about 0.08 of the nuclei
  for (let i = 0; i < N; i++) {
    const rr = Math.sqrt(h(i, 1)), a = 2 * Math.PI * h(i, 2), anti = h(i, 4) < 0.5, w = h(i, 5), w2 = h(i, 6), w3 = h(i, 7), ks = [];
    let u = h(i, 3) * tot, k = 'H'; for (const [kk, sh] of shares) { if (u < sh) { k = kk; break; } u -= sh; }
    for (let s = 0; s < FP_STAGES.length; s++) {
      if (s === 1 && (k === 'W' || k === 'H')) k = 'gam';
      if (s === 2 && (k === 'qL' || k === 'qR' || k === 'g')) k = w < 0.025 ? 'N' : 'gam';
      if (s === 3 && k === 'lep') k = w2 < 0.5 ? 'nu' : 'e';
      if (s === 4 && k === 'e') k = !anti && w3 < 0.5 ? 'e' : 'gam';
      if (s === 5 && k === 'N') k = nN++ % 13 === 6 ? 'He' : 'p';
      if (s === 6) k = k === 'p' ? 'Ha' : k === 'He' ? 'Hea' : k === 'e' ? null : k;
      ks.push(k);
    }
    P.push({ x: rr * Math.cos(a), y: rr * Math.sin(a), anti, ks, ph: 2 * Math.PI * h(i, 8), om: 0.6 + 1.2 * h(i, 9) });
  }
  FP.std = P;
}
function fpDrawStd(now) {
  const cv = $('cos-fp-std'); if (!cv || !FP.std) return;
  const g = cv.getContext('2d'), W = cv.width, C = W / 2, rad = W / 2 - 16, t = now / 1000, s = FP.stage, groups = new Map();
  fpDisc(g, W, C, rad);
  const at = (P) => [C + rad * (P.x + 0.006 * Math.sin(t * P.om + P.ph)), C - rad * (P.y + 0.006 * Math.cos(t * P.om * 1.3 + P.ph))];
  FP.std.forEach((P, j) => { const k = P.ks[s]; if (!k) return; const key = k + (P.anti && FP_KIND[k][2] === 'f' && !(s >= 4 && k === 'e') ? '~' : ''); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(j); });
  for (const key of [...groups.keys()].sort((a, b) => (FP_KIND[a.replace('~', '')][2] === 'p' ? -1 : 0) - (FP_KIND[b.replace('~', '')][2] === 'p' ? -1 : 0))) {
    const k = key.replace('~', ''), anti = key.endsWith('~'), [col, , type] = FP_KIND[k], js = groups.get(key);
    const r = type === 'p' ? 1.2 : type === 'b' ? 1.8 : type === 'n' || type === 'a' ? (k === 'He' || k === 'Hea' ? 4.4 : 3.2) : 2.5;
    g.globalAlpha = k === 'nu' ? 0.45 : type === 'p' ? 0.55 : 0.92; g.beginPath();
    for (const j of js) { const [x, y] = at(FP.std[j]); g.moveTo(x + r, y); g.arc(x, y, r, 0, 2 * Math.PI); }
    if (anti) { g.strokeStyle = col; g.lineWidth = 1.2; g.stroke(); } else { g.fillStyle = col; g.fill(); }
    if (type === 'a') { g.globalAlpha = 0.35; g.beginPath(); for (const j of js) { const [x, y] = at(FP.std[j]); g.moveTo(x + r + 3, y); g.arc(x, y, r + 3, 0, 2 * Math.PI); } g.strokeStyle = '#dfe9ff'; g.lineWidth = 1; g.stroke(); }
  }
  g.globalAlpha = 1;
  const fl = FP.flash; if (fl && now - fl.t0 < 1100) {
    const f = 1 - (now - fl.t0) / 1100; g.strokeStyle = `rgba(255,255,255,${(0.8 * f).toFixed(3)})`; g.lineWidth = 1.2; g.beginPath();
    for (const j of fl.list) { const [x, y] = at(FP.std[j]), rr = 3 + 8 * (1 - f); g.moveTo(x + rr, y); g.arc(x, y, rr, 0, 2 * Math.PI); }
    g.stroke();
  }
}
function fpStage(i) {
  const prev = FP.stage; FP.stage = i;
  const fl = []; if (i === prev + 1) FP.std.forEach((P, j) => { const a = P.ks[prev], b = P.ks[i]; if (a !== b && a) fl.push(j); });
  FP.flash = { t0: performance.now(), list: fl };
  document.querySelectorAll('#cos-fp-steps li').forEach((li, j) => li.classList.toggle('on', j === i));
  $('cos-fp-stdtext').textContent = FP_STAGES[i][2];
  const seen = new Set(FP.std.map((P) => P.ks[i]).filter(Boolean));
  $('cos-fp-stdkey').innerHTML = Object.keys(FP_KIND).filter((k) => seen.has(k)).map((k) => `<span><i class="cos-fp-dot" style="background:${FP_KIND[k][0]}"></i>${FP_KIND[k][1]}</span>`).join('');
}

// ---- the four cards -----------------------------------------------------------------------------------------------------
function fpPairsSVG() {
  const L = FP.lv, W = 620, lab = 62, bw = 240, gap = 36, x1 = lab, x2 = lab + bw + gap, top = 26, rh = 22, bh = 14, H = top + L.length * rh;
  let s = `<svg class="cos-fp-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Share of the three kinds of pair birth per level, in the first slice and over the whole run">` +
    `<text x="${x1}" y="13" class="cos-fp-ax">first slice of the run</text><text x="${x2}" y="13" class="cos-fp-ax">all pair births of the run</text>`;
  L.forEach((l, i) => {
    const y = top + i * rh, last = i === L.length - 1;
    s += `<text x="${lab - 10}" y="${y + bh - 3}" text-anchor="end" class="cos-fp-ax">level ${l.level}</text>`;
    for (const [v, x0, what] of [[l.first, x1, 'first slice'], [l.all, x2, 'whole run']]) {
      if (!v) continue; const tot = fpSum(v); let x = x0;
      PAIRS.forEach((P, j) => {
        const w = bw * v[j] / tot;
        if (w >= 1) {
          const ww = Math.max(1, w - 2);
          s += `<rect x="${x.toFixed(2)}" y="${y}" width="${ww.toFixed(2)}" height="${bh}" rx="3" fill="${fpCol(P.a)}" tabindex="0" data-tip="${fpEsc(`Level ${l.level}, ${what}: ${fmt(v[j])} ${P.name} of ${fmt(tot)}, ${fpPct(v[j], tot)}`)}"/>`;
          if (last && ww > 46) s += `<text x="${(x + 5).toFixed(1)}" y="${y + bh - 3.5}" class="cos-fp-in">${fpPct(v[j], tot)}</text>`;
        }
        x += w;
      });
    }
  });
  return s + '</svg>';
}
function fpPairsCard() {
  const L = FP.lv, top = L[L.length - 1]; if (!top || !top.first) return;
  const a = top.all, ta = fpSum(a), f = top.first, tf = fpSum(f), fi = f.indexOf(Math.max(...f)), ai = a.indexOf(Math.max(...a));
  const big = L.filter((l) => l.level >= 5), dev = Math.max(...big.flatMap((l) => PAIRS.map((P, j) => Math.abs(100 * l.all[j] / fpSum(l.all) - 100 * a[j] / ta))));
  const rows = L.map((l) => `<tr><td>level ${l.level}</td>${(l.first || [null, null, null]).map((v) => `<td class="num">${v == null ? '' : fmt(v)}</td>`).join('')}${l.all.map((v) => `<td class="num">${fmt(v)}</td>`).join('')}</tr>`).join('');
  $('cos-fp-pairs').innerHTML = `<h3>Which pairs come first ${fpTag('executed')}</h3>
    <div class="cos-fp-key">${PAIRS.map((P) => `<span>${fpPair(P)}${P.one} (${P.a} + ${P.b})</span>`).join('')}</div>
    <p>The register starts at zero. An accepted current-driven twist on two blank faces creates conjugate Z6 defects: a current of one, two or three units produces classes (1,5), (2,4) or (3,3). These are event-born bookkeeping structures during the prepared load quench. The rule has no vacuum pair-production amplitude or cosmological reheating history.</p>
    ${fpPairsSVG()}
    <p>At level ${top.level} the first slice alone makes ${fmt(tf)} of the ${fmt(ta)} pair births, ${fpPct(tf, ta)}. In that slice ${PAIRS[fi].name} lead with ${fpPct(f[fi], tf)}. Over the whole run ${PAIRS[ai].name} lead: ${PAIRS.map((P, j) => `${fpPct(a[j], ta)} ${P.name}`).join(', ')}. From level 5 up the shares of the whole run agree within ${dev.toFixed(1)} percentage points.</p>
    <details><summary>Numbers</summary><table class="t"><tr><th></th><th colspan="3">first slice: pairs ${PAIRS.map((P) => `${P.a} + ${P.b}`).join(', ')}</th><th colspan="3">whole run: pairs ${PAIRS.map((P) => `${P.a} + ${P.b}`).join(', ')}</th></tr>${rows}</table>
    <div class="note">Sources: view/class_stats.json of the runs L3 to L10 (screen tier, pair births from blank faces) and their run.json (census of the cut at causal depth 1, where every defect is new; a class-(3,3) pair contains two identities).</div></details>`;
}
function fpCursor() {
  const c = $('cos-fp-cur'); if (!c || !FP.cx) return;
  const x = FP.cx(FP.T[FP.k]).toFixed(1); c.setAttribute('x1', x); c.setAttribute('x2', x);
}
function fpCensusCard() {
  const el = $('cos-fp-census'); if (!FP.R) { el.innerHTML = ''; return; }
  const F = FP.frames, T = FP.T, cs = S.classStats.tiers.screen, C = cs.classes, ch = cs.channels, lvl = S.meta.geometry.level;
  const ser = [[1, 5], [3], [4, 2]].map((ks) => F.map((f) => ks.reduce((s, k) => s + f.counts_screen[k], 0)));
  const names = ['classes 1 and 5 (qL candidates)', 'class 3 (lepton candidate)', 'classes 4 and 2 (qR candidates)'], cols = [fpCol(1), fpCol(3), fpCol(4)];
  const W = 620, l = 50, r = 92, t = 12, b = 34, pw = W - l - r, ph = 200, H = t + ph + b, Tmax = T[T.length - 1];
  const lx = (d) => l + pw * Math.log10(1 + d) / Math.log10(1 + Tmax); FP.cx = lx;
  const ticks = fpTicks(Math.max(...ser.flat()), 4), ytop = ticks[ticks.length - 1], ly = (v) => t + ph * (1 - v / ytop), compact = ytop >= 1e4;
  let s = `<svg class="cos-fp-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Number of each kind on the screen at every stored slice">`;
  for (const v of ticks) s += `<line x1="${l}" x2="${l + pw}" y1="${ly(v).toFixed(1)}" y2="${ly(v).toFixed(1)}" stroke="rgba(147,163,192,.14)"/><text x="${l - 6}" y="${(ly(v) + 4).toFixed(1)}" text-anchor="end" class="cos-fp-ax">${fpTickLab(v, compact)}</text>`;
  for (const d of [1, 10, 100, 1000, 10000, 100000].filter((d) => d <= Tmax)) s += `<line x1="${lx(d).toFixed(1)}" x2="${lx(d).toFixed(1)}" y1="${t + ph}" y2="${t + ph + 4}" stroke="rgba(147,163,192,.5)"/><text x="${lx(d).toFixed(1)}" y="${t + ph + 16}" text-anchor="middle" class="cos-fp-ax">${fmt(d)}</text>`;
  s += `<text x="${l + pw / 2}" y="${H - 3}" text-anchor="middle" class="cos-fp-ax">causal depth, log scale</text><line x1="${l}" x2="${l + pw}" y1="${t + ph}" y2="${t + ph}" stroke="rgba(147,163,192,.35)"/>`;
  ser.forEach((v, i) => { s += `<polyline fill="none" stroke="${cols[i]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${v.map((y, j) => `${lx(T[j]).toFixed(1)},${ly(y).toFixed(1)}`).join(' ')}"/>`; });
  const ends = ser.map((v, i) => ({ i, y: ly(v[v.length - 1]), v: v[v.length - 1] })).sort((p, q) => p.y - q.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 13);
  for (const e of ends) s += `<circle cx="${l + pw + 8}" cy="${(e.y).toFixed(1)}" r="4" fill="${cols[e.i]}"/><text x="${l + pw + 16}" y="${(e.y + 4).toFixed(1)}" class="cos-fp-ax">${fmt(e.v)}</text>`;
  s += `<line id="cos-fp-cur" x1="0" x2="0" y1="${t}" y2="${t + ph}" stroke="rgba(255,255,255,.55)" stroke-width="1"/><line id="cos-fp-hl" x1="0" x2="0" y1="${t}" y2="${t + ph}" stroke="rgba(255,255,255,.25)" stroke-width="1" visibility="hidden"/>`;
  s += `<rect id="cos-fp-hov" x="${l}" y="${t}" width="${pw}" height="${ph}" fill="transparent"/></svg>`;
  FP.census = { ser, names, W };
  const mv = (k) => { const v = C[String(k)].moved_fraction; return fpPct(v, 1, v < 0.01 ? 2 : 1); };
  const b15 = ch['0 -> 1 + 5'] || 0, a15 = ch['1 + 5 -> 0'] || 0, al = (k) => fmt(C[String(k)].alive_at_end);
  const sub = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].filter((i) => i < F.length - 1).concat([F.length - 1]);
  el.innerHTML = `<h3>Who keeps moving and who stays ${fpTag('executed')}</h3>
    <div class="cos-fp-key">${names.map((n, i) => `<span><i class="cos-fp-dot" style="background:${cols[i]}"></i>${n}</span>`).join('')}</div>
    <p>Class 1 is mobile: ${mv(1)} of its identities move at least once. There are ${Math.round(100 * a15 / Math.max(1, b15))} annihilation events per 100 pair births; this compares channel totals, not matched birth cohorts. Classes 4 and 3 are much less mobile: ${mv(4)} and ${mv(3)} of their identities ever move. Their annihilation channels are rare: ${fmt(ch['2 + 4 -> 0'] || 0)} against ${fmt(ch['0 -> 2 + 4'] || 0)} pair births, and ${fmt(ch['3 + 3 -> 0'] || 0)} against ${fmt(ch['0 -> 3 + 3'] || 0)}.</p>
    ${s}
    <p>Left at the end of level ${lvl}: ${al(1)} and ${al(5)} of classes 1 and 5, against ${al(4)}, ${al(2)} and ${al(3)} of classes 4, 2 and 3. These are surviving register defects, censored at the run end; they are not an identified thermal relic population. The white line is the slice shown in the disc.</p>
    <div id="cos-fp-why"></div>
    <details><summary>Numbers</summary><table class="t"><tr><th>causal depth</th>${names.map((n) => `<th>${n}</th>`).join('')}</tr>${sub.map((i) => `<tr><td class="num">${i === F.length - 1 ? 'end, ' : ''}${fmt(T[i])}</td>${ser.map((v) => `<td class="num">${fmt(v[i])}</td>`).join('')}</tr>`).join('')}</table>
    <div class="note">Sources: run.json of level ${lvl} (census of the screen tier on every stored cut) and its view/class_stats.json. The table lists the first slices and the end. The paragraph Why: repair-law/work/TR/tr_exact.json, report repair-law/reports/TR.md.</div></details>`;
  fpCursor(); fpWhy();
}
function fpBalanceCard() {
  const L = FP.lv.find((l) => l.level === 10) || FP.lv[FP.lv.length - 1]; if (!L) return;
  const C = L.cls, al = (k) => C[String(k)].alive_at_end, dm = S.pmap.defect_map || {}, G = dm.groups || [];
  const g4 = G.find((g) => g.set === '{4, 4, 4}'), g2 = G.find((g) => g.set === '{2, 2, 2}');
  const lv = /differ by ([\d.]+ ± [\d.]+) percent/.exec(dm.balance || ''), sets = (str) => [...String(str).matchAll(/\{([^}]*)\}/g)].map((m) => m[1].split(',').map((x) => parseInt(x)));
  const withDef = dm.carriers && dm.empty_share != null ? Math.round(dm.carriers * (1 - dm.empty_share)) : null, maxShare = Math.max(...G.map((g) => g.share));
  const row = (mk, name, a, b) => `<div class="cos-fp-brow"><span>${mk}</span><span>${name}</span><b>${fmt(a)}</b><b>${fmt(b)}</b></div>`;
  const grp = G.map((g) => `<div class="cos-fp-grp"><span>${sets(g.set).map((st) => st.map((k) => marker(k, 11)).join('')).join('<em>or</em>')}</span><span>${fpEsc(GROUP_LABEL[g.set] || g.name)}</span>` +
    `<span class="cos-fp-bar"><i style="width:${(100 * g.share / maxShare).toFixed(1)}%"></i></span><span class="num">${fmt(g.carriers)} · ${fpPct(g.share, 1)}</span></div>`).join('');
  const d = g4 && g2 ? g4.carriers - g2.carriers : 0, spread = g4 && g2 ? Math.sqrt(g4.carriers + g2.carriers) : 0;
  $('cos-fp-bal').innerHTML = `<h3>Matter and antimatter</h3>
    <div class="cos-fp-two"><div class="cos-fp-stat"><div class="k" style="margin:0 0 4px;color:#ffb07a">Standard model</div>
      <p>Particles and antiparticles nearly balance at first. About one extra baryon per 1.6 billion photons survives the annihilations, η about 6.1 × 10⁻¹⁰. That excess accounts for the net baryonic component, not all cosmic matter. What made it is an open question.</p></div>
    <div class="cos-fp-stat"><div style="display:flex;align-items:center;gap:8px;margin:0 0 4px"><span class="k" style="margin:0;color:#b197ff">OPH run, level ${L.level}, end</span>${fpTag('executed')}</div>
      <div class="cos-fp-brow cos-fp-bhead"><span></span><span></span><b>class</b><b>conjugate</b></div>
      ${row(marker(1, 12) + marker(5, 12), 'classes 1 and 5', al(1), al(5))}${row(marker(4, 12) + marker(2, 12), 'classes 4 and 2', al(4), al(2))}
      ${g4 && g2 ? row(marker(4, 10) + marker(4, 10) + marker(4, 10), '{4,4,4} and {2,2,2} carrier counts', g4.carriers, g2.carriers) : ''}
      <p>The class-4 triples ${d < 0 ? 'trail the class-2 triples' : 'lead the class-2 triples'} by ${fmt(Math.abs(d))}, compared with an illustrative independent-Poisson scale √(N₄ + N₂) = ${fmt(Math.round(spread))}. Correlated carrier counts have no validated uncertainty model here.${lv ? ` Over levels 5 to 10 the two differ by ${lv[1]} percent.` : ''} This centre-class census does not establish baryon number or a matter excess. An excess like the measured one is not derived. ${fpTag('not derived')}</p></div></div>
    <p style="margin-top:10px">Groups on one carrier at level 10. The twenty face classes sum to zero modulo six. This centre neutrality does not establish a full gauge singlet or a fermionic baryon; the flux groups themselves are bookkeeping defects. ${fpTag('executed')}</p>
    ${grp}
    <div class="note">Share of the ${withDef ? fmt(withDef) + ' ' : ''}carriers that hold any defect${dm.empty_share != null ? `; ${fpPct(dm.empty_share, 1)} of all ${fmt(dm.carriers)} carriers hold none` : ''}. Sources: view/class_stats.json of level ${L.level} and defect_map in /particle_map.json.</div>`;
}
function fpConfSVG(pts, cols) {
  const W = 620, l = 50, r = 70, t = 12, b = 34, pw = W - l - r, ph = 190, H = t + ph + b, Rs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const ser = pts.map((p) => Rs.map((R) => p.V_minus_V1_T8[String(R)]));
  const ticks = fpTicks(Math.max(...ser.flat().map((v) => v[0] + (v[1] || 0))), 4), ytop = ticks[ticks.length - 1];
  const lx = (R) => l + pw * (R - 1) / 9, ly = (v) => t + ph * (1 - v / ytop);
  let s = `<svg class="cos-fp-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Finite Wilson-loop decay statistic in classical Z3 and Z2 ensembles">`;
  for (const v of ticks) s += `<line x1="${l}" x2="${l + pw}" y1="${ly(v).toFixed(1)}" y2="${ly(v).toFixed(1)}" stroke="rgba(147,163,192,.14)"/><text x="${l - 6}" y="${(ly(v) + 4).toFixed(1)}" text-anchor="end" class="cos-fp-ax">${fpTickLab(v)}</text>`;
  for (const R of Rs) s += `<text x="${lx(R).toFixed(1)}" y="${t + ph + 16}" text-anchor="middle" class="cos-fp-ax">${R}</text>`;
  s += `<line x1="${l}" x2="${l + pw}" y1="${t + ph}" y2="${t + ph}" stroke="rgba(147,163,192,.35)"/><text x="${l + pw / 2}" y="${H - 3}" text-anchor="middle" class="cos-fp-ax">loop width R, in lattice cells</text>`;
  ser.forEach((v, i) => {
    s += `<polyline fill="none" stroke="${cols[i]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${v.map((p, j) => `${lx(Rs[j]).toFixed(1)},${ly(p[0]).toFixed(1)}`).join(' ')}"/>`;
    v.forEach((p, j) => {
      if (p[1]) s += `<line x1="${lx(Rs[j]).toFixed(1)}" x2="${lx(Rs[j]).toFixed(1)}" y1="${ly(p[0] - p[1]).toFixed(1)}" y2="${ly(p[0] + p[1]).toFixed(1)}" stroke="${cols[i]}" stroke-width="1.2" opacity=".7"/>`;
      s += `<circle cx="${lx(Rs[j]).toFixed(1)}" cy="${ly(p[0]).toFixed(1)}" r="4" fill="${cols[i]}" stroke="#101a2e" stroke-width="2" tabindex="0" data-tip="${fpEsc(`${i ? 'weak centre' : 'colour centre'}, R = ${Rs[j]}: ${p[0].toFixed(4)} ± ${(p[1] || 0).toFixed(4)}`)}"/>`;
    });
    const e = v[v.length - 1]; s += `<text x="${l + pw + 10}" y="${(ly(e[0]) + 4).toFixed(1)}" class="cos-fp-ax">${i ? 'weak' : 'colour'}</text>`;
  });
  return s + '</svg>';
}
function fpConfCard() {
  const el = $('cos-fp-conf'), H4 = FP.h4, pt = (lab) => H4 && H4.points && H4.points.find((p) => p.label === lab);
  const p3 = pt('Z3_carrier'), p2 = pt('Z2_carrier'), p2p = pt('Z2_plaquette');
  if (!p3 || !p2) { el.innerHTML = '<h3>Classical centre-gauge loop comparison</h3><p>The reference ensemble is not available.</p>'; return; }
  const s3 = p3.sigma_from_creutz_R5to8, s2 = p2.sigma_from_creutz_R5to8, s2p = p2p && p2p.sigma_from_creutz_R5to8, cols = [fpCol(4), fpCol(3)];
  const rows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((R) => `<tr><td class="num">${R}</td>${[p3, p2].map((p) => { const v = p.V_minus_V1_T8[String(R)]; return `<td class="num">${v[0].toFixed(4)} ± ${(v[1] || 0).toFixed(4)}</td>`; }).join('')}</tr>`).join('');
  el.innerHTML = `<h3>Classical centre-gauge loop comparison ${fpTag('computed comparison')}</h3>
    <div class="cos-fp-key"><span><i class="cos-fp-dot" style="background:${cols[0]}"></i>colour centre, the flux of classes 2 and 4 (Z3)</span><span><i class="cos-fp-dot" style="background:${cols[1]}"></i>weak centre, the flux of class 3 (Z2)</span></div>
    <p>A separate three-dimensional classical Gibbs study draws Z3 and Z2 link fields from exp(κ Σ Re Wp). Its κ values are tuned to a selected density normalization of the run. The finite Wilson loops show area-law-compatible Z3 decay and perimeter-law-compatible Z2 decay over the reported range. This is not the run’s own dynamics, a full SU(3) quantum gauge theory, or a derived phase of nature.</p>
    ${fpConfSVG([p3, p2], cols)}
    <p>Colour centre at κ* = ${p3.kappa}: ${s3.law} law, √σ = ${s3.sqrt_sigma.toFixed(3)} per cell. Weak centre at κ* = ${p2.kappa}: ${s2.law.split(' (')[0]} law, σ below ${fpSci(s2.sigma_upper_bound_2err)}. These fitted finite-loop quantities are in the declared ensemble’s lattice units; no physical string tension or GeV scale is inferred.</p>
    <p class="note">The link fields come from a declared plaquette action tuned to the density of the run, not from the run's own dynamics${s2p && s2p.sqrt_sigma ? `, and with the density counted per plaquette instead of per carrier the weak centre is confined too, close to its transition (√σ = ${s2p.sqrt_sigma.toFixed(3)})` : ''}.</p>
    <details><summary>Numbers</summary><table class="t"><tr><th>R</th><th>colour centre: V(R) − V(1)</th><th>weak centre: V(R) − V(1)</th></tr>${rows}</table>
    <div class="note">V(R) is the finite-length logarithmic Wilson-loop ratio at length 8, with mean and standard error (sample scatter / √5) from five draws of 96 × 48 × 48 carriers. Source: repair-law/work/SIM/cf1/runs/H4_wilson_draws.json, report repair-law/reports/MA.md.</div></details>`;
}
function fpTexts() {
  const L = FP.lv.find((l) => l.level === 10) || FP.lv[FP.lv.length - 1];
  if (L) {
    $('cos-fp-lead').innerHTML = `The level ${L.level} run starts from iid integer loads and a blank Z6 register. Its first consistent cut contains ${fmt(L.firstDefects)} screen defects ${fpTag('executed')}. ` +
      `The three pair channels are register classes; qL, qR and lepton are candidate assignments, not identified Standard Model particles. ` +
      `The preparation supplies statistical homogeneity. The simulator derives no inflation, reheating, expansion temperature or physical cosmic age. ` +
      `The comparison illustrates a possible thermal history after reheating; its duration and initial species depend on the inflationary model. The observed 10⁻⁵ anisotropy belongs to the CMB at last scattering, not an instantaneous reheating measurement. ` +
      `<a href="run_evidence.json">Run evidence, exact counts and source hashes</a>.`;
    fpTable(L);
  }
  fpPairsCard(); fpBalanceCard(); fpConfCard(); fpWhy(); if (FP.R) fpRunHeader();
}
function fpTable(L) {
  const a = L.all, ta = fpSum(a), f = L.first || [0, 0, 0], tf = fpSum(f), C = L.cls, al = (k) => fmt(C[String(k)].alive_at_end), ch = L.ch;
  const G = (S.pmap.defect_map || {}).groups || [], g4 = G.find((g) => g.set === '{4, 4, 4}'), g2 = G.find((g) => g.set === '{2, 2, 2}');
  const p3 = FP.h4 && FP.h4.points && FP.h4.points.find((p) => p.label === 'Z3_carrier'), sq = p3 ? p3.sigma_from_creutz_R5to8.sqrt_sigma.toFixed(3) : null;
  const R = [
    ['Where they appear', `The first consistent cut of level ${L.level} holds ${fmt(L.firstDefects)} defects on ${fmt(L.plaq)} plaquettes. Local clocks need not be equal on that cut.`, 'executed', 'Reheating and thermalization occur over a model-dependent interval; no universal first-particle surface is measured.', `run.json of level ${L.level}, causal cut 1`],
    ['Why everywhere', 'Every port starts with an independent load in 0..5; the register starts blank. Statistical homogeneity is prepared.', 'premise', 'Inflation can produce a nearly homogeneous background. The observed CMB temperature anisotropy is about 10⁻⁵ at last scattering.', `run.json preparation ${fpEsc(L.prep.name || '')}; NASA CMB source linked above`],
    ['What comes first', 'Accepted twists create conjugate centre-class pairs (1,5), (2,4), (3,3). Particle species assignments remain candidates.', 'executed', 'Production channels depend on the reheating model. All Standard Model degrees are populated only in a sufficiently hot, thermalized plasma.', 'class_stats.json, pair births; PDG reheating'],
    ['In what proportion', `${PAIRS.map((P, j) => `${fpPct(a[j], ta)} ${P.name}`).join(', ')}, of ${fmt(ta)} pair births at level ${L.level}. First-cut shares: ${PAIRS.map((P, j) => fpPct(f[j], tf)).join(', ')}.`, 'executed', 'Ideal relativistic particle counts weight fermion degrees by 3/4; energy density weights them by 7/8. Finite masses and non-equilibrium production change the proportions.', 'class_stats.json and run.json; PDG thermal distributions'],
    ['What happens next', `Class (1,5): ${fmt(ch['1 + 5 -> 0'] || 0)} annihilations versus ${fmt(a[0])} pair births. Classes (2,4) and (3,3): ${fmt(ch['2 + 4 -> 0'] || 0)} and ${fmt(ch['3 + 3 -> 0'] || 0)} annihilations. Most heavy-class identity endings are fusion, split or exchange.`, 'executed', 'The expanding plasma cools; interactions, masses and expansion determine annihilation and freeze-out.', `class_stats.json of level ${L.level}`],
    ['What is left', `Surviving defects: class 4 ${al(4)}, class 2 ${al(2)}, class 3 ${al(3)}, class 1 ${al(1)}, class 5 ${al(5)}. Their survival is censored at the run end.`, 'executed', 'Baryons, electrons, photons and relic neutrinos, among other cosmic components.', `class_stats.json of level ${L.level}`],
    ...(FP.tr && FP.tr.X1_superselection && FP.tr.X1_superselection.lasting_33_all_proposals_raise_cost ? [['Why class 3 can be pinned',
      `After balance there is no released mismatch work. Every tested unit proposal on an isolated lasting carrier pair raises the declared cost by at least ${FP.tr.X1_superselection.min_rise_over_lasting_33} units, so it is rejected. This carrier result does not identify leptons or prove that all screen defects stay fixed.`,
      'derived', 'Thermal freeze-out compares an interaction rate with the cosmic expansion rate.', 'repair-law work/TR/tr_exact.json; PDG cosmology']] : []),
    ['Matter and antimatter', g4 && g2 ? `Centre-class contents {4,4,4}: ${fmt(g4.carriers)} carriers; {2,2,2}: ${fmt(g2.carriers)}. These are not identified baryons. Finite conjugate counts need not be equal.` : 'No physical baryon count is supplied.', 'executed', 'The measured baryon asymmetry requires a physical baryogenesis mechanism.', 'defect_map in /particle_map.json'],
    ['Excess of matter', 'The register has no baryon-number observable. A physical baryon asymmetry is not derived.', 'not derived', 'About one extra baryon per 1.6 billion photons, η about 6.1 × 10⁻¹⁰.', 'defect_map in /particle_map.json'],
    ['Centre-gauge comparison', `Separate classical Gibbs ensemble: finite Z3 loops fit an area law${sq ? ` with √σ = ${sq} per cell` : ''}; the selected Z2 ensemble fits a perimeter law. Density normalization and action are declared.`, 'computed comparison', 'The QCD crossover and electroweak phase require their own full quantum theories; centre labels alone do not establish either.', '<a href="/rl/SIM/cf1/runs/H4_wilson_draws.json">H4 finite Wilson-loop data</a>'],
    ['Clock', 'Causal depth, incident seam ticks (holds included), and solver sweeps are distinct. No temperature or cosmic seconds are derived.', 'not derived', '160 GeV at about 10⁻¹¹ s, 155 MeV at about 10 µs, 1 MeV at about 1 s.', 'run.json'],
    ['First nuclei and atoms', 'Not computed. The thermal history is imported.', 'not derived', 'Helium-4 takes about 0.245 of the mass at 3 minutes. Atoms and the CMB at about 380,000 years.', 'cosmology papers, rows above'],
  ];
  $('cos-fp-table').innerHTML = '<tr><th></th><th>OPH run or named comparison</th><th>status</th><th>standard model of cosmology</th><th>source of the OPH entry</th></tr>' +
    R.map((r) => `<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td>${fpTag(r[2])}</td><td>${r[3]}</td><td class="small">${r[4]}</td></tr>`).join('');
}

// ---- page wiring --------------------------------------------------------------------------------------------------------
function fpSkeleton() {
  $('cos-first').innerHTML = `<h2>The first defects and the early plasma</h2>
  <p class="cos-fp-lead" id="cos-fp-lead">Reading the runs.</p>
  <div class="cos-grid cos-fp-grid">
    <div class="cos-col cos-fp-col"><h3><span style="color:#b197ff">OPH run</span><span class="cos-fp-sub" id="cos-fp-runlab"></span><span id="cos-fp-runtag"></span></h3>
      <canvas id="cos-fp-oph" width="640" height="640"></canvas>
      <div class="cos-fp-ctl"><button id="cos-fp-play">▶ play the run</button><input type="range" id="cos-fp-slice" min="0" max="1" value="1" step="1" title="stored slice of the run"></div>
      <div class="cos-fp-read" id="cos-fp-read"></div><div class="note" id="cos-fp-cap"></div></div>
    <div class="cos-col cos-fp-col"><h3><span style="color:#ffb07a">Standard model of cosmology</span>${fpTag('illustration')}</h3>
      <canvas id="cos-fp-std" width="640" height="640"></canvas>
      <div class="cos-fp-ctl"><button id="cos-fp-stdplay">▶ play the history</button><span>textbook times and temperatures</span></div>
      <ol class="cos-fp-steps" id="cos-fp-steps">${FP_STAGES.map((s, i) => `<li data-i="${i}"><span>${s[0]}</span><b>${s[1]}</b></li>`).join('')}</ol>
      <div class="cos-fp-read" id="cos-fp-stdtext"></div><div class="cos-fp-key" id="cos-fp-stdkey"></div>
      <div class="note">A schematic thermal history with random positions, not tracked particles or calculated reaction yields. In the ideal high-temperature plasma, particle-number weights use 3/4 per fermionic degree: gₙ = 95.5. Energy-density weights instead use 7/8: g* = 106.75. Later marker conversions and the nucleon count are display choices. The measured baryon-to-photon ratio is about 6.1 × 10⁻¹⁰. <a href="https://pdg.lbl.gov/2025/reviews/rpp2025-rev-bbang-cosmology.pdf">PDG cosmology</a> · <a href="https://pdg.lbl.gov/2025/reviews/rpp2025-rev-inflation.pdf">PDG reheating</a> · <a href="https://lambda.gsfc.nasa.gov/resources/graphic_history/microwaves.html">NASA CMB anisotropy</a>.</div></div>
  </div>
  <div class="cos-fp-cards"><div class="card cos-fp-card" id="cos-fp-pairs"></div><div class="card cos-fp-card" id="cos-fp-census"></div><div class="card cos-fp-card" id="cos-fp-bal"></div><div class="card cos-fp-card" id="cos-fp-conf"></div></div>
  <h2 style="margin:24px 0 6px">Register history and thermal history, side by side</h2>
  <table class="t" id="cos-fp-table" style="width:100%"></table>
  <div class="note">${fpTag('executed')} the run does it, read from its files. ${fpTag('derived')} an exact property of the run's rule. ${fpTag('premise')} put in, not derived. ${fpTag('not derived')} OPH has no result yet. ${fpTag('illustration')} a drawing, not a calculation.</div>
  <div class="cos-fp-tip" id="cos-fp-tip"></div>`;
}
function fpPlay(on) { FP.playing = on; $('cos-fp-play').textContent = on ? '❚❚ pause' : '▶ play the run'; }
function fpStdPlay(on) { FP.stdPlay = on; FP.stdLast = performance.now(); $('cos-fp-stdplay').textContent = on ? '❚❚ pause' : '▶ play the history'; }
function fpTipAt(e) {
  const tip = $('cos-fp-tip'), hov = e.target.closest && e.target.closest('#cos-fp-hov'), hl = $('cos-fp-hl');
  if (hov && FP.census && FP.cx) {
    const box = hov.ownerSVGElement.getBoundingClientRect(), px = (e.clientX - box.left) / box.width * FP.census.W;
    let best = 0; FP.T.forEach((T, i) => { if (Math.abs(FP.cx(T) - px) < Math.abs(FP.cx(FP.T[best]) - px)) best = i; });
    tip.textContent = `causal depth ${fmt(FP.T[best])}${best === FP.T.length - 1 ? ', end of the run' : ''}: ` + FP.census.names.map((n, i) => `${fmt(FP.census.ser[i][best])} ${n}`).join(', ');
    if (hl) { const x = FP.cx(FP.T[best]).toFixed(1); hl.setAttribute('x1', x); hl.setAttribute('x2', x); hl.setAttribute('visibility', 'visible'); }
  } else {
    if (hl) hl.setAttribute('visibility', 'hidden');
    const t = e.target.closest && e.target.closest('[data-tip]'); if (!t) { tip.style.display = 'none'; return; }
    tip.textContent = t.dataset.tip;
  }
  tip.style.display = 'block'; tip.style.left = Math.min(window.innerWidth - 340, e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 16) + 'px';
}
function fpWire() {
  $('cos-fp-slice').oninput = (e) => { fpPlay(false); fpSetSlice(parseInt(e.target.value)); };
  $('cos-fp-play').onclick = () => { if (!FP.playing && FP.k >= FP.T.length - 1) fpSetSlice(0); fpPlay(!FP.playing); };
  $('cos-fp-stdplay').onclick = () => { if (!FP.stdPlay && FP.stage >= FP_STAGES.length - 1) fpStage(0); fpStdPlay(!FP.stdPlay); };
  document.querySelectorAll('#cos-fp-steps li').forEach((li) => li.onclick = () => { fpStdPlay(false); fpStage(parseInt(li.dataset.i)); });
  const root = $('cos-first'), tip = $('cos-fp-tip');
  root.addEventListener('pointermove', fpTipAt);
  root.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
  root.addEventListener('focusin', (e) => { const t = e.target.closest('[data-tip]'); if (!t) return; const b = t.getBoundingClientRect(); tip.textContent = t.dataset.tip; tip.style.display = 'block'; tip.style.left = Math.min(window.innerWidth - 340, b.right + 8) + 'px'; tip.style.top = (b.bottom + 6) + 'px'; });
  root.addEventListener('focusout', () => { tip.style.display = 'none'; });
  new IntersectionObserver((es) => { for (const en of es) { FP.seen = en.isIntersecting; if (FP.seen) FP.dirty = true; } }).observe(root.querySelector('.cos-fp-grid'));
}
function fpSync() { if (S.frames && S.frames.length && S.cat && S.classStats && S.meta && S.frames !== FP.frames) fpBuildRun(); }
function fpLoop(now) {
  requestAnimationFrame(fpLoop);
  if (!$('page-cosmos').classList.contains('on')) return;
  fpSync();
  if (FP.playing && !FP.loading && FP.R && now - FP.last > (FP.k < 12 ? 480 : 110)) { FP.last = now; if (FP.k < FP.T.length - 1) fpSetSlice(FP.k + 1); else fpPlay(false); }
  if (FP.stdPlay && now - FP.stdLast > 2800) { FP.stdLast = now; if (FP.stage < FP_STAGES.length - 1) fpStage(FP.stage + 1); else fpStdPlay(false); }
  if (!FP.seen) return;
  if (FP.R && (FP.dirty || now - FP.t0 < 1700)) { fpDrawOPH(now); FP.dirty = false; }
  if (now - FP.stdDrawn > 33) { FP.stdDrawn = now; fpDrawStd(now); }
}
function fpShown() {
  if (!$('cos-first')) return;
  if (!FP.on) {
    FP.on = true; fpSkeleton(); fpStdBuild(); fpWire();
    const q = new URLSearchParams(location.search);
    if (q.has('cosslice')) FP.k = Math.max(0, parseInt(q.get('cosslice')) || 0);
    fpStage(q.has('cosstage') ? Math.max(0, Math.min(FP_STAGES.length - 1, parseInt(q.get('cosstage')) || 0)) : 0);
    fpLoad().then(fpTexts).catch((e) => { $('cos-fp-lead').textContent = 'Could not read the runs: ' + e.message; });
    requestAnimationFrame(fpLoop);
    if (q.get('cosch') === 'first') setTimeout(() => $('cos-first').scrollIntoView(), 60);
  }
  fpSync(); FP.dirty = true;
}
window.__firstParticles = FP;
