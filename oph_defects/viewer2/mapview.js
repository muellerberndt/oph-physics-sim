// The Map tab: every known particle beside the object of the model that matches it best by type, group and charge.
// A best guess by structure (repair-law/MAP.md, version 0), with the evidence from the runs and the open issues.
// Data: particle_map.json key defect_map (built by oph_defects/particle_table.py from the run analyses).
import { $, S, FAM, marker } from './app.js';

const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const COL = { quark: '#ff7a45', lepton: '#b197ff', neutrino: '#3fe0d0', gluon: '#ffcf5c', photon: '#7dd3fc', weak: '#f0abfc', higgs: '#ffe45c', dark: '#8f9bd6' };
// The source map of the theory session (repair-law/work/MAP/source_map.json, built by work/MAP/build_source_map.py):
// every fermion kind as a pattern of the five levels, left and right, with its properties.
let SM = null, BOS = null, MC = null, picked = 'u0';
const KEY = (f) => (f.symbol === 'ν' || /neutrino/.test(f.name) ? 'nu' : /lepton/.test(f.name) ? 'e' : f.symbol);
const TARGET = { u: ['2.16 MeV', '1.27 GeV', '172.6 GeV'], d: ['4.70 MeV', '93.5 MeV', '4.18 GeV'], nu: ['individual mass unknown', 'individual mass unknown', 'individual mass unknown'], e: ['0.511 MeV', '105.7 MeV', '1.777 GeV'] };
const NAME3 = { u: ['up', 'charm', 'top'], d: ['down', 'strange', 'bottom'], nu: ['electron neutrino', 'muon neutrino', 'tau neutrino'], e: ['electron', 'muon', 'tau'] };
const SYM3 = { u: ['u', 'c', 't'], d: ['d', 's', 'b'], nu: ['ν<sub>e</sub>', 'ν<sub>μ</sub>', 'ν<sub>τ</sub>'], e: ['e', 'μ', 'τ'] };
const ROW = { u: 2, d: 3, e: 4, nu: 5 }, GCOL = { u: COL.quark, d: COL.quark, e: COL.lepton, nu: COL.neutrino };
const FREE = { quarkL: 'about 10 carriers', quarkR: 'about 12 carriers', leptonL: 'about 123 carriers', leptonR: 'no limit' };
const ISSUES = [
  ['The catalogue belongs to the earlier rule', 'The defect classes were recorded under the committed integer rule with a declared register. Under the coherent law the protected objects have to be listed again.', 'theory worker X1'],
  ['Defects are fluxes, matter are charges', 'The objects with the family charges are the charged quanta. The catalogued defects are neutral under the gauge currents.', 'theory worker X2 scores the readings'],
  ['The register is too coarse', 'It records the charge modulo 6. It cannot tell up from down singlets, the lepton doublet from the Higgs, or copies.', 'theory worker X3; then a new census at level 8'],
  ['Nothing in the catalogue moves with a velocity', 'The frozen classes sit still. The mobile ones wander and annihilate. A charged quantum travels, and it can carry a defect as cargo.', 'theory worker R2 done; observer cards from the runs (running)'],
  ['Spin and statistics', 'Flux pairs are neutral and bosonic. Fermions need the spinor of the cover and antisymmetry.', 'theory worker R4'],
  ['Three copies', 'Nothing repeats three times yet. Four candidates are listed above.', 'theory workers X4, T5'],
  ['Force carriers and the Higgs are operators', 'They are not yet excitations that travel.', 'theory worker X5'],
  ['Mass', 'The law gives a massless family and a tower at the scale of the lattice. The flux background of the runs is a relic of the preparation.', 'theory workers R8, R6, T3, T5; this session executes'],
  ['Unit and comparison', 'One measured anchor, then a frozen comparison.', 'theory worker R7: protocol frozen'],
];
const boxes = (levels) => '<span class="mp-lv5">' + ['v1', 'v2', 'v3', 's1', 's2'].map((l) => `<i class="${l[0]}${levels.includes(l) ? ' on' : ''}${l === 's2' ? ' tog' : ''}" title="${l}${l === 's2' ? ': the level the mass step toggles' : ''}"></i>`).join('') + '</span>';
const cls = (k) => (k ? `<span class="mp-cls" title="register class ${k}">${marker(k, 15)}</span>` : '<span class="mp-cls flat" title="no register class: a flat face">∅</span>');
function fermiTile(f, g) {
  const k = KEY(f);
  return `<div class="mp-tile${g ? ' later' : ''}${picked === k + g ? ' sel' : ''}" data-k="${k}${g}" style="--c:${GCOL[k]};grid-column:${g + 2};grid-row:${ROW[k]}">
    <div class="mp-top"><span class="mp-sym">${SYM3[k][g]}</span><span class="mp-mass" title="measured value: a target, not an output">${TARGET[k][g]}</span></div>
    <div class="mp-name">${NAME3[k][g]} · charge ${esc(f.electric_charge)} · spin ½</div>
    <div class="mp-row"><em>left</em>${boxes(f.left.pattern.levels)}${cls(f.left.pattern.Q6_mod_6)}</div><div class="mp-row"><em>right</em>${boxes(f.right.pattern.levels)}${cls(f.right.pattern.Q6_mod_6)}</div>
    <div class="mp-tag">${g === 0 ? 'charges exact · mass open' : 'copy ' + (g + 1) + ' · carrier of the copy open'}</div></div>`;
}
function detail() {
  const k = picked.slice(0, -1), g = parseInt(picked.slice(-1)), f = SM.fermion_types.find((x) => KEY(x) === k); if (!f) return '';
  const L = f.left.pattern, R = f.right.pattern, quark = f.group === 'quark', tag = (t, c) => `<span class="cos-tag ${c}">${t}</span>`;
  const seen = (o) => Object.entries(o).filter(([, v]) => v !== '0').map(([c, v]) => `class ${c}: ${v} turn`).join(', ') || 'none';
  const rows = [
    ['Pattern, left-handed', `${boxes(L.levels)} ${L.levels.join(', ')} · ${L.colour}, weak ${L.weak}, 6Y = ${L.six_Y}, T3 = ${L.T3}`, tag('exact label', 'ok')],
    ['Pattern, right-handed', `${boxes(R.levels)} ${R.levels.join(', ')} · ${R.colour}, weak ${R.weak}, 6Y = ${R.six_Y}, T3 = ${R.T3}`, tag('exact label', 'ok')],
    ['Electric charge', esc(f.electric_charge) + ' for both', tag('exact label', 'ok')],
    ['Handedness', 'the candidate walk assigns opposite low-momentum Weyl blocks to the two pattern parities; full interacting Lorentz chirality requires its own proof', tag('candidate free walk', 'open')],
    ['Spin', 'one half: the cover of the frame group acts on a pointer doublet. The pointer is declared: the native doublet of the carrier is spent on the weak index', tag('exact at rest; pointer declared', 'open')],
    ['Statistics', 'antisymmetric', tag('declared', 'open')],
    ['Antiparticle', 'the complement pattern: ' + f.antiparticle_left.levels.join(', ') + ' and ' + f.antiparticle_right.levels.join(', '), tag('exact', 'ok')],
    ['Speed', 'the declared free walk has a common leading low-momentum cone. Its finite lattice is anisotropic: axis and diagonal packets do not have identical finite-step speeds', tag('finite walk; not a physical speed fit', 'open')],
    ['Flux classes it sees, left', seen(f.flux_seen_left) + ' · gets ' + FREE[quark ? 'quarkL' : 'leptonL'] + ' through the flux gas', tag('exact phases; path measured', 'ok')],
    ['Flux classes it sees, right', seen(f.flux_seen_right) + ' · gets ' + FREE[quark ? 'quarkR' : 'leptonR'] + ' through the flux gas', tag('exact phases; path measured', 'ok')],
    ['Magnetic factor', '2, with a declared link phase: in a uniform field the lowest level stays at the rest angle', tag('declared extension, executed', 'open')],
    ['Mass link', 'the toggle of the level s2 joins the left and the right pattern', tag('representation', 'open')],
    ['Copy', (g + 1) + ' of 3: which object carries the copy is open, best guess the band of three resolved along the axes of the fixed frame', tag('open', 'open')],
    ['Mass', 'target ' + TARGET[k][g] + '. This displayed free-walk member has zero rest phase without an added mass step; other native quantum constructions have positive source energy windows', tag('open', 'bad')],
    ['Lifetime and decays', 'needs the weak field as a link that moves', tag('open', 'open')],
  ];
  return `<div class="card"><div class="k" style="margin-top:0">${NAME3[k][g]} · everything the map says</div><table class="t" style="width:100%">${rows.map((r) => `<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('')}</table></div>`;
}
function hero() {
  const F = SM.fermion_types, B = SM.bosons, G = SM.generations;
  return `<div class="mp-grid5">
      <div class="mp-lab" style="grid-column:2;grid-row:1">copy 1</div><div class="mp-lab" style="grid-column:3;grid-row:1">copy 2</div><div class="mp-lab" style="grid-column:4;grid-row:1">copy 3</div>
      <div class="mp-side" style="grid-column:1;grid-row:2 / span 2"><span>quarks</span></div><div class="mp-side" style="grid-column:1;grid-row:4 / span 2"><span>leptons</span></div>
      ${F.map((f) => [0, 1, 2].map((g) => fermiTile(f, g)).join('')).join('')}
    </div>
    <div class="mp-key"><span>${boxes(['v1', 's1'])} five levels: three vector levels (colour) and two spinor levels (weak charge); filled is occupied</span><span><span class="mp-lv5"><i class="s tog"></i></span> the level s2: the mass step toggles it, which turns left into right</span><span>${cls(1)}${cls(4)}${cls(3)} register class of the run with the same charge</span><span><span class="mp-cls flat">∅</span> no class</span></div>
    <div class="note" style="margin:6px 0 10px">Which object carries the copy is open. Best guess: ${esc(G.candidates[0].label)}. The measured masses are targets, not outputs. Click a tile.</div>
    <div id="mp-detail">${detail()}</div>
    <h2 style="margin:26px 0 6px">Gauge and scalar operators: candidate particle roles</h2>
    <div class="mp-bosons">${B.map((b) => `<div class="mp-tile bose" style="--c:${/gluon/.test(b.name) ? COL.gluon : /photon/.test(b.name) ? COL.photon : /Higgs/i.test(b.name) ? COL.higgs : COL.weak}"><div class="mp-top"><span class="mp-sym" style="font-size:22px">${esc(b.name)}</span><span class="mp-mass">${b.count > 1 ? '× ' + b.count : ''}</span></div>
        <div class="mp-what">${esc(b.source)}</div><div class="mp-name">charge ${esc(b.charge)} · ${esc(b.colour)} · acts on ${esc(b.acts_on)}</div><div class="mp-tag">travels: ${esc(b.travelling_excitation)}</div><div class="mp-tag" style="color:#c5d0e6">mass: ${esc(b.mass)}</div></div>`).join('')}
      <div class="mp-tile bose" style="--c:${COL.dark}"><div class="mp-top"><span class="mp-sym" style="font-size:22px">dark part</span><span class="mp-mass">best guess</span></div><div class="mp-what">the flux pair that travels from carrier to carrier, or a quantum with a flux pair as cargo</div><div class="mp-name">no gauge charge · protected</div><div class="mp-tag">not in the standard list</div></div></div>
    <h2 style="margin:26px 0 6px">Which object is a copy: four candidates</h2>
    <p class="small" style="max-width:1100px;margin:0 0 8px">${esc(G.required)}. ${esc(G.public_theorem)}</p>
    <div class="mp-cands">${G.candidates.map((c, i) => `<div class="card"${i === 0 ? ' style="border-color:rgba(255,228,92,.5)"' : ''}><b>${esc(c.id)}${i === 0 ? ' · best guess' : ''}</b><p>${esc(c.label)}</p><span>${esc(c.status)}</span></div>`).join('')}</div>
    <div class="card" style="margin-top:12px;border-color:rgba(125,211,252,.5)"><b>G-e · the second index of the frame group on the ports</b> <span class="cos-tag ok">finite port structure</span>
      <p style="margin:6px 0 6px;font-size:13.5px">Fix one cubic frame. Its twelve rotations move the twelve ports one onto another without a fixed port, so the ports are the elements of that group. Then a second action exists that commutes with the rotations and that the seams do not respect. Fixing the frame creates a threefold label, and the seams split it with no free number. In the frame axes of the second index the seam matrix is</p>
      <div class="mono" style="font-size:13px;margin:0 0 6px 12px">[−1, 0, 0] · [0, −1, 2] · [0, 2, 1] &nbsp; with levels √5, −1, −√5</div>
      <p style="margin:0;font-size:13px;color:var(--soft)">One axis stays unmixed. The other two are turned by 31.7 degrees, with squared components 0.7236 and 0.2764. This is the owner’s suggestion that fixing the gauge can explain three copies, in exact form. Its weakness: the three levels differ by factors of order one, 0.3047, 0.4510 and 0.4962 per tick, where nature needs factors up to 3,500.</p></div>
    ${BOS ? `<h2 style="margin:26px 0 6px">What it takes for a force carrier to travel</h2><table class="t" style="width:100%"><tr><th>carrier</th><th>as an excitation</th><th>mass</th><th>declared for it</th><th>native link alphabet can carry it</th></tr>${BOS.bosons.map((b) => `<tr><td><b>${esc(b.boson)}</b></td><td>${esc(b.excitation)}</td><td>${esc(b.mass)}</td><td class="small">${esc((b.declared || []).join('; '))}</td><td><span class="cos-tag ${b.native_alphabet_can_carry_it ? 'ok' : 'bad'}">${b.native_alphabet_can_carry_it ? 'yes' : 'no'}</span></td></tr>`).join('')}</table><div class="note">This table concerns the specified finite link alphabet. A gauge-generator count alone does not construct travelling bosons. An extended finite quantum carrier can instead be tested through its complete field records and controlled refinement; no infinite variable is required at a finite capacity.</div>` : ''}
    <div class="note" style="margin-top:8px">Mixing: ${esc(G.mixing)}. ${esc(G.owner)}.</div>`;
}
function kindCard(k, maxPath) {
  const c = COL[k.group], marks = k.classes.map((x) => (x ? marker(x, 26) : '<span class="mp-flat">∅</span>')).join('');
  const fp = k.free_path, bar = k.id === 'ls' ? 100 : fp ? Math.max(3, 100 * fp / maxPath) : 0;
  return `<div class="mp-kind" style="--c:${c}"><div class="mp-kind-head">${marks}<div><b>${esc(k.name)}</b><span>${k.classes[0] ? 'class ' + k.classes.join(' and ') : 'no class'} · colour charge: ${k.colour} · weak charge: ${k.weak}</span></div></div>
    <div class="mp-guess">same charge class as: ${esc(k.guess)}</div><ul>${k.run.map((r) => `<li>${esc(r)}</li>`).join('')}</ul>
    <div class="small" style="margin-bottom:5px;color:#7dd3fc">${esc(k.law || '')}</div><div class="mp-path"><div style="width:${bar}%"></div></div><div class="small">${k.id === 'ls' ? 'gets through the flux gas without limit' : 'gets about ' + fp.toFixed(0) + ' carriers through the flux gas'}</div></div>`;
}
// The defect catalogue read directly (owner's instruction): the families of protected defects of the committed runs, what
// each does in the run, and the particle with the same charges. Counts: level-10 run, repair-law/reports/CO.md and CH.md.
const FAMILIES = [
  { cls: [1, 5], name: 'left-handed quark', anti: 'antiquark', sym: 'q<sub>L</sub>', col: COL.quark, charges: 'colour and weak charge', made: '24.1 % + 24.1 %', expect: '21.4 % + 21.4 %',
    does: ['moves across the screen; on a carrier it stays on that carrier', 'the particle steps to the left of the local flow, the antiparticle to the right, in every one of 3.4 million steps', 'particle and antiparticle annihilate: 609,494 times', 'two of them fuse into one right-handed antiquark class: 915,000 times'],
    members: 'up and down, three colours each, three copies: 6 states per copy' },
  { cls: [4, 2], name: 'right-handed quark', anti: 'antiquark', sym: 'q<sub>R</sub>', col: '#ff5f7a', charges: 'colour charge only', made: '19.6 % + 19.6 %', expect: '21.4 % + 21.4 %',
    does: ['does not move once its faces are not neighbours', 'never alone: only in groups without net colour', 'in pairs with its antiparticle on 69 % of the occupied carriers, in threes on 2.4 %', 'changes class when a left-handed quark reaches it'],
    members: 'up and down, three colours each, three copies: 6 states per copy' },
  { cls: [3], name: 'left-handed lepton', anti: 'antilepton (same class)', sym: 'ℓ<sub>L</sub>', col: COL.lepton, charges: 'weak charge only', made: '12.7 %', expect: '14.3 %',
    does: ['does not move', 'comes in pairs, on 11 % of the occupied carriers', 'made when a left-handed quark meets a right-handed antiquark: 630,000 times', 'the Higgs has the same class'],
    members: 'electron and neutrino, three copies: 2 states per copy, and their antiparticles' },
  { cls: [0], name: 'right-handed electron', anti: 'and every force carrier', sym: 'e<sub>R</sub>', col: COL.neutrino, charges: 'no centre charge', made: 'not counted', expect: '',
    does: ['a flat face: leaves no defect', 'the photon, the gluons, W and Z are in this class too', 'so the register cannot see them'],
    members: 'needs a finer register' },
];
const REACTIONS = [
  ['q<sub>L</sub> + q̄<sub>L</sub>', 'nothing', '1 + 5 → 0', 609494, 'a quark meets its antiquark'],
  ['q<sub>L</sub> + q<sub>L</sub>', 'q̄<sub>R</sub>', '1 + 1 → 2', 458209, 'two quarks make a diquark, with the colour of one antiquark'],
  ['q<sub>L</sub> + q<sub>R</sub>', 'q̄<sub>L</sub>', '1 + 4 → 5', 394861, 'the same, with one of each handedness'],
  ['q<sub>L</sub> + q̄<sub>R</sub>', 'Higgs class', '1 + 2 → 3', 315024, 'centre classes compatible with a quark–scalar channel; no Yukawa amplitude is measured'],
  ['q<sub>L</sub> + Higgs class', 'q<sub>R</sub>', '1 + 3 → 4', 238523, 'centre-class transfer compatible with the candidate scalar assignment; no mass term follows'],
  ['class 3', 'q<sub>L</sub> + q̄<sub>R</sub>', '3 → 1 + 2', 182353, 'an order-two class splits into classes 1 and 2'],
  ['q̄<sub>R</sub>', 'q<sub>L</sub> + q<sub>L</sub>', '2 → 1 + 1', 377308, 'the reverse of the second row'],
  ['class 3 + class 3', 'nothing', '3 + 3 → 0', 17, 'two order-two register classes cancel; spin and the interaction operator are not recorded'],
];
const SMFAM = [
  { fam: 'A', cls: [1, 5], col: COL.quark, charges: 'colour and weak charge', run: 'moves; annihilates with its opposite; fuses',
    sm: 'Left-handed quarks', gens: [['u<sub>L</sub>', 'd<sub>L</sub>'], ['c<sub>L</sub>', 's<sub>L</sub>'], ['t<sub>L</sub>', 'b<sub>L</sub>']], extra: 'three colours each, and their antiquarks', states: '6 + 6', smShare: 21.4, runShare: 24.1 },
  { fam: 'B', cls: [4, 2], col: '#ff5f7a', charges: 'colour charge only', run: 'frozen; only in groups without net colour',
    sm: 'Right-handed quarks', gens: [['u<sub>R</sub>', 'd<sub>R</sub>'], ['c<sub>R</sub>', 's<sub>R</sub>'], ['t<sub>R</sub>', 'b<sub>R</sub>']], extra: 'three colours each, and their antiquarks', states: '6 + 6', smShare: 21.4, runShare: 19.6 },
  { fam: 'C', cls: [3], col: COL.lepton, charges: 'weak charge only', run: 'frozen pairs',
    sm: 'Left-handed leptons, and the Higgs', gens: [['ν<sub>eL</sub>', 'e<sub>L</sub>'], ['ν<sub>μL</sub>', 'μ<sub>L</sub>'], ['ν<sub>τL</sub>', 'τ<sub>L</sub>']], extra: 'and their antileptons; the Higgs doublet has the same charges', states: '2 + 2', smShare: 14.3, runShare: 12.7 },
  { fam: '·', cls: [0], col: COL.neutrino, charges: 'no charge the register can see', run: 'leaves no defect',
    sm: 'Right-handed leptons, and the force carriers', gens: [['ν<sub>eR</sub>', 'e<sub>R</sub>'], ['ν<sub>μR</sub>', 'μ<sub>R</sub>'], ['ν<sub>τR</sub>', 'τ<sub>R</sub>']], extra: 'and photon, gluons, W, Z', states: '2 + 2', smShare: null, runShare: null },
];
function massCheckSection() {
  if (!MC) return '';
  const E = MC.readings['end of the level-10 run'], Lm = MC.readings['classes 1 and 5 gone (limit)'];
  const fmt = (mev) => (mev >= 1000 ? (mev / 1000).toPrecision(4) + ' GeV' : mev >= 1 ? mev.toPrecision(4) + ' MeV' : mev >= 1e-3 ? (mev * 1000).toPrecision(3) + ' keV' : (mev * 1e6).toPrecision(2) + ' eV');
  const verdict = (r) => { const q = r.model_over_measured; if (r.upper_limit) return '<span class="cos-tag bad">more than 10⁶ too heavy</span>'; if (Math.abs(r.symbol === 'e' ? 0 : Math.log10(q)) < 0.2) return r.symbol === 'e' ? '<span class="cos-tag open">anchor</span>' : '<span class="cos-tag ok">within 1.6</span>'; if (Math.abs(Math.log10(q)) < 0.55) return '<span class="cos-tag open">within 3.5</span>'; const n = (x) => Number(x.toPrecision(2)).toLocaleString('en-US'); return `<span class="cos-tag bad">${q < 1 ? n(1 / q) + ' times too light' : n(q) + ' times too heavy'}</span>`; };
  const rows = E.rows.map((r, i) => { const l = Lm.rows[i]; return `<tr><td><b>${r.symbol}</b> ${r.name}</td><td class="num">${r.rest_phase_left.toFixed(5)} · ${r.rest_phase_right.toFixed(5)}</td><td class="num">${fmt(r.model_MeV_left)}</td><td class="num">${fmt(l.model_MeV_left)}</td><td class="num">${r.upper_limit ? 'below ' : ''}${fmt(r.measured_MeV)}</td><td>${verdict(r)}</td><td>${verdict(l)}</td></tr>`; }).join('');
  return `<h2 style="margin:30px 0 4px">Masses in GeV: the check <span class="mp-badge" style="background:rgba(255,95,122,.16);color:#ff9db0">calibrated comparison; full spectrum fails</span></h2>
    <p class="small" style="max-width:1150px;margin:0 0 8px">This declared background-walk reading assigns rest angles from the frozen flux of the level-10 run. Its numerical flux estimate uses 16.8 million carriers at the quoted 0.3 percent precision. These angles are not the source energies of the separate interacting quantum calculation in the Spectrum tab. <b>SI units:</b> one cell per tick is the speed of light, so a cell is c times a tick, and a rest phase θ per tick is a rest energy ħθ divided by the tick. The tick is not derived, so it is set by one anchor: the electron. That gives a tick of ${E.tick_s.toExponential(2)} s and a cell of ${E.cell_fm.toFixed(2)} fm. The largest energy on this chosen principal phase branch is ${E.heaviest_mass_the_clock_holds_MeV.toFixed(0)} MeV. A tick phase alone aliases energies modulo 2πħ per tick; it does not prove a universal maximum physical mass.</p>
    <table class="t" style="width:100%"><tr><th>particle</th><th>rest phase per tick, left · right</th><th>model, end of run</th><th>model, frozen limit</th><th>measured</th><th>end of run</th><th>limit</th></tr>${rows}
      ${MC.bosons.map((b) => `<tr><td><b>${b.name}</b></td><td colspan="3">${b.model}</td><td class="num">${b.measured}</td><td colspan="2">${/exact/.test(b.model) ? '<span class="cos-tag open">model zero; physical comparison required</span>' : '<span class="cos-tag open">no value</span>'}</td></tr>`).join('')}</table>
    <div class="mp-two" style="margin-top:12px">
      <div class="card" style="border-color:rgba(63,224,208,.5)"><div class="k" style="margin-top:0;color:#7ff0e2">Calibrated scale comparison</div><p style="margin:4px 0 0;font-size:13.5px">With the electron as anchor, the proposed up/down reading gives about 1.6 MeV at the end of the run and 3.3 MeV in the frozen limit, compared with running-mass references of about 2.2 and 4.7 MeV. Factor-level similarity is not agreement within measurement errors or a proved scheme match. Zero photon/gluon mass is a target of the supplied gauge action; photon mass has experimental upper bounds, and a confined gluon is not an isolated measured particle.</p></div>
      <div class="card" style="border-color:rgba(255,95,122,.5)"><div class="k" style="margin-top:0;color:#ff9db0">What does not fit</div><p style="margin:4px 0 0;font-size:13.5px">All three generations get the same value, because the defect class cannot tell them apart: the strange, charm, bottom and top quarks, the muon and the tau miss by factors of 20 to 100,000. Up equals down. The neutrino gets the electron's mass, a million times too heavy. The right-handed electron gets zero. The quark-to-electron ratio changes with how the run was prepared. <b>The test that needs no anchor:</b> if the masses fitted, every particle would need the same tick. They need ticks from ${E.needed_tick_min_s.toExponential(1)} s to ${E.needed_tick_max_s.toExponential(1)} s, a spread of ${E.needed_ticks_span.toExponential(1)}.</p></div></div>
    <div class="note" style="margin-top:8px"><b>Second reading, from the run's own energy bookkeeping.</b> When a defect appears, the repair that makes it pays the energy stored in its face, the register cost: 5 for a left-handed quark, 8 for a right-handed quark, 10 for a left-handed lepton, in units of the load mismatch. That is register bookkeeping cost. Equating it with a relativistic rest energy would require a Hamiltonian, clock and dispersion map; a stationary register label does not supply that map. The cost table is a declared choice among four in the code (the runs used the binding table; additive, flat and Wilson tables exist). Read as rest energies it makes leptons twice as heavy as quarks, the wrong way round, and again gives all three generations one value.</div>
    <div class="note" style="margin-top:8px"><b>Unit choice and comparison:</b> assigning a Planck-scale tick gives lattice-scale energies, not a match to measured leptons. Changing a common unit cannot repair equal generation ratios. The separate particle formulas are target-informed or conditional on declared response parameters. Their printed values do not reproduce the complete measured spectrum within the printed intervals. The <a href="https://github.com/FloatingPragma/observer-patch-holography/blob/main/model/particle_construction_scope/PROOF.md" target="_blank" rel="noopener">audited candidate calculations</a> keep fitted parameters and failed comparisons explicit.</div>`;
}
function familiesSection() {
  const row = (f) => `<div class="fm-row" style="--c:${f.col}">
      <div class="fm-def"><div class="fm-marks">${f.cls.map((k) => (k ? marker(k, 30) : '<span class="mp-flat">∅</span>')).join('')}</div>
        <div><b>${f.fam === '·' ? 'no defect' : 'defect family ' + f.fam}</b><span>${f.cls[0] ? 'class ' + f.cls.join(' and ') : 'flat face'} · ${f.charges}</span><em>${f.run}</em></div></div>
      <div class="fm-link"><i></i></div>
      <div class="fm-sm"><b>${f.sm}</b><div class="fm-gens">${f.gens.map((g, i) => `<div><span class="fm-gl">${i + 1}</span>${g.map((s) => `<span class="fm-sym">${s}</span>`).join('')}</div>`).join('')}</div><em>${f.extra}</em></div>
      <div class="fm-share">${f.smShare === null ? '<span class="small">not counted by the register</span>' : `<div class="fm-bar"><i style="width:${(f.smShare * 3).toFixed(1)}%;background:#93a3c0"></i></div><div class="fm-bar"><i style="width:${(f.runShare * 3).toFixed(1)}%;background:var(--c)"></i></div><span class="small">${f.smShare}% of the charged states of a generation · ${f.runShare}% of the defects made in the run (each sign)</span>`}</div></div>`;
  return `<h2 style="margin:0 0 4px">Defect families and Standard Model families <span class="mp-badge">centre-class compatibility; candidate assignments</span></h2>
    <p class="small" style="max-width:1150px;margin:0 0 10px">A defect class retains colour triality and weak centre parity, together with hypercharge modulo the centre relation. Different full representations, spins and generations can share it. The table shows proposed particle-family assignments compatible with those labels; it is not a one-to-one identification of source defects with Standard Model particles. Level-10 run, 20,971,520 carriers.</p>
    <div class="fm-head"><span>in the run</span><span></span><span>in the Standard Model, generations 1, 2, 3</span><span>how many</span></div>
    ${SMFAM.map(row).join('')}
    <div class="note" style="margin-top:10px"><b>Why the three defect families are not the three generations.</b> The three generations carry identical charges, and the three defect families carry different ones. A generation assignment would therefore need information beyond the centre label. <b>The shares</b> are quench-remnant counts, not a derived thermal occupation law. The displayed 24 : 24 : 20 : 20 : 13 versus 21 : 21 : 21 : 21 : 14 comparison tests a proposed equal-per-internal-state census, not early-Universe abundances.</div>
    <div class="card" style="margin-top:12px"><div class="k" style="margin-top:0">Testing a geometric generation candidate</div>
      <p style="margin:4px 0 6px;font-size:13.5px">The finer geometric register tells pairs apart by how far apart their two faces sit on the carrier. Frozen quark pairs come at separations 2, 3, 4 and 5 in shares 44, 35, 16 and 5 percent; lepton pairs in 50, 34, 13 and 4 percent. Guess: separation 2, 3, 4 are generations 1, 2, 3, the commonest being the lightest. A quantum carrying such a pair gets rest phases that rise with the separation, in ratios 1 : 1.15 : 1.22 for quarks and 1 : 1.20 : 1.34 for leptons.</p>
      <p style="margin:0;font-size:13.5px"><span class="cos-tag bad">fails</span> Nature has 1 : 207 : 3,477 for the electron, muon and tau. Separation 5 gives another geometric label; identifying it with an extra active neutrino would need an additional field and mass assignment before the Z-width constraint applies. <b>Verdict of the field session, checked on every kind:</b> mirror kinds are one placement seen from two sides, and the counts fall smoothly with separation as local creation predicts. The kinds are positions of a pair on its carrier. The defect class carries no generation.</p></div>
    <h3 style="margin:18px 0 4px;font-size:15px">Kinds inside each family</h3>
    <p class="small" style="max-width:1150px;margin:0 0 6px">A defect never sits alone on a carrier. The finer geometric register tells the groups apart by how far the two faces are from each other on the carrier and by whether the path between them turns left or right. Only groups whose faces are not neighbours last. Counts at the end of the level-10 run.</p>
    <table class="t" style="width:100%"><tr><th>group</th><th>kinds that last</th><th>the kinds and how many carriers hold each</th></tr>
      <tr><td><b>a right-handed quark with its antiquark</b></td><td class="num">6</td><td>2 faces apart, turning left 1,107,653 · turning right 1,109,692 · 3 apart, left then right 861,343 · right then left 861,518 · 4 apart 792,175 · 5 apart 259,535</td></tr>
      <tr><td><b>a lepton with an antilepton</b></td><td class="num">5</td><td>2 apart 395,030 · 3 apart, left then right 131,873 · right then left 132,124 · 4 apart 98,431 · 5 apart 28,090</td></tr>
      <tr><td><b>three right-handed quarks</b></td><td class="num">13</td><td>13 shapes, every one present, with very unequal shares; the same 13 for three antiquarks. 86,476 carriers with three quarks, 86,788 with three antiquarks</td></tr>
      <tr><td><b>a left-handed quark with its antiquark</b></td><td class="num">not lasting</td><td>all separations occur, 1 to 5; the pair annihilates with a half-life of 75 sweeps</td></tr></table>
    <div class="note" style="margin-top:6px">The quoted mirror counts are close in this run; their agreement does not prove the absence of every parity-breaking observable. Nature has six kinds of quark and six kinds of lepton. Whether a kind of pair is a kind of particle is not settled; this table is where to look.</div>
    <h3 style="margin:18px 0 4px;font-size:15px">Class addition in the recorded repair events</h3>
    <table class="t" style="width:100%"><tr><th>what meets</th><th>what comes out</th><th>classes</th><th>times seen</th><th>in words</th></tr>
      ${REACTIONS.map((r) => `<tr><td><b>${r[0]}</b></td><td><b>${r[1]}</b></td><td class="mono">${r[2]}</td><td class="num">${r[3].toLocaleString('en-US')}</td><td>${r[4]}</td></tr>`).join('')}</table>
    <div class="note" style="margin-top:8px">The recorded events conserve their class sum modulo 6. Opposite-class pairs and triples can be neutral under the colour centre. Centre neutrality is weaker than a full colour singlet, and this census does not measure a quantum scattering amplitude or exchange statistics.</div>
    <div class="note" style="margin-top:6px"><b>What the catalogue cannot tell yet:</b> up from down, the three colours, the electron from the neutrino, and the three copies. The register records only the class modulo 6. A finer geometric register: it finds 20 kinds of pair and 13 shapes of three-quark group in the same run, but no theorem identifies those geometric labels with generations. <b>Caution from the field session:</b> read strictly, a defect is a flux and a particle is a charge, and the two readings are being scored against each other. The charges and the reaction rule are the same in both.</div>`;
}
export async function mapShown() {
  const host = $('map-host'); if (!host || !S.pmap || !S.pmap.defect_map) return;
  if (!SM) { try { SM = await (await fetch('/rl/MAP/source_map.json')).json(); } catch (e) { host.innerHTML = '<p class="small">the source map could not be loaded</p>'; return; }
    try { const r = await fetch('/rl/X5/bosons.json'); if (r.ok) BOS = await r.json(); } catch (e) { BOS = null; }
    try { const r = await fetch('/rl/SIM/cf1/runs/MASS_CHECK.json'); if (r.ok) MC = await r.json(); } catch (e) { MC = null; } }
  const D = S.pmap.defect_map, maxPath = Math.max(...D.kinds.map((k) => k.free_path || 0));
  const groupsBar = D.groups.map((g) => `<i style="width:${(100 * g.share).toFixed(2)}%;background:${COL[g.group]}" title="${esc(g.name)}"></i>`).join('');
  const groupsRows = D.groups.map((g) => `<tr><td><i class="dot" style="background:${COL[g.group]}"></i><b>${esc(g.name)}</b></td><td class="mono">${esc(g.set)}</td><td>${esc(g.reading)}</td><td class="num">${g.carriers.toLocaleString('en-US')}</td><td class="num">${(100 * g.share).toFixed(2)} %</td></tr>`).join('');
  const A = [
    ['Quarks and leptons', 'the occupation patterns of the five internal levels of a quantum: three vector levels, two spinor levels', 'colour, weak charge, hypercharge and electric charge of all sixteen states: exact', 'spin, three copies, mass'],
    ['Antiparticles', 'the complementary pattern: every filled level empty, every empty level filled', 'opposite charges: exact', ''],
    ['Gluons, W, Z, photon', 'the twelve responses of a carrier on the five levels: 8 + 3 + 1', 'the algebra: exact', 'they do not travel yet'],
    ['Higgs', 'the toggle of the neutral spinor level: the one operator that keeps colour and electric charge and links left to right', 'one common link for all four kinds of fermion', 'a state and a strength'],
    ['The defect classes', 'vortices of the fields: colour-centre flux (classes 2, 4), weak-centre flux (class 3), generating flux (classes 1, 5)', 'each is seen by exactly the patterns that carry the matching charge', 'no particle name in this reading'],
  ];
  host.innerHTML = `
  <div class="mp-wrap">
    <h2 style="margin:0">The particle map <span class="mp-badge">best guess</span></h2>
    <p class="small" style="max-width:1100px;margin:4px 0 14px">Candidate assignments connect the register classes and internal representations to familiar particle names. Each column distinguishes an observed target from a computed source property. The Spectrum tab gives the authenticated quantum energy, current and detector results on their declared sources.</p>
    ${familiesSection()}
    ${massCheckSection()}
    <h2 style="margin:34px 0 8px">The candidate internal representation: patterns of five levels</h2>
    <div class="mp-two" style="margin-bottom:14px">
      <div class="card" style="border-color:rgba(63,224,208,.5)"><div class="k" style="margin-top:0;color:#7ff0e2">Exact internal weights in the declared lift</div>
        <ul style="margin:4px 0 0 18px;padding:0;font-size:13.5px;color:#dfe6f5"><li>the gauge weights of one fermion family, with an optional sterile singlet</li><li>the two handedness classes of each, and a weak current that acts on the left-handed class only</li><li>antiparticles as complement patterns</li><li>a common leading cone in the separate free-walk construction</li><li>the algebra of the forces: 8 + 3 + 1</li></ul></div>
      <div class="card" style="border-color:rgba(255,95,122,.5)"><div class="k" style="margin-top:0;color:#ff9db0">Not selected by the law as it stands</div>
        <ul style="margin:4px 0 0 18px;padding:0;font-size:13.5px;color:#dfe6f5"><li>this candidate walk does not select the observed masses; four independent mass-step strengths are allowed</li><li>three copies: five candidates, none with the measured ratios</li><li>physical force-carrier propagation and interaction on the same quantum source</li><li>the heavy tower of the carrier is not W, Z, Higgs and top: a blind test misses by 6 percent</li></ul></div>
    </div>
    ${hero()}

    <h2 style="margin:30px 0 4px">What the particle comparison establishes</h2>
    <p class="small" style="max-width:1100px;margin:0 0 8px">A representation match is useful evidence about internal quantum numbers. It does not identify the measured particle before its propagation, statistics and interaction records are connected on the same source.</p>
    <table class="t" style="width:100%"><tr><th>comparison</th><th>exact finite content</th><th>physical interpretation</th></tr>
      ${[
        ['One fermion family', 'the declared current lift has the fifteen Standard Model Weyl weights; a sixteen-state extension adds a singlet', 'a sterile right-handed neutrino is an option, not an observed extra state required by oscillations'],
        ['Electric charge', 'Q = T3 + Y gives the conventional quark and lepton weights in this representation', 'a proton or neutron also needs a physical composite state and its current'],
        ['Gauge algebra', 'the complete matrices realize the 8 + 3 + 1 Lie algebra in the declared lift', 'the generator count does not establish twelve propagating particles'],
        ['Anomaly sums', 'the family weights have the usual vanishing gauge-anomaly sums', 'the complete chiral regulated dynamics and its limit require their own construction'],
        ['Weak currents', 'the source current map acts on the designated left components', 'a weak decay or scattering prediction also needs the interacting relativistic field map'],
        ['Spin and magnetic response', 'a declared free spinor/link construction gives the corresponding finite comparison', 'a tree-level magnetic factor is not the complete measured anomalous moment'],
        ['Masses and generations', 'native source energy windows and calibrated candidate ratios are distinct results', 'no complete source-derived Standard Model GeV spectrum is established'],
      ].map((r) => `<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('')}
    </table>
    <div class="note" style="margin-top:8px">Particle targets: <a href="https://pdg.lbl.gov/2025/reviews/rpp2025-rev-standard-model.pdf" target="_blank" rel="noopener">PDG electroweak review</a> and <a href="https://pdg.lbl.gov/2025/reviews/rpp2025-rev-neutrino-mixing.pdf" target="_blank" rel="noopener">neutrino review</a>. Neutrino flavour labels are mixtures of mass states; their individual masses and Dirac/Majorana nature are not fixed by the displayed family tiles. Quark mass comparisons require an explicit renormalization scheme and scale. <a href="https://github.com/FloatingPragma/observer-patch-holography/blob/main/model/particle_construction_scope/PROOF.md" target="_blank" rel="noopener">Replayed finite candidate calculations and their exact scope ↗</a></div>

    <h2 style="margin:30px 0 4px">The protected defects of the run</h2>
    <p class="small" style="max-width:1100px;margin:0 0 10px">${esc(D.rule)} The defects themselves are not named as particles. The passive register is gauge-neutral in the audited implementation. Its geometric classes and repair-event counts are recorded directly; their mismatch with a proposed equal-state census is not a thermal-abundance measurement. The matching centre label is shown on the candidate tiles above.</p>
    <div class="mp-kinds">${D.kinds.map((k) => kindCard(k, maxPath)).join('')}</div>
    <div class="note" style="margin-top:8px">${esc(D.h2.note)} ${esc(D.h2.composite)} Colour flux of the gas: ${(D.h2.plaquettes_per_carrier.Z3 || 0).toFixed(3)} per carrier, weak flux ${(D.h2.plaquettes_per_carrier.Z2 || 0).toFixed(3)}. The working law has one bit per link, so only the weak flux is its own; the colour flux of these runs uses a declared link register of order six. Of the 32 patterns 28 see some flux.</div>

    <h2 style="margin:30px 0 4px">What sits together on a carrier</h2>
    <p class="small" style="max-width:1100px;margin:0 0 8px">${esc(D.constraint)} Level-10 run, ${D.carriers.toLocaleString('en-US')} carriers, ${(100 * D.empty_share).toFixed(1)} % of them empty. Shares are of the carriers that hold defects.</p>
    <div class="cos-bar" style="height:22px">${groupsBar}</div>
    <table class="t" style="width:100%;margin-top:8px"><tr><th>content of the carrier</th><th>classes</th><th>same charge content as</th><th>carriers</th><th>share</th></tr>${groupsRows}</table>
    <div class="note" style="margin-top:8px">${esc(D.balance)} ${esc(D.hadron_gas)} ${esc(D.fusion || '')}</div>
    <div class="mp-react">${D.reactions.map((r) => `<div><b>${r[1].toLocaleString('en-US')}</b><span>${esc(r[0])}</span><em>${esc(r[2])}</em></div>`).join('')}</div>

    <h2 style="margin:30px 0 4px">Two readings of the same catalogue</h2>
    <div class="mp-two">
      <div class="card"><div class="k" style="margin-top:0">Reading A · particles are charged quanta, defects are vortices</div>
        <table class="t" style="width:100%"><tr><th>in nature</th><th>in the model</th><th>match</th><th>missing</th></tr>${A.map((r) => `<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td>${r[2]}</td><td class="small">${r[3]}</td></tr>`).join('')}</table></div>
      <div class="card"><div class="k" style="margin-top:0">Reading B · a defect of class k is the multiplet of charge k <span class="cos-tag open">centre data do not determine a particle</span></div>
        <table class="t" style="width:100%"><tr><th>class</th><th>multiplet with the same charge class</th><th>as a flux</th></tr>
        ${[[1, 'quark doublet, left-handed'], [5, 'antiquark doublet'], [4, 'up and down quark, right-handed'], [2, 'their antiparticles'], [3, 'lepton doublet, left-handed, and its antiparticle']].map(([k, n]) => `<tr><td>${marker(k, 18)} ${k}</td><td><b>${n}</b></td><td class="small">${esc(FAM[k].label)} · ${esc(FAM[k].detail)}</td></tr>`).join('')}
        <tr><td><span class="mp-cls flat">∅</span> 0</td><td><b>electron and neutrino, right-handed</b></td><td class="small">flat face, no flux</td></tr></table>
        <div class="note"><b>What the scoring establishes:</b> the recorded quench shares differ from the proposed equal-per-internal-state census. The quoted score uses variation across five lattice levels, not a demonstrated thermal ensemble or independent particle-production samples. It does not exclude every possible particle assignment. The exact obstruction is information loss: a centre class does not fix the full representation or physical parity. Neutrality also does not decide between the readings. <b>Reading C</b>, a quantum carrying a flux pair, remains a declared candidate construction, without a quantified probability of identification.</div></div>
    </div>

    <div class="card" style="margin-top:14px;border-color:rgba(255,228,92,.45)"><div class="k" style="margin-top:0">Reading C · a quantum that carries a defect <span class="mp-badge">closest to "a particle that is a defect"</span></div>
      <p style="margin:6px 0 8px;font-size:13.5px">A flux pair of a carrier does not travel by itself: it has no pointer and no charge. A charged quantum can carry it as cargo, with one added declared gate. Then the pattern gives the charges, and the defect gives a conserved label and a rest phase. On a line of 256 carriers its speed equals the prediction to 1 part in 10⁸.</p>
      <table class="t" style="width:100%"><tr><th>cargo: a flux pair whose two faces are</th><th>1 apart</th><th>2</th><th>3</th><th>4</th><th>5</th></tr>
        <tr><td>charge phase 1/6 (a quark doublet on classes 1, 5)</td><td class="num">0.0424</td><td class="num">0.0510</td><td class="num">0.0562</td><td class="num">0.0583</td><td class="num">0.0593</td></tr>
        <tr><td>charge phase 1/3 (any quark on classes 2, 4)</td><td class="num">0.0751</td><td class="num">0.0928</td><td class="num">0.1064</td><td class="num">0.1131</td><td class="num">0.1169</td></tr>
        <tr><td>charge phase 1/2 (a weak doublet on class 3)</td><td class="num">0.0878</td><td class="num">0.1106</td><td class="num">0.1324</td><td class="num">0.1478</td><td class="num">0.1712</td></tr></table>
      <div class="note">Rest phase per tick of the composite. Each row is a ladder of five levels of one kind of particle. These are numbers of one carrier at the scale of the lattice, not masses in physical units. Stationary pairs of the run never sit one face apart.</div></div>

    <h2 style="margin:30px 0 4px">Mass, and what is not in the standard list</h2>
    <div class="mp-two">
      <div class="card">${D.mass.map((m) => `<p style="margin:0 0 8px;font-size:13.5px">${esc(m)}</p>`).join('')}</div>
      <div class="card"><div class="k" style="margin-top:0">${esc(D.dark.name)} · best guess: ${esc(D.dark.guess)}</div><ul style="margin:4px 0 10px 18px;padding:0;font-size:13px;color:var(--soft)">${D.dark.facts.map((f) => `<li>${esc(f)}</li>`).join('')}</ul>
        <div class="k">${esc(D.tower.name)} · best guess: ${esc(D.tower.guess)}</div>
        <div class="mp-tower">${D.tower.angles.map((a) => `<div><b>${a[1].toFixed(4)}</b><span>${esc(a[0])}</span></div>`).join('')}</div><div class="small">rest phase per tick, no free parameter</div></div>
    </div>


    <h2 style="margin:30px 0 4px">Electron comparison for this candidate walk</h2>
    <p class="small" style="max-width:1100px;margin:0 0 8px">This table compares a specified candidate walk with known electron properties. It is a retrospective comparison, not a blind test or a statement about every native quantum construction.</p>
    <table class="t" style="width:100%"><tr><th>property</th><th>model</th><th>verdict</th></tr>
      ${[['Electric charge, colour, weak charge, hypercharge', 'both handedness classes carry the electron’s values, as labels of the pattern', 'ok', 'equal'],
         ['Spin and statistics', 'spin one half by the routing; statistics by declaration', 'open', 'declared'],
         ['Charge as a coupling, stability, magnetic moment', 'no electromagnetic link is coupled to the charge yet. With declared link phases the lowest level in a field stays at the rest angle, which is the value 2 of the magnetic factor', 'open', 'not comparable'],
         ['Mass', 'no option gives it: the law leaves the family massless', 'bad', 'fails'],
         ['Mass from the frozen flux of the run', 'the right-handed electron stays massless, the neutrino gets the electron’s angle, up equals down. Quark to lepton ratio 3.2 at the end of the run and 6.7 in the limit, against 4.2 for up and 9.2 for down over the electron', 'bad', 'fails'],
        ].map((r) => `<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td><span class="cos-tag ${r[2]}">${r[3]}</span></td></tr>`).join('')}
    </table>
    <div class="note" style="margin-top:8px"><b>What a passing mass row needs:</b> one angle that joins the two handedness classes of a particle; a register that takes the charge the toggle moves; a strength per particle; a law for the electron’s angle, which is below 10⁻⁵ per tick if the top quark shares the clock, four orders below every carrier number so far; and one cone per particle.</div>

    <h2 style="margin:30px 0 4px">Three copies of every particle</h2>
    <div class="mp-two">
      <div class="card"><div class="k" style="margin-top:0">What the finite theorem stack offers</div>
        <ul style="margin:4px 0 0 18px;padding:0;font-size:13.5px;color:var(--soft)">
          <li><b>The candidate:</b> the band of three port vectors of the twelve-port carrier, at seam cost 5 − √5. A machine-checked theorem selects it, under three premises that are not proved: one complete faithful band, a size between 3 and 5, and comparison by seam cost.</li>
          <li><b>Gauge fixing does not give the number three.</b> The stack says so itself: the cancellation of anomalies holds copy by copy and does not fix the number of copies. In its own registers three copies stay a selection.</li>
          <li><b>Equal masses unless something breaks the symmetry.</b> If the three copies are the unbroken band and the Higgs does not tell them apart, all three get the same mass. A background has to split them.</li>
        </ul></div>
      <div class="card"><div class="k" style="margin-top:0">Finite calculation: a flux pair on the carrier splits the band</div>
        <table class="t" style="width:100%"><tr><th>faces of the pair</th><th>copy 1</th><th>copy 2</th><th>copy 3</th><th>spread</th></tr>
          <tr><td>1 apart</td><td class="num">0.2772</td><td class="num">0.3047</td><td class="num">0.3360</td><td class="num">38 %</td></tr>
          <tr><td>2 apart</td><td class="num">0.2525</td><td class="num">0.3047</td><td class="num">0.3555</td><td class="num">66 %</td></tr>
          <tr><td>3 apart</td><td class="num">0.2192</td><td class="num">0.3179</td><td class="num">0.3768</td><td class="num">96 %</td></tr>
          <tr><td>4 apart</td><td class="num">0.1979</td><td class="num">0.3316</td><td class="num">0.3856</td><td class="num">110 %</td></tr></table>
        <div class="note">Rest phase per tick of the three levels that grow out of the band, for a pattern that sees the pair with half a turn. Spread is the difference of the seam costs over their mean. Without the pair all three sit at 0.3047. So a defect does give three copies three different values, by 17 to 120 percent. Nature needs factors of 17 to 3,500 between copies. The invariant that measures a difference between matter and antimatter is zero for one pair.</div></div>
    </div>

    <div class="card" style="margin-top:14px"><div class="k" style="margin-top:0">Finite calculation: a point-like source of mass makes the copies unequal</div>
      <p style="margin:6px 0 8px;font-size:13.5px">The tick has four copies of every species on one lattice, its four candidate low-energy cones. A source of mass that sits on single carriers does not treat them alike. Exact calculation on 8³ carriers:</p>
      <table class="t" style="width:100%"><tr><th>source carriers in the box</th><th>copies that get a rest phase</th><th>their values, in units of the average</th></tr>
        <tr><td>1</td><td>1 of 4</td><td class="mono">4 · 0 · 0 · 0</td></tr>
        <tr><td>2</td><td>2 of 4</td><td class="mono">3 · 1 · 0 · 0</td></tr>
        <tr><td>4</td><td>3 of 4</td><td class="mono">2.4 · 1.3 · 0.3 · 0</td></tr>
        <tr><td>every carrier</td><td>4 of 4</td><td class="mono">1 · 1 · 1 · 1</td></tr></table>
      <div class="note">One heavy copy and lighter ones is the pattern of nature. Here it appears when the state that gives mass is sparse or ordered with a period of four carriers, and it disappears when that state is uniform. The heavy copy is a pair of both handedness classes, as a massive particle must be. Two species whose sources sit on the same cells mix by a plain reordering; on the other half of the cells every mixing entry is one quarter; a split between the two gives angles in between. This is a mechanism to test, not a derivation: the model has four copies where nature has three, two of them mirror copies, and nothing yet fixes the order of the vacuum.</div></div>

    <h2 style="margin:30px 0 4px">What a rest mass needs, premise by premise</h2>
    <p class="small" style="max-width:1100px;margin:0 0 8px">Dependencies for this candidate walk, compared with the public source constructions. Its table of source-derived Standard Model GeV masses is empty. Finite source energies and controlled free-field rest frequencies have their own proved entries.</p>
    <table class="t" style="width:100%"><tr><th>premise</th><th>what it supplies</th><th>status in the register</th></tr>
      ${[['one clock and one unit of energy', 'the anchor that turns a rest phase into a mass', 'imported', 'open'],
         ['rules that select the band of three', 'three copies', 'built on declared inputs', 'open'],
         ['spin and handedness', 'fermions with the right chirality', 'declared', 'open'],
         ['the Higgs carrier and its kinetic term', 'the field that gives mass', 'declared', 'open'],
         ['the Higgs potential and the breaking', 'the value of the field in the vacuum', 'declared', 'open'],
         ['the Yukawa matrices', 'one mass per particle and copy', 'declared', 'open'],
         ['propagation, frames, comparison', 'a moving particle with a cone', 'proved on distinct free-field constructions; same-source interacting attachment required', 'open'],
         ['gauge fields, currents and action', 'finite physical Gauss/current and field-response constructions exist; observed interaction matching is separate', 'finite results; common field limit required', 'open'],
         ['the effective field sector and its physical scale', 'controlled free Dirac and Proca record limits on declared finite families', 'interacting limit and physical scale required', 'open'],
        ].map((r) => `<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td><span class="cos-tag ${r[3]}">${r[2]}</span></td></tr>`).join('')}
    </table>
    <div class="mp-issues" style="margin-top:12px">${[
      ['A reference state with interaction', 'One state under the gauge constraints on which the native repair acts.', 'audit; theory worker R1'],
      ['A coupling that is not zero', 'The native repair acts as zero on the fifteen-state module. A mass needs a coupling that does not, and that passes the charge it moves to a register.', 'audit; theory worker R8; proposal: the class-3 faces'],
      ['Handedness without mirror partners', 'A chiral family, not a vector-like one.', 'audit; theory worker R4'],
      ['An interacting limit with a cone', 'The interaction must keep the relativistic behaviour.', 'theory workers R3, R5; runs A, B, G of this session'],
      ['The state of the Higgs and the strengths', 'A vacuum value that belongs to a stationary state and one strength per particle.', 'open; the flux of the runs is a relic of the start'],
      ['The readout of a mass', 'A pole or a stable resonance from complete detector records, on the interacting program.', 'theory worker R6; runs D, E'],
      ['The unit', 'One measured anchor.', 'theory worker R7'],
      ['Three copies with different masses', 'A background that splits the band strongly, and a nonzero difference between matter and antimatter.', 'theory worker X4; band splitting executed: 17 to 120 percent'],
    ].map((x, i) => `<div class="card"><b>${i + 1}. ${x[0]}</b><p>${x[1]}</p><span>${x[2]}</span></div>`).join('')}</div>
    <h2 style="margin:30px 0 4px">Finite outputs of these declared candidate programs</h2>
    <p class="small" style="max-width:1100px;margin:0 0 8px">Rest phases per tick, exact or executed. The clock is the period of two ticks. No row is a mass of a known particle.</p>
    <table class="t" style="width:100%"><tr><th>object</th><th>rest phase per tick</th><th>preparation and readout</th><th>status</th></tr>
      ${[['every species of the map, no mass step', '0', 'blank packet; shell radius and record statistics', 'executed to 16.8 million carriers', 'ok'],
         ['internal sectors of a carrier as the mass step', '0 · 0.3047 · 0.3672 · 0.4510 · 0.4962 · 0.5857', 'rest state of each sector; record after one tick. From sampled records: 0.3015 ± 0.0030 and 0.4510 ± 0.0045', 'no free number; lattice scale; executed check running', 'open'],
         ['a quantum with a flux pair as cargo', '0.042 to 0.171 by charge phase and face separation', 'packet on a line of 256 carriers; speed equals the prediction to 1 part in 10⁸', 'declared gate; lattice scale', 'open'],
         ['a quantum in the frozen flux of the level-10 run', 'quark doublets 0.0351 · quark singlets 0.0323 · lepton doublets 0.0114 · singlets 0', 'plane wave at rest; return amplitude. Density times angle to 0.3 percent at 16.8 million carriers', 'four declared readings stacked; set by the preparation; fails for the electron', 'bad'],
         ['the travelling flux pair of the audit', '2 source units', 'particle and hole of a filled reference; energy instrument with 36 records', 'declared face-count rule; gauge singlet', 'open'],
         ['the toggle of the level s2 with its register (the Higgs reading)', 'not fixed', 'one bit per carrier takes the charge the toggle moves; the tick then commutes with the constraint to 2 × 10⁻¹⁶. A register in number states converts handedness and gives no rest phase; a rest phase needs a register state with a common phase', 'four strengths allowed, one each for up type, down type, charged lepton and neutrino; none fixed; no flux background can supply them', 'bad'],
        ].map((r) => `<tr><td><b>${r[0]}</b></td><td class="mono">${r[1]}</td><td class="small">${r[2]}</td><td><span class="cos-tag ${r[4]}">${r[3]}</span></td></tr>`).join('')}
    </table>

    <h2 style="margin:30px 0 4px">What keeps this from being a literal table</h2>
    <div class="mp-issues">${ISSUES.map((x, i) => `<div class="card"><b>${i + 1}. ${x[0]}</b><p>${x[1]}</p><span>${x[2]}</span></div>`).join('')}</div>
  </div>`;
  host.querySelectorAll('.mp-tile[data-k]').forEach((el) => el.onclick = () => { picked = el.dataset.k; host.querySelectorAll('.mp-tile[data-k]').forEach((t) => t.classList.toggle('sel', t.dataset.k === picked)); $('mp-detail').innerHTML = detail(); });
}
