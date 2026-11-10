// The particle chart: colour-coded groups, every known property of each particle, the best-guess carrier in the
// model, what the selected run shows for it, and where rest masses stand.
import { $, S, FAM, fmt, marker } from './app.js';

const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
let anti = false, picked = 'electron';

function massText(t) {
  const m = t.measured_mass_GeV;
  if (m === null || m === undefined) return t.target.includes('neutrino') ? '&lt; 0.8 eV' : '–';
  if (m === 0) return '0';
  if (m < 1e-3) return (m * 1e3).toPrecision(3) + ' MeV';
  if (m < 1) return (m * 1e3).toPrecision(4).replace(/\.?0+$/, '') + ' MeV';
  return m.toPrecision(m < 10 ? 4 : 5).replace(/\.?0+$/, '') + ' GeV';
}
const flip = (q) => (q === '0' ? '0' : q.startsWith('+') ? '−' + q.slice(1) : '+' + q.slice(1));
function classesOf(target) { return (S.pmap.targets.find((t) => t.target === target) || {}).z6_classes_of_components || []; }
const pat = (p) => '(' + p[0] + ',' + p[1] + ')';
function patternsOf(target) { return (S.pmap.target_sectors || {})[target] || []; }
function carrierShort(c) {
  const ps = patternsOf(c.target);
  if (ps.length) return ps.map((x) => pat(x.pattern)).join(' · ');
  if (c.group === 'higgs') return 'link';
  return c.group === 'photon' ? 'wave' : 'response';
}

function tile(c, G) {
  const t = S.pmap.targets.find((x) => x.target === c.target), g = G[c.group];
  const q = c.properties['electric charge'];
  const self = c.symbol === c.antisymbol;
  return `<div class="tile${anti && !self ? ' anti' : ''}${picked === c.target ? ' sel' : ''}" data-t="${c.target}" style="--c:${g.color};grid-row:${c.row + 1};grid-column:${c.column + 1}">
    <div class="t-top"><span>${massText(t)}</span><span>${anti && !self ? flip(q) : q}</span></div>
    <div class="t-sym">${anti ? c.antisymbol : c.symbol}</div>
    <div class="t-name">${anti && !self ? 'anti-' : ''}${esc(c.name)}</div>
    <div class="t-foot"><span>spin ${c.properties.spin}</span><span class="t-car">${carrierShort(c)}</span></div></div>`;
}

function runRows(classes) {
  const st = S.classStats && S.classStats.tiers && S.classStats.tiers.screen ? S.classStats.tiers.screen.classes : null;
  if (!st || (S.meta && S.meta.config.twists)) return '';
  let h = '';
  for (const k of classes) {
    const r = st[k] || st[String(k)];
    if (!FAM[k]) continue;
    h += `<div class="row"><span style="display:flex;gap:7px;align-items:center">${marker(k, 14)} class ${k}: ${FAM[k].label}</span>
      <b>${r && r.ever ? fmt(r.ever) + ' seen · ' + fmt(r.alive_at_end) + ' at the end · ' + (r.hops_per_depth ? r.hops_per_depth.toFixed(4) : '0') + ' moves per causal step' : 'none in this run'}</b></div>`;
  }
  return h;
}

function patternRows(c) {
  const secs = S.pmap.sectors || [], ps = patternsOf(c.target);
  let h = '';
  for (const x of ps) {
    const s = secs.find((r) => r.n3 === x.pattern[0] && r.n2 === x.pattern[1]);
    if (!s) continue;
    const cp = anti ? s.conjugate : x.pattern, cs = anti ? secs.find((r) => r.n3 === cp[0] && r.n2 === cp[1]) : s;
    h += `<div class="row"><span>pattern ${pat(cp)} · ${esc(x.role)}</span><b>${cs.states} state${cs.states > 1 ? 's' : ''} · 6Y = ${cs.six_Y > 0 ? '+' : ''}${cs.six_Y} · ${cs.handedness}-handed</b></div>`;
  }
  const r = (S.pmap.response_of || {})[c.target];
  if (r) h += `<div class="row" style="display:block"><span>a response, not a pattern</span><div class="small" style="color:var(--soft);margin-top:3px">${esc(r)}</div></div>`;
  return h;
}

function sectorSection(G) {
  const secs = S.pmap.sectors, M = S.pmap.internal_modes;
  if (!secs || !M) return '';
  const lv = M.levels.map((l) => `<span class="lvl ${l.kind}"><b>${l.kind === 'vector' ? 'v' : 's'}</b><em>${l.charge > 0 ? '+' : ''}${l.charge}</em></span>`).join('');
  const cell = (n3, n2) => {
    const s = secs.find((r) => r.n3 === n3 && r.n2 === n2), g = G[s.group] || { color: '#93a3c0' };
    const dots = '●'.repeat(n3) + '○'.repeat(3 - n3) + ' ' + '◆'.repeat(n2) + '◇'.repeat(2 - n2);
    const names = s.targets.length ? s.targets.map((x) => (S.pmap.chart.find((c) => c.target === x) || {}).symbol || x).join(' ') : '';
    return `<div class="sec-cell" style="--c:${g.color}" title="${esc(s.multiplet)}"><div class="sec-top"><span>${pat([n3, n2])}</span><span>${s.states} state${s.states > 1 ? 's' : ''}</span></div>
      <div class="sec-dots">${dots}</div><div class="sec-name">${esc(s.multiplet)}</div>
      <div class="sec-foot"><span>6Y ${s.six_Y > 0 ? '+' : ''}${s.six_Y}</span><span>charge mod 6: ${s.z6_class}</span><span>${s.handedness}</span></div>
      ${names ? `<div class="sec-sym">${names}</div>` : ''}</div>`;
  };
  let grid = '';
  for (let n2 = 0; n2 < 3; n2++) for (let n3 = 0; n3 < 4; n3++) grid += cell(n3, n2);
  return `<h2 id="patterns" style="margin-top:30px">Internal patterns · revised best guess</h2>
    <p>${esc(M.statement)}</p>
    <div class="lvl-row"><span class="small">the five levels and their charges</span>${lv}</div>
    <div class="sec-grid">${grid}</div>
    <div class="note" style="margin-top:10px">${esc(M.origin)}</div>
    <div class="card" style="margin-top:10px"><div class="k" style="margin-top:0">The Higgs and the gauge fields are not patterns</div>
      <div style="font-size:13px;color:var(--soft)">${esc(M.higgs || '')}</div>
      ${(S.pmap.yukawa_links || []).map((l) => `<div class="row"><span>${esc(l.what)}</span><b>${pat(l.left)} ⟷ ${pat(l.right)} · mass term: declared, no value</b></div>`).join('')}</div>
    <div class="note">${esc(M.checks || '')}</div>
    <div class="note">${esc(M.fluxes || '')}</div>
    <div class="note">${esc(M.splits)}</div>
    <div class="note" style="color:var(--amber)">${esc(M.status)}</div>`;
}

// ------------------------------------------------------------------ best-guess identification
let occ = [0, 1, 0, 1, 0];
function stateOf(o) {
  const n3 = o[0] + o[1] + o[2], n2 = o[3] + o[4];
  const sec = S.pmap.sectors.find((r) => r.n3 === n3 && r.n2 === n2), names = S.pmap.state_names[n3 + ',' + n2];
  const nm = names['*'] || names[o[3] ? 's1' : 's2'];
  const q3 = -n3 + 3 * o[3], q = q3 === 0 ? '0' : (q3 % 3 === 0 ? (q3 > 0 ? '+' : '−') + Math.abs(q3 / 3) : (q3 > 0 ? '+' : '−') + Math.abs(q3) + '/3');
  const cols = S.pmap.levels.slice(0, 3).map((l) => l.colour);
  const colour = n3 === 1 ? cols[o.slice(0, 3).indexOf(1)] : n3 === 2 ? 'anti-' + cols[o.slice(0, 3).indexOf(0)] : 'none';
  return { n3, n2, sec, symbol: nm[0], name: nm[1], group: nm[2], charge: q, colour };
}
function builderSection(G) {
  if (!S.pmap.levels || !S.pmap.state_names) return '';
  const st = stateOf(occ), g = G[st.group] || { color: '#93a3c0' }, sec = st.sec;
  const chips = S.pmap.levels.map((l, i) => `<button class="lvl-btn ${l.kind}${occ[i] ? ' on' : ''}" data-i="${i}"><b>${l.kind === 'vector' ? 'v' + (i + 1) : 's' + (i - 2)}</b><em>${occ[i] ? 'filled' : 'empty'}</em><span>${l.kind === 'vector' ? esc(l.colour) + ' · −1/3' : (i === 3 ? 'charged · +1' : 'neutral · 0')}</span></button>`).join('');
  const seen = [1, 5, 4, 2, 3].filter((k) => ((k * sec.six_Y) % 6 + 6) % 6 !== 0);
  const hig = occ.slice(); hig[4] = 1 - hig[4]; const hs = stateOf(hig);
  const ant = occ.map((x) => 1 - x), as = stateOf(ant);
  const partners = [];
  if (st.n2 === 1) { const o2 = occ.slice(); o2[3] = 1 - o2[3]; o2[4] = 1 - o2[4]; partners.push('weak partner: ' + stateOf(o2).name); }
  if (st.n3 === 1 || st.n3 === 2) partners.push('two more colour states of the same particle');
  const row = (a, b) => `<div class="row"><span>${a}</span><b>${b}</b></div>`;
  return `<h2 id="builder" style="margin-top:26px">Identify a particle: fill the five levels</h2>
    <p>Matter in the coherent model is a quantum with five internal levels. Which levels are filled is the best guess for which particle it is. Click the levels.</p>
    <div class="builder"><div class="lvl-btns">${chips}</div>
      <div class="card b-card" style="--c:${g.color}"><div class="d-head" style="--c:${g.color}"><div class="d-sym">${st.symbol}</div>
        <div><div class="d-name">${esc(st.name)}</div><div class="d-group">pattern ${pat([st.n3, st.n2])} · ${sec.states} state${sec.states > 1 ? 's' : ''} in this multiplet · best guess</div></div></div>
        ${row('electric charge', st.charge)}${row('colour', st.colour)}${row('weak isospin', sec.weak + (st.n2 === 1 ? (occ[3] ? ' · upper member' : ' · lower member') : ''))}
        ${row('six times the hypercharge', (sec.six_Y > 0 ? '+' : '') + sec.six_Y)}${row('handedness', sec.handedness + '-handed')}
        ${row('centre fluxes that see it', seen.length ? seen.map((k) => marker(k, 13) + ' ' + k).join(' &nbsp; ') : 'none')}
        ${row('gauge partners', partners.length ? partners.join(' · ') : 'none: a singlet')}
        ${row('across the mass link (the neutral spinor level flipped)', esc(hs.name) + ' ' + pat([hs.n3, hs.n2]))}
        ${row('antiparticle (every level flipped)', esc(as.name) + ' ' + pat([as.n3, as.n2]))}
        ${row('in the executable model', st.n3 + st.n2 === 1 ? 'one quantum of the public walk' : st.n3 + st.n2 === 0 ? 'the empty state' : 'needs ' + (st.n3 + st.n2) + ' quanta in one orbital: the occupation lift')}
        ${row('rest mass from the model', '<span style="color:var(--amber)">none yet</span>')}</div></div>
    <div class="note">Generations are not separated: an up quark, a charm quark and a top quark are the same filling. The names are a labelling of charges, not an identification.</div>`;
}
function objectsSection(G) {
  const O = S.pmap.model_objects; if (!O) return '';
  let rows = '';
  for (const o of O) { const g = G[o.group] || { color: '#93a3c0' };
    rows += `<tr><td><b style="color:${g.color}">${esc(o.object)}</b></td><td>${esc(o.best_guess)}</td><td class="small">${esc(o.in_model)}</td><td class="small">${esc(o.confirmed)}</td><td class="small">${esc(o.missing)}</td><td><span class="cand" style="color:${g.color};border-color:${g.color}55">${esc(o.status)}</span></td></tr>`; }
  return `<h2 id="objects" style="margin-top:30px">What the executable model has, and what each object is at best guess</h2>
    <p>Objects of the coherent programs of the theory repository and of this simulation. The audit's own table of candidates has the same content.</p>
    <table class="t objs"><tr><th>object in the model</th><th>best guess</th><th>exists as</th><th>confirmed</th><th>missing</th><th>status</th></tr>${rows}</table>`;
}
function generationsSection() {
  const N = S.pmap.generations; if (!N) return '';
  const rows = N.mixing.map((m) => `<div class="row"><span>sin² ${esc(m.angle)}</span><b>pattern ${m.model.toFixed(4)} = ${esc(m.model_exact)} · measured ${esc(m.measured)}</b></div>`).join('');
  return `<h2 id="generations" style="margin-top:30px">Three generations · best guess</h2><p>${esc(N.statement)}</p>
    <div class="card" style="max-width:760px">${rows}<div class="row"><span>status</span><b style="color:var(--amber)">${esc(N.status)}</b></div><div class="note">${esc(N.caveats)}</div></div>`;
}
function scoreSection(G) {
  const TS = S.pmap.target_sectors, RO = S.pmap.response_of; if (!TS) return '';
  let rows = '';
  for (const c of S.pmap.chart.filter((x) => x.target !== 'W_minus')) {
    const t = S.pmap.targets.find((x) => x.target === c.target), g = G[c.group], ps = TS[c.target] || [];
    const obj = ps.length ? 'patterns ' + ps.map((x) => pat(x.pattern)).join(' and ') : (RO[c.target] || '');
    const exists = ps.length ? 'labels on the five levels' : c.group === 'higgs' ? 'a declared operator' : 'exact algebra of the port responses';
    const status = ps.length ? 'labelled only' : c.group === 'higgs' ? 'declared, no value' : 'algebra only';
    rows += `<tr><td><b style="color:${g.color}">${c.symbol}</b> ${esc(c.name)}</td><td class="num">${massText(t)}</td><td>${esc(obj)}</td><td class="small">${exists}</td><td class="num" style="color:var(--amber)">none</td><td><span class="cand" style="color:${g.color};border-color:${g.color}55">${status}</span></td></tr>`;
  }
  return `<h2 id="scoreboard" style="margin-top:30px">Scoreboard: every measured particle against the model</h2>
    <p>One row per particle. The last two columns fill when a source-derived number exists and has been frozen before comparison. Today no row has one.</p>
    <table class="t objs"><tr><th>particle</th><th>measured mass</th><th>best-guess object in the model</th><th>exists as</th><th>mass from the model</th><th>status</th></tr>${rows}</table>
    <div class="note" style="margin-top:8px">Every massive branch computed so far sits at the lattice cutoff, and one mass link gives one common mass to all four kinds of fermion. Both have to change before a row can carry a number.</div>`;
}

function censusSection() {
  const st = S.classStats && S.classStats.tiers && S.classStats.tiers.screen ? S.classStats.tiers.screen.classes : null;
  const secs = S.pmap.sectors;
  if (!st || !secs || (S.meta && S.meta.config.twists)) return '';
  const order = [1, 5, 4, 2, 3], ex = {}, ever = {}, alive = {};
  let et = 0, te = 0, ta = 0;
  for (const k of order) {
    ex[k] = secs.filter((s) => s.z6_class === k).reduce((a, s) => a + s.states, 0); et += ex[k];
    const r = st[k] || st[String(k)] || {}; ever[k] = r.ever || 0; alive[k] = r.alive_at_end || 0; te += ever[k]; ta += alive[k];
  }
  if (!te) return '';
  const pct = (x) => (100 * x).toFixed(1) + ' %';
  const bar = (x, c, dash) => `<span class="cbar"><i style="width:${(100 * x / 0.3).toFixed(1)}%;background:${dash ? 'repeating-linear-gradient(90deg,' + c + ' 0 6px,transparent 6px 10px)' : c}"></i></span>`;
  let rows = '';
  for (const k of order) rows += `<tr><td style="white-space:nowrap">${marker(k, 14)} <b style="color:${FAM[k].color}">${FAM[k].label}</b></td>
    <td class="num">${fmt(ever[k])}</td><td>${bar(ever[k] / te, FAM[k].color)}<span class="small"> ${pct(ever[k] / te)}</span></td>
    <td>${bar(ex[k] / et, FAM[k].color, true)}<span class="small"> ${pct(ex[k] / et)} · ${ex[k]} states</span></td>
    <td class="num">${fmt(alive[k])}</td><td class="num">${ta ? pct(alive[k] / ta) : '–'}</td></tr>`;
  const sig = (a, b) => { const d = a - b, s = Math.sqrt(Math.max(1, a + b)); return `${fmt(a)} against ${fmt(b)}: difference ${Math.abs(d / s).toFixed(1)} standard deviations`; };
  return `<h2 id="census" style="margin-top:30px">Census of the flux classes of run ${esc(S.name)}</h2>
    <p>The defects of this run are fluxes of the centre, counted by class. The solid bars are what the run produced. The dashed bars are a comparison only:
      the shares that particles would have in a hot early universe if flux class k were read as the particle of charge k, 6, 6, 6, 6 and 4 states per generation. Nothing in the construction supplies that reading.</p>
    <table class="t census"><tr><th>flux class</th><th>ever seen</th><th>share in the run</th><th>comparison: charges in a hot universe</th><th>at the end</th><th>share at the end</th></tr>${rows}</table>
    <div class="note" style="margin-top:8px">A flux and its conjugate: ${sig(ever[1], ever[5])} for classes 1 and 5, ${sig(ever[4], ever[2])} for classes 4 and 2. The run makes them in equal numbers.</div>
    <div class="note">The mobile classes 1 and 5 meet and annihilate; the classes that cannot move remain. The defect rule of these runs is a declared register rule of the simulation, so this census tests that rule, not the theory. Matter is not in these runs: quanta with internal levels exist only in the coherent model of the Carrier tab so far.</div>`;
}

function detail(G) {
  const c = S.pmap.chart.find((x) => x.target === picked), t = S.pmap.targets.find((x) => x.target === picked), g = G[c.group];
  const self = c.symbol === c.antisymbol;
  const props = Object.entries(c.properties).map(([k, v]) => `<div class="row"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('');
  const secs0 = S.pmap.sectors || [];
  const charges = patternsOf(c.target).map((x) => (secs0.find((r) => r.n3 === x.pattern[0] && r.n2 === x.pattern[1]) || {}).six_Y).filter((q) => q !== undefined);
  const ks = [1, 5, 4, 2, 3].filter((k) => charges.some((q) => ((k * q) % 6 + 6) % 6 !== 0));
  const missing = (t.missing || []).map((m) => `<li>${esc(m)}</li>`).join('');
  $('chart-detail').innerHTML = `
    <div class="d-head" style="--c:${g.color}"><div class="d-sym">${c.symbol}</div>
      <div><div class="d-name">${esc(c.name)}</div><div class="d-group">${esc(g.label)}${c.generation ? ' · generation ' + c.generation : ''}</div></div></div>
    <div class="k">Known properties</div>
    <div class="card">${`<div class="row"><span>rest mass, measured</span><b>${massText(t)}${t.measured_mass_note ? ' <span class="small">(' + esc(t.measured_mass_note) + ')</span>' : ''}</b></div>`}
      ${props}<div class="row"><span>antiparticle</span><b>${self ? 'its own' : c.antisymbol + ' · ' + esc((t.benchmark.antiparticle || '').replace(/^[^ ]+ ?/, '').replace(/[()]/g, ''))}</b></div></div>
    <div class="k">In the model · best guess</div>
    <div class="card"><div style="color:var(--soft);font-size:13.5px;margin-bottom:6px">${esc(c.carrier)}</div>
      ${patternRows(c)}
      ${runRows(ks) ? `<div class="small" style="margin:8px 0 2px">fluxes of run ${esc(S.name)} that this particle would see</div>` + runRows(ks) : (charges.length ? '<div class="small" style="margin:8px 0 2px">no flux class of the runs is seen by this particle</div>' : '')}
      <div class="row"><span>status</span><b style="color:var(--amber)">${esc(t.status)}: not an identified particle</b></div>
      <div class="row" style="display:block"><span>what the source has</span><div class="small" style="color:var(--soft);margin-top:3px">${esc(t.source_test_summary || t.best_guess || '')}</div></div></div>
    <div class="k">Rest mass from the source</div>
    <div class="card"><div class="row"><span>derived value</span><b style="color:var(--amber)">none yet</b></div>
      <div class="row" style="display:block"><span>what a derivation has to deliver</span><div class="small" style="color:var(--soft);margin-top:3px">${esc(t.required_mass || '')}</div></div>
      ${missing ? `<div class="row" style="display:block"><span>still missing</span><ul class="small" style="margin:4px 0 0 16px;padding:0;color:var(--soft)">${missing}</ul></div>` : ''}</div>`;
}

function massSection() {
  const M = S.pmap.mass_status;
  if (!M) return '';
  const cost = M.register_cost.table;
  let h = `<h2 id="mass">Rest mass: where the model stands</h2><p>${esc(M.statement)} Several kinds of numbers exist on the model side today. None is assigned to a particle.</p>
    <div class="mass-grid"><div class="card"><div class="k" style="margin-top:0">Declared cost of a register defect</div>`;
  for (const k of [1, 5, 4, 2, 3]) h += `<div class="row"><span style="display:flex;gap:7px;align-items:center">${marker(k, 14)} ${FAM[k].label}</span><b>${cost[String(k)]} cost units</b></div>`;
  h += `<div class="note">${esc(M.register_cost.note)}</div></div>`;
  if (M.wave_bands) {
    const W = M.wave_bands, c2 = W.massless_c_squared;
    h += `<div class="card"><div class="k" style="margin-top:0">Bands of the wave operator, symmetric wiring</div>
      <div class="row"><span>massless band</span><b>speed ${Math.sqrt(c2).toFixed(4)} edges per unit pulse</b></div>`;
    if (W.crossing) h += `<div class="row"><span>two lowest bands cross in a cone</span><b>at level ${W.crossing.two_lowest_levels[0].toFixed(4)}, slope ${W.crossing.linear_slope_of_lambda_by_direction[0].toFixed(4)}</b></div>`;
    for (const b of W.moving_bands) h += `<div class="row"><span>moving band, rest frequency ${Math.sqrt(b.level).toFixed(3)}</span><b>curvature ${b.curvature_over_massless.toFixed(2)} × massless</b></div>`;
    h += `<div class="row"><span>levels that do not move</span><b>${W.immobile_levels.map((v) => Math.sqrt(v).toFixed(3)).join(' · ')}</b></div>
      <div class="note">Rest frequencies are in units of the pulse rate. A band with a rest frequency is what a mass looks like in this code, and it shares the light cone only where its curvature is 1.
        The cone where the two lowest bands cross is the one place where everything moves at one speed. ${esc(W.note)}</div></div>`;
  }
  if (M.transverse_bands) {
    const T = M.transverse_bands;
    h += `<div class="card"><div class="k" style="margin-top:0">Seam and face modes, the light candidate</div>
      <div class="row"><span>massless band</span><b>speed ${Math.sqrt(T.massless_c_squared).toFixed(4)} edges per unit pulse</b></div>`;
    for (const b of T.moving_bands) h += `<div class="row"><span>moving band, rest frequency ${Math.sqrt(b.level).toFixed(3)}</span><b>curvature ${b.curvature_over_massless.toFixed(2)} × massless</b></div>`;
    h += `<div class="row"><span>levels that do not move</span><b>${T.immobile_levels.map((v) => Math.sqrt(v).toFixed(3)).join(' · ')}</b></div><div class="note">${esc(T.note)}</div></div>`;
  }
  if (M.cone_wirings) {
    h += `<div class="card"><div class="k" style="margin-top:0">The cone sector: which wirings give a mass</div>`;
    for (const w of M.cone_wirings) {
      const v = w.slope / 2, gap = w.half_gap;
      h += `<div class="row"><span>up ports (${w.up.join(',')}) · down ports (${w.down.join(',')})</span><b>${gap < 1e-9 ? 'cone, speed ' + v.toFixed(4) : 'gap ' + gap.toFixed(4) + ', speed ' + v.toFixed(4) + ', Compton length ' + (w.slope / gap).toFixed(1) + ' edges'}</b></div>`;
    }
    h += `<div class="note">Wirings of three glued ports per carrier whose massless band is the same in every direction. The triples are given by the seam distances between the glued ports.
      Where the up and down carriers use inequivalent triples the two lowest bands separate. These are bands of the load operator, whose levels are relaxation rates of the committed rule, read near the crossing; cubic corrections are present.
      They are energies only under a declared wave clock. The router of the model never produces such a wiring.</div></div>`;
  }
  if (M.coherent_walk) {
    const W = M.coherent_walk;
    h += `<div class="card"><div class="k" style="margin-top:0">Coherent seam walk · public program</div>`;
    for (const l of W.levels) h += `<div class="row"><span>${esc(l.name)} · ${l.states} state${l.states > 1 ? 's' : ''} · kernel ${l.kernel.toFixed(6)}</span><b>${l.frequency < 1e-12 ? 'frequency 0' : 'frequency ' + l.frequency.toFixed(5) + ' rad per step'}</b></div>`;
    h += `<div class="row"><span>back at the starting port after two steps</span><b>${W.two_step_return} · committed rule ${W.committed_two_step_return}</b></div>
      <div class="row"><span>five-level factor on the walk</span><b>charge ${W.fibre.charge.join(' · ').replace(/-/g, '−')} conserved</b></div>
      <div class="row"><span>one flipped link, three-step return amplitude</span><b>${W.fibre.central_link_amplitude_change.replace('-', '−')}</b></div>
      <div class="note">${esc(W.note)} The Carrier tab runs this walk on one carrier.</div></div>`;
  }
  if (M.family_walk) {
    for (const G of M.family_walk.gluings) {
      h += `<div class="card"><div class="k" style="margin-top:0">Family walk · ${esc(G.gluing)}</div>
        <div class="row"><span>loads</span><b>massless · speed ${G.massless_speed.toFixed(5)} edges per step</b></div>`;
      for (const r of G.rows) h += `<div class="row"><span>${esc(r.sector)} · rest phase ${r.rest_phase.toFixed(4)}</span><b>${r.flat ? 'does not move' : (r.speed_ratio > 0 ? 'speed ' + r.speed_ratio.toFixed(3) + ' × massless' : 'band maximum') + ' · Compton length ' + r.compton_edges.toFixed(3) + ' edges'}</b></div>`;
      h += `<div class="note">${esc(M.family_walk.note)}</div></div>`;
    }
  }
  if (M.flux_gaps) {
    const F = M.flux_gaps;
    h += `<div class="card"><div class="k" style="margin-top:0">Flux gap of the recharted sectors, on the federation</div>
      <div class="row"><span>loads</span><b>no gap · rises as ${F.scalar_curvature.toFixed(5)} q²</b></div>`;
    for (const r of F.rows) h += `<div class="row"><span>${esc(r.sector)}</span><b>gap ${r.level.toFixed(4)} · ${r.flat_branches} of ${r.branches} branches flat${r.moving_curvature.length ? ' · moving branch ' + r.moving_curvature.map((c) => c.toFixed(5)).join(', ') + ' q²' : ''}</b></div>`;
    h += `<div class="note">${esc(F.note)}</div></div>`;
  }
  if (M.integer_filling) h += `<div class="card"><div class="k" style="margin-top:0">A native particle and antiparticle</div>
    <div class="row"><span>cost of a raised or lowered port</span><b>${M.integer_filling.rest_cost} of the mismatch potential</b></div>
    <div class="row"><span>release when a pair annihilates</span><b>exactly ${M.integer_filling.pair_release}</b></div><div class="note">${esc(M.integer_filling.note)} They diffuse and do not travel.</div></div>`;
  return h + '</div>';
}

export function renderChart() {
  const host = $('chart');
  if (!host || !S.pmap || !S.pmap.chart) return;
  const G = Object.fromEntries(S.pmap.groups.map((g) => [g.id, g]));
  const shown = S.pmap.chart.filter((c) => c.target !== 'W_minus');
  const chips = S.pmap.groups.map((g) => `<span class="chip" style="--c:${g.color}"><i></i><b>${esc(g.label)}</b><em>${esc(g.blurb)}</em></span>`).join('');
  host.innerHTML = `
    <div class="chart-head"><div><h2 style="margin:0">Particles of nature and their best-guess place in the model</h2>
      <p style="margin:4px 0 0">Every tile is a measured particle. Colour is the group. The small tag is the internal pattern that the revised best guess gives it, or "response" and "wave" where it is not a pattern.
      Click a tile for everything known about it.</p></div>
      <button id="antiBtn" class="${anti ? 'on' : ''}">${anti ? 'showing antiparticles' : 'show antiparticles'}</button></div>
    <div class="chips">${chips}<span class="chip" style="--c:#c2cde2"><i class="ringi"></i><b>Antiparticles</b><em>drawn as rings in the colour of their particle</em></span></div>
    <div class="chart-wrap"><div class="chart-grid">
      <div class="g-lab" style="grid-row:1;grid-column:2">I</div><div class="g-lab" style="grid-row:1;grid-column:3">II</div><div class="g-lab" style="grid-row:1;grid-column:4">III</div>
      <div class="g-lab" style="grid-row:1;grid-column:5">force carriers</div><div class="g-lab" style="grid-row:1;grid-column:6">Higgs</div>
      <div class="g-side" style="grid-row:2 / span 2;grid-column:1"><span>quarks</span></div><div class="g-side" style="grid-row:4 / span 2;grid-column:1"><span>leptons</span></div>
      ${shown.map((c) => tile(c, G)).join('')}
    </div><div id="chart-detail"></div></div>
    <div class="k" style="margin-top:22px">Markers on the screen · flux classes of the big runs</div>
    <div class="legend-row">${[1, 5, 4, 2, 3].map((k) => `<span class="lg">${marker(k, 18)}<span><b style="color:${FAM[k].color}">${FAM[k].label}</b><br><span class="small">class ${k} · ${FAM[k].detail}</span></span></span>`).join('')}</div>
    ${builderSection(G)}
    ${objectsSection(G)}
    ${sectorSection(G)}
    ${generationsSection()}
    ${scoreSection(G)}
    ${censusSection()}
    ${massSection()}
    <h2 style="margin-top:30px">Ledgers</h2><p class="small">The detailed tables behind the chart: register classes, reactions, comparison targets, protected objects and excitations of the source.</p>`;
  host.querySelectorAll('.tile').forEach((el) => el.onclick = () => { picked = el.dataset.t; renderChart(); });
  host.querySelectorAll('.lvl-btn').forEach((el) => el.onclick = () => { const i = parseInt(el.dataset.i); occ[i] = 1 - occ[i]; const y = document.querySelector('#page-particles .scroll').scrollTop; renderChart(); document.querySelector('#page-particles .scroll').scrollTop = y; });
  $('antiBtn').onclick = () => { anti = !anti; renderChart(); };
  detail(G);
  const sc = new URLSearchParams(location.search).get('scrollto');
  if (sc && document.getElementById(sc) && !renderChart.scrolled) { renderChart.scrolled = true; setTimeout(() => document.getElementById(sc).scrollIntoView(), 50); }
}
