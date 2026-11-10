// The particle table and the dataset page.
import { $, S, FAM, fmt, classInfo, candidateName } from './app.js';

const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const pill = (t, cls = '') => `<span class="pill ${cls}">${esc(t)}</span>`;
const num = (v, d = 0) => (v === undefined || v === null ? '–' : typeof v === 'number' ? (d ? v.toFixed(d) : fmt(v)) : v);

function classStatCells(st) {
  if (!st || !st.ever) return '<td class="num">0</td><td class="num">–</td><td class="num">–</td><td class="num">–</td><td>–</td><td>–</td>';
  const b = st.born, e = st.ended;
  return `<td class="num">${fmt(st.ever)}</td><td class="num">${fmt(st.alive_at_end)}</td><td class="num">${num(st.lifetime_depth_median, 0)}</td>
    <td class="num">${st.hops_per_depth ? st.hops_per_depth.toFixed(4) : '0'}</td>
    <td>${fmt(b.conjugate_pair)} pair · ${fmt(b.split_product)} split · ${fmt(b.fusion_or_exchange_product)} fusion/exchange</td>
    <td>${fmt(e.annihilation)} annihilated · ${fmt(e.fusion)} fused · ${fmt(e.split)} split · ${fmt(e.exchange)} exchanged</td>`;
}
function weights(k) {
  const ms = S.pmap.multiplets.filter((m) => m.z6_class === k);
  const cj = S.pmap.multiplets.filter((m) => m.z6_class_conjugate === k && m.z6_class !== k);
  const line = (m, conj) => `<div><b>${esc(m.label)}${conj ? ' conjugate' : ''}</b>: SU(3) ${esc(m.SU3)}${conj ? ' bar' : ''}, SU(2) dim ${m.SU2_dimension}, Y ${conj ? '−(' + esc(m.Y) + ')' : esc(m.Y)},
    Q6 ${conj ? '−(' + esc(m.Q6) + ')' : esc(m.Q6)}<br><span class="small">T3 ${m.T3.join(', ')} · Q ${m.Q.join(', ')}${conj ? ' (signs reversed)' : ''}</span></div>`;
  return ms.map((m) => line(m, false)).join('') + cj.map((m) => line(m, true)).join('') || '–';
}
async function closedLoop() {
  try {
    const list = await (await fetch('/api/closed_loop')).json();
    if (!list.length) return '';
    let h = `<h2>Closed-loop charges: law C with transported contrast</h2>
      <p>A separate declared extension proposed by the theory session, in which a defect acts back on the loads: every seam carries an integer offset, a seam compares
      one end with the transported other end, and a face with offset curl q drives a permanent circulation of the committed current. The charge is the integer curl, not a Z6 class.
      For a pinned pair the potential released per sweep is fixed exactly by the offsets: (1/2)|d̄|², the lattice Coulomb energy of the two charges.
      It is the first quantity in these runs that plays the role of an energy of a defect, in native units of the mismatch potential per sweep. No conversion to GeV exists.</p>`;
    for (const item of list) {
      const rows = item.rows.filter((r) => r.q);
      h += `<h3>${esc(item.name)}: level ${rows[0].level}, ${fmt(rows[0].seams)} seams, ${fmt(rows[0].sweeps)} sweeps per run</h3>
        <table class="t"><tr><th>background</th><th>charge q</th><th>separation</th><th>release, exact</th><th>release, measured per sweep</th><th>per realised draw</th>
        <th>work of the offsets</th><th>odd seams</th><th>nonzero-current events per sweep</th><th>rings |z| max</th></tr>`;
      let last = '';
      for (const r of rows) {
        const key = r.background + ' q' + r.q;
        const zs = r.rings.filter((g) => g.z !== null).map((g) => Math.abs(g.z));
        h += `<tr${key !== last ? ' style="border-top:2px solid var(--edge)"' : ''}><td>${esc(r.background === 'frozen' ? 'frozen (all loads equal)' : 'density ' + r.background)}</td><td class="num">±${r.q}</td><td class="num">${r.r}</td>
          <td class="num">${r.release_exact_per_sweep.toFixed(4)}</td><td class="num">${r.release_measured_per_sweep.toFixed(4)}</td>
          <td class="num">${r.release_per_draw !== undefined ? r.release_per_draw.toFixed(4) : '–'}</td><td class="num">${r.offset_work_measured_per_sweep.toFixed(4)}</td>
          <td class="num">${r.odd_seams_mean.toFixed(0)}</td><td class="num">${r.nonzero_per_sweep.toFixed(0)}</td><td class="num">${zs.length ? Math.max(...zs).toFixed(2) : '–'}</td></tr>`;
        last = key;
      }
      h += '</table>';
    }
    return h;
  } catch (e) { return ''; }
}

export async function renderParticles() {
  const P = S.pmap, st = S.classStats.tiers, g = S.meta.geometry;
  if (S.meta.config.twists) {
    const rows = st.screen ? st.screen.classes : {};
    let h = `<h2 style="margin-top:0">Charges of the closed-loop run ${esc(S.name)}</h2>
      <p>This run is a declared extension, not the exact layer: law C with transported contrast and unit twists. A charge is the integer holonomy of the seam offsets
      around a screen plaquette. It acts back on the loads, so its effects can be read from native records: a permanent circulation of the committed current,
      a release of mismatch potential per sweep equal to the lattice Coulomb energy of the charges, and a shifted local event rate. Charges carry no candidate particle names.</p>
      <table class="t"><tr><th>charge</th><th>ever</th><th>alive at end</th><th>median life (depth)</th><th>hops per depth</th><th>born as</th><th>ended as</th></tr>`;
    for (const [k, label] of [[1, '+1'], [5, '−1'], [2, '+2'], [4, '−2'], [3, '±3']]) h += `<tr><td><b style="color:${FAM[k].color}">${label}</b></td>${classStatCells(rows[String(k)])}</tr>`;
    h += '</table>';
    if (st.screen) h += `<h3>Reactions (charges mod 6: 5 = −1, 4 = −2)</h3><p>${Object.entries(st.screen.channels).slice(0, 24).map(([k, v]) => pill(k + ' : ' + fmt(v))).join(' ')}</p>`;
    $('particles').innerHTML = h + await closedLoop();
    return;
  }
  const src = P.sources;
  let h = `<h2 style="margin-top:0">Defects, candidate names and comparison targets</h2>
    <p><b style="color:var(--amber)">${esc(P.status)}</b></p>
    <p class="small">Candidate map read from the theory repository at commit ${esc(src.repository_commit.slice(0, 9))}${src.repository_dirty ? ' (working tree with uncommitted changes)' : ''}:
      ${esc(src.results.path)} (sha256 ${esc(src.results.sha256.slice(0, 12))}…) and ${esc(src.correspondence.path)} (sha256 ${esc(src.correspondence.sha256.slice(0, 12))}…).
      Table built ${esc(P.built)}. Rebuild with <code>python -m oph_defects.particle_table</code> after the repository changes.</p>
    <h2>1 · Defects of the declared register in run ${esc(S.name)}, by family</h2>
    <p>${esc(P.z6_reading.statement)} <b>${esc(P.z6_reading.status)}.</b> The register records the currents of the exact run and does not act back on it,
      so no energy, force or mass of these defects can be read from the run. Lifetimes are in units of causal depth (about ${(S.classStats.depth_at_end / S.meta.sweeps).toFixed(1)} per sweep in this run).</p>
    <table class="t"><tr><th>class</th><th>candidate name</th><th>internal weights of the candidate components</th><th>comparison targets with such a component</th>
      <th>ever</th><th>alive at end</th><th>median life</th><th>hops per depth</th><th>born as</th><th>ended as</th><th>rest mass</th></tr>`;
  for (const fam of P.families) {
    h += `<tr class="fam"><td colspan="11"><span class="dot" style="background:${fam.color};margin-right:8px"></span>${esc(fam.label)} <span class="small" style="font-weight:400"> · ${esc(fam.note)}</span></td></tr>`;
    for (const k of fam.classes) {
      const ci = classInfo(k);
      for (const [tier, label] of [['screen', 'screen tier'], ['internal', 'internal tier']]) {
        if (!st[tier]) continue;
        const first = tier === 'screen';
        h += `<tr id="${first ? 'class-row-' + k : ''}">` + (first ? `<td rowspan="${st.internal ? 2 : 1}"><b style="color:${fam.color};font-size:16px">${k}</b><br><span class="small">conjugate ${ci.conjugate_class}</span></td>
          <td rowspan="${st.internal ? 2 : 1}"><b style="color:${fam.color}">${esc(ci.short)}?</b> ${esc(candidateName(k))} <span class="cand">candidate</span>
            <div class="small">gives a phase to ${esc(ci.visible_to.join(', ') || 'none')}; invisible to ${esc(ci.invisible_to.join(', ') || 'none')}</div></td>
          <td rowspan="${st.internal ? 2 : 1}">${weights(k)}</td>
          <td rowspan="${st.internal ? 2 : 1}">${ci.targets.length ? ci.targets.map((t) => pill(t)).join('') : '<span class="small">conjugates of the targets of class ' + ci.conjugate_class + '</span>'}</td>` : '')
          + classStatCells(st[tier].classes[String(k)]).replace('<td class="num">', `<td class="num"><span class="small">${label}</span><br>`)
          + (first ? `<td rowspan="${st.internal ? 2 : 1}">${pill('not derived', 'warn')}</td>` : '') + '</tr>';
      }
    }
  }
  const c0 = classInfo(0);
  h += `<tr class="fam"><td colspan="11">Flat faces (class 0): no flux</td></tr><tr><td><b>0</b></td><td>${esc(c0.multiplets.join(', '))} and the gauge fields <span class="cand">candidate</span></td>
    <td>${weights(0)}</td><td>${c0.targets.map((t) => pill(t)).join('')}</td><td colspan="6" class="small">a flat face is not a defect; components of class 0 are invisible to the register</td><td>${pill('not derived', 'warn')}</td></tr></table>`;
  for (const tier of ['screen', 'internal']) {
    if (!st[tier]) continue;
    const ch = Object.entries(st[tier].channels).slice(0, 24);
    h += `<h3>Reactions counted in the ${tier} tier (classes; a + b → c)</h3><p>${ch.map(([k, v]) => pill(k + ' : ' + fmt(v))).join(' ')}</p>`;
  }
  // targets
  h += `<h2>2 · Named comparison targets of the theory repository</h2>
    <p>Eighteen targets grouped by the repository's families. Status and requirements are copied verbatim. Measured masses are comparison data
    (${esc(src.measured_masses)}); they are not inputs and not predictions.</p>
    <table class="t"><tr><th>target</th><th>benchmark</th><th>source lead</th><th>class of the lead components</th><th>finite source evidence</th><th>required before identification</th>
    <th>source-derived mass</th><th>measured mass</th><th>status</th></tr>`;
  const fams = [...new Set(P.targets.map((t) => t.theory_family))];
  for (const f of fams) {
    h += `<tr class="fam"><td colspan="9">${esc(f.replace(/_/g, ' '))}</td></tr>`;
    for (const t of P.targets.filter((x) => x.theory_family === f)) {
      const b = t.benchmark || {};
      h += `<tr><td><b>${esc(t.target.replace(/_/g, ' '))}</b><br><span class="small">${esc(b.symbol || '')}</span></td>
        <td>spin ${esc(b.spin ?? '?')}<br>charge ${esc(b.charge ?? '?')}<br>colour ${esc(b.color ?? '?')}</td>
        <td>${t.source_component_ids.map((c) => pill(c)).join('')}<div class="small">${esc(t.source_multiplets.join(', '))}</div><div class="small">${esc(t.best_guess || '')}</div></td>
        <td>${t.z6_classes_of_components.length ? t.z6_classes_of_components.map((k) => `<b style="color:${k ? FAM[k].color : 'var(--muted)'}">${k}</b>`).join(', ') : '–'}</td>
        <td>${esc(t.source_test_summary)}</td>
        <td><div><span class="small">response</span> ${esc(t.required_response || '')}</div><div><span class="small">mass</span> ${esc(t.required_mass || '')}</div></td>
        <td>${t.source_derived_mass_GeV === null ? pill('none', 'warn') : t.source_derived_mass_GeV + ' GeV'}</td>
        <td class="num">${t.measured_mass_GeV === null || t.measured_mass_GeV === undefined ? '–' : t.measured_mass_GeV + ' GeV'}<div class="small">${esc(t.measured_mass_note || '')}</div></td>
        <td>${pill(t.status, 'warn')}</td></tr>`;
    }
  }
  h += '</table>';
  // protected objects
  h += `<h2>3 · Protected objects of the theory repository</h2><p>Eleven entries in six families, each with its own construction domain. None is assigned a particle or a rest mass.</p>
    <table class="t"><tr><th>object</th><th>source id</th><th>domain</th><th>what is protected</th><th>topology</th><th>first required attachment</th><th>particle</th><th>rest mass</th></tr>`;
  for (const fam of P.protected_families) {
    h += `<tr class="fam"><td colspan="8">${esc(fam.label)}</td></tr>`;
    for (const r of fam.rows) h += `<tr><td><b>${esc(r.label)}</b><div class="small">${esc(r.description)}</div></td><td>${pill(r.source_id)}<br>${pill(r.classification || '')}</td>
      <td>${esc(r.domain)}</td><td>${esc(r.protected_object)}</td><td>${esc(r.topology)}</td><td>${esc(r.first_required)}</td>
      <td>${r.physical_particle_id === null ? pill('none', 'warn') : esc(r.physical_particle_id)}</td><td>${r.physical_rest_mass === null || r.physical_rest_mass === undefined ? pill('none', 'warn') : esc(r.physical_rest_mass)}</td></tr>`;
  }
  h += '</table>';
  h += `<h2>4 · Source excitations of the theory repository</h2><p>Sixteen identifiers in six families. Their protected-defect field is empty: the attachment of a defect to an excitation is unproved.</p>
    <table class="t"><tr><th>excitation</th><th>source id</th><th>exact internal weights</th><th>class Q6 mod 6</th><th>description</th><th>protected defect</th></tr>`;
  for (const fam of P.excitation_families) {
    h += `<tr class="fam"><td colspan="6">${esc(fam.label)}</td></tr>`;
    for (const r of fam.rows) {
      const ex = r.exact || {};
      h += `<tr><td><b>${esc(r.label)}</b></td><td>${pill(r.source_id)}</td><td>${Object.keys(ex).length ? `SU(3) ${esc(ex.SU3)}, SU(2) dim ${esc(ex.SU2_dimension)}, Y ${esc(ex.Y)}, Q6 ${esc(ex.Q6)}, T3 ${esc(ex.T3)}, Q ${esc(ex.comparison_Q)}` : '–'}</td>
        <td>${r.z6_class === null ? '–' : `<b style="color:${r.z6_class ? FAM[r.z6_class].color : 'var(--muted)'}">${r.z6_class}</b>`}</td><td>${esc(r.description)}</td>
        <td>${r.protected_defect_id === null ? pill('none', 'warn') : esc(r.protected_defect_id)}</td></tr>`;
    }
  }
  h += '</table>';
  $('particles').innerHTML = h + await closedLoop() + `<p class="small">${esc(P.chronology_scope || '')}</p>`;
}

export async function renderData() {
  const m = S.meta, g = m.geometry, c = m.config, t = m.totals, pv = m.provenance;
  let h = `<h2 style="margin-top:0">Dataset: run ${esc(S.name)}</h2>
    <p>The exact layer of this directory is the executable specification with nothing added: the federation, the declared preparation, the uniform seam schedule with fair tie coins and
    the integer law (law C). Depths, clocks, slices and the register are declared layers computed from the exact records without further randomness.
    Every array file is a NumPy <code>.npz</code> archive and every table is JSON; <code>DATASET.md</code> in the directory defines each file and field.</p>
    <table class="t"><tr><th>quantity</th><th>value</th></tr>
    <tr><td>law</td><td>${esc(m.law.name)}<div class="small">${esc(m.law.formula)}</div></td></tr>
    <tr><td>schedule</td><td>${esc(m.law.schedule)}<div class="small">seed ${c.seed} = ${esc(m.law.seed_rule)}</div></td></tr>
    <tr><td>preparation</td><td>${esc(m.preparation.name)}: ${esc(m.preparation.rule)}</td></tr>
    <tr><td>regime</td><td>${esc(m.regime)}</td></tr>
    <tr><td>federation</td><td>level ${g.level}: ${fmt(g.carriers)} carriers, ${fmt(g.ports)} ports, ${fmt(g.seams)} seams (${fmt(g.glue_seams)} between carriers), ${fmt(g.faces)} faces
      <div class="small">${esc(g.gluing || '')}. ${esc(g.gluing_note || '')}</div></td></tr>
    <tr><td>length</td><td>${fmt(m.sweeps)} sweeps = ${fmt(m.sweeps * g.seams)} attempts; scaled time τ = ${(m.sweeps * m.scaled_time.tau_per_sweep).toExponential(3)}</td></tr>
    <tr><td>balanced class</td><td>${t.balanced ? 'reached after ' + fmt(t.attempts_to_balanced_class) + ' attempts' : 'not reached'}; V from ${fmt(t.V_initial)} to ${fmt(t.V_final)} (minimum ${fmt(t.V_minimum)})</td></tr>
    <tr><td>causal depth</td><td>${esc(m.layers.depth)}</td></tr>
    <tr><td>slices</td><td>${m.cuts.length} consistent cuts plus the end of the run. ${esc(m.layers.cuts)}</td></tr>
    <tr><td>clocks</td><td>${esc(m.layers.clocks)}</td></tr>
    <tr><td>register</td><td>${esc(m.layers.register.kind)}: ${esc(m.layers.register.rule)}; cost ${m.layers.register.cost.join(', ')}; kappa ${m.layers.register.kappa}; randomness ${esc(m.layers.register.randomness)}</td></tr>
    <tr><td>identities</td><td>${fmt(m.identities)} (${Object.entries(m.tracker_totals).map(([k, v]) => k + ' ' + fmt(v)).join(', ')})</td></tr>
    <tr><td>stored histories</td><td>${m.checkpoints.length} checkpoints (exact replay from each), ${m.records.length} sweeps of complete per-attempt records, ${(m.series || []).reduce((a, b) => a + b.snapshots, 0)} density snapshots,
      ${m.observers.length} observer files (tape, patch, readouts)</td></tr>
    <tr><td>backend</td><td>${esc(c.backend)}; kernel attempt body sha256 ${esc(pv.kernel_core_sha256.slice(0, 16))}…; numpy ${esc(pv.numpy)}; ${esc(pv.platform)}</td></tr>
    </table>
    <h3>Source custody</h3><table class="t"><tr><th>source</th><th>role</th><th>equal to the hash pinned by the model</th></tr>`;
  for (const [rel, p] of Object.entries(pv.source_pins)) h += `<tr><td><code>${esc(rel)}</code></td><td>${p.defines_the_run ? 'defines the federation of this run' : 'reference check of the law only'}</td><td>${p.equal ? pill('yes', 'ok') : pill('no', 'warn')}</td></tr>`;
  h += `</table><h3>All runs served here</h3><table class="t"><tr><th>run</th><th>level</th><th>observers</th><th>ports</th><th>seams</th><th>sweeps</th><th>slices</th><th>identities</th><th>preparation</th><th>backend</th></tr>`;
  for (const r of S.runs) h += `<tr><td><b>${esc(r.name)}</b></td><td class="num">${r.level}</td><td class="num">${fmt(r.carriers)}</td><td class="num">${fmt(r.ports)}</td><td class="num">${fmt(r.seams)}</td>
    <td class="num">${fmt(r.sweeps)}</td><td class="num">${r.cuts}</td><td class="num">${fmt(r.identities)}</td><td>${esc(r.preparation)}</td><td>${esc(r.backend)}</td></tr>`;
  h += `</table><h3>Reading it</h3><pre>from oph_defects.dataset import Dataset
ds = Dataset('runs/universe/${esc(S.name)}')
cut = ds.cut(ds.cuts()[-1]['k'])        # one slice of synchronised time: defects, loads summary, clocks
cat = ds.catalogue()                     # one row per defect identity
obs = ds.observer(${m.observers.length ? m.observers[0].carrier : 0})                    # tape, patch and readouts of one observer
rec = ds.records(0)                      # every attempt of sweep 0: seam, coin, current, contrast, total, twist, depth
for sweep, rec in ds.replay(40, 42):     # exact replay from the nearest checkpoint
    ...</pre>
    <h3>File and field definitions</h3><pre id="schemaDoc">loading DATASET.md…</pre>`;
  $('data').innerHTML = h;
  try { const r = await fetch('/DATASET.md'); $('schemaDoc').textContent = r.ok ? await r.text() : 'DATASET.md not found'; } catch (e) { $('schemaDoc').textContent = 'DATASET.md not found'; }
}
