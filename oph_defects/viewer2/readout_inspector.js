import { physicalReadout, recordSnapshot } from './readout_observables.js';

const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
const compact = x => Number.isFinite(x) ? (Math.abs(x) >= 1e5 || (x !== 0 && Math.abs(x) < 1e-3) ? x.toExponential(3) : x.toFixed(3)) : '…';
const vector = a => a.map(compact).join(', ');

function momentumPlots(obs, units, grid) {
  if (!obs?.momentumMarginals) return '';
  const factor=units.hbar/(grid.h*units.tick), colors=['#73c8ff','#bca6ff','#63dfbb'];
  const plots=obs.momentumMarginals.map((row,a)=>{
    const order=obs.momentumGrid.map((p,i)=>({p,i})).sort((x,y)=>x.p-y.p), peak=Math.max(...row,1e-12), n=row.length;
    const bars=order.map(({i},j)=>{const h=48*row[i]/peak;return `<rect x="${8+j*304/n}" y="${61-h}" width="${Math.max(1,304/n-2)}" height="${h}" rx="1.5" fill="${colors[a]}"><title>${compact(obs.momentumGrid[i]*factor)} MeV/c: ${(100*row[i]).toFixed(3)}%</title></rect>`;}).join('');
    return `<div class="ri-distribution"><b>Momentum ${'xyz'[a]} · probability by mode</b><svg viewBox="0 0 320 88" role="img" aria-label="Momentum ${'xyz'[a]} distribution. Spread ${compact(Math.sqrt(row.reduce((v,w,i)=>v+w*(obs.momentumGrid[i]-obs.meanMomentum[a])**2,0))*factor)} MeV per c."><line x1="8" y1="62" x2="312" y2="62" stroke="#39516c"/>${bars}<text x="8" y="80" fill="#91a4bf" font-size="10">${compact(order[0].p*factor)}</text><text x="312" y="80" text-anchor="end" fill="#91a4bf" font-size="10">${compact(order.at(-1).p*factor)} MeV/c</text></svg></div>`;
  }).join('');
  return `<div class="ri-distributions">${plots}</div><p class="ri-plot-note">The glow in the box shows position density. These three plots show the same packet in momentum space; each axis sums to 100%. They describe alternative measurements. A detector record changes the state and therefore changes its momentum spread.</p>`;
}

export function createReadoutInspector(getState, onSelect, onPause) {
  const css = document.createElement('link'); css.rel = 'stylesheet'; css.href = new URL('./readout_inspector.css', import.meta.url); document.head.append(css);
  const host = document.createElement('section'); host.className = 'ri-panel'; host.id = 'roJourney'; host.setAttribute('aria-label', 'Follow one source defect');
  document.getElementById('ro-cards').before(host);
  host.innerHTML = `<div class="ri-top"><div><div class="ri-eyebrow">Source → preparation → evolution → record</div><h3>Follow one defect</h3></div>
    <div class="ri-controls"><select aria-label="Defect to follow"></select><button type="button" data-action="pause">Pause to inspect</button><button type="button" data-action="export">Save this readout</button></div></div>
    <p class="ri-caption">Keep the same source ID in view as a run record becomes a prepared wave packet and then a detector record. Click a label in the box, or a followed defect on the screen.</p>
    <div data-content></div><details class="ri-detail"><summary>Detector history and source details</summary><div data-records></div></details>
    <p class="ri-foot">The patch is read from the run. The box evolves a separately prepared free Dirac packet on a finite grid. Family names are candidate assignments; momentum and energy below are state expectations. The MeV and metre-per-second values use the viewer’s supplied clock and length scale.</p>
    <p class="ri-status" role="status" aria-live="polite"></p>`;
  const select = host.querySelector('select'), content = host.querySelector('[data-content]'), history = host.querySelector('[data-records]'), status = host.querySelector('[role=status]');
  let selected = null, lastChoices = '', lastStamp = -Infinity, hiddenSelection = null;
  const setSelected = id => { selected = Number(id); hiddenSelection = null; lastStamp = -Infinity; update(performance.now()); };
  select.addEventListener('change', () => { setSelected(select.value); onSelect(selected); });
  host.querySelector('[data-action=pause]').onclick = () => { onPause(); status.textContent = 'Playback paused. The selected packet and its recorded positions remain available.'; };
  host.querySelector('[data-action=export]').onclick = () => {
    const s = getState(), p = s.parts.get(selected); if (!p) return;
    const data = recordSnapshot({ run:s.run, observer:s.observer?.carrier[0], slice:s.slice, fieldTime:s.time, particle:p, family:s.classes[p.cls], units:s.units, grid:s.grid });
    const blob = new Blob([JSON.stringify(data,null,2)+'\n'], { type:'application/json' }), url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = `oph-${String(s.run).replace(/[^a-zA-Z0-9_-]/g,'_')}-defect-${p.id}.json`; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent = `Saved defect #${p.id}: source, preparation, units and ${data.records.length} complete packet records and replay inputs.`;
  };
  document.getElementById('roScreen').addEventListener('click', e => {
    const s = getState(), o = s.observer; if (!o?.radius) return;
    const cv = e.currentTarget, rect = cv.getBoundingClientRect(), dpr = cv.width/rect.width;
    const x = (e.clientX-rect.left)*dpr, y = (e.clientY-rect.top)*dpr, scale = (Math.min(cv.width,cv.height)/2-34*dpr)/o.radius;
    const a = o.byId[s.slice], b = o.byId[Math.min(o.byId.length-1,s.slice+1)]; let nearest = null, distance = 20*dpr;
    for (const [id,d] of a) {
      if (!s.parts.has(id)) continue;
      const next = b.get(id), u = next ? d.u*(1-s.frac)+next.u*s.frac : d.u, v = next ? d.v*(1-s.frac)+next.v*s.frac : d.v;
      const delta = Math.hypot(x-(cv.width/2+u*scale), y-(cv.height/2+6*dpr-v*scale));
      if (delta<distance) { distance=delta; nearest=id; }
    }
    if (nearest!==null) { setSelected(nearest); onSelect(nearest); }
  });
  function update(now) {
    if (now-lastStamp<250) return; lastStamp=now;
    const s=getState(), particles=[...s.parts.values()], choices=particles.map(p=>`${p.id}:${p.cls}`).join('|');
    if (choices!==lastChoices) {
      lastChoices=choices;
      select.innerHTML=particles.map(p=>`<option value="${p.id}">#${p.id} · ${escape(s.classes[p.cls].name)}</option>`).join('');
      if (selected!==null && !s.parts.has(selected)) { hiddenSelection=selected; selected=null; }
      if (selected===null && particles.length) selected=particles[0].id;
      if (hiddenSelection!==null) status.textContent=`Defect #${hiddenSelection} left the displayed selection. Following ${selected===null ? 'no packet' : '#'+selected}.`;
    }
    select.disabled=!particles.length; host.querySelector('[data-action=export]').disabled=!particles.length;
    for (const p of particles) p.lab.dataset.followed=String(p.id===selected);
    if (!particles.length) { content.innerHTML='<p class="ri-empty">No defect is displayed at this slice. Move the time slider or select another observer.</p>'; history.innerHTML=''; return; }
    select.value=String(selected); const p=s.parts.get(selected); if (!p) return;
    const family=s.classes[p.cls], obs=p.stats?.observables, physical=obs ? physicalReadout(obs,s.units,s.grid.h) : null;
    const last=p.recs.at(-1), theta=(p.cls===1||p.cls===5 ? s.units.theta.qL : p.cls===4||p.cls===2 ? s.units.theta.qR : s.units.theta.lL);
    const step=(n,title,value,description,tag)=>`<article class="ri-step"><span class="ri-number">${n}</span><h4>${title}</h4><strong>${escape(value)}</strong><p>${escape(description)}</p><span class="ri-tag">${tag}</span></article>`;
    const metric=(label,value,note)=>`<div class="ri-metric"><span>${label}</span><b>${escape(value)}</b><small>${escape(note)}</small></div>`;
    content.innerHTML=`<div class="ri-flow">${step(1,'Protected source record',`Defect #${p.id}`,`Centre class k = ${p.cls}; run ${s.run}, observer ${s.observer?.carrier[0]}. The same ID is used in both views.`,'Run data')}
      ${step(2,'Prepared field packet',family.name,`A Gaussian packet is initialized at the projected screen position, with the displayed spinor and rest angle θ = ${theta.toFixed(6)}.`,'Declared preparation + candidate label')}
      ${step(3,'Between detector records','A spreading wave',`The ${s.grid.n}³ periodic grid evolves the free Dirac equation. The detector averages its density over a Gaussian window; the symbol marks the last record.`,'Calculated finite field')}
      ${step(4,'Recorded by the detector',`${p.nrec} position record${p.nrec===1?'':'s'}`,last ? `Latest at ${compact(last.t*s.grid.h)} field ticks. Both energy-band outcomes are retained; ${p.nconverted||0} records required the detector’s band-conversion operation.` : 'No detector record for this packet yet. Its initial marker shows where it was prepared.','Complete finite Born instrument')}</div>
      <div class="ri-metrics">${metric('Mean packet energy',physical ? `${compact(physical.energyMeV)} MeV` : 'Calculating…','Includes momentum; this is not a rest-mass identification.')}
      ${metric('Mean speed',physical ? `${compact(physical.meanSpeedMetresPerSecond)} m/s` : 'Calculating…','Average |p|/E across the actual momentum distribution.')}
      ${metric('Mean momentum (x, y, z)',physical ? vector(physical.momentumMeVPerC)+' MeV/c' : 'Calculating…','A state expectation, not a simultaneous position measurement.')}
      ${metric('Model rest energy',physical ? `${compact(physical.restEnergyMeV)} MeV` : `${compact(theta*s.units.hbar/s.units.tick)} MeV`,'Set by the displayed rest angle and supplied unit conversion.')}</div>${momentumPlots(obs,s.units,s.grid)}`;
    const rows=p.recs.slice(-5).reverse().map(r=>`<tr><td>${compact(r.t*s.grid.h)}</td><td>${escape(vector(r.x.map(x=>(x-s.grid.n/2)*s.grid.h*s.units.cell)))}</td><td>${r.band==='converted-negative'?'converted by detector':r.band==='positive'?'positive band':'—'}</td><td>${Number.isFinite(r.probability)?compact(100*r.probability)+'%':'—'}</td></tr>`).join('');
    history.innerHTML=`<p class="ri-caption">Run <b>${escape(s.run)}</b> · observer <b>${s.observer?.carrier[0]}</b> · source slice <b>${s.slice+1}</b>. Field ticks and the run’s repair ticks are separate clocks in this display.</p>
      <table><thead><tr><th>Field time (ticks)</th><th>Recorded position x, y, z (fm)</th><th>Band outcome</th><th>Joint probability</th></tr></thead><tbody>${rows||'<tr><td colspan="4">No position records yet.</td></tr>'}</tbody></table>
      <p class="ri-foot">Latest five of ${p.nrec} records; the display keeps eighteen and the saved readout keeps the complete packet transcript. The momentum spread is ${physical ? escape(vector(physical.momentumStdMeVPerC)) : '…'} MeV/c. These spreads are not a joint measurement of a sharp position and a sharp momentum. Displayed packets are independent single-packet experiments, not one antisymmetrized many-fermion state. The four spinor entries do not implement the full internal gauge representation. The band conversion is a declared detector action with recorded energy transfer; it is not free evolution or an antiparticle identification. <a href="readout_instrument.md" target="_blank" rel="noopener">Complete detector proof ↗</a> · <a href="/rl/SIM/cf1/runs/MASS_CHECK.json" target="_blank" rel="noopener">Rest-angle and unit reference ↗</a></p>`;
  }
  return { select:setSelected, update, get selectedId() { return selected; } };
}
