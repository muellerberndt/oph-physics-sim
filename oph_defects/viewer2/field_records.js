const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sci = text => { const [m,e] = text.split('e'); return Number(e)===0?m:`${m} × 10<sup>${Number(e)}</sup>`; };
const range = value => {
  const [lo,e]=value.lowerText.split('e'),[hi,f]=value.upperText.split('e');
  if(e===f)return Number(e)===0?`${lo}–${hi}`:`(${lo}–${hi}) × 10<sup>${Number(e)}</sup>`;
  return `${sci(value.lowerText)} to ${sci(value.upperText)}`;
};
const metric = (value,label) => `<div class="metric"><b>${value}</b><span>${label}</span></div>`;
const steps = items => `<div class="steps">${items.map((s,i)=>`<div class="step"><b>0${i+1}</b>${s}</div>`).join('')}</div>`;
const svg = content => `<svg class="diagram" viewBox="0 0 720 300" role="img" aria-label="${esc(content.label)}">${content.body}</svg>`;
let data;

function showTab(name, focus=false) {
  document.querySelectorAll('[data-tab]').forEach(button=>{
    const on=button.dataset.tab===name;button.setAttribute('aria-selected',String(on));button.tabIndex=on?0:-1;
    if(on&&focus)button.focus();
  });
  document.querySelectorAll('[role=tabpanel]').forEach(panel=>panel.hidden=panel.id!==name);
}
document.querySelectorAll('[data-tab]').forEach((button,index)=>{
  button.addEventListener('click',()=>showTab(button.dataset.tab));
  button.addEventListener('keydown',event=>{
    const buttons=[...document.querySelectorAll('[data-tab]')];let target;
    if(event.key==='ArrowRight')target=(index+1)%buttons.length;
    if(event.key==='ArrowLeft')target=(index+buttons.length-1)%buttons.length;
    if(event.key==='Home')target=0;if(event.key==='End')target=buttons.length-1;
    if(target!==undefined){event.preventDefault();showTab(buttons[target].dataset.tab,true);}
  });
});

function transfer() {
  $('transfer').innerHTML=`<span class="tag">Same interacting source · complete record law</span><h2 style="margin-top:16px">An excitation arrives. Then we measure energy.</h2>
  <p>The native quark-type probes transfer charge between two blocks through a gauge link. The detector first records arrival, then measures a spectral bin of the active interacting source plus its fixed reference. The preparation partner remains present.</p>
  <div class="controls"><label for="family">Source probe</label><select id="family"><option value="uc">uᶜ · source Y +2/3</option><option value="dc">dᶜ · source Y −1/3</option><option value="Q">Q · source Y −1/6</option></select><label><input type="checkbox" id="energy-control"> Compare energy-first control</label></div>
  <div class="grid"><div id="transfer-picture"></div><div id="transfer-numbers" aria-live="polite"></div></div>
  <p class="explain">A source gauge charge is a representation label. It does not by itself establish Lorentz spin, chirality, a quark flavour or a relativistic mass. The full detector includes the charge-compensating reference. The displayed source excitation subtracts its fixed cost of 4 and the actual neutral-source ground energy.</p>`;
  $('family').addEventListener('change',transferDraw);$('energy-control').addEventListener('change',transferDraw);transferDraw();
}
function transferDraw() {
  const row=data.quark.find(r=>r.id===$('family').value), control=$('energy-control').checked;
  const color=control?'#89b8ff':'#ffbd63';
  $('transfer-picture').innerHTML=svg({label:control?'Energy-first control: a conserved energy cannot enter a disjoint energy bin without an intervening measurement.':'Charge transfer through a gauge link, with a retained compensating reference and ordered arrival and energy records.',body:`
    <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="${color}"/></marker></defs>
    <circle cx="125" cy="115" r="48" fill="#17283f" stroke="${color}" stroke-width="2"/><circle cx="415" cy="115" r="48" fill="#17283f" stroke="${color}" stroke-width="2"/>
    <path d="M180 115 H355" stroke="${color}" stroke-width="3" marker-end="url(#arrow)"/><text x="270" y="90" text-anchor="middle" fill="#b3c2dc" font-size="15">actual evolution U(t)</text>
    <text x="125" y="120" text-anchor="middle" fill="#edf3ff" font-size="24">${esc(row.id)}</text><text x="415" y="120" text-anchor="middle" fill="#edf3ff" font-size="24">${control?'?':esc(row.id)}</text>
    <text x="125" y="185" text-anchor="middle" fill="#b3c2dc" font-size="15">prepared block</text><text x="415" y="185" text-anchor="middle" fill="#b3c2dc" font-size="15">${control?'disjoint energy bin':'arrival detector'}</text>
    <rect x="535" y="76" width="150" height="82" rx="12" fill="#162a2a" stroke="#43e0ce"/><text x="610" y="109" text-anchor="middle" fill="#43e0ce" font-size="16">${control?'exact zero':'energy record'}</text><text x="610" y="135" text-anchor="middle" fill="#b3c2dc" font-size="13">${control?'operator identity':'after arrival'}</text>
    <path d="M464 115 H527" stroke="${color}" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M125 164 V225 H415 V164" fill="none" stroke="#536a8c" stroke-dasharray="5 5"/><text x="270" y="251" text-anchor="middle" fill="#b3c2dc" font-size="14">compensating charge and every failure record retained</text>`})+
    steps(control?['Prepare an energy band','Keep its record','Evolve with the same H','Measure a disjoint band','Probability exactly zero']:['Prepare & retain flags','Pump and localize','Wait one source unit','Record arrival','Record source + reference energy'])+
    `<div class="equation">${control?'Π₂ exp(−itH) Π₁ = 0 for disjoint energy bands':'Pr(arrival e, energy k) = q ‖Πₖ Pₑ U(t) B P₀ z‖²'}</div>`;
  $('transfer-numbers').innerHTML=control?metric('0','Probability of a disjoint energy result in the energy-first control')+`<p>This exact operator identity is a comparison, not an additional simulated run. Energy is conserved during the autonomous wait. An intervening arrival measurement changes the state and therefore the later energy distribution.</p>`:
    metric(range(row.energy),'Certified source excitation interval · not a rest mass')+metric(range(row.probability),'Original probability for one route and this energy bin')+metric(String(row.records),'Complete coarse record labels; failures and outside-bin results included');
}

function ground() {
  $('ground').innerHTML=`<span class="tag">Same completed cube · different apparatus settings</span><h2 style="margin-top:16px">A ground-state theorem needs a preparation.</h2>
  <p>The source is not silently placed in its interacting ground. Recorded code tests and energy-agreement filters prepare an approximation, while every rejection remains part of the experiment.</p>
  <div class="controls"><label for="prep">Preparation setting</label><select id="prep"><option value="0">2 copies · 64 pointer labels · one filter</option><option value="1">4 copies · 4096 pointer labels · three filters</option></select></div><div class="grid"><div id="ground-picture"></div><div id="ground-numbers" aria-live="polite"></div></div>`;
  $('prep').addEventListener('change',groundDraw);groundDraw();
}
function groundDraw() {
  const r=data.preparation[Number($('prep').value)];
  $('ground-picture').innerHTML=svg({label:`${r.copies} prepared source copies pass recorded code tests and ${r.filters} energy agreement filters; both accepted and rejected records are retained.`,body:`
    ${Array.from({length:r.copies},(_,i)=>`<circle cx="95" cy="${60+i*48}" r="17" fill="#22443e" stroke="#43e0ce"/><text x="95" y="${65+i*48}" text-anchor="middle" fill="#edf3ff" font-size="13">${i+1}</text>`).join('')}
    <path d="M120 130 H225 M365 130 H465 M605 130 H660" stroke="#43e0ce" stroke-width="3"/>
    <rect x="225" y="83" width="140" height="94" rx="13" fill="#14263c" stroke="#43e0ce"/><text x="295" y="120" text-anchor="middle" fill="#edf3ff" font-size="16">code admission</text><text x="295" y="148" text-anchor="middle" fill="#b3c2dc" font-size="14">every flag kept</text>
    <rect x="465" y="83" width="140" height="94" rx="13" fill="#14263c" stroke="#a995ff"/><text x="535" y="120" text-anchor="middle" fill="#edf3ff" font-size="16">energy agreement</text><text x="535" y="148" text-anchor="middle" fill="#b3c2dc" font-size="14">${r.pointer} labels</text>
    <path d="M295 178 V232 H535 V178" fill="none" stroke="#ffbd63" stroke-width="2"/><text x="415" y="260" text-anchor="middle" fill="#ffbd63" font-size="15">rejected branches stay in the record</text>`})+
    `<div class="rows"><span>Elapsed source clock</span><b>${r.slots} service slots</b><span>Total photon occupation cap</span><b>2⁶⁴</b><span>Source, references and pointer capacity</span><b>≈ ${r.capacity.display[0].toLocaleString(undefined,{maximumFractionDigits:0})} nats</b></div>
    <p class="explain">These are declared finite matrix control services with explicit background cancellation and work bounds. They have not been compiled into the passive local repair controller. The very large photon matrix is bounded analytically, not enumerated. Capacity excludes controller coefficient workspace.</p>`;
  $('ground-numbers').innerHTML=metric('> '+sci(r.success.lowerText),'Original admission and filter success probability')+metric('< '+sci(r.groundError.upperText),'Conditional error in the prepared ground-state purification')+metric('> '+sci(r.recordProbability.lowerText),'Original probability of the complete preparation and selected matter–field record')+metric('< '+sci(r.historyError.upperText),'Complete two-field-history cutoff comparison error');
}

function cone() {
  $('cone').innerHTML=`<span class="tag">An explicit change of source architecture</span><h2 style="margin-top:16px">Make the free matter and photon slopes agree.</h2>
  <p>The original Wilson regulator contributes to the same matrix as the mass. At the chosen spacing a = 1/m, that doubles the low-momentum squared-dispersion coefficient. An orthogonal matrix Γ separates the regulator from the mass.</p>
  <div class="grid"><div><div id="cone-plot"></div><div class="legend"><span><i style="background:#ffbd63"></i>Original matter</span><span><i style="background:#43e0ce"></i>Revised matter = photon, on an axis</span></div>
  <div class="equation">Original: E² − m² = 2 Ωγ²<br>Revised: E² − m² = Ωγ²<br><small>Along a lattice axis, with a = 1/m. Off-axis lattice corrections remain.</small></div>
  <p class="explain">The plot shows free squared dispersion, not maximum signal speed. The original maximum axial group velocity is already 1. This correction does not establish an interacting Lorentz symmetry, weak chirality or an ultraviolet limit.</p></div>
  <div>${metric('> '+sci(data.cone.gapLower.lowerText),'Finite revised cube: full physical ground gap, in source units')}<p>The revised source is checked again with its new vacuum, fixed charge operators and full Gauss constraints. The old energy offset is retained. A declared complement coefficient <b>21/2</b> prevents discarded native sectors from undercutting the code ground.</p><p class="explain">This is a vectorlike four-component construction. Ordinary parity and time reversal are broken by the finite-spacing regulator. Its gap is not a measured particle mass.</p>
  <details><summary>What the photon vacuum changes</summary><p>Compressing the kinetic Hamiltonian onto the prepared Gaussian photon vacuum multiplies each hopping term by r = exp(−g²v/2). In this unbounded-field comparison, the squared curvature becomes 3r − 2r², and an additional Γ term appears at rest.</p>
  ${metric('> '+sci(data.cone.kineticCompressionCoefficientExcessLower.lowerText),'Excess of that compressed squared curvature over 1')}
  <p class="explain">This is a calculated partial response, not the full interacting self-energy or particle dispersion. The coupled evolution leaves this photon-vacuum subspace. A free common slope therefore does not finish the interacting mass calculation.</p></details></div></div>`;
  const points=(factor)=>Array.from({length:121},(_,i)=>{const q=i*Math.PI/120;return `${60+600*i/120},${250-factor*4*Math.sin(q/2)**2*26}`;}).join(' ');
  $('cone-plot').innerHTML=svg({label:'Free squared dispersion versus axial lattice momentum: the original matter curve is twice the photon curve; the revised matter curve equals it.',body:`
    <path d="M60 25 V250 H670" fill="none" stroke="#536a8c"/><path d="M60 146 H670 M60 42 H670" fill="none" stroke="#263750" stroke-dasharray="4 5"/>
    <polyline points="${points(2)}" fill="none" stroke="#ffbd63" stroke-width="3"/><polyline points="${points(1)}" fill="none" stroke="#43e0ce" stroke-width="3"/>
    <text x="50" y="255" text-anchor="end" fill="#b3c2dc" font-size="13">0</text><text x="50" y="151" text-anchor="end" fill="#b3c2dc" font-size="13">4</text><text x="50" y="47" text-anchor="end" fill="#b3c2dc" font-size="13">8</text><text x="60" y="278" fill="#b3c2dc" font-size="13">q = 0</text><text x="660" y="278" text-anchor="end" fill="#b3c2dc" font-size="13">π</text><text x="345" y="278" text-anchor="middle" fill="#b3c2dc" font-size="13">axial momentum q = ap</text><text x="75" y="32" fill="#b3c2dc" font-size="13">(E² − m²) / m²</text>`});
}

let interactionFamily='revised';
function interaction() {
  const revised=interactionFamily==='revised', r=revised?data.revisedInteraction:data.interaction;
  $('interaction').innerHTML=`<span class="tag">${revised?'Revised orthogonal':'Original beta-Wilson'} growing family · actual coupled evolution</span><h2 style="margin-top:16px">The field changes a matter-current record.</h2>
  <p>A fair recorded setting chooses an electric-field kick or no kick. Both branches evolve with the same interacting Hamiltonian. A complete current detector then distinguishes the two settings with a strictly positive probability contrast.</p>
  <div class="controls"><label for="interaction-family">Declared source family</label><select id="interaction-family"><option value="revised" ${revised?'selected':''}>Revised orthogonal regulator</option><option value="original" ${revised?'':'selected'}>Original beta-Wilson regulator</option></select></div>
  <div class="grid"><div>${svg({label:'A fair setting selects electric kick or identity, followed by actual coupled evolution and both current-detector outcomes.',body:`
    <rect x="30" y="96" width="110" height="80" rx="12" fill="#14263c" stroke="#89b8ff"/><text x="85" y="130" text-anchor="middle" fill="#edf3ff" font-size="16">fair setting</text><text x="85" y="153" text-anchor="middle" fill="#b3c2dc" font-size="13">bit recorded</text>
    <path d="M140 125 L215 70 H365 L430 130 M140 145 L215 215 H365 L430 145" fill="none" stroke="#43e0ce" stroke-width="2"/>
    <text x="285" y="56" text-anchor="middle" fill="#43e0ce" font-size="16">electric kick exp(iEₑ)</text><text x="285" y="243" text-anchor="middle" fill="#b3c2dc" font-size="16">identity</text>
    <rect x="430" y="96" width="125" height="80" rx="12" fill="#14263c" stroke="#43e0ce"/><text x="492" y="128" text-anchor="middle" fill="#edf3ff" font-size="16">coupled wait</text><text x="492" y="152" text-anchor="middle" fill="#b3c2dc" font-size="13">m t = 2⁻³²</text>
    <path d="M555 136 H605" stroke="#43e0ce" stroke-width="2"/><circle cx="645" cy="107" r="23" fill="#1c3b35" stroke="#43e0ce"/><circle cx="645" cy="170" r="23" fill="#302c24" stroke="#ffbd63"/><text x="645" y="114" text-anchor="middle" fill="#edf3ff" font-size="23">+</text><text x="645" y="177" text-anchor="middle" fill="#edf3ff" font-size="23">−</text>`})}
    <div class="equation">[Eₑ, F𝒇] = g δₑ𝒇 F𝒇<br>Jₑ = i(Fₑ − Fₑ†)<br>Pr(+) = ½ [1 − ⟨sin(uJₑ)⟩]</div>
    <p class="explain">These identities describe the unbounded-field comparison. Finite caps use literal matrix exponentials with the error allowance shown. The longitudinal charge field and the transverse photon field combine to make the comparison kick local to the selected link. At zero coupling the kick cannot change the matter current, so the corresponding contrast is exactly zero.</p></div>
    <div>${metric('> '+sci(r.contrast.lowerText),'Successful setting-conditioned contrast · unbounded-field comparison')}${metric('> '+sci(r.completeLawSeparation.lowerText),'Complete finite record-law separation from the zero-coupling comparison')}${metric('< '+sci(r.cutoffError.upperText),'Finite field-cap comparison error')}
    <p class="explain">The conservative finite witness uses ${r.cellsPerAxis.toLocaleString()} cells per axis and about ${sci(r.capacity.upperText)} nats, excluding controller workspace. This is an analytic sufficient construction, not an enumerated simulation or a resource optimum.</p></div></div>
    <div class="rows"><span>Complete finite interaction record</span><b class="yes">proved in the stated source</b><span>Local record limit at fixed spatial spacing</span><b class="yes">controlled</b><span>Interacting relativistic continuum and particle scattering</span><b class="open">still open</b><span>Standard Model identities and source-derived GeV masses</span><b class="open">still open</b></div>
    ${revised?`<div class="equation">One source family, fresh sea preparation, same charge frame<br>c(V) = 18V + 1 ⇒ ground lies in the code</div>
    <p class="explain">The revised family now has its own preparation and spatial record proof. The old cube penalty 21/2 fails on sufficiently large boxes. A separate source-defined penalty 18V+1 fixes that sector-selection problem for every finite box and cap, without fitting data. The prepared-state record law is unchanged because it stays in the code. This does not supply the coupled ground vector or its internal dispersion.</p>`:
    `<p class="explain">This is the original beta-Wilson source. Its free squared-dispersion coefficient differs from the revised source. The two families have separate state and record proofs; switch above to inspect the revised construction.</p>`}`;
  $('interaction-family').addEventListener('change',event=>{interactionFamily=event.target.value;interaction();$('interaction-family').focus();});
}

try {
  const response=await fetch('./field_records.json');if(!response.ok)throw Error(`Record data unavailable (${response.status})`);data=await response.json();
  if(data.schema!=='oph.viewer.field_records.v1'||data.scope.sourceDerivedMassesGeV.length||data.scope.physicalParticleIdentities.length)throw Error('Unreviewed record scope');
  transfer();ground();cone();interaction();$('loading').hidden=true;showTab('transfer');
  $('proofs').innerHTML=Object.entries(data.evidence).map(([name,e])=>`<div><a href="${esc(e.proof)}" target="_blank" rel="noopener">${esc(name.replaceAll('_',' '))} · proof</a><code>${esc(e.path)}<br>SHA256 ${esc(e.sha256)}<br>Original research receipt ${esc(e.researchSha256)}</code></div>`).join('');
} catch(error) {$('loading').className='error';$('loading').textContent=error.message;}
