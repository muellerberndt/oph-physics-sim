// One frozen native source, with a new bounded vector readout and every outcome.
export function validateProcaEvidence(d) {
  if(d.schema!=='oph.viewer.native_proca.v1'||d.members.length!==2||d.source.canonicalComponents!==3||d.source.fieldComponents!==4||d.source.comparisonPolarizations!==3)throw Error('Vector source changed');
  if(!d.source.sameHamiltonianStateReferenceClock||d.source.referenceDimension!==3072||!d.filters.timeComponentUsesCanonicalMomentum)throw Error('Source, reference or temporal filter changed');
  for(const key of ['finiteSourceExactLorentzSymmetry','interactingRelativisticLimit','unfilteredTripletAutomaticallyVector','standardModelIdentification','sourceDerivedGeV','fullNativeMatrixExecuted','replicasDerivedFromMemoryBudget'])if(d.scope[key]!==false)throw Error('Unreviewed vector claim: '+key);
  for(const key of ['newObservableMap','freeProcaComparison','sameNativeSource','allSixteenOutcomesRetained'])if(d.scope[key]!==true)throw Error('Missing vector scope: '+key);
  const interval=(b,probability=true)=>{if(!b||b.exact.length!==2||b.display.length!==2||b.display.some(x=>!Number.isFinite(x)||x<0||(probability&&x>1))||b.display[0]>b.display[1])throw Error('Invalid vector interval');};
  for(const m of d.members) {
    if(m.comparisonModes!==19||m.records.length!==16||m.signals.length!==2||m.referenceDimension!==3072||m.readoutBits!==4||m.apparatusDimension!==49152)throw Error('Incomplete vector execution or apparatus');
    if(!/^\d+$/.test(m.replicasPerSite)||BigInt(m.replicasPerSite)<=0n||!/^\d+$/.test(m.nativeCells)||BigInt(m.nativeCells)!==8n*BigInt(m.replicasPerSite)*BigInt(m.macroSites))throw Error('Wrong replica/source count');
    if(!(m.completeError.display>0&&m.completeError.display<=9.50000000001e-10)||m.anyLeakageFlagUpper!==m.completeError.display)throw Error('Incomplete record error');
    const menu=new Set();let lo=0,hi=0;
    for(const r of m.records) {
      if(![1,-1].includes(r.a0Sign)||![1,-1].includes(r.a1Sign)||![0,1].includes(r.a0Flag)||![0,1].includes(r.a1Flag))throw Error('Invalid vector outcome');
      const key=[r.a0Sign,r.a0Flag,r.a1Sign,r.a1Flag].join(',');if(menu.has(key))throw Error('Duplicate vector outcome');menu.add(key);
      interval(r.actual);interval(r.ideal);lo+=r.actual.display[0];hi+=r.actual.display[1];
      if((r.a0Flag||r.a1Flag)&&(r.ideal.display[0]!==0||r.ideal.display[1]!==0))throw Error('Comparison leakage invented');
    }
    if(lo>1+1e-12||hi<1-1e-12)throw Error('Incomplete vector probability law');
    for(const signal of m.signals){interval(signal.probability);if(!(signal.probability.display[0]>0))throw Error('Signal below complete error');}
    interval(m.capacity.nats,false);
    if(m.capacity.universeCapacityIdentification!==false||m.capacity.extraSynthesisWorkspaceIncluded!==false)throw Error('Unreviewed capacity promotion');
    if(m.capacity.below1e122!==(m.capacity.nats.display[1]<1e122))throw Error('Capacity comparison changed');
  }
  return d;
}

const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const big=x=>{const s=String(x);return `${s[0]}.${s.slice(1,4)} × 10<sup>${s.length-1}</sup>`;};
function intervalText(b) {
  const [lo,hi]=b.display;if(hi===0)return '0';
  const exponent=Math.floor(Math.log10(hi)),scale=10**exponent;
  const lower=Math.floor(lo/scale*1000)/1000,upper=Math.ceil(hi/scale*1000)/1000;
  return `[${lower.toFixed(3)}, ${upper.toFixed(3)}] × 10<sup>${exponent}</sup>`;
}
const scientific=x=>`${(x/10**Math.floor(Math.log10(x))).toFixed(2)} × 10<sup>${Math.floor(Math.log10(x))}</sup>`;
const cube=`<svg viewBox="0 0 90 52" aria-hidden="true"><path d="M20 16L48 7L73 20L45 30ZM20 16V39L45 49L73 39V20M45 30V49M48 7V29" fill="none" stroke="currentColor" stroke-width="1.6"/>${[[20,16],[48,7],[73,20],[45,30],[20,39],[48,29],[73,39],[45,49]].map(([x,y])=>`<circle cx="${x}" cy="${y}" r="3.1" fill="currentColor"/>`).join('')}</svg>`;

export function procaCardMarkup(d) {
  return `<div class="pe-head"><div><div class="pe-kicker">Same native source · a new observable map</div><h3>An audited massive-vector readout</h3></div><span class="pe-badge">Free comparison limit</span></div>
    <p class="pe-intro">Explicit filters turn the controlled three-component field into a massive-vector comparison. The native Hamiltonian, selected actual ground, clock and retained reference stay the same.</p>
    <div class="pe-flow" role="img" aria-label="Native eight-cell blocks supply three canonical fields and momenta. Explicit temporal and spatial filters define a free Proca comparison with three polarizations.">
      <div class="pe-step">${cube}<b>Native blocks</b><span>8 cells per constituent</span></div>
      <div class="pe-step"><div class="pe-glyph">Φ₁&nbsp; Φ₂&nbsp; Φ₃<small>with conjugate momenta Π</small></div><b>3 canonical components</b><span>one state and clock</span></div>
      <div class="pe-step pe-filter"><div class="pe-glyph">A₀ ← div Π / Ωₐ<small>A ← M(Ω₀) Φ</small></div><b>Explicit vector filters</b><span>time component retained</span></div>
      <div class="pe-step"><div class="pe-glyph pe-vector"><span>A<sup>μ</sup></span><small>4 components · 3 polarizations</small></div><b>Free Proca comparison</b><span>controlled finite windows</span></div></div>
    <div class="pe-unit">Effective free mass parameter <strong>Δ/c²</strong> · source units, c = Δ/√6 <span>No W/Z or GeV assignment</span></div>
    <div class="pe-results"><div><span>Temporal A₀: minus-sign probability</span><b data-proca-a0></b></div><div><span>Spatial A₁: minus-sign probability</span><b data-proca-a1></b></div><div><span>Full logical capacity, including readout</span><b data-proca-capacity></b></div></div>
    <p class="pe-small">These are outcomes of bounded phase tests. The executed calculation is a 19-mode Gaussian comparison; the full native ground, large replica carrier and every leakage record are controlled by explicit error bounds.</p>
    <details class="pe-details"><summary>All 16 outcomes, source size and capacity</summary>
      <div class="pe-members" aria-label="Choose an audited finite source member">${d.members.map((m,i)=>`<button type="button" data-proca-member="${i}" aria-pressed="${i===0}">a = ${esc(m.spacing)} · L = ${m.side}</button>`).join('')}</div>
      <div class="pe-facts" data-proca-facts></div>
      <p>Two tests retain both signs and both internal flags. Their 16 joint outcomes form one complete law. Individual interval endpoints are not separately renormalized. A nonzero flag remains a recorded outcome.</p>
      <div class="pe-scroll"><table class="pe-records"><thead><tr><th>A₀ sign</th><th>Flag</th><th>A₁ sign</th><th>Flag</th><th>Actual source probability interval</th></tr></thead><tbody data-proca-records></tbody></table></div>
      <p data-proca-capacity-detail></p><p>N is the declared number of replicas chosen by sufficient accuracy bounds. It is not predicted by a memory budget or scheduler law. The capacity is the full kinematic log dimension, including reference and pointers; Gauss reduction can only lower it. Extra gate-synthesis and preparation workspace is excluded. The prepared state’s entropy is a different quantity.</p>
      <p>Both displayed logical ceilings are numerically below 10<sup>122</sup> nats. This arithmetic comparison does not identify the universe’s capacity.</p></details>
    <details class="pe-details"><summary>What the vector comparison proves</summary>
      <p>The filtered limit has the Proca polarization tensor, the Lorenz constraint, an explicit spin-one induced Lorentz action and a microcausal commutator. The finite source has bounded readout errors; it has no claimed exact boost symmetry. The finite-replica interaction corrections vanish along the stated refinement, whose limit is free.</p>
      <div class="pe-formula">${esc(d.filters.temporal)}<br>${esc(d.filters.spatial)}</div><p>The formulas use source-derived mass-one, speed-one coordinates. A₀ uses canonical momentum; replacing it by the field alone fails the reality check. The unfiltered triplet remains a distinct scalar-component readout. The same source spectrum does not select this observer map automatically.</p></details>
    <p class="pe-links"><a href="${esc(d.evidence.proof)}" target="_blank" rel="noopener">Full source proof ↗</a> · <a href="proca_evidence.json" target="_blank" rel="noopener">Complete intervals and source receipt ↗</a></p>`;
}

export async function mountProcaEvidence(host) {
  if(!document.getElementById('proca-evidence-style')) {
    const style=document.createElement('style');style.id='proca-evidence-style';style.textContent=`
.pe{margin:20px 0;padding:23px;border:1px solid #3b5768;border-radius:18px;background:linear-gradient(125deg,#172633,#121a2a 72%);color:#e4edf5}.pe-head{display:flex;align-items:center;justify-content:space-between;gap:16px}.pe h3{margin:5px 0 7px;font-size:25px}.pe-kicker{font-size:10.5px;letter-spacing:1.5px;text-transform:uppercase;color:#9bbfe2}.pe-badge{border:1px solid #42657a;border-radius:20px;padding:6px 10px;color:#a6d2ec;font-size:11px;white-space:nowrap}.pe p{font-size:12.5px;line-height:1.65;color:#afc2d0;max-width:1100px}.pe .pe-intro{margin:5px 0 18px}.pe-flow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:17px;margin:15px 0}.pe-step{position:relative;border:1px solid #334c60;border-radius:11px;padding:12px;text-align:center;background:#0b1725a6}.pe-step:not(:last-child):after{content:'→';position:absolute;right:-15px;top:42px;color:#7b9aad;font-size:16px}.pe-step svg{height:52px;width:90px;color:#80bfe6}.pe-step b{display:block;font-size:12px;margin:7px 0 4px;color:#d4e5ef}.pe-step>span{font-size:10.5px;color:#8fa9ba}.pe-glyph{height:52px;display:flex;flex-direction:column;justify-content:center;font-size:16px;color:#9fc7ee}.pe-glyph small{display:block;font-size:9.5px;color:#99afc0;margin-top:6px}.pe-filter{border-color:#6c5d80}.pe-filter .pe-glyph{color:#c9b2ef;font-size:13px}.pe-vector{color:#80e1ca;font-size:24px}.pe-vector sup{font-size:12px}.pe-unit{display:flex;gap:6px;align-items:baseline;flex-wrap:wrap;font-size:12px;color:#adc4d2;margin:14px 0}.pe-unit strong{color:#d7e9f5}.pe-unit>span{margin-left:auto;color:#9eafc0;font-size:10.5px}.pe-results{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}.pe-results>div{padding:14px 12px;border:1px solid #304657;border-radius:10px;background:#0a1622b3}.pe-results span{display:block;font-size:10.5px;color:#9ab1c1;margin-bottom:10px}.pe-results b{display:block;font-size:17px;font-variant-numeric:tabular-nums;color:#83dec9;font-weight:550}.pe-results>div:last-child b{color:#b5cceb}.pe-results sup{font-size:10px}.pe .pe-small{font-size:11px;margin:11px 0 2px}.pe-details{margin-top:14px;font-size:12px;border-top:1px solid #2b4051;padding-top:12px}.pe summary{cursor:pointer;color:#a8d4eb;font-size:12px}.pe-members{display:flex;gap:8px;margin:15px 0;flex-wrap:wrap}.pe-members button{border:1px solid #435a70;background:#15283a;color:#c2d3df;padding:7px 11px;border-radius:7px;cursor:pointer}.pe-members button[aria-pressed=true]{border-color:#7cbbd9;background:#21404f;color:#bdeaff}.pe-facts{display:grid;grid-template-columns:1fr 1fr;gap:7px 18px;color:#a9bfce;font-size:11.5px}.pe-facts b{color:#d7e4ed;font-weight:500}.pe-scroll{overflow-x:auto}.pe-records{width:100%;border-collapse:collapse;font-size:11px;font-variant-numeric:tabular-nums}.pe-records th,.pe-records td{text-align:left;padding:7px 8px;border-bottom:1px solid #293e4f;white-space:nowrap}.pe-records th{color:#89a6b9;font-weight:500}.pe-records td{color:#bfd2df}.pe-records tr[data-leak=true]{opacity:.75}.pe-formula{border-left:2px solid #9c8abf;padding:8px 12px;color:#c9b7e6;font:12px/1.9 ui-monospace,monospace}.pe a{color:#9ad3ed}.pe .pe-links{font-size:11px;margin:15px 0 0}.pe code{overflow-wrap:anywhere}.pe button:focus-visible,.pe summary:focus-visible{outline:2px solid #9bd8f4;outline-offset:3px}
@media(max-width:850px){.pe-results{grid-template-columns:1fr 1fr}.pe-results>div:last-child{grid-column:1/-1}.pe-unit>span{margin-left:0;width:100%}}
@media(max-width:560px){.pe{padding:15px}.pe-head{align-items:flex-start;flex-direction:column;gap:4px}.pe h3{font-size:22px}.pe-flow{grid-template-columns:1fr 1fr;gap:10px}.pe-step:not(:last-child):after{content:none}.pe-results{grid-template-columns:1fr}.pe-results>div:last-child{grid-column:auto}.pe-facts{grid-template-columns:1fr}.pe-glyph{font-size:14px}.pe-filter .pe-glyph{font-size:11px}.pe-glyph small{font-size:8.5px}}
`;document.head.appendChild(style);
  }
  host.className='pe';host.innerHTML='<p role="status">Loading the audited vector readout…</p>';
  try {
    const response=await fetch('proca_evidence.json');if(!response.ok)throw Error('Evidence unavailable');
    const data=validateProcaEvidence(await response.json());host.innerHTML=procaCardMarkup(data);
    function draw(index) {
      const m=data.members[index];host.querySelectorAll('[data-proca-member]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.procaMember)===index)));
      host.querySelector('[data-proca-a0]').innerHTML=intervalText(m.signals[0].probability);
      host.querySelector('[data-proca-a1]').innerHTML=intervalText(m.signals[1].probability);
      host.querySelector('[data-proca-capacity]').innerHTML=`≈ ${scientific(m.capacity.nats.display[1])} nats`;
      host.querySelector('[data-proca-facts]').innerHTML=`<div>Replica count N per macro site: <b>${big(m.replicasPerSite)}</b></div><div>Macro sites: <b>${m.macroSites.toLocaleString()}</b></div><div>Native cells: <b>${big(m.nativeCells)}</b></div><div>Retained reference + readout: <b>${m.apparatusDimension.toLocaleString()} states</b></div><div>Complete record error: <b>≤ ${scientific(m.completeError.display)}</b></div><div>Any nonzero leakage flag: <b>≤ ${scientific(m.anyLeakageFlagUpper)}</b></div>`;
      host.querySelector('[data-proca-records]').innerHTML=m.records.map(r=>`<tr data-leak="${Boolean(r.a0Flag||r.a1Flag)}"><td>${r.a0Sign===1?'+':'−'}</td><td>${r.a0Flag}</td><td>${r.a1Sign===1?'+':'−'}</td><td>${r.a1Flag}</td><td>${intervalText(r.actual)}</td></tr>`).join('');
      host.querySelector('[data-proca-capacity-detail]').innerHTML=`<strong>Capacity:</strong> ${esc(m.capacity.outwardDisplay.join(' – '))} nats, outward rounded. Formula: <code>${esc(m.capacity.formula)}</code>. The exact replica count is <code>${esc(m.replicasPerSite)}</code>. Source-box side: ${esc(m.sourceBoxSide)}; momentum radius: ${esc(m.momentumRadius)} in unit-speed coordinates. Test times: Δt = ${m.times.map(esc).join(', ')}.`;
    }
    host.querySelectorAll('[data-proca-member]').forEach(b=>b.onclick=()=>draw(Number(b.dataset.procaMember)));draw(0);
  } catch(error) {host.textContent='The vector readout could not be validated. '+error.message;host.setAttribute('role','alert');}
}
