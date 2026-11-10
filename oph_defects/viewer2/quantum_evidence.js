// Quantitative finite interacting results, kept separate from the classical run and free viewer packets.
export function validateQuantumEvidence(data) {
  if(data.schema!=='oph.viewer.interacting_energy_evidence.v1' || data.channels.length!==2 || !data.allFailuresRetained)throw Error('Incomplete interacting evidence');
  for(const key of ['isolatedChargedPole','physicalRestMass','sameAsClassicalLoadRun','sourceDerivedGeV'])if(data.scope[key]!==false)throw Error('Unreviewed physical promotion: '+key);
  for(const row of data.channels) {
    if(!['ec','L'].includes(row.id) || row.massGeV!==null || row.empiricalParticle!==null || row.referenceEnergy!=='4')throw Error('Source/reference assignment changed');
    for(const interval of [row.sourceEnergy,row.totalEnergy,...Object.values(row.probabilities)]) {
      const [lo,hi]=interval.display;
      if(!Number.isFinite(lo)||!Number.isFinite(hi)||lo>hi)throw Error('Invalid evidence interval');
    }
    for(const key of ['line','outside','failure']) {
      const [lo,hi]=row.probabilities[key].display;
      if(!(lo>0&&hi<=1))throw Error('Detector omitted a positive outcome');
    }
  }
  if(!(data.preparation.traceDistanceUpper>0 && data.preparation.traceDistanceUpper<1e-6))throw Error('Preparation bound changed');
  const h=data.higgsConnection;
  if(!h || h.detectorRecords!==16 || h.potentiallyNonzeroRecords!==6 || !h.allFailureAndComplementRecordsRetained || !h.exactFiniteSpectralService)throw Error('Incomplete quantum Higgs experiment');
  for(const key of ['usesClassicalScalarBackground','compositeCanonicalOnAllOccupations','sameAsMaximumElementaryResidueSelector','weakScatteringOrSMIdentityProved','QPEImplementedForThisCompositeExperiment'])if(h[key]!==false)throw Error('Unreviewed Higgs promotion: '+key);
  if(!(rational(h.crossResidueLower)>.2898 && rational(h.compositeBinWeightLower)>.1999 && rational(h.originalSettingHitContrastLower)>.0724 && rational(h.originalSettingHitContrastLower)<1))throw Error('Invalid Higgs response bound');
  if(h.settings.length!==2 || h.sourceEnergy.display[0]<.69 || h.sourceEnergy.display[1]>.7)throw Error('Higgs setting or energy changed');
  const c=data.centroid;
  if(!c || c.pointerLabels!==1048576 || c.rationalFactors!==536870400 || !c.allFailuresAndPointerLabelsRetained || !c.priorExactSpectralBinService)throw Error('Incomplete finite centroid clock');
  for(const key of ['extraGroundProjection','exactGroundPhaseInput','fullSourcePointerExecuted','singleEigenvalueSelected','restMassOrSMIdentity'])if(c[key]!==false)throw Error('Unreviewed centroid promotion: '+key);
  const [lo,hi]=c.sourceEnergyMean.display;
  if(!Number.isFinite(lo)||!Number.isFinite(hi)||lo<.6916566792||hi>.6916566827||lo>=hi||hi-lo>=3.4e-9)throw Error('Centroid enclosure changed');
  if(!(c.originalBinSuccessLower.display[0]>.419706 && c.originalBinSuccessLower.display[1]<1 && c.baselineErrorUpper.display[0]>0))throw Error('Preparation or baseline was omitted');
  return data;
}
function rational(x){if(typeof x!=='string'||!/^\d+(?:\/\d+)?$/.test(x))throw Error('Invalid exact bound');const [n,d='1']=x.split('/');const value=Number(n)/Number(d);if(!Number.isFinite(value))throw Error('Non-finite exact bound');return value;}
const lowerPercent=x=>(Math.floor(x*1e8)/1e6).toFixed(6);
const interval=x=>x.display.map((v,i)=>((i?Math.ceil:Math.floor)(v*1e6)/1e6).toFixed(6)).join(' – ');
export function centroidHTML(data){const c=data.centroid;return `<section class="qe-centroid" aria-labelledby="qe-centroid-title"><div class="qe-tag">Finite physical preparation · one complete energy clock</div><h4 id="qe-centroid-title">A sharper prediction for the charged detector’s mean</h4><div class="qe-mean">${c.sourceEnergyMean.display.map(v=>v.toFixed(13)).join(' – ')}</div><p>Source energy units above the interacting ground. This is the predicted <strong>average clock reading within the admitted bin</strong>, with preparation and finite-pointer errors included. It is not a claim that every outcome has this energy or that a single particle mass has been found.</p><div class="qe-stats"><span><b>${lowerPercent(c.originalBinSuccessLower.display[0])}%</b>minimum bin probability per original attempt</span><span><b>1,048,576</b>retained phase-clock labels</span><span><b>No extra ground projection</b>uses the finite prepared state</span></div><details><summary>What is counted, and what apparatus is supplied?</summary><p>Preparation failure, pump failure, the whole outside-bin outcome and every phase label are retained. The energy reference contributes 4 source units. The decoder subtracts a known rational bound midpoint for the neutral ground; its residual uncertainty is ${c.baselineErrorUpper.display[1].toExponential(3)} source units, rather than an unknown exact-ground phase.</p><p>Admission to this narrow bin is still a declared exact spectral operation. The subsequent 20-bit clock has an explicit rational implementation of at most ${c.rationalFactors.toLocaleString('en-US')} factors. These bounds specify a finite instrument; the enormous full source-plus-pointer matrix has not been numerically executed. Controller-synthesis workspace is not included in the logical register count.</p><p><a href="${data.evidence.native_charge_centroid.proof}" target="_blank" rel="noopener">Physical preparation, centroid and clock proof ↗</a></p></details></section>`;}
export function higgsConnectionHTML(data){
  const h=data.higgsConnection;
  return `<section class="qe-join" aria-labelledby="qe-higgs-title"><div class="qe-tag">Two probes · one quantum source</div>
    <h4 id="qe-higgs-title">The scalar and the fermion meet at the same energy</h4>
    <p>The source lets us disturb its state in two different ways. Their energy-filtered responses overlap: they are not simply two labels given to unrelated objects.</p>
    <div class="qe-probes"><div><span>Elementary probe</span><b>A = eᶜ†</b><small>Change one charged fermion occupation.</small></div><div class="qe-shared"><span>Shared response</span><b>${interval(h.sourceEnergy)}</b><small>source units above the same ground</small></div><div><span>Quantum composite probe</span><b>B = −ε H†L</b><small>Change the scalar and lepton doublet together.</small></div></div>
    <p class="qe-equation">[h, A] = B + 2⁻³⁴[T, A]</p>
    <p>The Hamiltonian itself links the two probes through this equation of motion. The measured scalar is a finite quantum register; no classical scalar value is inserted. Both probes can reach a common odd, charged, two-component source excitation.</p>
    <details><summary>How an apparatus distinguishes a shared response from matching labels</summary>
      <p>Combine the probes with a plus sign or a minus sign. If their contributions reach the same energy state, they interfere. The complete experiment uses the two settings <strong>(A ± B/2)/2</strong>. After counting unsuccessful preparations and both settings, its difference in joint setting-and-bin-hit probabilities is at least <strong>${lowerPercent(rational(h.originalSettingHitContrastLower))} percentage points</strong>.</p>
      <p>All ${h.detectorRecords} setting, spin-occupation and energy labels remain, including ten exact-zero labels and every preparation failure. The energy-bin projector is a declared exact finite apparatus service; this composite experiment has not been compiled into the phase-clock circuit shown for the other experiment.</p>
      <p>This is a shared spectral response. It does not make H†L a canonical fermion on every occupied state, prove a weak-scattering amplitude, or identify an electron mass. The selected shared line also need not be the line chosen by the separate maximum-residue Dirac construction.</p>
    </details><p><a href="${data.evidence.native_higgs_spectral_join.proof}" target="_blank" rel="noopener">Same-Hamiltonian composite-response proof ↗</a></p></section>`;
}
export async function mountQuantumEvidence(host) {
  const style=document.createElement('style');style.textContent=`
.qe{margin:18px 0 22px;border:1px solid #446b79;border-radius:18px;background:linear-gradient(125deg,#132b32,#111b30 60%);padding:22px;overflow:hidden}
.qe h3{font-size:22px;margin:2px 0 8px}.qe p{color:var(--soft);font-size:13px;max-width:1060px;line-height:1.55}.qe-tag{color:#73edc9;font-size:11px;letter-spacing:1.5px;text-transform:uppercase}
.qe-flow{display:flex;align-items:center;gap:16px;margin:16px 0}.qe-flow>div{flex:1;border:1px solid #33515f;border-radius:12px;padding:14px;background:#0d1826}.qe-flow b{display:block;color:#dcecf3;font-size:14px}.qe-flow span{font-size:12px;color:#b1c9d0}.qe-arrow{color:#69dec8;font-size:26px}
.qe-table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}.qe-table th{font-size:10px;letter-spacing:1px;color:#9eb7c5;text-transform:uppercase;text-align:left}.qe-table td,.qe-table th{padding:12px 10px;border-bottom:1px solid #2a4053}.qe-table td:nth-child(n+2){color:#83e7d5}.qe-table-wrap{overflow:auto}.qe-foot{font-size:12px!important;color:#9db3c4!important}.qe a{color:#8bd9fa}
.qe-join{margin-top:24px;padding-top:22px;border-top:1px solid #355368}.qe-join h4{font-size:21px;margin:8px 0}.qe-probes{display:grid;grid-template-columns:1fr 1.15fr 1fr;gap:12px;margin:18px 0}.qe-probes>div{padding:17px;border:1px solid #426779;border-radius:12px;background:#0b1928;min-width:0}.qe-probes b{display:block;margin:8px 0;font-size:23px;color:#81ead4}.qe-probes span,.qe-probes small{font-size:12px;color:#b9cad8}.qe-probes .qe-shared{border-color:#806bb3;background:#211e38}.qe-shared b{font-size:19px;color:#c4b0ff;font-variant-numeric:tabular-nums}.qe-equation{padding:12px 16px;border-left:3px solid #9980d8;font-size:20px!important;color:#d6cced!important}.qe-join summary{cursor:pointer;color:#b5e2ef;font-size:14px;line-height:1.5}.qe-join details{padding:12px 0}.qe-join details p{padding-left:12px}
.qe-centroid{margin-top:22px;padding:20px;border:1px solid #398575;border-radius:14px;background:#0d272a}.qe-centroid h4{margin:8px 0 14px;font-size:20px}.qe-mean{font-size:clamp(17px,2.3vw,28px);font-weight:650;color:#8df0d2;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.qe-stats{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}.qe-stats span{flex:1;min-width:150px;font-size:12px;color:#abc5cb}.qe-stats b{display:block;font-size:17px;color:#d4efea;margin-bottom:6px}.qe-centroid summary{font-size:14px;color:#b5e2ef;cursor:pointer;line-height:1.5}
@media(max-width:650px){.qe{padding:15px}.qe-flow{flex-direction:column;align-items:stretch;gap:7px}.qe-arrow{transform:rotate(90deg);text-align:center}.qe h3{font-size:19px}.qe-table{min-width:610px}}
@media(max-width:750px){.qe-probes{grid-template-columns:1fr}.qe-probes .qe-shared{order:3}.qe-join h4{font-size:19px}}
`;document.head.append(style);
  host.className='qe';host.innerHTML='<p role="status">Loading authenticated interacting energy records…</p>';
  try {
    const response=await fetch('quantum_evidence.json');if(!response.ok)throw Error('Evidence unavailable');
    const data=validateQuantumEvidence(await response.json());
    host.innerHTML=`<div class="qe-tag">One Hamiltonian · physical preparation · complete outcomes</div>
      <h3>What the interacting quantum source actually measures</h3>
      <p>Eight cells hold all fermion occupations, quantum scalar states and gauge links. The coupled ground is proved at hopping 2⁻³⁴. A detector can remove one native fermion into a quantum reference and measure the resulting energy. This calculation uses a declared quantum program; it is separate from the classical load run and the free packet animation.</p>
      <div class="qe-flow"><div><b>1 · Prepare the physical ground</b><span>Finite controller; source error &lt; 2.36 × 10⁻⁷. Keep preparation failures.</span></div><i class="qe-arrow" aria-hidden="true">→</i><div><b>2 · Transfer one fermion</b><span>The reference keeps its charge. Every occupation and pump failure is retained.</span></div><i class="qe-arrow" aria-hidden="true">→</i><div><b>3 · Read an energy bin</b><span>Record the selected bin or its complete complement on the same clock.</span></div></div>
      <div class="qe-table-wrap"><table class="qe-table"><thead><tr><th scope="col">Native removal field</th><th scope="col">Source energy</th><th scope="col">Bin probability</th><th scope="col">Outside probability</th></tr></thead><tbody>${data.channels.map(r=>`<tr><th scope="row">${r.id==='ec'?'eᶜ · charged singlet':'L · weak lepton doublet'}</th><td>${interval(r.sourceEnergy)}</td><td>${interval(r.probabilities.line)}</td><td>${interval(r.probabilities.outside)}</td></tr>`).join('')}</tbody></table></div>
      <p><strong>Energy accounting:</strong> the apparatus reference costs 4 source units in either successful channel. Add that to the displayed source energy for the total transition. Pump failure is about one half. The probability intervals use the exact ground; the finite preparation adds the error stated above.</p>
      <p class="qe-foot">These are positive odd-fermion responses. A static charge-compensated energy transition does not yet identify a propagating electron, muon or neutrino, nor a rest mass in GeV. The preparation's ${data.preparation.scheduleTerms} terms specify a finite algebraic service, not an efficiently executed full-source circuit.</p>
      <p><a href="${data.evidence.interacting_charged_response.proof}" target="_blank" rel="noopener">Charged response proof ↗</a> · <a href="${data.evidence.interacting_ground_preparation.proof}" target="_blank" rel="noopener">Physical preparation proof ↗</a> · <a href="quantum_evidence.json" target="_blank">Source receipts and exact intervals ↗</a></p>${centroidHTML(data)}${higgsConnectionHTML(data)}`;
  } catch(error) {host.textContent='The interacting energy evidence could not be validated. '+error.message;host.setAttribute('role','alert');}
}
