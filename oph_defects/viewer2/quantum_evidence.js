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
  return data;
}
const interval=x=>x.display.map(v=>v.toFixed(6)).join(' – ');
export async function mountQuantumEvidence(host) {
  const style=document.createElement('style');style.textContent=`
.qe{margin:18px 0 22px;border:1px solid #446b79;border-radius:18px;background:linear-gradient(125deg,#132b32,#111b30 60%);padding:22px;overflow:hidden}
.qe h3{font-size:22px;margin:2px 0 8px}.qe p{color:var(--soft);font-size:13px;max-width:1060px;line-height:1.55}.qe-tag{color:#73edc9;font-size:11px;letter-spacing:1.5px;text-transform:uppercase}
.qe-flow{display:flex;align-items:center;gap:16px;margin:16px 0}.qe-flow>div{flex:1;border:1px solid #33515f;border-radius:12px;padding:14px;background:#0d1826}.qe-flow b{display:block;color:#dcecf3;font-size:14px}.qe-flow span{font-size:12px;color:#b1c9d0}.qe-arrow{color:#69dec8;font-size:26px}
.qe-table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}.qe-table th{font-size:10px;letter-spacing:1px;color:#9eb7c5;text-transform:uppercase;text-align:left}.qe-table td,.qe-table th{padding:12px 10px;border-bottom:1px solid #2a4053}.qe-table td:nth-child(n+2){color:#83e7d5}.qe-table-wrap{overflow:auto}.qe-foot{font-size:12px!important;color:#9db3c4!important}.qe a{color:#8bd9fa}
@media(max-width:650px){.qe{padding:15px}.qe-flow{flex-direction:column;align-items:stretch;gap:7px}.qe-arrow{transform:rotate(90deg);text-align:center}.qe h3{font-size:19px}.qe-table{min-width:610px}}
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
      <p><a href="${data.evidence.interacting_charged_response.proof}" target="_blank" rel="noopener">Charged response proof ↗</a> · <a href="${data.evidence.interacting_ground_preparation.proof}" target="_blank" rel="noopener">Physical preparation proof ↗</a> · <a href="quantum_evidence.json" target="_blank">Source receipts and exact intervals ↗</a></p>`;
  } catch(error) {host.textContent='The interacting energy evidence could not be validated. '+error.message;host.setAttribute('role','alert');}
}
