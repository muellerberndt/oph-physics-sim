import { familyReading, LEPTON_REFERENCE, FR_TWO_TICK, muonResidualPpm } from './lepton_reading.js';
// The Story tab, card "A charged-lepton mass-ratio candidate". Sources: repair-law/SPECTRUM.md (the coordinator's ledger)
// and the reports DA3.md (branch weights), FR.md (family register, record = endpoint), LNAT2.md (phase, ratios), KOI.md (node at pi/12), LS2.md (scale).
// Computed in the browser: the three amplitudes 1 + sqrt2 cos(delta + 2 pi k / 3) and the two mass ratios for any delta.
const PI = Math.PI, R2 = Math.SQRT2;
const MEAS = { mu: LEPTON_REFERENCE.muonGeV / LEPTON_REFERENCE.electronGeV, tau: LEPTON_REFERENCE.tauGeV / LEPTON_REFERENCE.electronGeV };            // API values frozen in family_clock.json; rounded display masses are separate.
const D29 = 2 / 9, NODE = PI / 12, DMAX = PI / 6;
const TEAL = '#3fe0d0', AMBER = '#ffb44a', PINK = '#ff6480', VIOLET = '#a995ff';
const L = { inited: false, d: D29 };

const STYLE = `
.lk-ch{display:grid;grid-template-columns:minmax(380px,1.05fr) minmax(320px,1fr);gap:22px;margin:34px 0 30px;align-items:start;border-top:1px solid var(--edge);padding-top:22px}
.lk-ch h2{margin:0 0 4px}
.lk-badge{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:50%;background:#ffe45c;color:#04060d;font-weight:700;margin-right:10px;font-size:17px}
.lk-lead{font-size:14px;color:#dfe6f5;margin:4px 0 8px;line-height:1.5}
.lk-formula{display:inline-block;font:600 18px/1.3 ui-monospace,Menlo,monospace;color:var(--text);margin:8px 0 10px;padding:9px 14px;border:1px solid var(--edge);border-radius:12px;background:rgba(10,16,30,.65)}
.lk-claims{display:grid;gap:7px;margin:8px 0 6px}
.lk-claim{display:grid;grid-template-columns:168px 1fr;gap:10px;align-items:baseline;font-size:13.5px;color:var(--soft);line-height:1.45}
.lk-tag{justify-self:start;font-size:11px;padding:1px 8px;border-radius:9px;white-space:normal;line-height:1.35}
.lk-tag.d{background:rgba(95,227,161,.15);color:#8ff0bf}.lk-tag.c{background:rgba(255,180,74,.15);color:#ffd08a}.lk-tag.o{background:rgba(255,100,128,.16);color:#ff9db0}
.lk-sub{font-size:11.5px;letter-spacing:1.6px;text-transform:uppercase;color:var(--teal);font-weight:700;margin:14px 0 4px}
.lk-cv{width:100%;display:block;border:1px solid var(--edge);border-radius:10px;background:#060a14;margin-bottom:8px}
.lk-ctl{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px;color:var(--muted);margin:6px 0 8px}
.lk-ctl input[type=range]{flex:1;min-width:180px}
.lk-ctl b{font:600 14px ui-monospace,Menlo,monospace;color:var(--text);min-width:92px}
.lk-tab{border-collapse:collapse;width:100%;font-size:13px;margin:4px 0}
.lk-tab td{padding:5px 8px;border-bottom:1px solid #16213a;color:var(--soft)}
.lk-tab td.n{text-align:right;font-family:ui-monospace,Menlo,monospace;white-space:nowrap}
.lk-note{color:var(--faint);font-size:12px;margin-top:6px;line-height:1.45}
.lk-lg{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12px;color:var(--soft);margin:-2px 0 10px}
.lk-lg i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
@media (max-width:1000px){.lk-ch{grid-template-columns:1fr}.lk-claim{grid-template-columns:1fr}}
`;
const tag = (k, label) => `<span class="lk-tag ${k}">${label}</span>`;
const claim = (k, label, text) => `<div class="lk-claim">${tag(k, label)}<span>${text}</span></div>`;
function html() {
  return `<div>
<h2><span class="lk-badge">5</span>A charged-lepton mass-ratio candidate</h2>
<p class="lk-lead">A declared neutral register has three characters. Under the FR preparation and analyzer, one tick produces a circulant with three eigenvalues. Squaring their relative amplitudes gives the candidate ratios below. This calculation does not make the three characters independently occupiable particle generations, and the repeated FR tick has a different spectrum.</p>
<div class="lk-formula">√w<sub>k</sub> ∝ 1 + √2 cos(δ + 2πk/3),&nbsp; k = 0, 1, 2</div>
<div class="lk-claims">
${claim('d', 'exact local check', 'All 360 tested local columns agree with native Observer.repair. Under the stated preparation the branch weights are agreement 1/6 and oriented repairs 1/12 + 1/12. The family register, controlled shifts and analyzer are additional declared operations.')}
${claim('d', 'one-tick effect', 'M=(√2/3)C(δ), with a=1/√6 and b=1/√12. In a chamber where the three eigenvalues are nonnegative, the squared-eigenvalue reading gives Q=2/3 exactly. The complete effects are M†M and I−M†M; mean success on a mixed family input is 2/27. Success-only conditioning is a different experiment.')}
${claim('o', 'literal repeated tick', 'With the same records reused, <Φ|U²|p>=√3/9 I for every phase: all three squared amplitudes are equal, and Q=1/3. The actual tick has multiple Floquet branches. Its one-tick compression does not establish a persistent three-mass clock.')}
${claim('c', 'a declared complete clock', 'A constructive extension uses fresh ancillas and P=U†(|Φ><Φ|⊗I)U, retaining every outcome. Its marginal converges to exp(−itA), A=M†M=(2/9)C², with family/reference error bounded by min(2,4t²/N). The generator, inverse tick and fresh-ancilla schedule are explicit additional operations.')}
${claim('c', 'probability versus phase', 'For independent flags p=1/12 and q=1/6, p+q−2pq=2/9 exactly. Converting that probability into δ=2/9 radians requires a specified clock coupling and time standard. The probability alone does not determine that coupling.')}
${claim('d', 'cancellation node', 'In this circulant, the smallest amplitude vanishes at δ=π/12 because 1+√2 cos(π/12+2π/3)=0. Near the node a small eigenvalue remains. This explains the small weight within the candidate reading.')}
${claim('c', 'δ=2/9 comparison', 'The squared-eigenvalue ratios are 206.77031597 and 3477.472837. The muon/electron residual is +9.83354 ppm against the shipped particle reference. The slider can fit that comparison; a fitted phase is then calibration, not an independent prediction.')}
${claim('c', 'independent occupations', 'The three characters are sectors of one neutral C³ register. Independent generations require Fock(K⊗C³), a larger declared state space; Fock(K)⊗C³ alone does not provide three independently occupiable copies.')}
${claim('d', 'same-H finite memory result', 'The declared local quantum scalar memory has two charged energy-transfer lines and nine dark states, with all residues retained. Elementary and composite operators share that Hamiltonian, reference and clock. This finite line identity does not establish continuum FMS transfer or relativistic rest masses.')}
${claim('o', 'residual and scale', 'For every positive memory coupling, its heavier/lighter line ratios strictly exceed the squared-circulant ratios, so this dressing cannot remove the positive muon residual. No absolute mass unit follows. Displayed GeV values elsewhere use the measured electron as calibration.')}
</div>
<p class="lk-note">A neutral Higgs background H=(0,v) instead gives m=|vY|, with v=v_EW/√2. Choosing Y=C gives ratios 14.3795 and 58.9701; Y=C² gives the squared ratios above. These are different declared operators. <a href="/rl/CODEX_JOIN_SPECTRUM/family_clock.json">Family instrument, repeated tick and clock evidence</a> · <a href="/rl/CODEX_JOIN_SPECTRUM/quantum_memory.json">same-H memory evidence</a>.</p>
</div>
<div>
<div class="lk-sub" style="margin-top:4px">Try the phase</div>
<div class="lk-ctl"><span>δ</span><input type="range" id="lk-d" min="0" max="${DMAX}" step="0.00005" value="${D29}"><b id="lk-d-val"></b></div>
<div class="lk-ctl"><button id="lk-29">2/9</button><button id="lk-fit">fit to μ/e</button><button id="lk-node">π/12, the node</button><button id="lk-zero">0</button></div>
<canvas class="lk-cv" id="lk-amp"></canvas>
<div class="lk-lg"><span><i style="background:${TEAL}"></i>agreement, 1</span><span><i style="background:${AMBER}"></i>disagreement, adds</span><span><i style="background:${PINK}"></i>disagreement, subtracts</span><span><i style="background:#ffffff"></i>the sum, √w up to one common factor</span></div>
<canvas class="lk-cv" id="lk-ratio"></canvas>
<table class="lk-tab" id="lk-num"></table>
<div class="lk-note">Literal reused-record second tick: compression √3/9 I = ${FR_TWO_TICK.compressionAmplitude.toFixed(9)} I; ratios ${FR_TWO_TICK.mu}, ${FR_TWO_TICK.tau}; Q=${FR_TWO_TICK.Q.toFixed(6)}. At δ=2/9 the one-tick comparison residual remains +${muonResidualPpm().toFixed(5)} ppm.</div>
<div class="lk-note">Candidate weights w<sub>k</sub> ∝ (1 + √2 cos(δ + 2πk/3))². The sorted weights are labelled by the e, μ, τ comparison targets. Reference ratios: ${MEAS.mu.toFixed(9)} and ${MEAS.tau.toFixed(6)}. No absolute energy scale is selected here. Q = Σw / (Σ√w)² stays 2/3 for every δ up to the node.</div>
</div>`;
}

// Character amplitudes and the declared squared-weight comparison reading.
const copies = familyReading;
// the phase that gives the measured mu/e left of the node (bisection)
function fitMu() { let lo = 0.15, hi = NODE - 1e-9; for (let i = 0; i < 200; i++) { const mid = (lo + hi) / 2; if (copies(mid).mu < MEAS.mu) lo = mid; else hi = mid; } return (lo + hi) / 2; }

function setup(cv, cssH) {
  const dpr = Math.min(2, window.devicePixelRatio || 1), W = Math.max(50, cv.clientWidth || cv.parentNode.clientWidth || 500);
  cv.style.height = cssH + 'px'; cv.width = Math.round(W * dpr); cv.height = Math.round(cssH * dpr);
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.fillStyle = '#060a14'; g.fillRect(0, 0, W, cssH); return { g, W, H: cssH };
}
function ampDraw() {
  const cv = document.getElementById('lk-amp'); if (!cv) return;
  const { g, W, H } = setup(cv, 210), c = copies(L.d), m = { l: 56, r: 86, t: 26, b: 26 }, lo = -1.6, hi = 2.6;
  const X = (v) => m.l + (W - m.l - m.r) * (v - lo) / (hi - lo), rows = [0, 1, 2].sort((i, j) => c.m[j] - c.m[i]), rh = (H - m.t - m.b) / 3;
  g.font = '11px ui-monospace,Menlo,monospace'; g.fillStyle = 'rgba(147,163,192,.85)'; g.fillText('amplitude of each character: agreement 1, plus disagreement √2 cos(δ + 2πk/3)', 10, 15);
  g.strokeStyle = 'rgba(147,163,192,.35)'; g.lineWidth = 1; g.beginPath(); g.moveTo(X(0), m.t - 4); g.lineTo(X(0), H - m.b + 4); g.stroke();
  for (const v of [-1, 1, 2]) { g.setLineDash([2, 4]); g.beginPath(); g.moveTo(X(v), m.t - 4); g.lineTo(X(v), H - m.b + 2); g.stroke(); g.setLineDash([]); }
  g.fillStyle = 'rgba(147,163,192,.85)'; for (const v of [-1, 0, 1, 2]) g.fillText(String(v), X(v) - 3, H - 8);
  rows.forEach((k, r) => {
    const y = m.t + r * rh + rh / 2, a = c.amp[k], dis = a - 1;
    g.fillStyle = 'rgba(223,230,245,.95)'; g.font = '600 14px ui-monospace,Menlo,monospace'; g.fillText(c.names[k], 14, y + 5);
    g.font = '11px ui-monospace,Menlo,monospace'; g.fillStyle = 'rgba(147,163,192,.8)'; g.fillText(`k=${k}`, 30, y + 5);
    g.fillStyle = TEAL; g.globalAlpha = 0.85; g.fillRect(X(0), y - 9, X(1) - X(0), 8); g.globalAlpha = 1;          // agreement: 1
    g.strokeStyle = dis >= 0 ? AMBER : PINK; g.lineWidth = 4; g.beginPath(); g.moveTo(X(1), y + 5); g.lineTo(X(a), y + 5); g.stroke();   // disagreement
    const dir = Math.sign(dis) || 1; g.fillStyle = dis >= 0 ? AMBER : PINK; g.beginPath(); g.moveTo(X(a), y + 5); g.lineTo(X(a) - dir * 7, y + 0.5); g.lineTo(X(a) - dir * 7, y + 9.5); g.fill();
    g.fillStyle = '#ffffff'; g.beginPath(); g.arc(X(a), y - 5, 4, 0, 2 * PI); g.fill();                          // the sum
    g.fillStyle = 'rgba(223,230,245,.95)'; g.fillText(a.toFixed(4), W - m.r + 8, y - 1); g.fillStyle = 'rgba(147,163,192,.85)'; g.fillText(`w ∝ ${a * a < 1e-3 ? (a * a).toExponential(1) : (a * a).toFixed(4)}`, W - m.r + 8, y + 12);
  });
}
function ratioDraw() {
  const cv = document.getElementById('lk-ratio'); if (!cv) return;
  const { g, W, H } = setup(cv, 250), m = { l: 46, r: 14, t: 24, b: 26 }, y0 = 0, y1 = 5;
  const X = (d) => m.l + (W - m.l - m.r) * d / DMAX, Y = (r) => H - m.b - (H - m.t - m.b) * (Math.log10(Math.max(1, r)) - y0) / (y1 - y0);
  g.font = '11px ui-monospace,Menlo,monospace'; g.fillStyle = 'rgba(147,163,192,.85)'; g.fillText('candidate weight ratios against δ; dashed: mass targets', 10, 15);
  g.strokeStyle = 'rgba(147,163,192,.3)'; g.lineWidth = 1;
  for (let e = 0; e <= 5; e++) { g.beginPath(); g.moveTo(m.l, Y(10 ** e)); g.lineTo(W - m.r, Y(10 ** e)); g.stroke(); g.fillText(e === 0 ? '1' : `10${'⁰¹²³⁴⁵'[e]}`, 8, Y(10 ** e) + 4); }
  for (const [d, lab] of [[0, '0'], [D29, '2/9'], [NODE, 'π/12'], [DMAX, 'π/6']]) { g.fillText(lab, X(d) - (d === DMAX ? 22 : 6), H - 8); }
  g.setLineDash([2, 3]); g.strokeStyle = 'rgba(147,163,192,.55)'; g.beginPath(); g.moveTo(X(NODE), m.t); g.lineTo(X(NODE), H - m.b); g.stroke(); g.setLineDash([]);
  for (const [key, col, lab] of [['mu', AMBER, 'μ/e'], ['tau', VIOLET, 'τ/e']]) {
    g.strokeStyle = col; g.globalAlpha = 0.65; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(m.l, Y(MEAS[key])); g.lineTo(W - m.r, Y(MEAS[key])); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    g.lineWidth = 2; g.beginPath(); let pen = false;
    for (let i = 0; i <= 700; i++) { const d = DMAX * i / 700, r = copies(d)[key]; if (!isFinite(r) || r > 1e6) { pen = false; continue; } const px = X(d), py = Math.max(m.t, Y(r)); if (pen) g.lineTo(px, py); else g.moveTo(px, py); pen = py > m.t; }
    g.stroke(); g.lineWidth = 1; g.fillStyle = col; g.fillText(lab, m.l + 6, Y(MEAS[key]) - 5);
  }
  g.strokeStyle = '#ffffff'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(X(L.d), m.t); g.lineTo(X(L.d), H - m.b); g.stroke();
  const c = copies(L.d); for (const [key, col] of [['mu', AMBER], ['tau', VIOLET]]) { if (c[key] > 1e5 || !isFinite(c[key])) continue; g.fillStyle = col; g.beginPath(); g.arc(X(L.d), Y(c[key]), 4.5, 0, 2 * PI); g.fill(); }
}
function numbers() {
  const el = document.getElementById('lk-num'); if (!el) return;
  const c = copies(L.d), f = (x) => (x > 1e7 || !isFinite(x) ? '∞' : x.toFixed(4));
  const ppm = (x, y) => { const d = x / y - 1; return Math.abs(d) < 1e-3 ? `${d >= 0 ? "+" : ""}${(d * 1e6).toFixed(5)} ppm` : Math.abs(d) < 1 ? `${(d * 100).toFixed(2)} %` : `${(x / y).toPrecision(3)} times`; };
  el.innerHTML = `<tr><td>m<sub>μ</sub>/m<sub>e</sub></td><td class="n">${f(c.mu)}</td><td class="n" style="color:var(--muted)">measured ${MEAS.mu.toFixed(4)}</td><td class="n" style="color:var(--muted)">${isFinite(c.mu) ? ppm(c.mu, MEAS.mu) : ''}</td></tr>
<tr><td>m<sub>τ</sub>/m<sub>e</sub></td><td class="n">${f(c.tau)}</td><td class="n" style="color:var(--muted)">measured ${MEAS.tau.toFixed(2)}</td><td class="n" style="color:var(--muted)">${isFinite(c.tau) ? ppm(c.tau, MEAS.tau) : ''}</td></tr>
<tr><td>Koide Q</td><td class="n">${c.Q.toFixed(6)}</td><td class="n" style="color:var(--muted)">measured 0.666664</td><td></td></tr>`;
  const v = document.getElementById('lk-d-val'); if (v) v.textContent = `δ = ${L.d.toFixed(7)}`;
}
function draw() { ampDraw(); ratioDraw(); numbers(); }
function setD(d) { L.d = Math.max(0, Math.min(DMAX, d)); const s = document.getElementById('lk-d'); if (s) s.value = String(L.d); draw(); }
export function lkInit() {
  if (L.inited) return; L.inited = true;
  if (!document.getElementById('lk-style')) { const st = document.createElement('style'); st.id = 'lk-style'; st.textContent = STYLE; document.head.appendChild(st); }
  const host = document.querySelector('#page-story .st-wrap') || document.getElementById('page-story'); if (!host) return;
  const sec = document.createElement('section'); sec.id = 'lk-lepton'; sec.className = 'lk-ch'; sec.innerHTML = html(); host.appendChild(sec);
  document.getElementById('lk-d').oninput = (e) => setD(parseFloat(e.target.value));
  document.getElementById('lk-29').onclick = () => setD(D29); document.getElementById('lk-fit').onclick = () => setD(fitMu());
  document.getElementById('lk-node').onclick = () => setD(NODE); document.getElementById('lk-zero').onclick = () => setD(0);
  window.addEventListener('resize', () => { if (document.getElementById('page-story')?.classList.contains('on')) draw(); });
  draw();
}
export function lkShown() { if (L.inited) draw(); }
window.__lepton = { L, copies, fitMu, setD };
