// Candidate constructions compared with particle targets. A complete source-to-species map is not established.
// Colours retain the assumptions and calibration of each displayed step.
// Sources: repair-law/SPECTRUM.md (the coordinator's ledger) and the lane reports named in the rows.
import { $ } from './app.js';
import { familyReading, LEPTON_REFERENCE } from './lepton_reading.js';
import { mountQuantumEvidence } from './quantum_evidence.js';
import { mountCollisionEvidence } from './collision_evidence.js';
import { mountProcaEvidence } from './proca_evidence.js';
import { mountChargedField } from './charged_field_evidence.js';
import { mountDiracEvidence } from './dirac_evidence.js';
import { projectReferenceMap, referenceForRow, REFERENCE_SOURCES, RETAINED_OM, escapeReferenceText as escRef } from './particle_reference_adapter.js';

const ST = {
  S: { label: 'exact in the stated construction', color: '#5fe3a1' },
  C: { label: 'conditional on named items', color: '#ffb44a' },
  I: { label: 'taken from outside or chosen after the data', color: '#a995ff' },
  O: { label: 'open', color: '#ff6480' },
};
const CLS = {   // the obstruction classes: k = 6Y mod 6 of the flux on a face of a carrier
  1: { short: 'qL', color: '#ff7a45', shape: 'disc' }, 5: { short: 'q̄L', color: '#ff7a45', shape: 'ring' },
  4: { short: 'qR', color: '#ffb347', shape: 'disc' }, 2: { short: 'q̄R', color: '#ffb347', shape: 'ring' },
  3: { short: 'ℓL', color: '#3ee6a8', rim: '#4cc3ff', shape: 'duo' }, 0: { short: 'flat', color: '#9fb0cf', shape: 'flat' },
};
const ME = LEPTON_REFERENCE.electronGeV;          // measured calibration, GeV
const LEPTON_RATIOS = familyReading(2 / 9);
const LEPTON = [['candidate classes: 3 and 0', 'C', 'The assigned lepton doublet and singlet have these centre classes. A Z6 class alone does not identify a species, parity or generation.'],
  ['branch weights', 'S', 'For the declared preparation, agreement and oriented repairs weigh 1/6, 1/12, 1/12. All 360 local columns were checked against native Observer.repair (CODEX_JOIN_SPECTRUM).'],
  ['one-tick effect', 'S', 'The declared FR instrument has M=(√2/3)C(θ), a=1/√6, b=1/√12. Squared eigenvalues give Q=2/3 only in a chamber where the eigenvalues are nonnegative. The complete binary effects are M†M and I−M†M; mean success is 2/27.'],
  ['repeated clock', 'C', 'The literal second tick is <Φ|U²|p>=√3/9 I. A declared fresh-ancilla collision extension instead realizes A=M†M, with complete marginal-channel error ≤4t²/N. Its projector phase gate and schedule are supplied; it is not the original tick logarithm.'],
  ['phase 2/9', 'C', 'The independent flags have XOR probability 2/9 exactly. Turning that probability into a phase requires the declared Hamiltonian coupling and time unit; the probability does not supply one radian per read.'],
  ['three characters', 'C', 'Three characters belong to one neutral C³ register. Independently occupiable generations require the additional Fock(K⊗C³) construction, not Fock(K)⊗C³.'],
  ['native and supplied operations', 'C', 'The oriented record is an exact product of native optional basis-phase controls. The neutral family register, its controlled shift, joint analyzer and tail preparation are additional operations. The analyzer has Schmidt rank two; the listed separate-object operations cannot implement it without a joint controller (LNAT3).'],
  ['scale and source transfer', 'I', 'Displayed GeV values calibrate the overall scale to the measured electron mass. A same-H local scalar-memory theorem proves two shared charged spectral lines with complete residues; it does not establish relativistic rest masses or continuum FMS transfer.']];
const QUARK = (x) => [['candidate classes: 1 and 4', 'C', 'The chosen quark multiplets have these centre classes; the class does not specify the complete representation, parity or generation.'],
  ['junction magnitude law', 'O', `needs x = p/q = ${x}. The bare junction gives 1/2 on every leg (QM). With the seam record bit and the two-member Higgs register it gives 1/2 for leptons, 2/3 for down quarks and at most 2/3 for up quarks, and no mixing (QM2). Lead: a second Higgs doublet that the up quarks read, occupied about 59 times more than the one down quarks and leptons read`]];
const EW = [['candidate class 0 bosons', 'C', 'On the declared five-level protected code, independent continuous port-phase controls generate S(U(3)×U(2)). Writing these controls into a dynamical seam register is an extra architectural choice; the passive finite transport alphabet does not do this (GL).' ],
  ['declared cell-scale reading', 'C', 'E_cell = E* P^(−1/2) uses the declared capacity/energy identification (DC). The equal bare couplings and sin²θ_W = 3/8 belong to GL’s chosen trace-form action; covariance alone does not choose that action.'],
  ['slopes', 'C', 'the one-loop slopes (33/5, 1, −3) of the declared D10 chart: the source does not supply them (DB, EB, SC); below the cell the couplings never meet again (SC)'],
  ['M_U form, P_fwd', 'C', 'the M_U form and the pixel root P_fwd of the declared D10 chart (LEW, SC)'],
  ['transmutation law', 'C', 'v/E* is declared; every cell-local mechanism reads the cut coupling and gives v/E* = 6.5e-12 (SC)'],
  ['v through G_F', 'C', 'the zero-momentum Fermi record is the only one that depends on v alone (exact tree identity), so v attaches through G_F; derived from the book\'s identification of v (LW)'],
  ['radiative corrections', 'I', 'Standard Model loop corrections from SMDR, an outside calculation. The tree values of the unsliced source chart are W 80.33006 and Z 91.58834 GeV. The hypercharge slice of the older chart, set three days after a 0.46 GeV Z miss, is removed (LW)']];
const HADRON = (c) => [['confinement pattern', 'C', 'measured in the register\'s Gibbs ensemble (H4): colour confined with an area law, the weak centre not; it sits at the cell scale, not yet the GeV phase (HD, argued)'],
  ['colour coupling α3', 'C', 'Λ(3) = 0.340 GeV from the D10 colour coupling; the source\'s own cut coupling would put hadrons at 10⁴ to 10¹⁵ GeV (HD)'],
  ['lattice QCD number', 'I', `${c}: computed by lattice QCD (BMW 08 with ALPHA 17), not by the source (HD)`]];
const ROWS = [
  { fam: 'Charged leptons' },
  { sym: 'e', name: 'electron', cls: [3, 0], q: '−1', spin: '½', steps: LEPTON, meas: ME, model: ME, unit: true, vs: 'I', note: 'sets the unit of mass until the scale is derived' },
  { sym: 'μ', name: 'muon', cls: [3, 0], q: '−1', spin: '½', steps: LEPTON, vs: 'C', meas: 0.1056583755, model: LEPTON_RATIOS.mu * ME, note: 'Declared squared-circulant μ/e = 206.77031597; residual +9.83354 ppm against the shipped reference. The electron fixes the unit.' },
  { sym: 'τ', name: 'tau', cls: [3, 0], q: '−1', spin: '½', steps: LEPTON, vs: 'C', meas: LEPTON_REFERENCE.tauGeV, err: 0.00009, model: LEPTON_RATIOS.tau * ME, note: 'Declared squared-circulant τ/e = 3477.472837; the electron fixes the unit.' },
  { fam: 'Neutrinos' },
  { sym: 'ν', name: 'three neutrinos', cls: [3, 0], q: '0', spin: '½', steps: [['candidate classes 3 and 0', 'C', 'the chosen neutral lepton representations; centre class alone does not determine their full identity'],
    ['conserved source number', 'C', 'The declared operations conserve B − L and exclude a number-violating Majorana term in that sector. They do not alone identify a physical Dirac neutrino or its mass (DG).'], ['mixing', 'O', 'with the charged-lepton structure alone the mixing matrix is a permutation, all angles zero, excluded; mixing needs a second lepton-family structure that breaks the face third-turn and time reversal (PM)'], ['neutrino amplitude', 'O', 'The tested neutral composite has a scalar onsite block and conserved source number. This excludes the tested Majorana/spin-flip term; it does not prove massless dispersion to all orders. Propagating neutral poles and their mass need a separate same-H calculation (NU3 scope audit).']],
    range: [5e-11, 4.5e-10], note: 'Comparison guide only: oscillations require a mass splitting near 0.05 eV; the 0.45 eV endpoint is an effective beta-decay mass limit, not three measured individual masses.' },
  { fam: 'Quarks' },
  { sym: 'u', name: 'up', cls: [1, 4], q: '+2/3', spin: '½', steps: QUARK('0.832'), meas: 2.16e-3, rel: 0.07 / 2.16 },
  { sym: 'd', name: 'down', cls: [1, 4], q: '−1/3', spin: '½', steps: QUARK('0.622'), meas: 4.70e-3, rel: 0.07 / 4.70 },
  { sym: 's', name: 'strange', cls: [1, 4], q: '−1/3', spin: '½', steps: QUARK('0.622'), meas: 93.5e-3, rel: 0.8 / 93.5 },
  { sym: 'c', name: 'charm', cls: [1, 4], q: '+2/3', spin: '½', steps: QUARK('0.832'), meas: 1.2730, err: 0.0046 },
  { sym: 'b', name: 'bottom', cls: [1, 4], q: '−1/3', spin: '½', steps: QUARK('0.622'), meas: 4.183, err: 0.007 },
  { sym: 't', name: 'top', cls: [1, 4], q: '+2/3', spin: '½', steps: [QUARK('0.832')[0], ['electroweak chart', 'C', 'the declared D10 chart: slopes (33/5, 1, −3), M_U and transmutation forms, P_fwd; v through G_F (LW)'], ['double criticality', 'C', 'the boundary scale E* e^(−π) P^(−1/6) is declared and is the same item as R(P) (HT)'], ['radiative corrections', 'I', 'Standard Model loop corrections from SMDR, an outside calculation (LW)']], meas: 172.56, err: 0.31, model: 171.98782, note: 'PDG 2025 direct-reconstruction reference: 172.56 ± 0.31 GeV. This Monte Carlo mass parameter is not automatically the pole mass from the supplied loop calculation; a scheme-matching uncertainty is still required.' },
  { fam: 'Force carriers and the Higgs' },
  { sym: 'W', name: 'W boson', cls: [0], q: '±1', spin: '1', steps: EW, meas: 80.3692, err: 0.0133, model: 80.53911, note: 'the source chart without the target-informed hypercharge slice, with Standard Model radiative corrections (LW)' },
  { sym: 'Z', name: 'Z boson', cls: [0], q: '0', spin: '1', steps: EW, meas: 91.1880, err: 0.0020, model: 91.36191, note: 'W/Z = 0.88154 against 0.88128 (LW)' },
  { sym: 'H', name: 'Higgs', cls: [3], q: '0', spin: '0', steps: [['candidate class 3 scalar', 'C', 'class 3 also occurs for a lepton doublet; the Higgs assignment needs the bosonic parity and full scalar representation (DB2)'],
    ['electroweak chart', 'C', 'the declared D10 chart: slopes (33/5, 1, −3), M_U and transmutation forms, P_fwd; v through G_F (LW)'], ['double criticality', 'C', 'The double-criticality boundary and its scale are declared. The tested fixed scalar memories do not supply an exact nonzero scalar eigenbackground or exact canonical coordinates on their full finite space. This does not exclude controlled collective scalar limits. The quoted Higgs values additionally use an imported nonlinear-scalar identification and a declared running-coupling chart (HT).'], ['radiative corrections', 'I', 'Standard Model loop corrections from SMDR, an outside calculation (LW)']], meas: 125.20, err: 0.11, model: 125.84492 },
  { sym: 'γ g', name: 'photon, gluons', cls: [0], q: '0', spin: '1', steps: [['candidate adjoint fields', 'C', 'Class 0 contains both singlet and adjoint representations. Gauge-wave polarizations and a common cone require the full declared field action and state; the centre class alone does not identify a photon or gluon (GL).'],
    ['massless', 'C', 'Unbroken classical colour and electromagnetic modes have no mass term in the declared action. This does not identify the quantum vacuum or physical asymptotic spectrum.']], meas: 0, model: 0, note: 'Zero is the Standard Model gauge-field target, not an exact measured mass for a free gluon. Photons have experimental upper mass bounds; confined gluons are not isolated asymptotic particles.' },
  { fam: 'Mixing' },
  { sym: 'V', name: 'quark mixing', cls: [1, 4], q: '–', spin: '–', mix: '|V_us| 0.225 · J 3.1 × 10⁻⁵', steps: [['junction identities', 'S', 'the up and down junctions are the class vertices 1 + 3 = 4 with H and H̃ (MATCH, QM)'],
    ['CP phase', 'O', 'The declared two-seam junction produces nonzero CP violation. Its W3 reading gives |Vus| = 0.0799, |Vcb| = 0.0231, |Vub| = 0.0186 and |J| = 3.424 × 10⁻⁵. None of 30 tested readings or two fitted assignments passes all four comparisons. The small-breaking single-site formula fails the observed Vub/Vcb hierarchy. Junction weights, family base and one-tick reading are explicit inputs (CM2).']] },
  { sym: 'U', name: 'neutrino mixing', cls: [3, 0], q: '–', spin: '–', mix: 'sin²θ13 0.022 · sin²θ12 0.31 · sin²θ23 0.55', steps: [['conserved source number', 'C', 'B − L conservation constrains allowed mass operators; a physical neutrino identification and mass still require their dynamics (DG).'],
    ['mixing angles', 'O', 'with the charged-lepton structure alone the mixing matrix is a permutation, all angles zero; it needs a second lepton-family structure breaking the face third-turn and time reversal (PM)']] },
  { fam: 'Bound states' },
  { sym: 'p', name: 'proton', cls: [4, 4, 4], q: '+1', spin: '½', steps: HADRON('c_N = 2.745 ± 0.137'), meas: 0.938272, model: 0.93331, err: 0.0467, errModel: true, note: 'A baryon is three fermionic quark quanta; the carrier content {4, 4, 4} has the same centre classes but is a boson (SS). The lattice ratio reproduces experiment by construction; the source number tested here is the strong coupling α3(m_Z) = 0.11834 of the D10 chain, lattice 0.1183.' },
  { sym: 'n', name: 'neutron', cls: [4, 4, 4], q: '0', spin: '½', steps: HADRON('c_N = 2.745 ± 0.137'), meas: 0.939565, model: 0.93331, err: 0.0467, errModel: true, note: 'The lattice ratio reproduces experiment by construction; the source number tested here is the strong coupling α3(m_Z) = 0.11834 of the D10 chain, lattice 0.1183.' },
  { sym: 'ρ⁰', name: 'neutral rho meson', cls: [4, 2], q: '0', spin: '1', steps: HADRON('c_ρ = 2.273 ± 0.123'), meas: 0.77526, model: 0.77278, err: 0.0418, errModel: true, note: 'A quark and an antiquark quantum; the carrier content {2, 4} has the same centre classes but is a boson (SS). The lattice ratio reproduces experiment by construction; the source number tested here is the strong coupling α3(m_Z) = 0.11834 of the D10 chain, lattice 0.1183.' },
  { sym: 'π±', name: 'charged pion', cls: [4, 2], q: '±1', spin: '0', steps: [HADRON('')[0], ['condensate from Λ(3)', 'C', 'B = 7.87 Λ(3) from the D10 colour coupling (HD)'], ['quark masses m_ud', 'O', 'needs m_ud near 3.4 MeV at 2 GeV from the quark lane (HD, QM)']], meas: 0.13957039 },
  { sym: 'K±', name: 'charged kaon', cls: [4, 2], q: '±1', spin: '0', steps: [HADRON('')[0], ['condensate from Λ(3)', 'C', 'B = 7.87 Λ(3) (HD)'], ['quark masses m_s, m_ud', 'O', 'needs m_s near 88.5 MeV at 2 GeV (HD, QM)']], meas: 0.493677 },
];
const DERIVED = [
  ['Centre-class arithmetic', 'The declared register conserves class sums. Matching a class does not determine the full gauge representation, fermion parity or generation.'],
  ['Native branch weights', 'Under the stated active-seam preparation, agreement and oriented repairs have weights 1/6, 1/12 and 1/12.'],
  ['Complete charged quantum records', 'The actual interacting eight-cell source has positive charge-compensated energy bins, including its reference cost and every complementary outcome. See the source table below.'],
  ['Complete packet detector', 'The separately prepared Dirac packet has a complete finite Born instrument, both energy-band outcomes, energy accounting and reproducible record export.'],
  ['Controlled field comparisons', 'Complete native-block constructions give controlled free scalar, spin-one and spin-half record limits. Their actions and detectors are explicit; a common interacting chiral limit remains open.'],
];
const CONDITIONAL = [
  ['Continuous gauge links', 'The full code-current algebra is S(U(3)×U(2)). Continuous port-phase controls, protection and a dynamical link register are explicit architectural choices; the passive run does not supply them.', 'GL'],
  ['Equal bare couplings', 'The chosen five-state trace plaquette has one invariant quadratic coefficient, giving GUT-normalized equality and sin²θ_W = 3/8. This is an action-specific statement, not a selection by covariance alone.', 'GL'],
  ['Finite-range Dirac walk', 'The selected tick has one vectorlike cone under its stated momentum and mass scaling. A chiral interacting family, loop census and physical field vacuum require separate proofs.', 'T3'],
  ['Lepton one-tick balance', 'Q = 2/3 for the declared squared-circulant reading in its positive chamber. Literal two-tick compression instead gives Q = 1/3.', 'FR; CODEX_JOIN_SPECTRUM'],
  ['Lepton ratios 206.77, 3477.5', 'Declared phase 2/9 and squared-eigenvalue reading; μ/e remains +9.83354 ppm high. GeV scale calibrated to the electron.', 'CODEX_JOIN_SPECTRUM'],
  ['A constructive effect clock', 'Fresh ancillas and generator P=U†(|Φ><Φ|⊗I)U realize A=M†M with all outcomes retained; the family/reference error is at most min(2,4t²/N). These are additional declared operations.', 'CODEX_JOIN_SPECTRUM'],
  ['W, Z, v, α', 'the one-loop slopes and the transmutation law, both declared. The coupling sector now reduces to one number: the source action\'s 1/α3 at the cell must reach 18.24; the bare value is 16.25', 'SC, DC3, BF'],
  ['Higgs and top', 'the criticality scale', 'LEW'],
  ['Confinement pattern', 'centre-variable Wilson-loop behavior in the declared Gibbs ensemble; this does not establish confinement or the quantum vacuum of the full continuous gauge–matter source', 'H4'],
  ['Nucleon and rho', 'the colour coupling α3, with one number from lattice QCD per hadron', 'HD'],
];
const OPEN = [
  ['Lepton scale', 'the weight of one unrewarded read; the Z6-only determinant law is excluded by the quarks', 'LS2, TR'],
  ['Quarks', 'the junction magnitude law, x_u = 0.832 and x_d = 0.622, and a label beyond Z6', 'QM'],
  ['Neutrinos', 'a neutrino-specific Dirac amplitude or a B − L-violating singlet register', 'DG, NU3'],
  ['Pion, kaon, n − p', 'the light quark masses m_ud, m_s and m_d − m_u', 'QM'],
  ['Quark and neutrino mixing', 'A tested two-seam construction gives a nonzero quark CP phase but fails the mixing magnitudes. A successful quark mixing matrix and a second lepton-family structure are still missing.', 'CM2, PM'],
  ['Lepton residual and pole transfer', 'The tested exact local memory increases the heavy/light ratios. In DR2, retaining the transported load’s clock energy leaves ratios unchanged. Omitting it gives the required sign, but the tested couplings overshoot; 4.782628 × 10⁻⁷ is a weight fitted to the muon ratio, not a derived correction.', 'CODEX_JOIN_SPECTRUM; DR2'],
];

// ---------------------------------------------------------------- drawing helpers
function glyph(k, size = 22) {
  const c = CLS[k], h = size / 2;
  const body = c.shape === 'ring' ? `<circle cx="${h}" cy="${h}" r="${h * 0.62}" fill="none" stroke="${c.color}" stroke-width="${size * 0.15}"/>`
    : c.shape === 'duo' ? `<circle cx="${h}" cy="${h}" r="${h * 0.42}" fill="${c.color}"/><circle cx="${h}" cy="${h}" r="${h * 0.76}" fill="none" stroke="${c.rim}" stroke-width="${size * 0.1}"/>`
      : c.shape === 'flat' ? `<circle cx="${h}" cy="${h}" r="${h * 0.62}" fill="none" stroke="${c.color}" stroke-width="${size * 0.07}" stroke-dasharray="2.5 2.5"/>`
        : `<circle cx="${h}" cy="${h}" r="${h * 0.62}" fill="${c.color}"/>`;
  return `<svg width="${size}" height="${size}" class="sp-g" style="filter:drop-shadow(0 0 ${c.shape === 'flat' ? 0 : 5}px ${c.color})">${body}</svg>`;
}
const LMIN = -11, LMAX = 3.45;                    // log10 GeV range of the ladder
const W = 360, H = 30;
const xOf = (m) => 8 + (W - 16) * (Math.log10(m) - LMIN) / (LMAX - LMIN);
function axisSvg() {
  const names = { '-9': 'eV', '-6': 'keV', '-3': 'MeV', '0': 'GeV', '3': 'TeV' };
  let s = `<svg viewBox="0 0 ${W} 34" width="100%" height="34" preserveAspectRatio="none" class="sp-axis">`;
  for (let e = LMIN; e <= Math.floor(LMAX); e++) {
    const x = xOf(10 ** e), major = names[e] !== undefined;
    s += `<line x1="${x}" y1="${major ? 16 : 22}" x2="${x}" y2="30" stroke="${major ? '#5d6f94' : '#2a3a5c'}" stroke-width="1"/>`;
    if (major) s += `<text x="${x}" y="12" text-anchor="middle" fill="#93a3c0" font-size="10.5">1 ${names[e]}</text>`;
  }
  return s + '</svg>';
}
function ladderSvg(r) {
  const col = ST[valueStatus(r)].color;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" preserveAspectRatio="none">`;
  for (let e = LMIN; e <= Math.floor(LMAX); e++) s += `<line x1="${xOf(10 ** e)}" y1="4" x2="${xOf(10 ** e)}" y2="${H - 4}" stroke="#16223b" stroke-width="1"/>`;
  s += `<line x1="8" y1="${H / 2}" x2="${W - 8}" y2="${H / 2}" stroke="#1f2c48" stroke-width="1"/>`;
  if (r.meas === 0) {
    s += `<text x="12" y="${H / 2 + 4}" fill="#c5d0e6" font-size="11">0, massless gauge-field target and construction</text>`;
    return s + '</svg>';
  }
  if (r.mix) { s += `<text x="12" y="${H / 2 + 4}" fill="#c5d0e6" font-size="11">a mixing matrix, not a mass</text>`; return s + '</svg>'; }
  if (r.range) s += `<rect x="${xOf(r.range[0])}" y="${H / 2 - 4}" width="${xOf(r.range[1]) - xOf(r.range[0])}" height="8" rx="4" fill="#e9eefb" fill-opacity=".55"/>`;
  else if (r.meas) {
    const lo = r.rel ? r.meas * (1 - r.rel) : r.meas, hi = r.rel ? r.meas * (1 + r.rel) : r.meas;
    if (r.rel) s += `<rect x="${xOf(lo)}" y="${H / 2 - 3}" width="${Math.max(2, xOf(hi) - xOf(lo))}" height="6" rx="3" fill="#e9eefb" fill-opacity=".35"/>`;
    s += `<line x1="${xOf(r.meas)}" y1="${H / 2 - 9}" x2="${xOf(r.meas)}" y2="${H / 2 + 9}" stroke="#ffffff" stroke-width="2"/>`;
  }
  if (r.model !== undefined && r.model > 0) {
    const x = xOf(r.model);
    if (r.errModel) s += `<line x1="${xOf(r.model - r.err)}" y1="${H / 2}" x2="${xOf(r.model + r.err)}" y2="${H / 2}" stroke="${col}" stroke-width="3" stroke-linecap="round" stroke-opacity=".55"/>`;
    s += r.unit ? `<circle cx="${x}" cy="${H / 2}" r="6.5" fill="none" stroke="${col}" stroke-width="2.2"/>`
      : `<circle cx="${x}" cy="${H / 2}" r="5.5" fill="${col}" style="filter:drop-shadow(0 0 4px ${col})"/>`;
  } else if (r.meas) {
    s += `<circle cx="${xOf(r.range ? r.range[0] : r.meas)}" cy="${H / 2}" r="5" fill="none" stroke="${ST.O.color}" stroke-width="1.6" stroke-dasharray="2 2"/>`;
  }
  return s + '</svg>';
}
const rank = { S: 0, C: 1, I: 2, O: 3 };
function worst(steps) { return steps.reduce((a, s) => (rank[s[1]] > rank[a] ? s[1] : a), 'S'); }
// standing of the value shown: the lepton ratios do not use the scale step; a row without a value is open
const valueStatus = (r) => (r.model === undefined ? 'O' : r.vs || worst(r.steps));
function fmtGeV(m) {
  if (m === 0) return '0';
  if (m >= 1) return m.toFixed(m >= 100 ? 2 : 3) + ' GeV';
  if (m >= 1e-3) return (m * 1e3).toFixed(m >= 0.1 ? 2 : 3) + ' MeV';
  if (m >= 1e-6) return (m * 1e6).toFixed(1) + ' keV';
  return (m * 1e9).toFixed(2) + ' eV';
}
function devText(r) {
  if (r.model === undefined) return `<span class="sp-open">no value yet</span>`;
  if (r.unit) return `<span class="sp-dim">unit</span>`;
  if (r.meas === 0) return `<span class="sp-dim">massless symmetry target</span>`;
  const d = (r.model - r.meas) / r.meas, ppm = Math.abs(d) < 1e-3;
  const txt = ppm ? `${d >= 0 ? '+' : '−'}${Math.abs(d * 1e6).toFixed(1)} ppm` : `${d >= 0 ? '+' : '−'}${Math.abs(d * 100).toFixed(2)} %`;
  const sig = r.errModel ? ` · lattice input uncertainty ±${(100 * r.err / r.model).toFixed(0)} %` : r.err ? ` · ${((r.model - r.meas) / r.err >= 0 ? '+' : '−')}${Math.abs((r.model - r.meas) / r.err).toFixed(1)} experimental σ only` : '';
  return `<b>${txt}</b><span class="sp-dim">${sig}</span>`;
}

// ---------------------------------------------------------------- page
let built = false;
export function spectrumShown() {
  if (built) return; built = true;
  const legend = Object.entries(ST).map(([k, s]) => `<span class="sp-key"><i style="background:${s.color};box-shadow:0 0 10px ${s.color}"></i>${s.label}</span>`).join('');
  const flow = [['Obstruction', 'a protected flux on a face of a carrier, class k of Z6'], ['Candidate field', 'centre class plus representation, parity and an independent generation construction'],
    ['Mass mechanism', 'rest angle, lepton circulant, quark junction or electroweak chain'], ['Readout', 'specified operator, preparation and clock; GeV conversion records calibration']];
  const card = (title, color, rows, lane) => `<div class="sp-col" style="--c:${color}"><div class="sp-colh">${title}</div>${rows.map((r) =>
    `<div class="sp-item"><b>${r[0]}</b><span>${r[1]}</span>${lane && r[2] ? `<em>${r[2]}</em>` : ''}</div>`).join('')}</div>`;
  let rowsHtml = '';
  for (const r of ROWS) {
    if (r.fam) { rowsHtml += `<div class="sp-fam">${r.fam}</div>`; continue; }
    const st = valueStatus(r);
    rowsHtml += `<div class="sp-row" style="--c:${ST[st].color}">
      <div class="sp-obs">${r.cls.map((k) => glyph(k)).join('')}<span>${r.cls.map((k) => CLS[k].short).join(' · ')}</span></div>
      <div class="sp-name"><b>${r.sym}</b><span>${r.name}</span><em>charge ${r.q} · spin ${r.spin}</em></div>
      <div class="sp-chain">${r.steps.map((s) => `<span class="sp-step" style="--s:${ST[s[1]].color}" title="${s[2].replace(/"/g, '&quot;')}">${s[0]}</span>`).join('<i class="sp-link"></i>')}</div>
      <div class="sp-lad">${ladderSvg(r)}</div>
      <div class="sp-num"><span class="sp-dim">model</span> ${r.model !== undefined ? fmtGeV(r.model) + (r.errModel ? ` ± ${Math.round(r.err * 1000)} MeV` : '') : '–'}<br><span class="sp-dim">${r.meas===0?'SM target':'measured'}</span> ${r.mix ? r.mix : r.range ? 'Δm scale ≈ 0.05 eV; mβ < 0.45 eV' : fmtGeV(r.meas)}</div>
      <div class="sp-dev">${devText(r)}</div>
      ${r.note ? `<div class="sp-note">${r.note}</div>` : ''}
      <div class="sp-why">${r.steps.map((s) => `<div><i style="background:${ST[s[1]].color}"></i><b>${s[0]}</b> ${s[2]}</div>`).join('')}</div></div>`;
  }
  $('sp-host').innerHTML = `
    <div class="sp-head"><h2>From obstruction to mass</h2>
      <p>Each row compares a candidate construction with an observed particle target. Centre classes, one-tick effects, Hamiltonian energy lines and calibrated GeV values are distinct steps. Click a row for its exact hypotheses and unresolved transfers. <a href="/rl/CODEX_JOIN_SPECTRUM/family_clock.json">Family instrument and clock evidence</a> · <a href="/rl/CODEX_JOIN_SPECTRUM/quantum_memory.json">same-H quantum memory evidence</a>.</p>
      <div class="sp-keys">${legend}<span class="sp-key"><i class="sp-tick"></i>measured</span><span class="sp-key"><i class="sp-ring"></i>unit</span></div></div>
    <div class="sp-flow">${flow.map((f, i) => `<div class="sp-fstep"><b>${i + 1} · ${f[0]}</b><span>${f[1]}</span></div>`).join('<div class="sp-farrow"></div>')}</div>
    <div class="sp-cols">${card('Exact results in their construction', ST.S.color, DERIVED)}${card('Conditional on named items', ST.C.color, CONDITIONAL, true)}${card('Required physical connections', ST.O.color, OPEN, true)}</div>
    <div class="sp-table">
      <div class="sp-row sp-hrow"><div>Centre class</div><div>Particle target</div><div>From obstruction to mass</div><div class="sp-lad">${axisSvg()}</div><div>Mass</div><div>Model − measured</div></div>
      ${rowsHtml}</div>
    <section id="sp-quantum" aria-label="Verified interacting source energies"></section>
    <section id="sp-charged-field" aria-label="Finite-clock charge position and field observations"></section>
    <section id="sp-dirac" aria-label="Native charged free Dirac field and complete detector records"></section>
    <section id="sp-collision" aria-label="Finite quantum collision experiment"></section>
    <section id="sp-proca" aria-label="Controlled massive-vector field readout"></section>
    <div id="sp-om"></div>
    <p class="note" style="max-width:1100px">Model values: lepton ratios at the phase 2/9 times the measured electron mass; W, Z, Higgs and top from the source chart with Standard Model radiative corrections and v fixed by G_F, Higgs and top at the declared criticality scale; proton, neutron and rho from Λ(3) of the D10 colour coupling times one lattice QCD number each. Measured values: shipped particle references (electron/muon/tau values as in the audited family_clock.json); light quark masses in the MS-bar scheme at 2 GeV, charm and bottom at their own scale. The u, d, s, c and b reference intervals are 90% confidence limits, not one-sigma errors. Charged pion/kaon and neutral rho masses refer to those specific charge states. References: <a href="https://pdg.lbl.gov/2025/tables/rpp2025-sum-quarks.pdf" target="_blank" rel="noopener">PDG 2025 quarks</a>, <a href="https://pdg.lbl.gov/2025/tables/rpp2025-sum-gauge-higgs-bosons.pdf" target="_blank" rel="noopener">gauge/Higgs targets</a>, and <a href="https://arxiv.org/abs/2406.13516" target="_blank" rel="noopener">KATRIN effective-mass limit</a>. Experimental σ labels on the other rows divide by the reference measurement error only; they are not combined statistical tests of these conditional constructions. The supplied top loop mass also requires matching to the direct-reconstruction mass convention. Ledger: repair-law/SPECTRUM.md; exact hypotheses in each row.</p>`;
  document.querySelectorAll('#sp-host .sp-row:not(.sp-hrow)').forEach((el) => { el.onclick = () => el.classList.toggle('open'); });
  mountQuantumEvidence($('sp-quantum'));
  mountChargedField($('sp-charged-field'));
  mountDiracEvidence($('sp-dirac'));
  mountCollisionEvidence($('sp-collision'));
  mountProcaEvidence($('sp-proca'));
  buildMap();
}

// ---------------------------------------------------------------- every state: the coordinator's obstruction map
const OMS = { SOURCE: '#5fe3a1', CONDITIONAL: '#ffb44a', FIT: '#a995ff', OPEN: '#ff6480', EXCLUDED: '#ff6480' };
const GEN = { u: ['u', 'c', 't'], d: ['d', 's', 'b'], e: ['e', 'μ', 'τ'], nu: ['νe', 'νμ', 'ντ'] };
const FLAV = { u: 'Up-type quarks', d: 'Down-type quarks', e: 'Charged leptons', nu: 'Neutrinos' };
let OM = null, OM_REFERENCES = null, OM_SOURCE_HASH = null;
function omStyle() {
  if (document.getElementById('sp-om-style')) return;
  const st = document.createElement('style'); st.id = 'sp-om-style';
  st.textContent = `
#sp-om{margin:26px 0 10px}#sp-om h2{margin:0;font-size:20px}#sp-om>p{margin:4px 0 12px;color:var(--soft);max-width:980px;font-size:13.5px}
.om-blocks{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:14px}
.om-block{min-width:0;border:1px solid var(--edge);border-radius:14px;padding:10px 12px;background:rgba(8,12,22,.6)}
.om-bh{font-size:11.5px;letter-spacing:2px;text-transform:uppercase;color:var(--teal);font-weight:700;margin-bottom:6px}
.om-grid{display:grid;grid-template-columns:54px repeat(3,minmax(0,1fr));gap:6px;align-items:stretch}
.om-gh{font-size:10.5px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;text-align:center}
.om-rh{font-size:11px;color:var(--muted);display:flex;align-items:center}
.om-cell{min-width:0;overflow-wrap:anywhere;border:1px solid color-mix(in srgb,var(--sc) 55%,transparent);border-radius:10px;padding:6px 7px;background:color-mix(in srgb,var(--sc) 7%,#0b1222);cursor:pointer;min-height:58px;position:relative}
.om-cell:hover{box-shadow:0 0 18px -6px var(--sc)}
.om-top{display:flex;align-items:center;gap:5px}.om-top b{font-size:15px}
.om-m{font-size:11px;color:var(--soft);margin-top:3px;font-variant-numeric:tabular-nums}.om-m em{font-style:normal;color:var(--faint)}
.om-st{position:absolute;right:6px;top:6px;width:8px;height:8px;border-radius:50%;background:var(--sc);box-shadow:0 0 8px var(--sc)}
.om-flat{display:flex;flex-wrap:wrap;gap:6px}.om-flat .om-cell{width:168px}
.om-panel{position:fixed;right:14px;top:70px;bottom:14px;width:440px;max-width:calc(100% - 28px);overflow:auto;z-index:20;background:rgba(9,14,26,.97);border:1px solid var(--edge);border-radius:16px;padding:14px 16px;box-shadow:0 30px 80px -30px #000;display:none}
.om-panel{box-sizing:border-box;overflow-wrap:anywhere}.om-panel h3{margin:0 0 2px;font-size:20px}.om-panel .k{margin-top:12px}
.om-v{border-top:1px solid #16213a;padding:6px 0;font-size:12.5px;color:var(--soft)}.om-v b{color:var(--text)}
.om-chip{display:inline-block;font-size:10.5px;font-weight:800;letter-spacing:.8px;padding:1px 7px;border-radius:999px;color:#0b1222;background:var(--sc);margin-right:6px}
.om-sum{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--soft);margin:0 0 12px}`;
  document.head.appendChild(st);
}
function omGlyph(r, size = 18) {
  const k = r.defect_class, anti = r.particle === 'antiparticle';
  if (k === null || k === undefined) return '';
  const base = CLS[k]; if (!base) return '';
  if (k === 0) return glyph(0, size);
  return glyph(k, size);
}
const UNIT = { GeV: 1, MeV: 1e-3, keV: 1e-6, eV: 1e-9 };
function bestValue(r) {                 // a model value only if it is SOURCE or CONDITIONAL; fits and excluded values stay in the panel
  const vals = ((r.mass && r.mass.values) || []).filter((v) => (v.status === 'SOURCE' || v.status === 'CONDITIONAL'));
  const mass = vals.find((v) => v.kind === 'mass' && typeof v.value === 'number' && UNIT[v.unit] !== undefined);
  if (mass) return { text: fmtGeV(mass.value * UNIT[mass.unit]), status: mass.status };
  const ratio = vals.find((v) => v.kind === 'mass_ratio');
  if (ratio) return { text: String(ratio.display || ratio.value).slice(0, 26), status: ratio.status };
  return null;
}
function omCell(r, label) {
  if (!r) return '<div></div>';
  const col = OMS[r.status] || '#93a3c0', bv = bestValue(r), view = referenceForRow(r);
  return `<div class="om-cell" style="--sc:${col}" data-id="${escRef(r.id)}" title="${escRef(view.basis)}"><span class="om-st"></span><div class="om-top">${omGlyph(r)}<b>${escRef(label)}</b></div>
    <div class="om-m"><em>${escRef(view.cardLabel)}</em> ${escRef(view.reference.short)}</div>
    <div class="om-m">${bv ? `<em>${view.category === 'higgs_multiplet' ? 'attached h' : 'model'}</em> <span style="color:${OMS[bv.status]}">${escRef(bv.text)}</span>` : `<em>model</em> <span style="color:${col}">${escRef(r.status.toLowerCase())}</span>`}</div></div>`;
}
function omPanel(id) {
  const r = OM.rows.find((x) => x.id === id), el = document.getElementById('om-panel'); if (!r) return;
  const view = referenceForRow(r), ref = view.reference;
  const col = OMS[r.status] || '#93a3c0', c = r.charges || {}, ss = r.spin_statistics || {}, gc = r.generation_character || {};
  const vals = ((r.mass && r.mass.values) || []).map((v) => `<div class="om-v"><span class="om-chip" style="--sc:${OMS[v.status] || '#93a3c0'}">${escRef(v.status)}</span><b>${escRef(v.quantity)}</b>: ${escRef(v.display ?? v.value ?? '')}${v.comparison ? `<div class="small" style="margin-top:2px">Retained lane comparison (original inputs): ${escRef(v.comparison)}</div>` : ''}${v.report ? `<div class="small">${escRef(v.report)}${v.where ? ', ' + escRef(v.where) : ''}</div>` : ''}</div>`).join('');
  const sources = ref.sources.map((key) => { const s = REFERENCE_SOURCES[key]; return `<div><a href="${s.url}" target="_blank" rel="noopener">${escRef(s.title)}</a> · ${escRef(s.where)}</div><details><summary>Source SHA-256</summary>${s.sha256}</details>`; }).join('');
  el.innerHTML = `<button class="ro-x" id="om-x">×</button>
    <h3>${omGlyph(r, 22)} ${r.symbol} <span style="font-size:13px;color:var(--soft);font-weight:600">${r.name}</span></h3>
    <div class="small">${escRef(view.basis)}${view.sharedSpeciesReference ? ' · L/R components and their conjugates share one species mass reference.' : ''}</div>
    <div><span class="om-chip" style="--sc:${col}">${r.status}</span><span class="small">${(r.status_items || []).join(', ')}</span></div>
    <div class="k">Comparison reference · fixed PDG 2025 edition</div><div class="small" style="color:var(--soft)"><b>${escRef(ref.display)}</b><br>${escRef(ref.convention)}${ref.uncertainty ? '<br>Uncertainty: ' + escRef(ref.uncertainty) + '.' : ''}<br>${escRef(view.scoreReason)}</div><div class="small">${sources}</div>
    <div class="k">Candidate obstruction assignment</div><div class="small" style="color:var(--soft)">defect class ${r.defect_class ?? '–'} (conjugate ${r.conjugate_class ?? '–'}); colour ${r.colour}, weak ${r.weak}; ${r.run ? r.run.run : ''}${r.run && r.run.mean_free_path_H2 ? '; mean free path ' + r.run.mean_free_path_H2 : ''}. A centre class alone does not identify a full gauge representation or an observed species.</div>
    <div class="k">Candidate charges in the retained map</div><div class="small" style="color:var(--soft)">Q ${c.Q ?? '–'}, T3 ${c.T3 ?? '–'}, Y ${c.Y ?? '–'}, B − L ${c.B_minus_L ?? '–'} · ${c.status || ''}</div>
    <div class="k">Spin, generation, form</div><div class="small" style="color:var(--soft)">spin ${ss.spin ?? '–'}, ${ss.statistics ?? ''} (${ss.status ?? ''}); register character ${gc.display ?? '–'} (${gc.status ?? ''}${gc.lanes ? ', ' + gc.lanes.join(', ') : ''}); ${r.composite_form || ''}. Three neutral-register sectors do not by themselves create three independently occupiable generations.</div>
    <div class="k">Retained model values and lane statuses</div><div class="small" style="color:var(--soft)">${r.mass ? r.mass.mechanism_text : ''} These quoted values and comparisons have not been refitted or rescored against this catalogue.</div>${vals}
    <details><summary>Original coordinator reference (retained)</summary><div class="small">${r.measured ? `${escRef(r.measured.display)} · ${escRef(r.measured.source || '')} · ${escRef(r.measured.label || '')}` : '–'}<br>This is the original input attribution, not a verification of its edition or species identification.</div></details>
    <div class="k">Lanes and reports</div><div class="small">${(r.lanes_running || []).join(', ')} · ${(r.reports || []).join(', ')}</div>
    <details><summary>Map and reference provenance</summary><div class="small"><a href="${RETAINED_OM.path}" target="_blank" rel="noopener">Loaded coordinator JSON</a><br>SHA-256: ${OM_SOURCE_HASH || 'unavailable in this browser'}<br><a href="./particle_reference_catalog.json" target="_blank" rel="noopener">Reference catalogue, conventions and source hashes</a><br>${OM_SOURCE_HASH === RETAINED_OM.sha256 ? 'Matches the 69-row reviewed input.' : 'This loaded input differs from the reviewed 69-row input; the adapter does not validate its scientific claims.'}</div></details>`;
  el.style.display = 'block'; document.getElementById('om-x').onclick = () => { el.style.display = 'none'; };
}
async function buildMap() {
  omStyle();
  const host = document.getElementById('sp-om');
  try {
    const response = await fetch(RETAINED_OM.path, { cache: 'no-store' });
    if (!response.ok) throw new Error('Coordinator map unavailable');
    const raw = await response.text(); OM = JSON.parse(raw); OM_REFERENCES = projectReferenceMap(OM);
    if (globalThis.crypto?.subtle) OM_SOURCE_HASH = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(raw))), (b) => b.toString(16).padStart(2, '0')).join('');
  } catch (e) { host.innerHTML = '<p>Candidate map unavailable: its source or row structure could not be read.</p>'; return; }
  const by = new Map(OM.rows.map((r) => [r.id, r]));
  const sum = OM.summary || {}, rs = sum.rows_by_status || {};
  let html = `<h2>Candidate fields and particle targets</h2><p>${OM_REFERENCES.sourceRows} coordinator rows share ${OM_REFERENCES.distinctMassReferenceQuantities} numeric mass references. Left/right field components and their conjugates are not independent measured particles. The ν_R rows are extension candidates beyond the minimal Standard Model; H and H̃ are field multiplets, with the observed scalar listed as h. Reference values use a fixed PDG 2025 edition, with CODATA 2022 constants. Click a row for its convention, sources, and the unchanged model values.</p><p>Generation columns are a comparison dictionary; three neutral-register characters do not supply independently occupiable generations. Dots retain the coordinator’s mass status. Centre-class assignments do not establish full particle identities.</p>
    <div class="om-sum">${Object.entries(rs).map(([k, n]) => `<span><span class="om-chip" style="--sc:${OMS[k] || '#93a3c0'}">${k}</span>${n} rows</span>`).join('')}<span class="small">built ${escRef(OM.built_utc || '')} by ${escRef(OM.builder || '')}</span><a href="./particle_reference_catalog.json" target="_blank" rel="noopener">Reference sources and conventions</a></div><div class="om-blocks">`;
  for (const f of ['u', 'd', 'e', 'nu']) {
    html += `<div class="om-block"><div class="om-bh">${FLAV[f]}</div><div class="om-grid"><div></div>${[1, 2, 3].map((g) => `<div class="om-gh">generation ${g}</div>`).join('')}`;
    for (const [rid, lab] of [['L', 'left'], ['R', 'right'], ['aL', 'conj. L<br>(right)'], ['aR', 'conj. R<br>(left)']]) {
      html += `<div class="om-rh">${lab}</div>`;
      for (let g = 1; g <= 3; g++) {
        const id = (rid[0] === 'a' ? 'anti_' : '') + `${f}_${rid.slice(-1)}_g${g}`, nm = GEN[f][g - 1];
        html += omCell(by.get(id), (rid[0] === 'a' ? 'anti ' : '') + nm);
      }
    }
    html += '</div></div>';
  }
  html += '</div>';
  const flat = (title, ids) => `<div class="om-block" style="margin-top:14px"><div class="om-bh">${title}</div><div class="om-flat">${ids.map((id) => omCell(by.get(id), by.get(id) ? by.get(id).symbol : id)).join('')}</div></div>`;
  html += flat('Gauge fields and Higgs representations', ['H', 'H_c', 'h', 'gluon', 'photon', 'Z', 'W_plus', 'W_minus']);
  html += flat('Composites of the run', OM.rows.filter((r) => r.id.startsWith('C_')).map((r) => r.id));
  html += flat('Hadrons', ['proton', 'neutron', 'pi0', 'rho0']);
  host.innerHTML = html + '<div class="om-panel" id="om-panel"></div>';
  host.querySelectorAll('.om-cell').forEach((el) => { el.onclick = () => omPanel(el.dataset.id); });
}
