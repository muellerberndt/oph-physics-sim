// Pure reference projection: coordinator rows, values, statuses and comparisons are never edited.
// This fixed PDG edition is a comparison catalogue, not a new source prediction.
function freeze(value) {
  if (value && typeof value === 'object') { Object.values(value).forEach(freeze); Object.freeze(value); }
  return value;
}
export const REFERENCE_SOURCES = freeze({
  "bosons": {
    "bytes": 83221,
    "edition": "PDG 2025 (fixed edition)",
    "sha256": "4a1b1afd334e5cda6134d526e43f4965d95d978511a2cd2efdc8365e57633a86",
    "title": "PDG 2025 gauge and Higgs summary",
    "url": "https://pdg.lbl.gov/2025/tables/rpp2025-sum-gauge-higgs-bosons.pdf",
    "where": "pp.1,2,4; notes on p.8"
  },
  "constants": {
    "bytes": 249001,
    "edition": "PDG 2025 (fixed edition)",
    "sha256": "596a9ed34c582e92a13c1fff7263a498844ea7e301511417587ecfb191516e6f",
    "title": "PDG 2025 physical constants, CODATA 2022",
    "url": "https://pdg.lbl.gov/2025/reviews/rpp2025-rev-phys-constants.pdf",
    "where": "Table 1.1"
  },
  "leptons": {
    "bytes": 87086,
    "edition": "PDG 2025 (fixed edition)",
    "sha256": "5aa9a3a33b554204056cb04c42319bb32c9664ae518907652ff5cdc1cb87e5bb",
    "title": "PDG 2025 lepton summary",
    "url": "https://pdg.lbl.gov/2025/tables/rpp2025-sum-leptons.pdf",
    "where": "pp.1,2"
  },
  "mesons": {
    "bytes": 150378,
    "edition": "PDG 2025 (fixed edition)",
    "sha256": "1fb5c743046459fafffd0f317fe2504e9b68e8da18d5b27bc0bf171dcd747606",
    "title": "PDG 2025 light unflavored mesons",
    "url": "https://pdg.lbl.gov/2025/tables/rpp2025-tab-mesons-light.pdf",
    "where": "pp.1,2,4"
  },
  "neutrinos": {
    "bytes": 1929704,
    "edition": "PDG 2025 (fixed edition)",
    "sha256": "d7067e2e3c9098cc924f10ffbca579c557fb8e848bf3acc17f9815598cdda7a6",
    "title": "PDG 2025 neutrino masses, mixing and oscillations",
    "url": "https://pdg.lbl.gov/2025/reviews/rpp2025-rev-neutrino-mixing.pdf",
    "where": "Sections 14.1–14.3"
  },
  "quarks": {
    "bytes": 49062,
    "edition": "PDG 2025 (fixed edition)",
    "sha256": "586adace46689cb8580ce068cbab242397b7f313256052bde8c69dcff65de5d1",
    "title": "PDG 2025 quark summary",
    "url": "https://pdg.lbl.gov/2025/tables/rpp2025-sum-quarks.pdf",
    "where": "p.1"
  }
});
export const RETAINED_OM = freeze({
  path: '/rl/OM/obstruction_map.json',
  shippedPath: './research/OM/obstruction_map.json',
  sha256: '28f25cf8ba2e09021ae27fe16080259cdf4154f358ec6a77cd2a19678ea60942',
  rows: 69,
});
const value = (key, short, display, valueGeV, errorGeV, source, convention, uncertainty = 'quoted uncertainty') =>
  ({ key, kind: 'mass_reference', short, display, valueGeV, errorGeV,
     sources: [source], convention, uncertainty });
export const MASS_REFERENCES = freeze({
  e: value('e', '0.511 MeV', '0.51099895069 ± 0.00000000016 MeV', .00051099895069, 1.6e-13,
    'constants', 'CODATA 2022 electron rest-mass reference, as tabulated in the PDG 2025 constants table.', 'one standard deviation'),
  mu: value('mu', '105.658 MeV', '105.6583755 ± 0.0000023 MeV', .1056583755, 2.3e-9,
    'leptons', 'Charged-muon mass reference; shared by the L/R components.'),
  tau: value('tau', '1776.93 MeV', '1776.93 ± 0.09 MeV', 1.77693, .00009,
    'leptons', 'Rounded PDG 2025 tau mass. The retained family calculation keeps its original full-precision API reference.'),
  u: value('u', '2.16 MeV', '2.16 ± 0.07 MeV', .00216, .00007,
    'quarks', 'MS-bar running quark mass at μ = 2 GeV; no isolated free-quark mass measurement.', '90% confidence-level interval, not a one-sigma error'),
  d: value('d', '4.70 MeV', '4.70 ± 0.07 MeV', .00470, .00007,
    'quarks', 'MS-bar running quark mass at μ = 2 GeV; no isolated free-quark mass measurement.', '90% confidence-level interval, not a one-sigma error'),
  s: value('s', '93.5 MeV', '93.5 ± 0.8 MeV', .0935, .0008,
    'quarks', 'MS-bar running quark mass at μ = 2 GeV; no isolated free-quark mass measurement.', '90% confidence-level interval, not a one-sigma error'),
  c: value('c', '1.2730 GeV', '1.2730 ± 0.0046 GeV', 1.2730, .0046,
    'quarks', 'MS-bar running charm mass at μ = m_c.', '90% confidence-level interval, not a one-sigma error'),
  b: value('b', '4.183 GeV', '4.183 ± 0.007 GeV', 4.183, .007,
    'quarks', 'MS-bar running bottom mass at μ = m_b.', '90% confidence-level interval, not a one-sigma error'),
  t: value('t', '172.56 GeV', '172.56 ± 0.31 GeV', 172.56, .31,
    'quarks', 'Direct-reconstruction top mass. Its Monte Carlo mass convention requires a matching uncertainty before comparison with a calculated pole or running mass.'),
  h: value('h', '125.20 GeV', '125.20 ± 0.11 GeV', 125.20, .11,
    'bosons', 'Observed Higgs scalar resonance; H and its conjugate are field-multiplet entries, not two additional measured scalars.'),
  W: value('W', '80.3692 GeV', '80.3692 ± 0.0133 GeV', 80.3692, .0133,
    'bosons', 'PDG 2025 W mass average; this quoted average excludes the CDF 2022 result. Charge-conjugate rows share this reference.'),
  Z: value('Z', '91.1880 GeV', '91.1880 ± 0.0020 GeV', 91.1880, .0020,
    'bosons', 'Breit–Wigner resonance parameter. PDG notes an approximately 34 MeV distinction from the real part of the propagator pole energy.'),
  proton: value('proton', '938.272 MeV', '938.27208943 ± 0.00000029 MeV', .93827208943, 2.9e-10,
    'constants', 'CODATA 2022 proton rest-mass reference in the PDG 2025 constants table.', 'one standard deviation'),
  neutron: value('neutron', '939.565 MeV', '939.56542194 ± 0.00000048 MeV', .93956542194, 4.8e-10,
    'constants', 'CODATA 2022 neutron rest-mass reference in the PDG 2025 constants table.', 'one standard deviation'),
  pi0: value('pi0', '134.9768 MeV', '134.9768 ± 0.0005 MeV', .1349768, .0000005,
    'mesons', 'Neutral-pion mass; it is distinct from the charged-pion reference.'),
  rho0: value('rho0', '775.26 MeV', '775.26 ± 0.23 MeV', .77526, .00023,
    'mesons', 'Neutral ρ(770) Breit–Wigner mass; a resonance parameter, not a stable-state eigenenergy or the complex pole mass.'),
});
const none = (kind, short, convention, sources = []) =>
  freeze({ key: null, kind, short, display: short, valueGeV: null, errorGeV: null, convention, sources });
const ACTIVE_NEUTRINO = none('no_flavour_mass', 'not measured',
  'These labels specify weak-flavour fields. Oscillations measure mass-squared differences and mixing; the rows are not separate measured flavour masses.', ['neutrinos']);
const SINGLET_NEUTRINO = none('extension_candidate', 'not established',
  'An independent right-handed neutrino is beyond the minimal SM. The listed H̃†L scalar-times-spinor composite retains L’s Lorentz chirality; naming it ν_R requires an additional chirality construction.', ['neutrinos']);
const HIGGS_MULTIPLET = none('field_multiplet', 'see physical h',
  'H and H̃ are conjugate descriptions of the scalar doublet, not extra observed Higgs resonances. The retained mass values refer to an attached h candidate.', ['bosons']);
const GLUON = freeze({ ...none('gauge_mass_term_target', '0 (theory)',
  'Zero is the unbroken-colour gauge-field mass-term target. PDG explicitly marks it theoretical; it is not a measured mass of an isolated free gluon.', ['bosons']), massTermTargetGeV: 0 });
const PHOTON = freeze({ ...none('upper_limit', '< 10⁻¹⁸ eV',
  'PDG 2025 photon-mass upper limit. A bound is not an exact measured zero; the declared unbroken electromagnetic action has a zero mass-term target.', ['bosons']),
  key: 'photon_limit', upperLimitGeV: 1e-27, massTermTargetGeV: 0 });
const COMPOSITE = none('unidentified_register_composite', 'unidentified',
  'A register-composite label has no measured particle mass until a source-to-species identification is established.');
const UNMAPPED = none('unmapped', 'unmapped', 'This row has no reviewed reference mapping.');
const FAMILIES = freeze({ u: ['u', 'c', 't'], d: ['d', 's', 'b'], e: ['e', 'mu', 'tau'] });

export function referenceForRow(row) {
  if (!row || typeof row.id !== 'string') throw new TypeError('A typed coordinator row is required');
  const match = /^(anti_)?(u|d|e|nu)_(L|R)_g([123])$/.exec(row.id);
  let reference = UNMAPPED, category = 'unmapped', basis = 'unmapped candidate', cardLabel = 'ref.';
  if (match) {
    const anti = !!match[1], family = match[2], sourceHand = match[3], generation = Number(match[4]);
    const hand = anti ? (sourceHand === 'L' ? 'R' : 'L') : sourceHand;
    if (row.handedness !== undefined && row.handedness !== hand) throw new Error('Coordinator chirality contradicts its conjugate-field ID');
    basis = anti ? `conjugate of ${sourceHand} (${hand} component)` : `${hand} field component`;
    if (family === 'nu') {
      reference = sourceHand === 'L' ? ACTIVE_NEUTRINO : SINGLET_NEUTRINO;
      category = sourceHand === 'L' ? 'active_neutrino_component' : 'singlet_neutrino_candidate';
      cardLabel = sourceHand === 'L' ? 'mass' : 'SM extension';
    } else {
      reference = MASS_REFERENCES[FAMILIES[family][generation - 1]];
      category = 'charged_fermion_component';
    }
  } else if (row.id === 'H' || row.id === 'H_c') {
    reference = HIGGS_MULTIPLET; category = 'higgs_multiplet'; basis = 'field multiplet'; cardLabel = 'field';
  } else if (row.id === 'h') {
    reference = MASS_REFERENCES.h; category = 'higgs_resonance'; basis = 'physical scalar target';
  } else if (row.id === 'gluon' || row.id === 'photon') {
    reference = row.id === 'gluon' ? GLUON : PHOTON;
    category = 'gauge_field'; basis = 'gauge-field target'; cardLabel = row.id === 'gluon' ? 'target' : 'limit';
  } else if (row.id === 'Z' || row.id === 'W_plus' || row.id === 'W_minus') {
    reference = MASS_REFERENCES[row.id === 'Z' ? 'Z' : 'W']; category = 'vector_boson'; basis = 'vector-boson target';
  } else if (['proton', 'neutron', 'pi0', 'rho0'].includes(row.id)) {
    reference = MASS_REFERENCES[row.id]; category = 'hadron'; basis = 'hadron target';
  } else if (row.id.startsWith('C_')) {
    reference = COMPOSITE; category = 'register_composite'; basis = 'register-composite candidate';
  }
  return Object.freeze({ id: row.id, sourceRow: row, retainedReference: row.measured ?? null,
    reference, category, basis, cardLabel,
    sharedSpeciesReference: category === 'charged_fermion_component',
    combinedStatisticalScore: null,
    scoreReason: 'This view does not supply the source-to-species/scheme transfer and combined model uncertainty required for a statistical comparison.' });
}

export function projectReferenceMap(map) {
  if (!map || !Array.isArray(map.rows)) throw new TypeError('A complete coordinator row array is required');
  const seen = new Set(), rows = map.rows.map((row) => {
    if (seen.has(row.id)) throw new Error('Duplicate coordinator row ID');
    seen.add(row.id); return referenceForRow(row);
  });
  const counts = {}, massKeys = new Set();
  for (const row of rows) {
    counts[row.category] = (counts[row.category] || 0) + 1;
    if (row.reference.kind === 'mass_reference') massKeys.add(row.reference.key);
  }
  return Object.freeze({ rows: Object.freeze(rows), categories: Object.freeze(counts),
    distinctMassReferenceQuantities: massKeys.size, sourceRows: map.rows.length,
    sourcePredictionsOrStatusesChanged: false, referenceEdition: 'PDG 2025 fixed edition; CODATA 2022 constants' });
}

export const REFERENCE_CATALOG = freeze({ schema: 'oph.viewer_particle_references.v1',
  edition: 'PDG 2025 fixed edition; CODATA 2022 constants',
  role: 'Comparison references only; no refit, model mutation or new particle identification',
  sources: REFERENCE_SOURCES, massReferences: MASS_REFERENCES, retainedInput: RETAINED_OM,
  nonMassReferences: { activeNeutrino: ACTIVE_NEUTRINO, singletNeutrino: SINGLET_NEUTRINO,
    HiggsMultiplet: HIGGS_MULTIPLET, gluon: GLUON, photon: PHOTON, registerComposite: COMPOSITE },
});
export function escapeReferenceText(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
}
