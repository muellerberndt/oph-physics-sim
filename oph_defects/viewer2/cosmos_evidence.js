// Pure data rules shared by the Cosmos page and its source-data checks.
export const HOT_GROUPS = [
  ['qL', 36, 'fermion'], ['qR', 36, 'fermion'], ['lep', 18, 'fermion'],
  ['g', 16, 'boson'], ['W', 8, 'boson'], ['H', 4, 'boson'],
];
export function hotWeights(quantity = 'number') {
  if (!['number', 'energy'].includes(quantity)) throw new Error('Unknown thermal quantity');
  const f = quantity === 'number' ? 3 / 4 : 7 / 8;
  return HOT_GROUPS.map(([id, degrees, stats]) => [id, degrees * (stats === 'fermion' ? f : 1)]);
}
export function displayedCut(cut, { stride = 1, hash = (id) => id, previousDepth = -1 } = {}) {
  const ids = cut.screen_id, cls = cut.screen_cls, xyz = cut.screen_xyz, birth = cut.screen_birth_depth;
  if (!ids || !cls || !xyz || !birth || ids.length !== cls.length || xyz.length !== 3 * ids.length || birth.length !== ids.length)
    throw new Error('The selected cut has no complete stored screen positions');
  const rows = [];
  for (let i = 0; i < ids.length; i++) if (stride === 1 || hash(ids[i]) % stride === 0) rows.push(i);
  return {
    ids: Int32Array.from(rows, (i) => ids[i]), cls: Uint8Array.from(rows, (i) => cls[i]),
    xyz: Float32Array.from(rows.flatMap((i) => [xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]])),
    born: rows.map((i, j) => birth[i] > previousDepth ? j : -1).filter((i) => i >= 0),
    storedRows: ids.length,
  };
}
export function aliveOnCut(birthDepth, deathDepth, depth, end = false) {
  return end ? deathDepth < 0 : birthDepth <= depth && (deathDepth < 0 || deathDepth > depth);
}

// Frozen seen-data projection of the public comparison ledger, not new predictions.
export const COSMOLOGY_LEDGER = {
  "source": "code/cosmology/postdiction_ledger/runtime/cosmology_postdiction_ledger.json",
  "sha256": "8d7c47beacb10e4072124acd3396ceaccaf81dacc7f534ea9a2132aaea1cbaf2",
  "url": "https://github.com/FloatingPragma/observer-patch-holography/blob/4623c6ec191026f2f470399282dc3458b3bb649f/code/cosmology/postdiction_ledger/COSMOLOGY_POSTDICTION_LEDGER.md",
  "P_C_tilt": 0.9660214956374175,
  "P_fwd_tilt": 0.9660214146696063
};
export const TILT_COMPARISONS = [
  {
    "row_id": "ns_edge_center_P_C_planck2018_vi_ns",
    "dataset": "Planck 2018 TT,TE,EE+lowE+lensing",
    "measurement": {
      "kind": "gaussian",
      "sigma": 0.0042,
      "unit": null,
      "value": 0.9649
    },
    "oph_value": 0.9660214956374175,
    "sigma_distance": 0.26702277081369397,
    "verdict": "consistent",
    "citation": "Planck Collaboration (2020), Planck 2018 results. VI. Cosmological parameters, A&A 641, A6, arXiv:1807.06209; the same value is quoted in Planck 2018 results. X, arXiv:1807.06211",
    "url": "https://arxiv.org/abs/1807.06209"
  },
  {
    "row_id": "ns_edge_center_P_C_spt3g_d1_planck_ns",
    "dataset": "SPT-3G D1 + Planck",
    "measurement": {
      "kind": "gaussian",
      "sigma": 0.0035,
      "unit": null,
      "value": 0.9636
    },
    "oph_value": 0.9660214956374175,
    "sigma_distance": 0.6918558964050041,
    "verdict": "consistent",
    "citation": "SPT-3G Collaboration, Camphuis et al. (2025), SPT-3G D1: CMB temperature and polarization power spectra and cosmology, arXiv:2506.20707",
    "url": "https://arxiv.org/abs/2506.20707"
  },
  {
    "row_id": "ns_edge_center_P_C_cmb_spa_ns",
    "dataset": "CMB-SPA (SPT-3G D1 + Planck + ACT DR6)",
    "measurement": {
      "kind": "gaussian",
      "sigma": 0.003,
      "unit": null,
      "value": 0.9684
    },
    "oph_value": 0.9660214956374175,
    "sigma_distance": -0.7928347875274951,
    "verdict": "consistent",
    "citation": "SPT-3G Collaboration, Camphuis et al. (2025), arXiv:2506.20707",
    "url": "https://arxiv.org/abs/2506.20707"
  },
  {
    "row_id": "ns_edge_center_P_C_spt3g_d1_alone_ns",
    "dataset": "SPT-3G D1 alone",
    "measurement": {
      "kind": "gaussian",
      "sigma": 0.011,
      "unit": null,
      "value": 0.951
    },
    "oph_value": 0.9660214956374175,
    "sigma_distance": 1.3655905124925014,
    "verdict": "consistent",
    "citation": "SPT-3G Collaboration, Camphuis et al. (2025), arXiv:2506.20707",
    "url": "https://arxiv.org/abs/2506.20707"
  },
  {
    "row_id": "ns_edge_center_P_C_act_dr6_p_act_ns",
    "dataset": "ACT DR6 P-ACT (ACT DR6 + Planck)",
    "measurement": {
      "kind": "gaussian",
      "sigma": 0.0038,
      "unit": null,
      "value": 0.9709
    },
    "oph_value": 0.9660214956374175,
    "sigma_distance": -1.2838169375217068,
    "verdict": "consistent",
    "citation": "ACT Collaboration, Louis et al. (2025), ACT DR6 power spectra, likelihoods and LambdaCDM parameters, arXiv:2503.14452; companion extended-models paper Calabrese et al. (2025), arXiv:2503.14454",
    "url": "https://arxiv.org/abs/2503.14452"
  },
  {
    "row_id": "ns_edge_center_P_C_act_dr6_p_act_lb_ns",
    "dataset": "ACT DR6 P-ACT-LB (ACT DR6 + Planck + CMB lensing + DESI 2024 (DR1) BAO)",
    "measurement": {
      "kind": "gaussian",
      "sigma": 0.0034,
      "unit": null,
      "value": 0.9743
    },
    "oph_value": 0.9660214956374175,
    "sigma_distance": -2.434854224288966,
    "verdict": "tension",
    "citation": "ACT Collaboration, Louis et al. (2025), arXiv:2503.14452; Calabrese et al. (2025), arXiv:2503.14454",
    "url": "https://arxiv.org/abs/2503.14452"
  }
];
