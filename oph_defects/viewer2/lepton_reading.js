// The declared squared-circulant reading; it is not the spectrum of repeated FR ticks.
// Comparison values: RER/code/particles/data/particle_reference_values.json.
export const LEPTON_REFERENCE = {
  electronGeV: 0.00051099895069,
  muonGeV: 0.1056583755,
  tauGeV: 1.77693246513409, // frozen API average; the source display rounds to 1776.93 MeV
  source: '/rl/CODEX_JOIN_SPECTRUM/family_clock.json',
};
export function familyReading(phase) {
  const amp = [0, 1, 2].map((k) => 1 + Math.SQRT2 * Math.cos(phase + 2 * Math.PI * k / 3));
  const m = amp.map((a) => a * a), ord = [0, 1, 2].sort((a, b) => m[a] - m[b]);
  const names = []; names[ord[0]] = 'e'; names[ord[1]] = 'μ'; names[ord[2]] = 'τ';
  return { amp, m, names, mu: m[ord[1]] / m[ord[0]], tau: m[ord[2]] / m[ord[0]],
    Q: m.reduce((s, v) => s + v, 0) / amp.reduce((s, a) => s + Math.abs(a), 0) ** 2,
    reading: 'squared eigenvalue ratios, with an independently calibrated overall scale' };
}
export const FR_TWO_TICK = Object.freeze({ compressionAmplitude: Math.sqrt(3) / 9,
  mu: 1, tau: 1, Q: 1 / 3, interpretation: 'literal reused-record two-tick compression, not the square of the one-tick effect' });
export function muonResidualPpm(phase = 2 / 9) {
  return 1e6 * (familyReading(phase).mu / (LEPTON_REFERENCE.muonGeV / LEPTON_REFERENCE.electronGeV) - 1);
}
