# KMS collar response probe: sector-repair replay (open task)

Date filed: 2026-07-14 (night session). Bisect chain in
`runs/kms_diag_4k_*` run dirs.

## Finding

Since commit `e460f34` ("Refactor for upcoming 128k run") the
perturb-remeasure response probe in `oph_fpe/bulk/modular_probe.py` is
gauge-covariant and fail-closed: when the production dynamics runs with
sector repair enabled, the probe returns an identity response with

    response_source = kms_collar_transport_response_fail_closed
    proof_blockers = ['production_sector_repair_not_replayed_by_response_probe']

The transition-scale selection then reads `response_degenerate: true`,
`two_pi_selected: false`, so `kms_bw_pass` fails, which cascades:

- the cosmology freezeout gate blocks `freezeout_fields.npz`,
  `freezeout_map_summary.json`, and the C_l proxy products;
- `declared_bw_kms_branch_replay_receipt` goes false, so the
  `observer_facing_3p1d_h3_experience_receipt` goes false.

Bisect facts (4k dense config, identical seed and machine):

| tree | config | kms_bw_pass |
|---|---|---|
| e56b1a9 | e56b1a9 config | PASS |
| e460f34 | e460f34 config | FAIL |
| e460f34 + working tree | any window combination (16/32, 4/32) | FAIL |
| e460f34 + working tree | b_a block disabled | FAIL |

The `history_window` bumps are innocent. The failure is the deliberate
fail-closed contract meeting a probe that lacks the replay.

## Honesty consequence (recorded in the epic-wins tracker)

The 07-11 runs whose `observer_facing_3p1d_h3_experience_receipt` reads
TRUE were certified by the pre-covariant probe, which ignored gauge and
sector repair. Under the covariant contract those receipts are unproven,
neither refuted nor certified. They must be re-earned once the replay
lands.

## The fix (queued)

Implement production sector-repair replay inside
`_perturb_remeasure_response_matrix`:

1. Thread the sector-label state and the production repair rule
   (`_repair_sector_labels` path in `oph_fpe/scale/bw_array.py`) into
   `graph_response`.
2. In the probe's repair loop, after the covariant port-pair repair step,
   apply the same sector-label update to the probe's local copy, using
   `repair_covariant_port_pairs` (imported, unused as of e460f34).
3. Remove the `production_sector_repair_enabled` fail-closed branch once
   the replay reproduces the production mismatch trajectory on a frozen
   4k fixture (add that fixture test).
4. Re-run the 4k dense config: `kms_bw_pass` must be earned, never
   assumed. If the covariant probe still reads the response degenerate,
   that is a physics statement about the 2 pi clock under sector repair,
   and the 3p1d receipt stays false with that reason.

## Interim ladder policy (2026-07-14 night runs)

Production sector repair stays ENABLED (the covariant dynamics is the
point of the K1 upgrade). Night ladder configs declare

    cosmology.freezeout.require_kms_bw_pass: false
    cosmology.freezeout.require_kms_override_reason: "..."

so screen-proxy products emit for visualization while the gate report
records the false check and the blocker verbatim. No receipt is
overridden; only the product gating is.
