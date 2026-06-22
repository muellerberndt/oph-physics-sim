# Small OPH Universe v1

This lane is the first exact finite-consensus calibration harness. It follows Pro's advice to stop using large CMB/H3 diagnostics as confirmation evidence until the finite theorem burden is closed.

## Claim Boundary

Small OPH Universe v1 can certify:

- strict finite consensus on a fixed 12-patch Z2 icosahedral fixture;
- zero accepted Phi increases;
- zero strict-descent violations;
- disjoint commutation;
- local-diamond and schedule confluence;
- repair completeness;
- exactly one globally consistent state;
- exactly one terminal normal form;
- a frustrated one-edge holonomy obstruction control.

It does not certify endogenous modular flow, inferred `2pi`, Lorentz symmetry, H3 bulk, particles, or CMB.

## Run

```bash
python3 tools/verify_small_universe.py \
  --config configs/sou_v1_icosa12.yml \
  --seed 20260620 \
  --schedule-replays 16 \
  --out runs/small_oph_universe_v1

cd runs/small_oph_universe_v1
shasum -a 256 -c SHA256SUMS
```

## Output Contract

The verifier writes:

```text
MANIFEST.json
config.yml
source_hashes.json
all_states.jsonl
repair_transition_table.jsonl
schedule_replays.jsonl
cycle_holonomy.json
exact_consensus_receipt.json
frustrated_control_receipt.json
small_oph_universe_evidence.json
NON_CLAIMS.md
SHA256SUMS
SHA256.txt
```

Tables are JSONL rather than Parquet so the exact verifier remains dependency-light and reproducible
under the current project dependencies. Concrete run counts and hashes belong in the generated run
directory, not in this stable documentation page.

## Next Work

The next promotion step is not CMB. Replace the calibration parent-copy repair generator with an actual OPH recovery-derived repair relation while keeping `tools/verify_small_universe.py` as the independent verifier. Counterexample states from that verifier should be preserved, not tuned away.
