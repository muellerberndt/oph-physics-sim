# OPH-FPE

OPH-FPE (Observer-Patch Fundamental Physics Emergence) is a finite simulator
for Observer-Patch Holography (OPH). It turns parts of the OPH consistency
program into reproducible computational experiments and writes the evidence
needed to audit what each experiment does and does not establish.

## What is OPH?

Observer-Patch Holography is a research program in which physical description
begins with bounded observers rather than a pre-existing global view. An OPH
observer is a concrete self-reading system: it has local state, ports or a
boundary, readback, records, and feedback or repair moves. Neighboring
observers compare information on their overlaps and attempt to make their
records mutually consistent.

The program asks whether stable shared records can support structures that are
normally assumed at the outset, such as common geometry, time, fields, and
particle-like defects. These are conditional emergence claims, not assumptions
built into the definition of a successful run.

## What does the simulator do?

OPH-FPE instantiates the observer-patch idea on finite graphs. A run:

1. creates a finite carrier of patches with local state and boundary data;
2. lets patches read overlaps and identify disagreements;
3. applies configured local repair rules;
4. records the resulting state, controls, and reconstruction diagnostics; and
5. emits machine-readable receipts for the claims tested by that run.

The simulator includes experiment lanes for finite consensus, observer-local
readback and modular time, observer-facing geometry, defect dynamics,
cosmology diagnostics, and distributed carriers. The lanes are kept separate:
a geometric chart is not automatically a neutral spacetime, a stable defect is
not automatically a physical particle, and a screen spectrum is not
automatically a physical CMB prediction.

The primary output is therefore an evidence bundle, not a single score. It
contains the resolved configuration, seed, records, traces, controls, hashes,
diagnostics, and pass/fail receipts. A failed receipt is meaningful: it marks
the precise bridge that the finite construction did not supply.

## How does it relate to OPH?

OPH-FPE is the finite experimental surface of OPH. It tests whether specific
finite systems instantiate hypotheses used by the theory and whether the
corresponding structures survive explicit controls. It does not replace the
analytic or theorem-level work, and a numerical receipt does not promote a
paper claim by itself.

The simulator is deliberately limited in scope. Its lattice dynamics does not
derive quantitative constants such as the fine-structure constant, particle
masses, the scalar spectral tilt, or the cosmological constant. Inputs added
for explanatory visualization are labeled as assumptions and cannot turn a
computed receipt into a physical claim.

In short:

```text
bounded patches -> local readback -> overlap repair -> shared records -> gated reconstruction
```

## Installation

OPH-FPE requires Python 3.11 or newer.

```bash
git clone https://github.com/muellerberndt/oph-physics-sim.git
cd oph-physics-sim

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

CAMB-backed cosmology diagnostics are optional:

```bash
python -m pip install -e '.[test,camb]'
```

Verify the installation:

```bash
python -m pytest -q
oph-fpe --help
```

## Run a simulation

The smallest general smoke run uses the bundled `e0_z2_patchnet` fixture:

```bash
oph-fpe run \
  --config configs/e0_z2_patchnet.yml \
  --out-dir runs
```

For the production icosahedral-tower array pipeline (a small,
non-evidential engineering rung):

```bash
oph-fpe run-array \
  --config configs/icosa_smoke.yml \
  --out-dir runs
```

The protected-authority consensus fixture exercises a theorem-audited finite
normalizer on the 81,920-cell tower rung:

```bash
oph-fpe run-bw-array \
  --config configs/icosa_82k_protected_consensus.yml \
  --out-dir runs
```

Its consensus theorem is conditional on the authority metadata frozen before
dynamics. The separate
`data/repair_closure/vertex12_signed_record_feedback_receipt.json` checks a
literal record-conditioned software feedback rule on eight twelve-port
carriers. That smaller receipt is not part of the 81,920-cell run and does not
constitute a laboratory or physical-observer attachment.

The older `fibonacci_sphere` fixtures remain available as explicit
support-chart/KNN controls; they do not instantiate the production A1 carrier.
For example:

```bash
oph-fpe run-array \
  --config configs/e1_s3_modular_screen_4k.yml \
  --out-dir runs
```

For the integrated observer, geometry, defect, and visualization pipeline,
the current bundled fixture is likewise a legacy-chart control rather than a
production-carrier run:

```bash
oph-fpe run-oph-universe \
  --config configs/e4_shared_observer_bulk_64k_object_chart.yml \
  --out-dir runs \
  --run-id my_oph_run
```

Run artifacts are written beneath the selected output directory. The exact
contents depend on the command and configuration, but the bundle keeps inputs,
computed outputs, visualization data, and claim receipts together. Generated
runs are ignored by Git.

Use a bundled configuration as a starting point for a local experiment:

```bash
mkdir -p configs/local
cp configs/e0_z2_patchnet.yml configs/local/my_experiment.local.yml
oph-fpe run --config configs/local/my_experiment.local.yml --out-dir runs
```

Each configuration should state its `claim_boundary`: what the run
instantiates, which diagnostics it requests, and which stronger
interpretations remain conditional on explicit receipts. See
[`configs/README.md`](configs/README.md) for the curated examples and
[`docs/configuration.md`](docs/configuration.md) for the configuration format.

## Reproducibility and further documentation

- [`REPRODUCTION.md`](REPRODUCTION.md) describes clean-environment setup and
  reproduction commands for specialized diagnostics.
- [`docs/WHAT_OPH_FPE_DOES.md`](docs/WHAT_OPH_FPE_DOES.md) gives a more detailed
  conceptual account of the simulator.
- [`docs/RUN_OUTPUTS_AND_VISUALIZATION.md`](docs/RUN_OUTPUTS_AND_VISUALIZATION.md)
  documents run artifacts and visualizer bundles.
- [`docs/CLAIM_LANES.md`](docs/CLAIM_LANES.md) defines the boundaries between
  the simulator's experiment lanes.
- [`docs/SIMULATION_ASSUMPTION_POLICY.md`](docs/SIMULATION_ASSUMPTION_POLICY.md)
  explains how computed and assumed visualization layers remain distinct.
- [`docs/README.md`](docs/README.md) indexes the full technical documentation.

Interactive visualizations built from simulator evidence bundles are available
at <https://simulation.floatingpragma.io>.

Compact display-only handoffs for carrier dynamics, observer-frame quantum
conditioning, refinement depth, repair confluence, observer cameras and modular
time, defects, finite EM, cosmology diagnostics, and the full theorem/evidence
atlas live under `visualizer_handoffs/oph-headlines-2026-08-20/`. Regenerate
them with `python tools/build_headline_visualizer_handoffs.py`; every directory
has renderer instructions, hashes, an explicit claim boundary, and a hard
200 MB size limit. The generated handoffs are intentionally ignored by Git.
Build the single upload-ready visualizer-builder archive, including its guide
and copy/paste prompt, with `python tools/build_visualizer_upload_bundle.py`.
