# OPH universe runs, defect layers and viewer

`oph_defects` runs the OPH executable model at federation scale, writes each run as a documented
dataset, and serves a viewer for it. The exact layer is the executable specification with nothing
added. Everything else is a declared layer computed from the exact records.

| Module | Purpose |
| --- | --- |
| `fedgeom.py` | the exact federation in compact form, built from the functions the model pins; a frame-free control gluing |
| `kernels.py` | one attempt body in C, compiled as a sequential kernel, as causal layers on the CPU and as CUDA; an exact C reproduction of NumPy's PCG64 bounded draws |
| `universe.py` | the run driver: `python -m oph_defects.universe --level 5 --sweeps 600 --records all --out runs/universe/L5` |
| `cuda.py` | the GPU engine; `python -m oph_defects.cuda 7 4` compares GPU and CPU on every array |
| `tracking.py` | identities of register defects and of moving charges |
| `dataset.py` | reader, exact replay, observer tapes, dataset card, viewer files |
| `closed_loop.py` | law C with transported contrast: pinned charge pairs and their exact stationary field |
| `pair_dynamics.py` | the same law with unit twists: charges that move, repel and annihilate |
| `gluing_control.py` | low spectrum of the port Laplacian for the declared and the control gluing |
| `ou_test.py` | density correlations of a canonical run against the law of the native operator |
| `particle_table.py` | the particle table of the viewer, built from the candidate map of the theory repository |
| `serve.py`, `viewer2/` | the viewer: `python -m oph_defects.serve runs/universe --port 8777` |
| `DATASET.md` | definition of every file and field of a run directory |

The first-generation files (`run.py`, `dynamics.py`, `geometry.py`, `federation_run.py`,
`federation_geometry.py`, `data.py`, `report.py`, `exact_run.py`, `serve_legacy.py`, `viewer/`,
`SCHEMA.md`) are kept for the old run formats.

## The exact layer

Law C (integer nearest agreement with a fair tie coin) on the federation of
`oph_exact.federation.build_federation(level, 'port_pair')`, with the declared preparation and the
uniform seam schedule drawn exactly as `run_integer_law` draws it. Tests
(`tests/test_universe_run.py`) assert:

* the geometry equals the pinned builder seam for seam;
* the loads, the tallies and the attempt at which the balanced class is reached equal
  `run_integer_law` on the pure Python engine;
* level 5 reproduces the stored receipt of 2026-09-09 (seed 914100: 209,717 descents, 7,673,840 swaps,
  26,307,803 waits, balanced after 33,820,528 attempts);
* executing each causal layer of disjoint attempts in forward or reverse order gives every output file
  of the schedule order (this is what the GPU does);
* a consistent cut equals the state after re-executing exactly its own attempts;
* resume and replay are bit-exact.

The compact level-5 reference receipt is shipped as
`references/federation_level5.json`; the regression re-executes its integer
run even when the large local run directory is absent. The receipt is an
exact copy of that run's summary, not a new particle-identification result.

`python -m oph_defects.cuda LEVEL SWEEPS` repeats the GPU against CPU comparison on any machine with
CuPy. A complete level-5 run on the GPU equals the CPU run file by file.

## Declared layers

Causal depth, consistent cuts (slices of synchronised observer time), incident clocks, the
energy-accounted Z6 register and its identities. They never change a load. `DATASET.md` states each
rule. The register is a bookkeeping layer: no load or current record can detect one of its defects, so
names attached to its classes in the viewer are candidates and carry no energy or mass.

## Closed-loop experiments (declared extension)

`closed_loop.py` and `pair_dynamics.py` run law C with transported contrast: every seam carries an
integer offset `b_e` and compares `x_a` with `x_b + b_e`; `b = 0` is law C. A plaquette with offset
holonomy `q` drives a permanent circulation of the committed current. The theory session derived, and
the runs confirm at level 3 to 0.2 percent:

* the mean contrast is the divergence-free part of `-b`: `dbar = B x* - b`;
* the potential released per sweep by a pinned configuration is `(1/2) |dbar|^2`;
* `sum d^2 = 2 |dbar|^2 + number of odd seams`;
* the net current around every port cycle is `-holonomy / 2` per sweep.

With unit twists (`pair_dynamics.py`) the offsets themselves change and charges move.

## Scale

| Level | Carriers | Seams | Backend | Seconds per sweep |
| --- | --- | --- | --- | --- |
| 3 | 1,280 | 40,320 | CPU | 0.001 |
| 5 | 20,480 | 645,120 | CPU | 0.02 |
| 7 | 327,680 | 10,321,920 | CPU / GPU (L40S) | 0.6 / 0.09 |
| 8 | 1,310,720 | 41,287,680 | CPU / GPU | 3.6 / 0.38 |
| 9 | 5,242,880 | 165,150,720 | CPU | about 20 |
| 10 | 20,971,520 | 660,602,880 | GPU | about 10 |

Kernel times only; a production run adds the schedule draws, cut files and records.

## Viewer

The viewer presents the stored universe, individual observers, particle candidates,
spectrum, story, field readouts and a cosmology comparison. The run's carrier loads,
register identities and local clocks come from its records. Candidate family names
do not establish measured particle identities. The cosmology view separates these
finite records from imported thermal-history illustrations; its numerical counters
are checked by `run_evidence.py` and `tests/test_run_evidence.py`.

In the field readout, select a defect on the screen or click its label in the box.
**Follow one defect** retains the same ID through a declared preparation, finite
free Dirac evolution and position detector. Its momentum plots and energy/velocity
values are operator expectations, not simultaneous sharp position and momentum
measurements. MeV and metre-per-second labels retain the supplied length and clock
conversion. The box is a separate finite field experiment, not a derived
interacting continuum of the classical load run.

The Spectrum tab also includes an authenticated interacting-source table.
`python -m oph_defects.quantum_evidence` rebuilds it from the public quantum
source receipts; `--check` verifies the saved table and every inherited source
hash. Its charged energy bins, preparation error and separately costed reference
are shown in source units. The card does not identify those transitions with a
measured lepton or with the classical-run register.

The detector retains both energy-band outcomes. Its explicit band-conversion
operation has an energy cost; it is not an antiparticle identification or free
evolution. [The instrument proof](viewer2/readout_instrument.md) gives the complete
probability law. **Save this readout** exports the exact preparation, every packet
record (including those beyond the eighteen-point trail), detector widths and
realized sampling variates. These variates belong to the viewer apparatus, not the
native repair tape. Replay the saved experiment with:

```sh
node oph_defects/viewer2/readout_replay.mjs saved-readout.json
node oph_defects/viewer2/readout_instrument_check.mjs
node oph_defects/viewer2/readout_observables_check.mjs
node oph_defects/viewer2/readout_replay_check.mjs
node oph_defects/viewer2/readout_limit_check.mjs
```

Replay compares the positions, band outcomes, probabilities and apparatus energy
changes, and checks the final packet norm. It neither fits a mass nor promotes a
candidate label to an observed Standard Model species.

```sh
.venv/bin/python -m oph_defects.particle_table          # rebuild the table after the theory repository changes
.venv/bin/python -m oph_defects.serve runs/universe --port 8777
.venv/bin/python -m pytest -q tests/test_universe_run.py
```

The [routing comparison](viewer2/readout_limit.md) gives an explicit two-tick
error bound. Its continuum scaling shrinks momentum and mass angle together.
It does not promote a fixed-angle, finite-grid comparison into an interacting
continuum theorem.

To rebuild the bundled run evidence after changing stored datasets:

```sh
python -m oph_defects.run_evidence runs/universe --out oph_defects/viewer2/run_evidence.json
```

Reference-only builders accept `--rer` or `--repository`, respectively, or
`OPH_THEORY_REPOSITORY` for a theory checkout in another location. Serving the
saved viewer table and receipts does not require that checkout. Large run
files remain outside Git; create a small run with the command above or use
an existing documented run directory.

Research charts referenced by `/rl/` are served from the frozen
`viewer2/research` bundle. Its manifest lists exactly the referenced JSON
artifacts and their hashes; inclusion does not promote an artifact's claims.
Refresh and verify it explicitly with:

```sh
python -m oph_defects.bundle_research --source /path/to/repair-law/work
python -m oph_defects.bundle_research --check
```

The Spectrum tab also shows complete finite interacting quantum records.
These are separate from the classical register run and the prepared free
Dirac packets. The five-block momentum-exchange panel retains native
preparation error and both detector outcomes. Its component occupations
are relative to the retained quantum reference. Rebuild these source-bound
tables from the specified theory checkout with:

```sh
python -m oph_defects.quantum_evidence --repository /path/to/reverse-engineering-reality
python -m oph_defects.collision_evidence --repository /path/to/reverse-engineering-reality
node oph_defects/viewer2/collision_evidence_check.mjs
```
