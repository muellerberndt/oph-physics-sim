# OPH universe runs: dataset definition (`oph.universe.run.v2`)

A run directory written by `python -m oph_defects.universe` holds one execution of the OPH
executable model at federation scale. This file defines every file and field. All array
files are NumPy `.npz` archives (a zip of `.npy` members, little-endian, C order); all tables
are JSON. `oph_defects/dataset.py` reads everything and replays any interval exactly.

## 1. What is exact and what is declared

**Exact layer: the executable specification, nothing added.**

* *Federation.* `N = 20 * 4^level` twelve-port carriers on the cells of the geodesic
  icosahedral tower, glued by the declared `port_pair` convention (three ports of every
  carrier meet a neighbour). It is the federation of
  `oph_exact.federation.build_federation(level, 'port_pair')`, built from the two functions
  the model pins (`geodesic_icosahedral_patch_arrays`, `assign_echosahedral_ports`).
  `run.json` records the hash of each defining source against the model's literal pin.
* *Preparation.* `exact06`: `numpy.random.default_rng(20260909 + level).integers(0, 6, size=ports)`
  (`oph_exact.federation.initial_loads`). `canonical:<num>/<den>:<seed>`: a uniform subset of
  fixed size, loads 0 or 1, drawn by the sequential exact sampler of
  `model/observer_time_fields/canonical_sector.draw_sector` with `random.Random(seed)`.
* *Schedule.* Seed `909000 + 1000 * level + 100 + schedule`. Per sweep, with `S` the number of
  seams: `seq = rng.integers(0, S, size=S, dtype=int64)`, then
  `coin = rng.integers(0, 2, size=S, dtype=int64)`; attempt `i` acts on seam `seq[i]` with tie
  coin `coin[i]`. Exactly the draws of `oph_exact.federation.run_integer_law`.
* *Law C.* Attempt on seam `(a, b)` with loads `x_a, x_b` and coin `c`: `t = x_a + x_b`;
  `x_a` becomes `ceil(t / 2)` if `c = 1`, else `floor(t / 2)`; `x_b = t - x_a`. The record of the
  attempt is `(c, j)` with current `j = old x_a - new x_a`, the units moved from `a` to `b`.
  `j = 0` is a hold; `|x_a - x_b| = 1` with `j != 0` is a tie swap; `|x_a - x_b| >= 2` is a descent.

The loads of a run equal those of `run_integer_law` for the same seed (tested against the pure
Python engine, including the tallies and the attempt at which the balanced class is reached; level 5
reproduces the stored receipt of 2026-09-09).

**Declared layers: deterministic functions of the exact event stream, no further randomness.**

* *Causal depth.* Two attempts are ordered when they share a resource and one comes earlier in
  the schedule. The resources of an attempt are its two ports and, for a glue seam, its two
  screen plaquettes. `depth = 1 + max(depth of the earlier attempts on its resources)`; the
  preparation has depth 0. `depth_native` uses the two ports only. Attempts of equal depth are
  disjoint and commute, so any execution order that respects depth gives the same result.
* *Consistent cuts (slices of synchronised observer time).* Cut `T` is the set of all attempts of
  depth at most `T`. It is closed under causal predecessors, so the state "after exactly these
  attempts" is well defined: every port, face and carrier is read after its last attempt of depth
  at most `T`. The sweep index is a solver counter and plays no role in a cut.
* *Incident clocks.* `ticks[c]` counts the attempts on the 33 seams of carrier `c` (30 own, 3 glue),
  holds included. On a cut it counts the attempts of depth at most `T`.
* *Register (declared extension).* One Z6 link per seam, zero at the start. A repair with current
  `j != 0` on seam `(a, b)` with contrast `d = x_a - x_b` proposes `delta = j mod 6`: the face that
  traverses the seam `a -> b` changes its class by `+delta`, the face that traverses it `b -> a` by
  `-delta`. The twist is applied when `cost[new] - cost[old] <= kappa * j * (d - j)`, where
  `j * (d - j)` is the drop of the mismatch potential `1/2 sum x^2` in that repair. A face of
  nonzero class is a defect; classes sum to zero mod 6 on every closed surface (each carrier's twenty
  faces; all screen plaquettes). The register never changes a load, so no load or current record can
  detect a defect: it is a bookkeeping layer, and names attached to its classes are candidates.
* *Identities.* Every accepted twist changes two faces; pre and post classes decide: pair birth,
  move, split, annihilation, fusion or exchange (`oph_defects/tracking.py`).

**Regime.** With `exact06` the run is a quench: the seeded loads have variance 35/12 per port
against 1/4 in the balanced band, block totals are conserved, and the long density modes relax
at rate `2 lambda_k` in scaled time `tau = sweeps / (2 * 4^level)`. With `canonical` the run is
stationary from the first attempt.

## 2. Indexing

* Port `p` of carrier `c`: global port `12 c + p`.
* Seam `s < 30 N`: template seam `k = s % 30` of carrier `s // 30`, ends
  `(12 c + template_seams[k][0], 12 c + template_seams[k][1])`. Seam `30 N + g`: glue seam `g`
  with ends `(glue_a[g], glue_b[g])`. Every seam is oriented from its first end `a` to its second `b`.
* Face `20 c + f`: template face `f` of carrier `c` (internal tier). Face `20 N + v`: the screen
  plaquette around vertex `v` of the tower mesh (screen tier), bounded by glue seams.
* Cells are hierarchical: the children of cell `c` at the next level are `4 c .. 4 c + 3`, the
  central child is `4 c + 3`.
* `run.json: geometry` holds `template_seams`, `template_faces`, `template_face_plus/minus`
  (the face that traverses template seam `k` in direction `a -> b` / `b -> a`) and `port_vectors`.
  `oph_defects.fedgeom.build(level)` rebuilds everything; `geometry.npz` stores it up to level 7.

## 3. Files

| File | Content |
| --- | --- |
| `run.json` | manifest: `schema`, `config`, `law`, `preparation`, `regime`, `scaled_time`, `layers`, `geometry`, `totals`, `cuts`, `end`, `checkpoints`, `records`, `series`, `observers`, `identities`, `tracker_totals`, `provenance` |
| `stats.npz` | one row per sweep (section 4) |
| `cuts/cut_KKKKK.npz`, `cuts/end.npz` | one consistent cut each; `end.npz` is the state after every attempt (section 5) |
| `catalogue.npz` | one row per defect identity (section 6) |
| `observers/obs_<carrier>.npz` | tape, patch and readouts of one featured observer (section 7) |
| `records/sweep_SSSSSS.npz` | every attempt of one sweep (section 8); `.rng.json` beside it is the generator state at the start of the sweep |
| `series/block_KKKKK.npz` | density snapshots on a fixed sweep cadence (section 9) |
| `checkpoints/ckpt_SSSSSS.npz` + `.json`, `aux_SSSSSS.npz` | complete state at the start of sweep `S` (section 10) |
| `geometry.npz` | `glue_a`, `glue_b`, `glue_fp`, `glue_fm` (plaquette faces of each glue seam), `tri` (mesh vertices of each cell), `mesh_xyz`, `carrier_xyz`, `neighbours`, `loads_initial`; levels up to 7 |
| `final_loads.npy` | int8 `[ports]`, the loads at the end |
| `view/` | derived files for the viewer: `display.npz` (display mesh), `catalogue.npz` (rows of identities that appear on a stored cut), `class_stats.json` |
| `README.md`, `MANIFEST.sha256`, `DATASET.md`, `particle_map.json` | dataset card, sha256 of every file, this definition, the candidate table used for names |

## 4. `stats.npz` (per sweep, solver bookkeeping)

`V` (sum of squared loads after the sweep), `descents`, `swaps`, `waits`, `potential_released`
(sum of `j (d - j)`), `unsettled_ports` (ports outside the balanced band), `depth_min/max/mean`,
`depth_native_min/max/mean`, `ticks_min/max`, `twists_accepted`, `twists_blocked` (and `_screen`),
`work_offered` (`kappa` times the released potential), `work_used`, `energy_released`,
`counts_screen[6]`, `counts_intra[6]` (faces per class), `births`, `moves`, `annihilations`,
`fusions`, `splits`, `exchanges` (tracked tiers), `identities`, `open_cuts`, `seconds`.

## 5. Cut files

`depth` (int64; -1 in `end.npz`). Screen tier, one entry per defect on the cut:
`screen_face` int32, `screen_cls` uint8 (1..5), `screen_id` int32 (identity),
`screen_birth_depth` int32, `screen_xyz` float32 `[n, 3]` (unit vector). The same five arrays with
prefix `intra_` when the internal tier is tracked (`config.track_intra`). `load_hist` int64 `[6]`
(ports per load value), `load_map` uint16 (sum of loads over the descendants of each cell of the
display level), `feat_ticks` int32 (own tick of each featured observer on the cut). Up to
`config.full_cut_level`: `loads` int8 `[ports]` and `ticks` int32 `[carriers]`, the complete state on
the cut. The entry of the cut in `run.json: cuts` repeats the counts, `V`, `unsettled_ports`,
`ticks_min/max/mean` and the sweeps at which the cut was opened and completed.

## 6. `catalogue.npz`

`cls` uint8, `tier` uint8 (1 screen, 0 internal), `birth_sweep`, `birth_depth`, `death_sweep`,
`death_depth` (-1 while alive), `birth_face`, `last_face`, `hops`, `partner` (the other member of a
pair birth, or the other party of the ending event), `parent1`, `parent2`, `cause` uint8 (0 alive,
1 annihilation, 2 fusion, 3 split, 4 exchange), `child1`, `child2`; `birth_xyz`, `last_xyz`
float32 when there are at most 8 million identities. Row index = identity.

## 7. Observer files

Featured observers are the central cell of every base face (carrier `f * 4^level + 4^level - 1`;
`f = 0` is the root observer of the model) and one pentagonal cell at every base vertex.

`carrier`, `xyz`, `frame` (3 x 3, columns tangent_x, tangent_y, normal: the routing frame of the
cell), `loads_initial`, `loads_final` (twelve ports). Tape, one row per attempt on the 33 seams in
schedule order (row `n` is own tick `n + 1`): `tape_sweep`, `tape_pos` (position in the sweep),
`tape_seam` (global seam), `tape_slot` (0..29 own template seam, 30..32 glue seam), `tape_end`
(0: the observer holds end `a`, 1: end `b`), `tape_coin`, `tape_current`, `tape_twist`,
`tape_depth`. `loads_by_tick` int8 `[n + 1, 12]` (reconstructed from the tape alone and checked
against the run), `displacement` float32 `[n + 1, 3]` (`x(n) = sum_p n_p u_p`), `axial` float32
`[n + 1, 3]` (`Z_n = sum (u_a x u_b) j` over the carrier's own seams). Patch: `patch_cells`
(carriers within `config.patch_radius` glue steps, breadth first), `patch_dist`, `patch_cell_xyz`,
`patch_frames`, `patch_seams` int32 `[k, 3]` (local port `a`, local port `b`, global seam; local
port = `12 * index_in_patch + p`), `patch_loads_initial`, `patch_loads_final`, and per stored cut
`cut_depth`, `cut_ticks`, `cut_patch_loads` uint8 `[cuts, 12 m]`.

## 8. Record files

Arrays of length `S` in schedule order: `seam` int32, `coin` uint8, `current` int8,
`contrast` int8 (`x_a - x_b` before the attempt), `total` int8 (`x_a + x_b` before the attempt; with
the contrast it gives both loads), `twist` int8 (accepted delta, or minus the proposed delta when it
was blocked, 0 when no current moved), `energy` int8 (register cost change of the proposal),
`depth` int32. Sweeps without a record file are reproduced by `Dataset.replay`.

## 9. Series blocks

Stacked snapshots after every `config.series_every` sweeps: `sweep`, `cells` (sum of the twelve loads
of every carrier), `nonzero_current_events` and `descent_events` (uint16 per carrier, counted since
the previous snapshot; a glue seam counts for both carriers), `level5_sums` (sums over the
descendants of each level-5 cell, levels above 5), `ports` (int8 loads of every port, when
`series_ports`).

## 10. Checkpoints

`ckpt_SSSSSS.npz`: `sweep`, `loads`, `register`, `classes`, `depth_native` (per port), `depth`
(per port, then per plaquette), `ticks`, `face_id` (identity on each tracked face), the catalogue
columns as `cat_*`, `tracker_stats`, and the open cuts (`open_cut_index/depth/opened/loads/ticks/
classes/ids`). `ckpt_SSSSSS.json`: the generator state (a NumPy PCG64 state), the next cut, and the
lists of cuts, records and series blocks written so far. `aux_SSSSSS.npz`: the per-sweep tables and
the observer tapes so far. `python -m oph_defects.universe --resume <ckpt>` continues the same
history; `Dataset.replay(start, stop)` regenerates the records of any interval from the nearest
checkpoint.

## 11. Checks a reader can repeat

* `MANIFEST.sha256` against the files.
* `Dataset.replay` against any stored record file.
* A cut: re-execute, in schedule order, only the attempts with `depth <= T`; loads, classes and clocks
  equal the cut file (`tests/test_universe_run.py::test_a_cut_is_the_state_after_exactly_its_own_events`).
* An observer tape: apply the currents to `loads_initial`; the result equals `loads_final`.
* Conservation: the sum of loads is constant; face classes sum to zero mod 6 per carrier and over the
  screen plaquettes, on every cut.

## 12. Closed-loop runs (`config.twists`)

`python -m oph_defects.universe --twists` writes the same directory layout for a different law, a declared
extension that is not the exact layer: law C with transported contrast and unit twists. Every seam carries
an integer offset `b_e` (oriented `a -> b`); an attempt takes `s = x_a + x_b + b_e`, sets `x_a` to
`ceil(s / 2)` or `floor(s / 2)` by the coin and `x_b` to `s - x_a - b_e`; the transported contrast is
`d = x_a - x_b - b_e`. Before that repair, an attempt on a glue seam with `d != 0` moves the offset by
`k = sign(d)` when `2 (cost(after) - cost(before)) < 2 kappa |d| - kappa` over the two plaquette
holonomies (equality decided by the coin; cost extended additively beyond 3), and then moves no load.
With all offsets zero and no twists this is law C. Differences in the files:

* `classes` holds signed integer charges on screen plaquettes (the holonomy of the offsets); cut files add
  `screen_charge` int8, and `screen_cls` is the charge mod 6 for display;
* the catalogue column `cls` holds the charge as two's-complement uint8 (read it with `.view(numpy.int8)`);
* checkpoints add `offsets`; the run directory adds `initial_offsets.npy` and `final_offsets.npy`;
* in record files `contrast` is the transported contrast and `twist` is the applied unit twist (`+1` or `-1`);
* `totals.balanced` and `unsettled_ports` have no meaning; `V` is the plain sum of squared loads;
* `preparation.offsets` states the prepared offsets (`all zero` or `iid:<seed>`: independent glue offsets in
  `{-1, 0, +1}` from `numpy.random.default_rng(seed).integers(-1, 2)`).

## 13. Related runs

`oph_defects/closed_loop.py` writes a separate family (declared extension): law C with transported
contrast, where every seam carries an integer offset `b_e` and compares `x_a` with `x_b + b_e`.
Each run directory holds `offsets.npy`, `arrays.npz` and `summary.json`; the module docstring
defines the fields.
