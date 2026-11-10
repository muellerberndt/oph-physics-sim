# Run data schema: `oph.defects.federation_run.v1`

A run directory written by `python -m oph_defects.federation_run` holds the
settled fixed point of one universe, exact checkpoints from which any
interval of its history replays bit for bit, dense per-attempt records for
chosen sweeps, defect snapshots and the complete defect catalogue. All
binary files are little-endian; all arrays are C order. `oph_defects/data.py`
reads every file and performs the replay.

## The universe

The carrier is the exact federation of `oph_exact.federation` at `level`:
`N = 20 * 4^level` twelve-port carriers on the cells of the geodesic
icosahedral tower, glued by the declared `port_pair` convention (three ports
of every carrier meet a neighbour). Port `p` of carrier `c` is global port
`12 c + p`. Seams are indexed intra first (seam `s < 30 N` is carrier
`s // 30`, template seam `s % 30` of `oph_exact.carrier.seams()`), then the
inter-carrier seams in dual-edge order; every seam has an ordered pair
`(a, b)` of global ports (`fed.seam_a`, `fed.seam_b`). Faces are indexed
intra first (`20 c + f`, template face `f` of `ORIENTED_BASE_FACES`), then
the screen vertices of the tower mesh (`20 N + v`). Rebuild all of it with
`oph_defects.federation_geometry.build(level)`; it is deterministic.

Observers are the carriers. Time is the attempt counter: one sweep is
`|S|` attempts, one per seam on average; attempts are drawn uniformly over
seams with `numpy.random.default_rng(seed)` (seed `909000 + 1000 level +
schedule`), and the tie coins are drawn after each sweep's seam draw, both
exactly as `oph_exact.federation.run_integer_law`. Law C on an attempt at
seam `(a, b)` with loads `x_a, x_b`: total `t = x_a + x_b`; new loads
`(ceil(t/2), floor(t/2))` when the coin is 1 and `(floor(t/2), ceil(t/2))`
when it is 0; equal loads hold. The current of the attempt is `x_a - x_a'`,
the units moved from `a` to `b` (0 for a hold).

The transport register is one Z6 element per seam, zero at the start.
Every attempt with nonzero current proposes `delta = current mod 6` on its
seam; the two faces that traverse the seam change by `+delta` (the face
that traverses `a -> b`) and `-delta`. The proposal is accepted when the
face cost `cost[class]` does not rise, otherwise with probability
`exp(-dE / T_s)`, where `T_s = T0 *` the previous sweep's descent activity
(balancing moves with `|x_a - x_b| >= 2` per attempt) of the seam's carrier
(mean of both carriers for an inter seam). Acceptance randomness comes from
a splitmix64 stream seeded per chunk from `default_rng(seed + 1)`. The loads
never depend on the register.

## Files

| File | Content |
| --- | --- |
| `run.json` | `schema`, `config` (law, cost table, `T0`, temperature law, level, `sweeps`, `chunks_per_sweep`, `ticks = sweeps * chunks`, seed, schedule, snapshot cadences, kernel sha256, initial-load rule, status), `geometry` (counts, `intra_faces`, `screen_vertices`, template faces and vertices), `stats` (per sweep: `V` descent functional, `phi` mismatch potential, `unsettled` seams with unequal ends, `descents`, `swaps`, `reg_accepted`, `reg_rejected`, `births`, `moves`, `annihilations`, `fusions`, `splits`, `exchanges`, `T_mean`, `counts_intra[6]`, `counts_macro[6]` classes 0..5), `snapshots`, `checkpoints`, `record_windows`, `identities`, `first_sweep`, `parent_checkpoint`, runtime, numpy and python versions. |
| `final_loads.npy` | int8 `[12 N]`, the settled loads (balanced class: every reading in `{q, q+1}` per component). |
| `final_register.npy` | int8 `[S]`, the Z6 register at the end. |
| `checkpoints/ckpt_SSSSSS.npz` | at the start of sweep `S`: `loads` int8, `register` int8, `classes` int8 `[faces]`, `activity` float32 `[N]`, `face_id` int32 `[faces]` (identity on each face or -1), the catalogue columns `cat_*` and `cat_n`. |
| `checkpoints/ckpt_SSSSSS.json` | the two generator states at that sweep start (`law_rng`, `register_rng`) and the tracker counters. A checkpoint plus the code replays every later attempt exactly; `--resume` continues a run from it. |
| `records/cur_SSSSSS.npy.gz` | int8 `[S]`, the current of every attempt of sweep `S` in attempt order (dense record window). |
| `records/rng_SSSSSS.json` | generator states at the start of sweep `S`; regenerate its seam and coin draws from them. |
| `snapshots/macro_TTTTTT.bin.gz` | `Z6SX` + uint32 tick + uint32 n, then int32 faces `[n]`, uint8 classes `[n]`, int32 identities `[n]`, float32 xyz `[n, 3]`: every screen-tier defect at tick `T = sweep * chunks`. |
| `snapshots/intra_TTTTTT.bin.gz` | same layout for the internal tier (sparser cadence). |
| `catalogue.bin.gz` | `Z6CT` + uint32 n, then columns `cls` u8, `birth` i32, `death` i32 (-1 alive), `bface` i32, `lface` i32, `hops` i32, `partner` i32, `p1` i32, `p2` i32, `cause` u8 (0 alive, 1 annihilation, 2 fusion, 3 split, 4 exchange), `c1` i32, `c2` i32. Ticks are chunk indices (`sweep * chunks + chunk`). |
| `catalogue_xyz.bin.gz` | `Z6XY` + uint32 n, float32 birth xyz `[n, 3]`, float32 last xyz `[n, 3]`. |
| `progress.log` / `*.log` | the printed sweep lines. |

## Reading observer histories

An observer's history is the ordered list of records on its seams. With
`data.Run(path)`: `observer_tape(carrier, sweeps)` returns rows
`(sweep, attempt, seam, coin, current)`; for sweeps inside a record window
the rows come from the stored currents, otherwise from an exact replay from
the nearest checkpoint. The attempt index orders every record globally;
two attempts commute when they share no port, so the causal order of a
history is the transitive closure of shared-port succession.

The settled fixed point is the balanced class, not a frozen configuration:
once `descents` reach zero, every later attempt is a hold or a tie swap,
and the swap records keep every observer's clock running. Modular-flow and
field-history reconstructions read those records on the observer's own
attempt count.

## Size at level 8 (1.31 M carriers)

Checkpoint about 60 MB compressed, dense record window about 10 MB per
sweep, screen snapshot under 1 MB, internal snapshot 10 to 30 MB,
catalogue 20 to 40 MB. A level-9 run is four times that, a level-10 run
sixteen times.
