# Lean settlement engine for large glued federations

`oph_exact/federation_huge.py` settles the production gluing of the geodesic icosahedral
tower at levels where the reference lane (`oph_exact/federation.py`, `federation_archive.py`)
no longer fits: level 8 (1,310,720 carriers), level 9 (5,242,880) and level 10 (20,971,520
carriers, 251,658,240 ports, 660,602,880 seams). Every declared convention of the reference
lane is unchanged; what differs is representation.

## Conventions (identical to the reference lane)

- Carriers: the `20 * 4^L` cells of the tower, twelve ports each, thirty intra-carrier seams
  from `carrier.seams()`; the production gluing `assign_echosahedral_ports` on the cell-dual
  graph glues three ports per carrier (`local_frame_hash` is platform-specific; the
  port-pairs digest is the portable identity).
- Seam order: intra seams carrier-major (seam `s < 30N`: carrier `s // 30`, template
  `s % 30`), then inter seams in dual-edge order.
- Loads `default_rng(20260909 + level).integers(0, 6)`; schedule seeds
  `909000 + 1000 level + 100 + k` (production gluing index one).
- One sweep: `|S|` seam indices drawn i.i.d. with replacement
  (`integers(0, |S|, size=|S|, dtype=int64)`), then for the integer law `|S|` coins
  (`integers(0, 2, ...)`), applied in draw order.
- Integer nearest-agreement law (ceiling on the first endpoint when the coin is one);
  `V = sum x^2`; termination at the exact attempt at which `V` reaches the balanced-class
  minimum, the sweep in progress completed; terminal hash = component multiset.
- Mean law `E_e = I - b b^T / 2` under a sweep budget; the lattice-snap hash is withheld
  unless the residual is below `LATTICE_SNAP_MARGIN = 0.25`.
- Response kernels `K_n = 12 C_n / tr C_n` from the twelve centered impulses propagated
  `2n` steps with per-step rescale; sample cells from `kernel_sample_cells` (one cell per
  pentagonal vertex, then a seeded draw).

## Representation

- `int8` state for the integer law, `float64` for the mean law; no neighbour lists, no
  Laplacian; the inter-seam table as two contiguous `int32` port arrays read from
  memory-mapped `.npy` files in a geometry cache (`geometry --level L --cache DIR`).
- Draws taken in contiguous chunks of `2^24` from one `default_rng(seed)`; the chunked
  stream equals the single-call stream bit for bit (the test suite checks a size that
  crosses chunk boundaries), and the archive's per-sweep draw digests
  `[sha256(seq int64), sha256(coin int64)]` are accumulated incrementally.
- A C kernel (`-O2 -ffp-contract=off`) applies one sweep; it returns the index of the first
  move after which `V` equals the minimum and accumulates descents, swaps, waits, unit
  transfers and decrement-identity violations. The mean-law kernel accumulates the ledger.
- After every sweep: `V` ledger entry, move counts, a SHA-256 of the state; every eighth
  sweep a checkpoint (the last two retained); one process per schedule, a spawn pool of
  `--workers` processes; kernels in worker processes as well.
- Response kernels are propagated on the ball of graph radius `2n + 1` around the probed
  carrier. The operator moves one hop per step, so the readback at the probed carrier after
  `2n` steps is exactly the whole-federation readback; the level-six kernels of the archive
  are reproduced to the last bit.

## Validation

- Level six, schedule seed 915100, against
  `evidence/exact_federation_L6_canonical_20260909` in the research repository: quotient
  hash `f384744f…`, 62 sweeps, attempts to the balanced class 157,922,110, descents, swaps,
  waits, unit transfers, `V` ledger, all 62 per-sweep draw digests, the terminal SHA-256 and
  the terminal vector are equal. The schedule takes 3.3 s (the reference lane needs
  minutes). The kernels of five archive cells at n = 1, 5, 30, 100, 300 agree to 0.0.
- Level one, three seeds, integer and mean law: equal to `federation.run_integer_law` and
  `run_mean_law` on every compared field and the terminal state.
- `tests/test_exact_federation_huge.py`: the comparisons above at level one, chunked-draw
  equality, local-ball kernel equality, and mutation tests of the verifier.

## Results at levels 8, 9 and 10 (2026-09-24)

Sixteen integer schedules per level, seeds `909000 + 1000 L + 100 + k`, loads
`default_rng(20260909 + L)`; one budgeted mean schedule of 64 sweeps; 64 kernel cells at
`n = 1, 5, 30, 100` and four at `n = 300`. Every schedule terminates at the balanced class and
every level returns one quotient hash equal to the multiset fixed by the loads; the host
verification replayed every draw digest and the last sweeps of every schedule.

| L | carriers | seams | sweeps (min to max, mean) | law `4.3 log2 N - 9` | unit transfers per port | glued slow-band share medians at n = 1, 5, 30, 100, 300 | float `Phi` after 64 sweeps | seconds per schedule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 (archive) | 81,920 | 2,580,480 | 51 to 79, 60.7 | 61.2 | 0.9408 to 0.9417 | 0.2987, 0.4132, 0.9062, 0.5954, 0.6413 | 17.3 after 256 sweeps | |
| 8 | 1,310,720 | 41,287,680 | 69 to 89, 75.5 | 78.4 | 0.94154 to 0.94187 | 0.298669, 0.413158, 0.906125, 0.595353, 0.640923 | 2.41e8 to 5.03e3 | 165 to 205 |
| 9 | 5,242,880 | 165,150,720 | 79 to 108, 89.4 | 87.0 | 0.94146 to 0.94159 | 0.298669, 0.413158, 0.906125, 0.595353, 0.641611 | 9.63e8 to 2.01e4 | up to 1,088 |
| 10 | 20,971,520 | 660,602,880 | 84 to 100, 91.1 | 95.6 | 0.94152 to 0.94159 | 0.298669, 0.413158, 0.906185, 0.595353, 0.641611 | 3.85e9 to 8.06e4 | up to 4,313 |

The kernel shares are level-independent to six digits at `n <= 100` and to three at
`n = 300`: the per-carrier readout is a property of the neighbourhood, not of the level. The
mean law's residual after its budget is not a settlement; its lattice-snap hash is withheld
at every level. The settling texture readouts (`oph_exact/federation_texture.py`) add the
universal descent curve (contraction 0.822 per sweep at every level), the transport constant
1.2555 and the settling horizon (535 cells at L8, 621 at L9).

## Readback of the settled states (`oph_exact/federation_readback.py`)

The carrier's readback is the rank-three slow-band projection of its twelve loads. On the
settled terminal states (`data/exact/readback/readback_L{6,8,9,10}.json`) the slow-band energy
share is the isotropic value 3/11 before and after settling (0.2725 to 0.2729 at every level),
the per-carrier slow norm falls from 2.75 to 0.79 as the loads contract from `{0..5}` to
`{q, q+1}`, about 1.9 percent of carriers end with no slow component, and the terminal slow
and fast fields are uncorrelated with the initial ones (slopes and correlations below `10^-3`
at levels 6, 8, 9 and 10; the coarse slow-norm retention is zero at every scale). Settling keeps
the coarse load density above the settling horizon and forgets the per-carrier orientation
everywhere: the public record retains how much, not which ports. Over the eleven-dimensional
zero-sum port space the settled energy splits into the three bands of the seam Laplacian
(dimensions 3, 5, 3) as 0.27273, 0.45452, 0.27276 at level ten against the isotropic
3/11, 5/11, 3/11 = 0.272727, 0.454545, 0.272727, at every level from six to ten: the settled
record has no preferred direction in the port space, and refinement does not change the split.

## Independent verifier

`oph_exact/verify_federation_huge_independent.py RUN_DIR` shares nothing with the engine
but the carrier seam template and the canonical-JSON convention. It re-derives the cell
count, checks every port glued at most once, recomputes the components and the port-pairs
digest, recomputes the loads and from them the balanced minimum, the expected multiset
hash and the mean minimum; for every integer schedule it checks the terminal digest,
dtype and range, membership in the balanced class, `V`, conservation, the recomputed
multiset hash against the expected hash, the monotone ledger, every draw digest
(re-drawn from `default_rng(seed)` with single-call draws) and a layered numpy replay from
the last retained checkpoint through every intermediate state digest to the terminal
state; for the mean schedule the digest, `Phi`, `V`, the deviation, the lattice residual and
the withheld-hash rule; for the kernels symmetry, trace, the slow-band share, and a dense
recomputation on the local ball at the two smallest step counts; and the module pins.

## Running a level

    python3 -m oph_exact.federation_huge geometry --level 10 --cache CACHE
    python3 -m oph_exact.federation_huge run --level 10 --cache CACHE --out RUN \
        --schedules 16 --workers 17 --float-sweeps 64 --float-schedules 1 \
        --kernel-cells 64 --kernel-steps 1 5 30 100 --long-steps 300 --long-cells 4
    python3 -m oph_exact.verify_federation_huge_independent RUN --schedules 0 --kernel-sample 4

Memory per integer schedule at level ten is about four gigabytes (state, one sweep of
draws, the inter-seam arrays); the geometry cache of level ten is about 1.3 GB.
