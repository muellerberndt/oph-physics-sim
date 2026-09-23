# Exact federation: confluence and the slow band (lane L1)

Module `oph_exact/federation.py`; receipt
`data/exact/federation_canonical_mean_receipt.json`; independent verifier
`oph_exact/verify_federation_independent.py`; tests
`tests/test_exact_federation.py`. Build with
`.venv/bin/python -m oph_exact.federation --write`, check staleness with
`--check`, run the scale demonstration with
`.venv/bin/python -m oph_exact.federation --level 5 --gluing port_pair --schedules 4 --out runs/exact_federation_L5`.

The lane runs the declared twelve-port carrier exactly inside a federation:
`N = 20 * 4^L` carriers on the cells of the geodesic icosahedral tower, the
canonical seam-mean law on every carrier's thirty seams, and a declared
inter-carrier gluing. It answers two questions at rungs `L = 0, 1, 2, 3`
(20, 80, 320, 1280 carriers): does every asynchronous schedule end at the same
consistent quotient (confluence), and does the rank-three slow band of the
single carrier survive the declared gluing (the slow band).

## What is exact

- The carrier: the oriented `(12, 30, 20)` incidence, the seam-mean law
  `E_e = I - b_e b_e^T / 2`, the repair mean `T = I - L_ico/60`, the exact
  `Q(sqrt 5)` band projectors and the intrinsic Gram `G = 4 P_slow`
  (`oph_exact/carrier.py`).
- The synchronous expectation operator `T_fed = I - L_fed/D` with
  `D = 2|S|/N` (one attempt per carrier per tick). On the isolated mode
  `D = 60` and every carrier block equals `carrier.repair_mean_exact()`
  entry for entry; the receipt asserts this with rational arithmetic. Per
  attempt the expectation operator is `I - L_fed/(2|S|)`; the per-carrier
  clock rescales time and leaves every dimensionless ratio fixed.
- Descent. The descent functional is `V(x) = sum_p x_p^2`, the flagship's
  quadratic descent functional
  (`from_observer_consensus_to_standard_physics.tex`, lines 889-895, where a
  conservative unit transfer across a seam with mismatch `d >= 2` lowers `V`
  by exactly `2(d - 1)`). For the real seam-mean law each move is an
  orthogonal projection, so `V` drops by exactly `(x_i - x_j)^2 / 2` on
  every non-wait move, and the centered norm `V - V_min = sum_p (x_p -
  mean_c)^2` drops by the same amount because the component total is
  conserved. The receipt reports `V` and the centered norm at the start and
  at termination, zero strict-descent violations on every schedule, and on
  the exact rungs the global ledger identity `V_0 - V_end = (1/2) sum d^2`
  over every move in dyadic integer arithmetic.
- Integer law. A nearest-agreement move with mismatch `|d| >= 2` is the
  composition of `floor(|d|/2)` unit transfers with mismatches `|d|, |d| - 2,
  ...`; each lowers `V` by `2(d_k - 1)`, so the move lowers `V` by
  `(d^2 - (d mod 2)) / 2`. Every descent move of every schedule is checked
  against this identity; the receipt reports the number of unit transfers
  and zero identity violations.
- The termination potential `Phi(x) = sum_seams (x_i - x_j)^2`. `Phi = 0`
  is consensus. `Phi` is not monotone along single moves:
  `Phi(E_e x) - Phi(x) = -d ((deg_i x_i - N_i) - (deg_j x_j - N_j)) + d^2
  (deg_i + deg_j + 2) / 4` with `N` the neighbour sums, and the receipt
  carries an exact single-carrier witness (`Phi` 128 to 136 while `V` drops
  60 to 58) together with, for every schedule, the number of accepted moves
  that raise `Phi` and the first such move with its pre-move readings.
- The terminal quotient. Every single seam move fixes the component-mean
  projection `Pi` (checked exactly for every seam of the small rungs), so
  every schedule shares one limit. Disjoint moves commute exactly.
  Conflicting single moves have no one-step diamond in general; their
  conflict component is joined exactly by the aggregate three-port mean,
  which is the canonical aggregate payload of the flagship's transactional
  diamond (`prop:transactional-diamond`, clause (ii)), and both orders have
  the same terminal quotient.
- Exact dyadic runs on `L = 0, 1`: readings `x_p = n_p / 2^K` with the
  precision raised on demand, the parity of every endpoint total asserted,
  exact conservation of every component total, the exact decision
  `Phi <= 10^-18` at a sweep end, the exact deviation from the component
  mean at termination, and the float companion's deviation from the exact
  state (below `2e-15`).
- Integer termination: every schedule reaches the balanced class (every
  reading in `{q, q+1}` per component, equivalently `V = V_min`), which is
  one orbit of the neutral unit swaps; the exact attempt index is reported.
  The canonical quotient is the multiset of readings per component, so the
  two odd-tie placements are identified; sixteen schedules give one hash on
  every rung.

## What is declared

- The gluing. `port_pair` is the production convention of
  `oph_fpe.core.screen_ports.assign_echosahedral_ports`: cells are adjacent
  through the cell-dual graph of `geodesic_icosahedral_patch_arrays(level,
  patch_basis="cells")` (three edge neighbours per cell); each dual-edge
  endpoint is routed to the port whose icosahedral template direction, read
  in the cell's local tangent frame (`tangent_x = normalize(reference x
  normal)` with reference `(0, 0, 1)`, or `(1, 0, 0)` when `|normal_z| >
  0.9`; `tangent_y = normal x tangent_x`), has the largest float32 inner
  product with the unit vector from the cell centre to the neighbour centre;
  duplicate choices inside one cell are repaired by an exact
  maximum-alignment assignment. Each routed pair is one inter-carrier seam,
  so three of the twelve ports of every carrier are glued. Because the
  neighbour directions are tangent, the convention only uses the ports
  `0, 1, 2, 3, 8, 10`; the port-usage histogram and the fraction of
  antipodally consistent pairs are recorded per rung. The source-derived
  gluing is the open (M1) item and is work in progress; the receipt measures
  what this convention does.
- The initial loads: seeded integers in `{0, ..., 5}`, one field per level
  shared by both gluings, `numpy.random.default_rng(20260909 + level)`.
- The schedule: uniform over all seams; one sweep is `|S|` attempts drawn
  as `default_rng(seed).integers(0, |S|, size=|S|)`; the integer law draws
  `integers(0, 2, size=|S|)` tie coins after each sweep's seam draw. Seeds
  are `909000 + 1000 level + 100 gluing_index + schedule_index`.
- Termination thresholds: float `Phi < 1e-18` at a sweep end; exact
  `Phi <= 10^-18`; integer `V = V_min`.
- The terminal canonicalizer of the mean law: the component-lattice snap
  `q_p = round(m_c x_p)` with `m_c` the port count of the component of `p`.
  The hash covers component sizes, labels and `q`; the maximal residual
  `|m_c x_p - q_p|` is reported (at most `6.2e-6`, on the glued `L = 3`
  rung whose single component has 15,360 ports, against the snapping
  margin `1/2`), and the exact rungs certify the snap
  in rational arithmetic. Every terminal hash equals the hash of the exact
  component means computed from the loads alone. The snap is a certificate
  only when it is unambiguous: a run whose maximal residual reaches the
  declared margin `0.25` (`LATTICE_SNAP_MARGIN`) reports
  `lattice_snap_unambiguous = false` and no terminal hash, and the schedule
  aggregate counts it under `ambiguous_terminal_count`. A budgeted float
  run on a large glued rung therefore records its residual and its deviation
  from the component mean instead of a hash.
- Portability of the gluing identity: the port pairs digest
  `port_pairs_sha256` is reproduced on arm64 (Accelerate) and x86 (OpenBLAS)
  at levels three to six, and equals the level-six archive's pin. The
  `local_frame_hash` (tangent frames rounded to fifteen decimals) is
  platform-specific and differs on x86 at every level; it is metadata, not
  an identity that any verifier checks.
- The kernel readout: probe port `p` of carrier `c` with a unit impulse,
  apply `T_fed` `2n` times, read carrier `c` (column `p` of `R_n`);
  `C_n = Q R_n Q`, `K_n = 12 C_n / tr C_n`, `n` in `{1, 5, 30, 100, 300}`.
  The centered impulses `R_n Q` are propagated directly with a common
  rescaling per step, which avoids cancelling the constant mode and keeps
  the readout accurate at `n = 300` (the isolated kernels agree with
  `carrier.normalized_response_kernel(n)` to `9e-16` at every `n`). Twelve
  carriers per rung are sampled, at least four of them touching the twelve
  pentagonal vertices of the tower.

## Engines

Three float engines apply identical IEEE operations: a pure Python
sequential loop (reference), a layered numpy engine that applies
pairwise-disjoint seams together in dependency order (two-hop dependency
layering, so every neighbour sum read for the `Phi` diagnostic is the
sequential one), and a native C kernel loaded through `ctypes`, compiled on
first use with `-ffp-contract=off`. Their states, wait counts and `Phi`
diagnostics are bit-identical; only the descent ledger differs in summation
order, and the receipt compares it against `V_0 - V_end` at `1e-9` relative.
The receipt is therefore engine independent. The native kernel runs at
about 2 ns per move; the reference loop at about 0.2 microseconds without
the diagnostic and about 1 microsecond with it. With the native kernel and
eight worker processes the receipt builds in a few minutes; without a C
compiler the same bytes are produced by the numpy and Python engines in
substantially more time (the glued `L = 3` rung dominates).

## Numbers

Receipt: eight rung blocks (`L = 0..3`, `isolated` and `port_pair`), sixteen
schedules each for the float mean law and the integer law, sixteen exact
dyadic schedules on `L = 0, 1`, all six verdict flags true. Build time with
the native kernel and eight workers: 568 s wall on a ten-core Mac shared with
another session (`runs/exact_federation_receipt/timing.json`); the exact
glued `L = 1` schedules (99 s each) and the glued `L = 3` float schedules
(116 s each for `2.2e9` attempts, about `1.9e7` attempts per second) dominate.

Federations (seams, components, port-graph gap `lambda_2`, attempts per e-fold of the slowest mode):

| rung | carriers | seams (intra + inter) | components | `lambda_2` | attempts per e-fold |
|-|-|-|-|-|-|
| L0/isolated | 20 | 600 (600 + 0) | 20 | 2.76393 | 434.164 |
| L0/port_pair | 20 | 630 (600 + 30) | 1 | 0.0441425 | 28543.9 |
| L1/isolated | 80 | 2400 (2400 + 0) | 80 | 2.76393 | 1736.66 |
| L1/port_pair | 80 | 2520 (2400 + 120) | 1 | 0.0103158 | 4.89e+05 |
| L2/isolated | 320 | 9600 (9600 + 0) | 320 | 2.76393 | 6946.63 |
| L2/port_pair | 320 | 10080 (9600 + 480) | 1 | 0.0025546 | 7.89e+06 |
| L3/isolated | 1280 | 38400 (38400 + 0) | 1280 | 2.76393 | 27786.5 |
| L3/port_pair | 1280 | 40320 (38400 + 1920) | 1 | 0.000638 | 1.26e+08 |

Mean law, float, sixteen schedules per rung (attempts to `Phi < 1e-18`, waits, moves raising `Phi` as a fraction of non-wait moves, terminal hashes):

| rung | attempts min / median / max | waits median | `Phi`-raising fraction median | max deviation from component mean | centered norm terminal max | unique hashes | equals expected |
|-|-|-|-|-|-|-|-|
| L0/isolated | 12000 / 13200 / 14400 | 1473.5 | 0.2353 | 3.23e-10 | 2.35e-19 | 1 | True |
| L0/port_pair | 5.73e+05 / 5.84e+05 / 5.93e+05 | 62432.5 | 0.39805 | 3.83e-10 | 1.3e-17 | 1 | True |
| L1/isolated | 52800 / 55200 / 57600 | 6316.5 | 0.23525 | 2.85e-10 | 2.03e-19 | 1 | True |
| L1/port_pair | 9.63e+06 / 9.66e+06 / 9.71e+06 | 1.03e+06 | 0.41075 | 3.93e-10 | 5.32e-17 | 1 | True |
| L2/isolated | 2.21e+05 / 2.21e+05 / 2.3e+05 | 25757.5 | 0.2341 | 2.8e-10 | 2.25e-19 | 1 | True |
| L2/port_pair | 1.45e+08 / 1.45e+08 / 1.46e+08 | 1.55e+07 | 0.4175 | 3.97e-10 | 2.05e-16 | 1 | True |
| L3/isolated | 9.22e+05 / 9.22e+05 / 9.98e+05 | 1.07e+05 | 0.2343 | 1.83e-10 | 2.18e-19 | 1 | True |
| L3/port_pair | 2.22e+09 / 2.22e+09 / 2.23e+09 | 2.38e+08 | 0.4218 | 4.01e-10 | 8.1e-16 | 1 | True |

Mean law, exact dyadic arithmetic, sixteen schedules (`L = 0, 1`):

| rung | attempts median | precision bits max | descent identity | global ledger | conservation | max deviation from component mean | float vs exact | unique hashes |
|-|-|-|-|-|-|-|-|-|
| L0/isolated | 13200 | 1024 | every move: True | True | True | 3.23e-10 | 1.78e-15 | 1 |
| L0/port_pair | 5.84e+05 | 12288 | every move: True | True | True | 3.83e-10 | 2.22e-15 | 1 |
| L1/isolated | 55200 | 1024 | every move: True | True | True | 2.85e-10 | 2.22e-15 | 1 |
| L1/port_pair | 9.66e+06 | 50176 | every 4096th move (33701 checked): True | None | True | 3.93e-10 | 3.11e-15 | 1 |

Integer law, sixteen schedules (attempts to the balanced class, descents / swaps / waits medians, unit transfers, odd-tie seams at termination):

| rung | attempts min / median / max | descents | swaps | waits | unit transfers median | identity violations | odd-tie seams min / median / max | unique quotient hashes |
|-|-|-|-|-|-|-|-|-|
| L0/isolated | 2474 / 3694 / 7343 | 195.5 | 719.5 | 3000.5 | 218.5 | 0 | 171 / 183 / 189 | 1 |
| L0/port_pair | 3556 / 5753 / 10023 | 205 | 1197.5 | 4853 | 228.5 | 0 | 212 / 259 / 275 | 1 |
| L1/isolated | 9685 / 16974.5 / 30400 | 746 | 3554.5 | 13727.5 | 824.5 | 0 | 842 / 878 / 896 | 1 |
| L1/port_pair | 30543 / 43398 / 60442 | 806.5 | 9215 | 35344 | 887 | 0 | 1036 / 1080.5 / 1154 | 1 |
| L2/isolated | 60976 / 91190.5 / 1.81e+05 | 3003 | 18811 | 74173.5 | 3348.5 | 0 | 3437 / 3526 / 3555 | 1 |
| L2/port_pair | 1.8e+05 / 2.41e+05 / 3.11e+05 | 3215 | 54458.5 | 1.89e+05 | 3568 | 0 | 4534 / 4731.5 / 4889 | 1 |
| L3/isolated | 3.7e+05 / 6.06e+05 / 9.52e+05 | 11908 | 1.17e+05 | 4.86e+05 | 13241.5 | 0 | 13934 / 14010 / 14102 | 1 |
| L3/port_pair | 1.15e+06 / 1.35e+06 / 2.08e+06 | 13126 | 2.98e+05 | 1.06e+06 | 14455.5 | 0 | 18501 / 18914.5 / 19503 | 1 |

Local confluence (exact rational arithmetic, `L = 0, 1`): disjoint pairs commuting, conflicting pairs with a one-step diamond, conflicting pairs joined by the aggregate three-port mean, single moves fixing the terminal quotient:

| rung | disjoint commute | one-step diamond | aggregate join | terminal join | moves fixing `Pi` |
|-|-|-|-|-|-|
| L0/isolated | 64 / 64 | 2 / 64 | 64 / 64 | 64 / 64 | 600 / 600 |
| L0/port_pair | 64 / 64 | 3 / 64 | 64 / 64 | 64 / 64 | 630 / 630 |
| L1/isolated | 64 / 64 | 3 / 64 | 64 / 64 | 64 / 64 | 2400 / 2400 |
| L1/port_pair | 64 / 64 | 2 / 64 | 64 / 64 | 64 / 64 | 2520 / 2520 |

Slow band. Isolated mode: `K_n` equals `carrier.normalized_response_kernel(n)` within `1e-12` at every `n` on every rung (measured deviations below `1e-14`), and `K_300` equals the intrinsic Gram (eigenvalues 4, 4, 4, 0, ...). Glued mode (`port_pair`), slow-band share `tr(P_slow K_n P_slow) / tr K_n` over the twelve sampled carriers (min / median / max) and the median top-four eigenvalues:

| rung | n = 1 | n = 5 | n = 30 | n = 100 | n = 300 | top four at n = 300 (median) |
|-|-|-|-|-|-|-|
| L0/isolated | 0.3000 / 0.3000 / 0.3000 | 0.4211 / 0.4211 / 0.4211 | 0.9449 / 0.9449 / 0.9449 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 4, 4, 4, 0 |
| L0/port_pair | 0.2987 / 0.2987 / 0.2987 | 0.4132 / 0.4132 / 0.4132 | 0.9061 / 0.9063 / 0.9091 | 0.5913 / 0.5954 / 0.6837 | 0.6371 / 0.6405 / 0.6734 | 4.846, 4.732, 2.431, 0.0002176 |
| L1/isolated | 0.3000 / 0.3000 / 0.3000 | 0.4211 / 0.4211 / 0.4211 | 0.9449 / 0.9449 / 0.9449 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 4, 4, 4, 0 |
| L1/port_pair | 0.2987 / 0.2987 / 0.2987 | 0.4132 / 0.4132 / 0.4132 | 0.9061 / 0.9063 / 0.9072 | 0.5912 / 0.5953 / 0.6071 | 0.6381 / 0.6407 / 0.6418 | 4.875, 4.747, 2.365, 0.0002665 |
| L2/isolated | 0.3000 / 0.3000 / 0.3000 | 0.4211 / 0.4211 / 0.4211 | 0.9449 / 0.9449 / 0.9449 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 4, 4, 4, 0 |
| L2/port_pair | 0.2987 / 0.2987 / 0.2987 | 0.4132 / 0.4132 / 0.4132 | 0.9061 / 0.9061 / 0.9076 | 0.5933 / 0.5954 / 0.6052 | 0.6335 / 0.6416 / 0.6420 | 4.833, 4.801, 2.348, 0.0002569 |
| L3/isolated | 0.3000 / 0.3000 / 0.3000 | 0.4211 / 0.4211 / 0.4211 | 0.9449 / 0.9449 / 0.9449 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 4, 4, 4, 0 |
| L3/port_pair | 0.2987 / 0.2987 / 0.2987 | 0.4132 / 0.4132 / 0.4132 | 0.9061 / 0.9061 / 0.9075 | 0.5933 / 0.5954 / 0.6069 | 0.6376 / 0.6419 / 0.6420 | 4.826, 4.823, 2.347, 0.0002565 |

Reference: the isotropic boundary Green's pattern of the glued ports has slow-band share 0.6669 (band weights `3/mu_s^2 : 5/mu_m^2 : 3/mu_f^2`).

`Phi` counterexample from the run (`L0/port_pair`, first schedule): attempt 4, seam 508 on ports [200, 201], readings 1 and 0, neighbour sums 11 and 5, degrees 6 and 5: `Phi` rises by 3.25 while `V` drops by 0.5.

Reading. Every schedule on every rung terminates at the same terminal hash,
which equals the hash of the exact component means computed from the loads
alone, with zero strict-descent violations and zero unit-transfer identity
violations. The seam sum `Phi` rises on 23 percent (isolated) to 42 percent
(glued) of the accepted non-wait moves while `V` drops on every one of
them. Consensus time under the production gluing scales as `N^2` attempts
(`5.8e5`, `9.7e6`, `1.45e8`, `2.2e9` for `N = 20, 80, 320, 1280`; the
port-graph gap scales as `lambda_2 = 0.88/N` and one sweep is `|S| = 31.5 N`
attempts), so a sphere of `N` carriers glued through three ports each needs
of order `N` sweeps to agree to `1e-18`; the isolated control needs a
constant number of sweeps. The integer law reaches its balanced class in
tens of sweeps on every rung, with the terminal states differing as vectors
(different odd-tie placements) and agreeing as multisets.

Slow band. On the isolated control the run kernel reproduces the carrier
kernel at every `n` to below `1e-15` and reaches the rank-three intrinsic
Gram at `n = 300` (deviation `2.8e-15`). Under the production gluing the
slow-band share follows the isolated curve up to `n = 30` (0.906 against
0.945), then turns down to 0.595 at `n = 100` and settles at 0.64 at
`n = 300` on every rung, with the median top-four eigenvalues about
`4.8, 4.8, 2.35, 3e-4`: the late-time centered readback is a rank-three
pattern again, spanned by the carrier Green's columns of the three glued
ports rather than by the slow band, and its share sits at the isotropic
Green's reference 0.667 up to the anisotropy of the actual inflow. The
measurement therefore reads: the rank-three slow band of the single carrier
survives this gluing at intermediate response depth and is replaced at
large depth by the boundary pattern of the glued ports; the answer is a
property of the declared three-port convention, and the deviation from
`4 P_slow` at `n = 300` is 2.3 in the maximum entry on every glued rung.

Scale demonstration (`runs/exact_federation_L5/summary.json`, gitignored):
`--level 5 --gluing port_pair --schedules 4` runs 20,480 carriers, 245,760
ports and 645,120 seams (30,720 inter-carrier) in 259.5 s in-run with the
native kernel and four workers (build 1.9 s, sparse shift-invert spectrum
63.9 s, schedules 40.7 s, eight-carrier kernels 153.0 s; 11.5 min wall while
sharing the cores with other jobs). The integer law reaches its balanced
class in 53 to 59 sweeps (`3.4e7` to `3.8e7` attempts), one quotient hash
equal to the expected multiset hash, zero unit-transfer identity violations,
about `3.1e5` odd-tie seams. The float law runs a declared budget of 256
sweeps (`1.65e8` attempts per schedule, about `5e6` attempts per second per
worker under contention): `Phi` falls from `3.8e6` to 4.6, the maximal
deviation from the component mean is 0.19, and 41 percent of the non-wait
moves raise `Phi` while `V` drops on every one; with `lambda_2 = 3.98e-5`
(`0.82/N`) the extrapolated cost of `Phi < 1e-18` is `7e11` attempts, the
`N^2` law of the rungs above, so the four float schedules end at four
distinct snapped states and the run reports them as such. The glued kernel
curve of eight sampled carriers (four pentagonal) repeats the rungs above:
slow-band share 0.299, 0.413, 0.906, 0.595, 0.642 at `n = 1, 5, 30, 100,
300`, median top four at `n = 300` about `4.83, 4.80, 2.36, 2.6e-4`.

## Production-law note

The finite-consensus receipt of the 64k production run
(`runs/e6_64k_dense_20260820/finite_consensus_replay_report.json`) is false
for a different law: the `bw_array` overwrite kernel
(`port_left <- g_ij * port_right` or `port_right <- inverse(g_ij) *
port_left`, the branch chosen by an independent Bernoulli(1/2) per selected
edge, plus a sector-link mutation with probability 0.08). Its structural
witness (`exact_endpoint_branch_structural_confluence_v1`: at a shared node
of degree twelve an unchanged incident edge fixes the node frame, so
repairing the left endpoint and repairing the right endpoint leave two
distinct local-frame quotient orbits; two terminal quotient hashes) is a
property of that law, which violates clause (ii) of the transactional local
diamond by construction. The canonical seam-mean law of this receipt is
symmetric between the two sides of a seam, conservative and idempotent; its
terminal quotient is the component mean on every schedule, and its
nonconfluence count is zero on every rung.

## Claim boundary

Supplied: the carrier incidence, the port-pair gluing convention, the loads,
the schedule law and the per-carrier clock normalization. Established: the
exact single-move descent of `V`, exact conservation, exact invariance of
the terminal quotient under every seam move, schedule-independent terminal
hashes on every rung, exact integer-law termination modulo the odd-tie
orbit, and the per-carrier normalized response kernel from the run. Not
claimed: physical position, physical length, cofinal gluing, the refinement
limit, the continuum limit and any field attachment. The slow-band numbers
under gluing are measurements of a declared convention; the source-derived
gluing and the refinement limit of the glued slow band across levels are
work in progress.
