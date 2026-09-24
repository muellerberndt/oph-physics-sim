# Exact carrier source net (lane A-real)

The OPH carrier stack realizes the source-net population of the r2039
construction at the finite level, with no external coordinates: exact
twelve-port carriers hold the integer source records as port loads, read
their own positions from those loads through the rank-three response, find
their metric neighbours from the readbacks, read each other for
`K_q = ceil(sqrt q)` rounds, and the provenance of those reads generates the
layered causal order on which the interior-diamond observations of lane L2
are made.  Levels `q = 5, 8, 13, 21, 34, 55, 89` (125 to 704,969 carriers).

Producer `oph_exact/carrier_source_net.py`, verifier
`oph_exact/verify_carrier_source_net_independent.py`, tests
`tests/test_exact_carrier_source_net.py`, receipt
`data/exact/carrier_source_net_receipt.json` (schema
`oph.exact.carrier-source-net.v1`), stored event logs
`data/exact/carrier_source_net_logs/q{5,8}_event_log.json.gz` (104 KB).

```
.venv/bin/python -m oph_exact.carrier_source_net --write --processes 6   # rebuild the receipt (about 3 min)
.venv/bin/python -m oph_exact.carrier_source_net --check --processes 6   # rebuild and compare byte for byte
.venv/bin/python -m oph_exact.carrier_source_net --repin                 # refresh the file pins
.venv/bin/python -m oph_exact.verify_carrier_source_net_independent       # about 1 s
.venv/bin/python -m pytest tests/test_exact_carrier_source_net.py -q     # 15 tests, about 14 s
```

## 1. What the carriers do themselves, what is supplied

Produced by the carriers, from their own records:

* positions: `x(b) = 2 P_slow N(b)`, the readback of the carrier's own loads
  through the exact rank-three response of the flagship theorem
  `thm:rank-three` (`oph_exact/carrier.py`: `slow_band_projector`,
  `intrinsic_gram = 4 P_slow` with entries `{1, 1/sqrt5, -1/sqrt5, -1}` by
  port distance);
* the neighbour graph: `||x(b) - x(b')|| <= a_q`, `a_q = L/sqrt q`, decided
  by each carrier from the readbacks alone, exactly in `Q(sqrt5)`;
* the reads: `K_q` rounds of complete-neighbour reads, every read an
  authenticated read-after-write event (reader, register, version) and every
  write versioned;
* the provenance order: the precedence relation generated from the log by the
  read-after-write rule and its transitive closure, with no declared parent
  lists and no layer labels as input.

Supplied, as in the paper (`paper/tex_fragments/SOURCE_NET_CAUSAL_LIMIT.tex`,
paragraph "A conservative-record family with a raw count law"): the integer
records `z(b)`, the load placement, the read law `q_i(0) = i + 1`,
`q_i(j) = 1 + sum` (RER `source_net_causet.trace`), the round rule (one event
per carrier and round, round `j` reads version `j` and writes version
`j + 1`), the edge radius `a_q`, and the fixed population.

Not claimed: selection of the population by native repair, a physical clock,
spacetime, the continuum limit, a population that moves during the reads.

## 2. Records as loads, readback, the metric identity

**Records.** For `b = (b1, b2, b3)` in `[0, q)^3` and `m_i = -floor(b_i phi)`,
`z(b) = (b2 - m1, b2 + m1, b3 - m2, b3 + m2, b1 - m3, b1 + m3)` (RER
`source_control`), an even-sum six-vector on the six antipodal axes (its
coordinate sum is `2(b1 + b2 + b3)`); the current section is RER
`source_currents` and its `l1` norm is the primitive seam-word length.  The
record digests equal RER's and lane L2's at every level.

**Load placement (declared).** Axis `k` of `z(b)` sits on the antipodal port
pair `(p_k, -p_k)` with `p_k = (0, 1, 4, 5, 8, 9)[k]`, the positive port basis
of the port-Gram completion bridge receipt
(`data/repair_closure/port_gram_completion_bridge_receipt.json`,
`antipodal_relations [[0,3],[1,2],[4,7],[5,6],[8,11],[9,10]]`, the same
pairing `carrier.antipode()` derives from graph distance three).  The split is
one-sided: `N_{p_k} = max(z_k, 0)`, `N_{-p_k} = max(-z_k, 0)`.  The
antipodal-odd quotient `Z^12 / {x_p = x_{-p}} = Z^6`,
`(N_{p_k} - N_{-p_k})_k`, returns `z(b)` exactly at every site of every level
(`load_placement.round_trip_exact`), the loads are nonnegative and their
total is `|z(b)|_1`.

**Readback.** `P_slow` is antipodal-odd, so `2 P_slow e_{-p} = -2 P_slow e_p`
and the readback of any placement with odd part `z` is the signed generator
sum `x(b) = sum_k z_k v_{p_k}`, `v_p = 2 P_slow e_p`, `<v_p, v_q> = G_pq`
(the "labeled generator" of the bridge receipt).  Checked in float at every
site (`float_readback_equals_signed_generator_sum`) and in exact `Q(sqrt5)`
on a sample of ten or eleven carriers per level, including the corner sites
and the centre, with `carrier.slow_band_projector_exact()`
(`exact_sample.exact_gram_identity`).  The antipodal-even part of a load
change leaves the readback fixed (test
`test_readback_is_the_signed_generator_sum_and_only_the_odd_part_moves_it`).

**The metric identity (the scale factor).** With `G6` the Gram of the six
positive ports (`1` on the diagonal, `+1/sqrt5` at port distance one,
`-1/sqrt5` at distance two; sign pattern in the receipt's
`carrier.gram_sign_pattern`),

```
||x(b) - x(b')||^2 = (z(b) - z(b'))^T G6 (z(b) - z(b'))
                   = L^2 * sum_i (xi_{b_i} - xi_{b'_i})^2
                   = ||s(b) - s(b')||^2,     s(b) = L (xi_b1, xi_b2, xi_b3).
```

The scale relative to the paper's position `s(b)` is exactly `1`.  The scale
relative to lane L2's dimensionless squared distances `(A + B phi)` (units of
`L^2`) is exactly `L^2 = 12/5 - (4/5) phi = 2 - (2/5) sqrt5`, the element
`("2", "-2/5")` of `Q(sqrt5)`.  Reason: the paper's source axes are the unit
vectors of the bridge receipt's raw generators `a_0 = (-1, phi, 0)`,
`a_1 = (1, phi, 0)`, `a_2 = (0, -1, phi)`, `a_3 = (0, 1, phi)`,
`a_4 = (phi, 0, -1)`, `a_5 = (phi, 0, 1)` (`|a_k|^2 = phi + 2`,
`a_k . a_l / (phi + 2) = G6_kl` exactly, `generator_frame_is_isometric_for_gram6`),
and `(1/2) sum_k z_k a_k = (xi_b1, xi_b2, xi_b3)` is an exact integer identity
in `Z[phi]` (`exact_position_equals_paper_contraction`), so
`s(b) = L (xi) = sum_k z_k a_k / |a_k|`.  The identity is verified in integer
form, `10 S2 + 2 X sqrt5 = 20 A + (8 B - 4 A) sqrt5` with
`S2 = sum dz_k^2`, `X = sum_{k != l} sigma_kl dz_k dz_l`, on all pairs at
`q <= 21` (7,750; 130,816; 2,412,306; 42,878,430 pairs) and on 1,000,000
seeded pairs at `q = 34`, with zero failures.  Discrepancy: none.

**Neighbour decision (exact).** `||x(b) - x(b')||^2 <= L^2/q` iff
`sign((5 q S2 - 10) + (q X + 2) sqrt5) <= 0`, the exact sign rule in the
`sqrt5` basis (`carrier.q5_sign`, vectorized in int64).  An orthonormal chart
of `range(P_slow)` is used only to propose candidates inside
`a_q (1 + 1e-9)` with a k-d tree; every candidate is decided exactly.  At
every level the candidate set equals the accepted set (zero rejected, zero
borderline candidates), and the neighbour digest (canonical JSON of the
sorted neighbour lists, waiting included) equals lane L2's
`neighbors_including_wait_sha256` and RER's:

| q | carriers | K | undirected edges | min/max neighbours (incl. wait) | digest equal to L2 / RER |
|:--|:--|:--|:--|:--|:--|
| 5 | 125 | 3 | 1,330 | 8 / 39 | yes / yes |
| 8 | 512 | 3 | 16,276 | 23 / 105 | yes / yes |
| 13 | 2,197 | 4 | 145,997 | 38 / 205 | yes / yes |
| 21 | 9,261 | 5 | 1,451,292 | 72 / 437 | yes / (no RER level) |
| 34 | 39,304 | 6 | 13,140,588 | 121 / 856 | yes / (no RER level) |
| 55 | 166,375 | 8 | 121,391,967 | 278 / 1767 | yes / (no RER level) |
| 89 | 704,969 | 10 | 1,091,925,058 | 501 / 3545 | yes / (no RER level) |

## 3. Layered reads and the log

Round 0 writes the seed `q_i(0) = i + 1` (`+1` at the centre in the
intervention run).  In round `j >= 1` every carrier reads version `j` of every
neighbour register (the same-site read included) and writes version `j + 1`
of its own register with value `1 + sum`.  Values are exact integers (87 bits
at `q = 34`).  The audit chain is RER's:
`chain = sha256(chain || canonical([[j, i], reads, [i, j + 1, [j, i], value]]))`
with `reads = [[r, j, [j - 1, r], value_read], ...]` in neighbour order; the
per-round layer digests are `sha256(canonical(values))`.

RER cross-check at `q = 5, 8, 13` against the pinned theory receipt
(`sha256 c0f790ad38...5681`): forward audit chain, forward layer digests and
sums, event and read counts, intervention audit chain, intervention layer
digests and sums, neighbour digest, record digest, intervention site and the
per-layer response supports all agree (`reads.rer_cross_check.all_agree`).

| q | events | authenticated reads | forward audit chain | intervention audit chain |
|:--|:--|:--|:--|:--|
| 5 | 500 | 8,355 | `35b50b2a...bde5c` (= RER) | `78a92371...de5be` (= RER) |
| 8 | 2,048 | 99,192 | `02a1c358...d4d9` (= RER) | `6bc330f3...1c72` (= RER) |
| 13 | 10,985 | 1,176,764 | `b409bd6f...94a3` (= RER) | `8f253c4f...be61` (= RER) |
| 21 | 55,566 | 14,559,225 | `901943975c17c307...` | `efc794275994af47...` |
| 34 | 275,128 | 157,922,880 | `c1e63647e1ec4191...` | `e78f6426bbbf37fb...` |
| 55 | 1,497,375 | 1,943,602,472 | `a3353b5e71ab0343...` | `c024282cb5f6316f...` |
| 89 | 7,754,659 | 21,845,550,850 | `00fbd424c8049da6...` | `c969c9df2c27aef8...` |

The complete logs of `q = 5` and `q = 8` are stored (canonical JSON, gzip,
`mtime 0`, uncompressed sha256 pinned in the receipt); the verifier derives
the order from the stored log alone as a second path.

## 4. Provenance-derived order

Rule (reimplemented from
`oph_fpe/bulk/source_derived_causal_order.py::generated_provenance_edges`,
which is not imported): every register version has exactly one writer; an
edge runs from the writer event of a version to every event that reads it; a
read of an unwritten version is an error (the seeds are the only events
without parents); no event reads its own committed version; event ids are the
ordinal positions in the log.  Layer labels are outputs: the derived
longest-path rank of every event is computed from the parent relation and
compared with the round it executed in.

Checks at every level (`provenance` block): one writer per version, every
read resolved, parents precede children in the log, derived rank equals the
round, the per-round read relation is identical across rounds and its digest
equals the neighbour digest, edge count `K (2E + n)`.  The future cone of the
centre's seed event and the past cone of its round-`K` event are grown from
the log records; their intersection is the centre interval:

| q | interval events | counts by round | strict pairs | equals L2's layered order (event-set digest, counts, pairs) |
|:--|:--|:--|:--|:--|
| 5 | 80 | 1, 39, 39, 1 | 968 | yes |
| 8 | 188 | 1, 93, 93, 1 | 4,244 | yes |
| 13 | 1,529 | 1, 179, 1,169, 179, 1 | 102,990 | yes |
| 21 | 6,482 | 1, 395, 2,845, 2,845, 395, 1 | 2,904,889 | yes |
| 34 | 32,265 | 1, 847, 5,761, 19,047, 5,761, 847, 1 | 47,881,819 | yes |
| 55 | 212,252 | 1, 1,656, 12,728, 42,442, 98,598, 42,442, 12,728, 1,656, 1 | 2,083,062,313 | yes |
| 89 | 1,081,730 | 1, 3,494, 27,069, 90,503, 213,535, 412,526, 213,535, 90,503, 27,069, 3,494, 1 | 55,108,871,450 | yes |

The event-set digest of the provenance interval equals the digest of the
layered order `(j, s) <= (j', t)` iff `d(s, t) <= j' - j` on the log-derived
relation and on lane L2's own site graph (`source_net.build_site_graph`,
rebuilt in the producer as the comparison target).  Strict pair counts are
exact (per-start searches on the common read relation, pruned to
`alpha_u <= K - alpha_s - m` at search layer `m`, the graph-distance form of
lane L2's cone pruning) and equal L2's `strict_pair_count` at every `k <= K`.

**Intervention.** The `+1` at the centre in round 0 changes exactly the
events of the future cone in every round (`intervention.equals_future_cone_all_rounds`);
the support digests equal lane L2's `reachable_ids_sha256` probes from the
centre and, at `q = 5, 8, 13`, RER's `support_ids_sha256`, support counts,
delta sums and delta maxima.  Support counts: `q = 21`: 1, 395, 2,845, 7,839,
9,247, 9,261; `q = 34`: 1, 847, 5,761, 19,047, 35,964, 39,279, 39,304; `q = 89`: 1, 3,494, 27,069, 90,503, 213,535, 412,526, 599,468, 690,084, 704,797, 704,969, 704,969.

## 5. Manifold readouts on the provenance order

Ordering fraction `2C/(N(N-1))` (target `1/10` in `1+3` dimensions), inverted
Myrheim-Meyer dimension, and the count clock between the `K`-layer and the
`floor(K/2)`-layer diamonds, all recomputed from the provenance order; every
value equals lane L2's (`manifold.all_intervals_equal_source_net`,
`count_clock.equals_source_net`).

| q | K | N | C | ordering fraction | MM dimension | count clock (K vs K/2) | model ratio |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 5 | 3 | 80 | 968 | 121/395 = 0.3063 | 2.634 | 2.515 | 3 |
| 8 | 3 | 188 | 4,244 | 2122/8789 = 0.2414 | 2.932 | 3.114 | 3 |
| 13 | 4 | 1,529 | 102,990 | 51495/584078 = 0.0882 | 4.149 | 1.705 | 2 |
| 21 | 5 | 6,482 | 2,904,889 | 2904889/21004921 = 0.1383 | 3.612 | 2.010 | 2.5 |
| 34 | 6 | 32,265 | 47,881,819 | 47881819/520498980 = 0.0920 | 4.099 | 2.088 | 2 |
| 55 | 8 | 212,252 | 2,083,062,313 | 2083062313/22525349626 = 0.0925 | 4.093 | 1.907 | 2 |
| 89 | 10 | 1,081,730 | 55,108,871,450 | 55108871450/585069355585 = 0.0942 | 4.071 | 2.051 | 2 |

Twelve-digit values, the rows for every `k < K`, and the distance to `1/10`
are in the receipt.  The even-`K` diamonds sit near `1/10` (0.088 at `q = 13`,
0.092 at `q = 34`); the odd-`K` rows carry lane L2's parity effect.  These are
the same finite diagnostics as lane L2, produced here from the reads.

## 6. Operation costs

Byte model (declared): site ids 4 bytes, versions 2 bytes, values in their
minimal unsigned big-endian length; a read record carries reader id, register
id, version and the value read; a write record carries register id, version
and the value written.  Reads per round are `2E + n`; the primitive seam-word
length of the population is RER's `sum_word_lengths` with bound `27(q - 1)`
per record.

| q | reads per round | total reads | total writes | total bytes | max value bytes (last round) | word length sum / max / bound |
|:--|:--|:--|:--|:--|:--|:--|
| 5 | 2,785 | 8,355 | 500 | 102,155 | 3 | 1,890 / 32 / 108 |
| 8 | 33,064 | 99,192 | 2,048 | 1,225,538 | 4 | 13,556 / 57 / 189 |
| 13 | 294,191 | 1,176,764 | 10,985 | 15,706,632 | 6 | 98,254 / 98 / 324 |
| 21 | 2,911,845 | 14,559,225 | 55,566 | 204,360,087 | 7 | 687,862 / 164 / 540 |
| 34 | 26,320,480 | 157,922,880 | 275,128 | 2,402,551,058 | 10 | 4,795,802 / 271 / 891 |
| 55 | 242,950,309 | 1,943,602,472 | 1,497,375 | 33,495,247,827 | 13 | 33,156,915 / 444 / 1458 |
| 89 | 2,184,555,085 | 21,845,550,850 | 7,754,659 | 424,060,324,906 | 18 | 228,612,164 / 724 / 2376 |

Per-round rows (reads, writes, read bytes, write bytes) are in
`operation_costs.per_round`.  The read traffic dominates: at `q = 34` the six
rounds move 2.40 GB of read records against 3.2 MB of writes, and the value
width grows by about one byte per round.

Wall-clock of the producer on the shared machine (6 workers for the pair
searches; everything else single process): `q = 5, 8, 13` under 5 s together,
`q = 21` 22 s, `q = 34` 142 s (neighbours 17 s, the two hash-chained traces
67 s, provenance 9 s, pair searches 48 s); total about 170 s.  On a 64-vCPU box with 62 workers: `q = 55` 1,106 s
(neighbours 113 s, traces 367 s, provenance 97 s, pair searches 529 s) and
`q = 89` 29,777 s (neighbours 1,028 s, traces 5,547 s, provenance 1,012 s,
pair searches 22,183 s over 412,526 starts).  The traces are sequential; the
neighbour search, the provenance pass and the cone passes stream the
neighbour table in bounded slices, so memory stays linear in the carriers.

## 7. Dynamic population

The population is held fixed during the reads, the paper's hypothesis of
keeping the full population at each layer.  A population whose records move
by canonical repairs during the reads is work in progress.  What changes:
every accepted seam repair changes the loads by the repair mean, so each
readback `x = 2 P_slow N` moves by `2 P_slow` of the load change (only the
antipodal-odd part of a repair moves the position; the even part leaves the
readback fixed), the neighbour decision `||x(b) - x(b')|| <= a_q` is
re-decided at every round from the moved readbacks, and the read relation,
hence the generated order, is time dependent: the per-round relations need
not coincide, the derived rank of an event is its position in a
round-dependent DAG, and the interval statistics have to be read on that DAG
rather than on one common site graph.

## 8. Claim boundary

Finite realization of the declared family by exact carriers.  The population,
the load placement, the read law, the round rule, one event per site and
round and the edge radius are supplied.  Nothing here selects the population
by native repair, identifies a physical clock or spacetime, or demonstrates
the continuum limit; the readouts are the finite diagnostics of lane L2,
produced here from the carriers' own reads.

## 9. Verification and pins

The verifier imports nothing from `oph_exact` or `oph_fpe`.  It rebuilds the
incidence from a copy of the twenty oriented faces, takes the slow band as the
numpy eigenprojector of the seam Laplacian and checks it against the exact
`Q(sqrt5)` matrix `G/4` (idempotent, trace three, `5 - sqrt5` eigenmatrix,
agreement to `1e-12`), rebuilds the records, the placement, the exact
readbacks on the receipt's sample, the metric identity on all pairs, the
neighbour lists by a dense all-pairs exact decision, the two traces with the
RER chain, the provenance order by a dictionary rule with explicit descendant
sets, the interval readouts, the intervention rows and the operation costs at
`q = 5` and `q = 8`, replays the stored logs on their own, compares the RER
digests with the pinned theory receipt when the checkout is present, and
checks the levels `q = 13, 21, 34` for internal consistency and against the
frozen lane L2 receipt (neighbour digests, shell counts, probe digests, strict
pair counts, fractions, dimensions, clocks).  Pins: producer, verifier, test
file, `oph_exact/carrier.py`, `oph_exact/source_net.py`, lane L2's receipt,
the two stored logs, and in the theory checkout
`code/causal_refinement/source_net_causet.py`, its receipt and
`paper/tex_fragments/SOURCE_NET_CAUSAL_LIMIT.tex`.  Lane L2's receipt is an
untracked file of a concurrent lane; a rewrite of it changes the pin and
calls for `--repin` (its compared values are deterministic).

Tests: load placement round trip (records and random six-vectors), readback
equals the signed generator sum and moves only with the odd part, readback
metric equals the source metric up to the global scale at `q = 5` (all pairs,
exact sample, frame isometry), neighbour digest equality at `q = 5` (and CSR
equality with lane L2's graph), RER trace digests at `q = 5`, provenance
order equals the layered order at `q = 5` with a brute-force descendant-set
recount of every diamond, intervention supports equal the future cone and
RER's supports, manifold readouts and costs at `q = 5`, frozen canonical
receipt free of wall-clock, the `q = 5` row rebuilds byte for byte, pins,
verifier accepts the receipt, verifier rejects sixteen mutations, verifier
import hygiene, strict loader.
