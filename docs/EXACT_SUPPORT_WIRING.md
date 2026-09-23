# Full S2 support wiring: twelve ports, A5 carriers, canonical repair, provenance order (lane A-wire)

Module `oph_exact/support_wiring.py`; receipt
`data/exact/support_wiring_receipt.json`; stored log
`data/exact/support_wiring_logs/W12_L3_provenance_log.npz`; independent
verifier `oph_exact/verify_support_wiring_independent.py`; tests
`tests/test_exact_support_wiring.py`. Build with
`.venv/bin/python -m oph_exact.support_wiring --write --workers 6`, check
staleness with `--check`.

The lane puts every architectural element of the declared substrate into one
run: twelve-port A5 icosahedral carriers on the cells of the geodesic S2
screen at tower level `L`, all twelve ports of every carrier wired to the
twelve neighbouring cells, the canonical seam-mean repair law, the integer
nearest-agreement law with fixed protected totals, records as cumulative port
loads, positions as each carrier's own rank-three readback `x = 2 P_slow N`,
and the provenance order generated from the log of the reads by the
authenticated read-after-write rule. It answers one question: what does an
observer read from the provenance order of the reads, and how does that
compare with the record-metric route of the source net (lane L2,
`oph_exact/source_net.py`)?

## The wiring W12

Cells are the `20 * 4^L` spherical triangles of the geodesic icosahedral
tower (`oph_fpe/core/icosahedral.py`). Two cells are neighbours when they
share a mesh vertex: a cell whose three vertices all have degree six has
three edge-adjacent and nine vertex-adjacent neighbours, twelve in all; the
sixty cells touching one of the twelve pentagonal vertices (degree five) have
eleven. Every neighbour pair is one glued seam between one port of each cell.

Port assignment. The production geometric rule
`oph_fpe.core.screen_ports.assign_echosahedral_ports(left, right, N,
points=cell_centres)` accepts the twelve-neighbour graph at every level
(routing mode `icosahedral_directional_assignment`, zero overflow, distinct
ports per cell): each endpoint is routed to the port whose icosahedral
template direction, read in the cell's local tangent frame, has the largest
float32 inner product with the unit chord to the neighbour centre, and the
collisions inside a cell (there are always some, because twelve tangent
chords meet a template with eight distinct azimuths) are repaired by an exact
maximum-alignment assignment over all of the cell's endpoints. The template
is the base icosahedron of `oph_fpe.core.icosahedral`, whose edges are
exactly the carrier seams of `oph_exact.carrier` and whose geometric antipode
is `carrier.antipode()`. An eleven-neighbour cell leaves its least aligned
port unglued; at every level that port is port 4 or port 5, the two template
directions along the outward normal. The receipt also records the float64
optimal matching on the same frames and templates (agreement fraction with
the production rule and the two total alignments) and the comparison with
the three-port production gluing of `oph_exact/federation.py` on the
cell-dual graph.

Antipodal consistency of a glued pair `((a, p), (b, q))` means `q =
antipode(p)`. It is a property of the frames: neighbouring cells carry nearly
parallel tangent frames, so the outgoing and return chords are nearly
antipodal in the template, and the bijection constraint of the matching
breaks the pairing on a fraction of the pairs that shrinks as the mesh
refines.

## Dynamics on W12

The seam set is the thirty intra-carrier seams of every carrier plus the
glued seams. The float law is the seam-mean retraction (both endpoints
replaced by their mean) under the L1 schedule (uniform over all seams with
replacement, one sweep `= |S|` attempts, `default_rng(seed).integers(0, |S|,
size=|S|)`), applied by a lean native kernel (mean, `V` ledger, waits); the
integer law is `federation.run_integer_law` (nearest agreement, fair-coin odd
ties, termination at the balanced class). The descent functional is the
flagship's `V = sum x^2`; the terminal canonicalizers are those of the L1
receipt (component lattice snap for the mean law, component multiset for the
integer law), so the hashes are comparable with `federation_canonical_mean_receipt.json`.

At `L = 3` the float law runs sixteen schedules to `Phi < 1e-18`
(confluence receipt); at `L = 4, 5` it runs four schedules under a declared
sweep budget (64 and 32 sweeps) and the receipt reports the `Phi` decay and
the extrapolated cost from the port-graph gap. The integer law runs sixteen
schedules to termination at every level.

## Slow band under W12

Per-carrier normalized centered response kernels `K_n = 12 C_n / tr C_n`
from the twelve centered impulses of one carrier propagated by `T_fed = I -
L_fed/D` for `2n` steps with per-step rescaling (the L1 readout), `n` in
`{1, 5, 30, 100, 300}`, for 64 sampled carriers at `L = 3, 4` and 32 at
`L = 5` (declared budget; the values are level independent), at least a third
of them pentagonal-adjacent. Slow-band share `tr(P_slow K_n P_slow) / tr
K_n`. Controls: the isolated federation (must equal `carrier.normalized_response_kernel(n)`
within `1e-12` and `4 P_slow` at `n = 300`) and the three-port production
gluing on the same sampled cells at `L = 3`.

## The provenance order

Events. One round is one random permutation of all seams
(`numpy.random.Generator(PCG64(seed))`: `rng.permutation(|S|)` then
`rng.integers(0, 2, size=|S|)` tie coins), every seam attempted once. The
intra-carrier repairs inside a round are carrier-internal. Every glued-seam
repair attempt is an event that reads both endpoint carrier records at their
current versions; it writes both records (version advance, writer identity)
when it changes a value (a unit transfer or a swap), and a wait reads only.
The log stores, per attempt in execution order, the seam, the outcome, the
read versions and the written versions.

Rule. From the log alone: every carrier record starts at version zero (a
root without a writer); the direct parents of an attempt are the writers of
the two versions it read; precedence is the transitive closure. No declared
parents and no round labels enter the rule. Under this rule a wait is a
maximal element (nobody reads a version it wrote), so intervals contain
writing events only. A declared control treats every attempt as a write
(the certified comparison counts as a write); its order is four times denser
and its causal front about four times faster.

Runs. Seeded loads in `{0..5}` (the L1 loads of the level), sixteen burn-in
rounds (the balanced class is reached by round 5 to 8 at every level; the
glued write fraction is then stationary at 0.25), then the logged window of
`R = 8, 16, 32` rounds at `L = 3, 4, 5`, plus two slack rounds for the
ladder tops. Tips: six carriers per level (three interior, three
pentagonal-adjacent), bottom tip the first write on the carrier after round
`R/4` of the window, top tips the first write on the same carrier after a
ladder of round separations `{1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24}` up to
`3R/4`; the main interval is the `R/2` separation (tips at rounds `R/4` and
`3R/4`). An interval is interior when every event lies within 60 degrees of
the tip cell centre.

Readouts per interval: the event count, the strictly comparable pairs
(exact ancestor bitsets up to 60,000 events, a stratified sample of 2,048
sources by round above), the ordering fraction `2C/(N(N-1))`, the inverted
Myrheim--Meyer dimension, the height (longest chain in events), the layer
width (largest antichain layer of the longest-chain layering), the waist
angle, the growth exponent of the event count against the round separation
over the interior ladder points (`1 +` spatial dimension for a manifold-like
order), the causal front speed (maximal angle reached per round ahead of the
bottom tip), and the link statistics of the direct read-after-write edges
(cover fraction, same-round fraction, spatial displacement and isotropy)
under two placements: (a) the writer's own readback, the event at the mean of
its two endpoint carriers' readbacks `x = 2 P_slow N` at the written
versions in the isometric coordinates `X = V^T N` (`G = 4 P_slow = V V^T`),
and (b) the screen chart, the event at the normalized midpoint of its two
cell centres on S2 (declared control).

Depth. A three-level federation on levels `(L-2, L-1, L)` with the W12
wiring on every level and the committed join transport as declared
inter-level seams: child port `k` glued to parent port `k` (twelve join seams
per child cell), the same mean or nearest-agreement law on every seam, every
seam attempted once per round. The stationary state of a parent port under
that law is the plain average of its four children's ports; the committed
conditional expectation weights are the children's spherical area fractions,
whose maximal deviation from `1/4` is recorded, and they enter no repair.
The coarsest level's front is four times faster in angle, so the depth
runs use an eight-round window at every `L`; tips sit on two carriers of
level `L-1` and two of level `L`.

Source net. The comparison block rebuilds the record-metric route at `q =
13` from `oph_exact.source_net` (site graph, centre site, layered order `(j,
s) <= (j', t)` iff `d(s, t) <= j' - j`) and runs the same pair counter,
height, width, link and growth readouts on the vertical intervals with `k =
2, 3, 4` layers. Its top interval reproduces the source-net receipt row
exactly (`N = 1529`, `C = 102990`, ordering fraction `0.0882`, dimension
`4.149`), which also validates the pair counter of this lane on a known
order.

## Numbers

NUMBERS_PLACEHOLDER

## Theory reading

The flagship places position in the rank-three record readback: a carrier's
position is `x = 2 P_slow N`, the slow-band projection of its own cumulative
port loads (theorem `thm:rank-three`), and the read law of the source net
lives in the record metric, the rank-three source Gram on the golden
populations, whose provenance order reads `3+1` (lane L2: dimension 4.15,
3.61, 4.10, 4.09 at `q = 13, 21, 34, 55`). The S2 wiring is the screen's
regulator: it fixes which carriers read each other on the geodesic screen at
a fixed level. What the S2 wiring contributes to the causal readout is what
this receipt measures, and the measurement reads: the provenance order of
the glued reads on one screen level is a `2+1` order (ordering fraction
toward `8/35`, Myrheim--Meyer dimension toward three, growth exponent three,
isotropic links of one neighbour spacing on the screen chart), the readback
placement puts the same events on a small internal ball with isotropic unit
steps unrelated to the screen, and the depth direction of a three-level
federation raises the readout above three without reaching four at depth
three. The record-metric route reads `3+1` because its read law is the
rank-three source metric; the screen wiring reads `2+1` because its read law
is the two-dimensional screen. Selection of the wiring by the axioms (M1) is
work in progress; the refinement limit of the provenance readout across
levels is work in progress.

## Claim boundary

Supplied: the W12 wiring convention, the production port assignment, the
join convention, the schedules, the initial loads, the write convention.
Produced: the repairs, the provenance order generated by the read-after-write
rule from the log alone, the readbacks, the slow band under W12, the
confluence receipt. Not claimed: selection of the wiring by the axioms (M1),
a physical clock, the continuum limit, physical position or length, field
attachment. The ordering fractions, dimensions and growth exponents are
finite diagnostics of the declared wiring.
