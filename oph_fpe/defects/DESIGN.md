# Defect census against the matter grammar (lane C6)

Status: exploratory, non-evidential, design-only. No instrument is frozen or
armed, no decision rule exists, and nothing here is a physical particle
claim. This document is written and fixed before any comparison code exists;
the census pipeline of sections 3 to 8 is constructed without reference to
the expected occupancy of the committed matter table, and the comparison
against the committed table runs only through the separate verifier stage of
section 9, after census outputs exist.

Campaign: `plan/SIM_ALIGNMENT_2026-08-20.md`, lane C6 (owner-requested).

## 1. Question

Do the stable defect classes of a Z/6 link configuration on the committed
icosahedral base carrier, evolved by a gauge-covariant local repair rule to a
repair fixed point, organize by the committed matter-grammar bookkeeping:
the 36-element label lattice Z/6 x Z/3 x Z/2, the descent congruence
2t + 3d + q = 0 mod 6, the ten committed component rows, and the blocked
control label (1, 0, 0)?

## 2. Committed inputs (read-only)

* Carrier: `oph_fpe/em/base_carrier.py` (lane C4), restating the committed
  twelve-port, thirty-seam, twenty-oriented-face icosahedral complex with
  the committed spanning tree (eleven seams) and its nineteen chords and
  fundamental cycles. The carrier is imported, never redefined.
* Sector classification: `Lean/Screen/SeamU1HolonomyClassification.lean`
  (RER): H^1(seam graph, U(1)) = U(1)^19 via the nineteen chord holonomies
  modulo port gauge. This lane discretizes U(1) to its sixth roots, i.e. to
  the additive group Z/6.
* Label lattice and congruence (verifier stage only):
  `Lean/Screen/GlobalFormCharacterDescent.lean`,
  `Lean/Screen/ExteriorSelection.lean`,
  `Lean/Screen/ExteriorComponentBridge.lean` (RER): the lattice
  W = Z/6 x Z/3 x Z/2 of (q = 6Y mod 6, triality t, duality d), the
  congruence 2t + 3d + q = 0 mod 6 with exactly 6 of 36 labels descending,
  the ten component rows with charge column
  (-2, 3, -4, 1, 6, 4, -1, -6, 2, -3) and bidegree column
  ((1,0), (0,1), (2,0), (1,1), (0,2), (1,2), (2,1), (3,0), (2,2), (3,1)),
  the two parity survivor masks (even {2,3,4,8,9}, odd {0,1,5,6,7}), and
  the blocked control label (1, 0, 0).

The concurrently owned `oph_fpe/core/icosahedral.py` is not imported; the
icosahedral symmetry action is derived inside this lane from the committed
base-carrier tables alone (section 7).

## 3. Configuration space and gauge action

* A configuration assigns to every seam e (canonical orientation
  left(e) -> right(e), left < right) a link label A(e) in Z/6, written
  additively as an integer 0..5. Z/6 carries the two canonical quotient
  readouts Z/6 -> Z/3 (mod 3) and Z/6 -> Z/2 (mod 2), matching the
  committed lattice reading of W.
* A port gauge move is a function g: ports -> Z/6 acting by
  A(e) -> A(e) + g(right(e)) - g(left(e)) mod 6, the additive form of the
  committed rechart action (endpoint conjugation, abelianized). Gauge moves
  form the group (Z/6)^12 acting through the coboundary d; constants act
  trivially.

## 4. Conserved sector data and the defect definition

* The sector data of a configuration is its 19-tuple of chord holonomies
  h_c(A) = sum_e cycle_c(e) A(e) mod 6, over the nineteen committed
  fundamental chord cycles in committed chord order. This is the committed
  classification discretized to Z/6.
* Exact conservation under gauge (receipted as finite arithmetic, not
  sampling): h_c(A + dg) - h_c(A) = <cycle_c, dg> = <boundary(cycle_c), g>
  = 0 because every fundamental cycle has zero boundary. The receipt
  enumerates boundary(cycle_c)(p) = 0 for all 19 x 12 pairs (c, p) over the
  integers; gauge moves are Z-linear combinations of unit port moves, so
  the identity holds for every configuration and every gauge move over Z/6.
* Completeness (receipted): tree reduction. Every configuration is carried
  by an explicitly constructed gauge move to the unique representative that
  vanishes on the eleven tree seams and equals its chord holonomies on the
  nineteen chords. Sectors are therefore in bijection with (Z/6)^19, and
  two configurations are gauge-equivalent exactly when their sector tuples
  agree.
* Curvature: F(A) = C A mod 6 on the twenty oriented faces. F is a sector
  invariant (C d = 0 mod 6), and it classifies sectors: over the sphere,
  ker(C mod 6) = im(d mod 6), receipted by rank(C) = 19 over GF(2) and over
  GF(3) together with C d = 0 (the mod-6 kernel then has 6^11 elements,
  equal to the coboundary image, forcing equality). Equal curvature
  therefore means equal sector, and each sector has one curvature pattern.
* A DEFECT is a nontrivial conserved sector class: a nonzero 19-tuple,
  equivalently a nonzero residual curvature pattern. The zero class is the
  vacuum and is reported separately, not as a defect.

Consequence fixed here, before any dynamics is coded: a flat configuration
(F = 0) is pure gauge, so any dynamics that conserves the sector exactly
cannot change F at all, and any dynamics that reduces curvature must change
the sector. There is no rule that both reduces face curvature and conserves
chord holonomy. This design therefore takes option (b) of the lane brief:

## 5. Repair rule (declared choice: option (b), sector-tracking)

Repair changes seam labels and therefore moves between sectors; the census
tracks which sectors are dynamically stable, i.e. survive as repair fixed
points. The exactly established properties are listed in section 6.

Rule `single_seam_strict_local_descent.v1`, deterministic:

* Mismatch functional: E(A) = sum_f rho(F(A)(f)) with
  rho(x) = min(x, 6 - x) in {0, 1, 2, 3}, the circular distance to flatness
  on Z/6. rho(-x) = rho(x).
* One sweep visits the thirty seams in committed index order. At seam e the
  two incident faces f1 < f2 carry opposite incidence signs s and -s
  (committed opposite-signs receipt). For each candidate increment
  delta = 1..5 the local energy change is
  dE(delta) = rho(F(f1) + s delta) + rho(F(f2) - s delta)
            - rho(F(f1)) - rho(F(f2)).
  The rule applies the candidate with minimal dE, tie-broken by smaller
  rho(delta) and then by smaller delta, and only when dE < 0 (strict).
  Applying the move updates A(e) and the two face curvatures.
* Sweeps repeat until one full sweep applies no move: the repair fixed
  point. Termination is guaranteed: E is a nonnegative integer, at most 60
  on the base carrier, and every applied move decreases it by at least 1.
* Locality: each decision reads exactly the two face curvatures incident to
  the seam and the incidence sign, nothing else.

## 6. Conservation and covariance properties established

1. Gauge moves conserve the sector exactly, for every configuration and
   every gauge move (finite arithmetic receipt, section 4). Under gauge
   moves alone, defects are frozen topological data.
2. The repair rule is gauge-covariant with identical move traces:
   repair(A + dg) = repair(A) + dg, because every decision reads only
   (F(f1), F(f2), s), which are gauge-invariant, and the sweep order is
   fixed. Receipted by trace equality on seeded samples; a raw-label mutant
   rule (reads A(e) itself) fails the receipt (mutation guard).
3. Corollary of 1 and 2: repair descends to a well-defined deterministic
   map on sector space, and the census is a census of sectors, not of
   representatives. Receipted: class(repair(A + dg)) = class(repair(A)).
4. E strictly decreases at every applied move and repair is idempotent:
   continued repair does not move a fixed point (stability under continued
   repair, receipted).
5. Stability is a sector property: whether some single-seam move strictly
   reduces E reads only the curvature pattern, a sector invariant. A
   stronger probe is recorded per class: NEUTRAL ESCAPABILITY, whether one
   energy-neutral single-seam move (dE = 0) unlocks a strictly improving
   second move. Classes with no such two-move escape are depth-2 stable
   under this rule. Both probes are declared conventions of this rule, not
   physical statements.

Not claimed: A5-equivariance of the repair map (the sweep order is the
committed seam index order, which the icosahedral action does not
preserve). The exact A5 statements are in section 7.

## 7. Icosahedral (A5) action

Derived from the committed base-carrier tables alone: the automorphism
group of the port graph (order 120) is computed by backtracking search, and
the orientation-preserving half (every oriented face maps to a cyclic
rotation of a committed oriented face row) is retained. Receipts: order 60,
identity present, closure under composition, element-order histogram
{1: 1, 2: 15, 3: 20, 5: 24} (the A5 profile), transitivity on ports, and
orientation preservation on all twenty faces.

* Signed seam action: sigma sends seam e = (l, r) to the seam e' with
  endpoint set {sigma(l), sigma(r)}; the label transports with sign +1 when
  the canonical orientation is preserved and -1 when reversed:
  (sigma . A)(e') = +/- A(e).
* Exact invariances receipted: gauge moves map to gauge moves
  (sigma . dg = d(g o sigma^{-1})); curvature transports by the face
  permutation with sign +1 (orientation preservation); E(sigma . A) = E(A);
  the induced sector action sigma_* on (Z/6)^19 (transport a tree-trivial
  representative, read holonomies) satisfies
  class(sigma . A) = sigma_*(class(A)) and respects composition and
  identity; stability and neutral escapability are invariant under
  sigma_* because the single-move set and E are invariant. Mutation guard:
  the unsigned action (orientation signs dropped) fails the curvature
  transport and energy receipts.
* Orbit data: the A5 orbit of a class is its set of images under all sixty
  elements; the canonical orbit representative is the lexicographically
  smallest image tuple; the orbit size is the image count.

Labels (section 8) are read through the committed tree and are not claimed
to be constant on A5 orbits; the census records per-class labels and
per-orbit label multisets, and the exploratory run reports what is
observed.

## 8. Label readout (declared convention) and census protocol

Readout `total_chord_holonomy_with_canonical_quotients.v1`, fixed a priori:

* q(k) = sum of the nineteen chord holonomies of the class k, mod 6;
* t(k) = q(k) mod 3 and d(k) = q(k) mod 2, the canonical quotients of Z/6,
  matching the committed lattice reading of W = Z/6 x Z/3 x Z/2.

Justification: the total holonomy is the only readout functional declared
here that uses no structure beyond the committed chord set and the abelian
group; the quotients are the canonical ones. The readout is a convention of
this lane, recorded here; it is not a committed identification.

Structural entailments, disclosed before any run:

* The readout image is the CRT diagonal {(q, q mod 3, q mod 2)}, six of the
  thirty-six labels, and 2t + 3d + q = 2q + 3q + q = 6q = 0 mod 6 on the
  diagonal: every label this readout can emit satisfies the descent
  congruence, and the diagonal coincides with the committed six-element
  descent kernel. Verifier check (a) is therefore structurally forced to
  fraction 1 under this readout, against the lattice baseline of one sixth;
  the verifier detects diagonality numerically and reports the vacuity
  rather than presenting the fraction as a finding.
* The blocked control label (1, 0, 0) is off-diagonal (1 mod 3 = 1, not 0)
  and is therefore structurally unreachable; check (c) is a coding control,
  receipted live by a planted-violation test, not an empirical result.
* The live, non-forced content of the comparison is check (b): which of the
  six diagonal label values are realized by stable defect classes, with
  what multiplicities and orbit structure, and hence which committed rows
  are label-matched. The committed rows carry exactly the six diagonal
  labels, all six values of q; equidistribution over q is the neutral
  expectation for an unstructured census.

Secondary per-class data (recorded, no comparison): the per-cycle label
multiset {h_c != 0} with each h_c in Z/6, chord support size, curvature
pattern, curvature support size.

Census protocol (`oph.sim.defect_census.z6_carrier.v1`):

* Ensemble streams, named, seeded, recorded in the receipt (Mersenne
  Twister via `random.Random(seed)`):
  * `uniform_iid`, seed 20260820, N = 160: every seam label uniform on Z/6.
  * `sparse_pair`, seed 20260821, N = 80: zero configuration, then two
    distinct uniformly chosen seams receive uniform nonzero labels.
* Every member is evolved by the section-5 rule to its repair fixed point.
* The census is the set of realized fixed-point sector classes with, per
  class: the 19-tuple, the curvature pattern, energy, the (q, t, d) label,
  secondary data, multiplicity (total and per stream), the count of members
  whose sector was unchanged by repair, stability (fixed point, asserted),
  neutral escapability, orbit representative and orbit size. The vacuum
  class is reported separately with its multiplicity.
* All arithmetic is exact integer arithmetic mod 6; the receipt records
  schema, flags (exploratory true, evidential false, frozen false,
  instrument_armed false), seeds, rule and readout identifiers, and the
  carrier pin (port/seam/face/tree/chord counts and the section-4 rank
  receipts).

Determinism: fixed seeds give byte-identical receipts (receipted by a
double-run test).

## 9. Verifier stage (separate module, after census outputs exist)

Module `z6_matter_grammar_verifier.py` is the only census-lane module that
contains committed matter-table values. It consumes a finished census
receipt and reports; it feeds nothing back, and no census parameter is
tuned in response to it.

* (a) Descent congruence: fraction of realized defect classes (by class
  and multiplicity-weighted) with 2t + 3d + q = 0 mod 6; lattice baseline
  exactly one sixth (committed equidistribution: every center-character
  fibre has six labels); numerical detection of readout diagonality with
  the structural vacuity note of section 8 when it holds.
* (b) Committed row occupancy: the ten component rows with weights
  (charge mod 6, color degree mod 3, weak degree mod 2) computed inside the
  verifier from the committed charge and bidegree columns; per row: whether
  its weight is realized by a defect class, matching class count and total
  multiplicity; plus per-parity-sector row counts for the two committed
  survivor masks.
* (c) Negative control: occupancy of the blocked label (1, 0, 0); expected
  count zero; the verifier's ability to detect a nonzero count is
  established by a planted-violation test, not assumed.
* Also reported: distinct labels realized out of 36 and out of the 6
  descending labels.

## 10. Target-cleanliness statement

This file is written before any pipeline or verifier code. The census
modules (`z6_carrier_defects.py`, `z6_a5_action.py`, `z6_defect_census.py`)
contain no committed matter-table row values, no descent congruence, and no
reference to expected occupancy; the readout is fixed above by canonicity,
with its structural consequences disclosed rather than discovered. The
verifier is a distinct module executed after census receipts exist and only
reports. Nothing in this lane is tuned against the comparison outcome.

## 11. Module and test map

* `oph_fpe/defects/z6_carrier_defects.py`: carrier interface (any
  port-graph with spanning tree and chord set; instantiated from
  `oph_fpe.em.base_carrier` for this lane), configurations, gauge action,
  chord holonomies, tree reduction, curvature, energy, structural and
  conservation receipts.
* `oph_fpe/defects/z6_a5_action.py`: derived icosahedral action, signed
  seam action, sector action, orbit data, receipts.
* `oph_fpe/defects/z6_defect_census.py`: repair rule, ensembles, census
  pipeline and receipt, exploratory run entry point.
* `oph_fpe/defects/z6_matter_grammar_verifier.py`: verifier stage.
* `tests/test_z6_defect_census.py`: receipts plus mutation guards
  (non-covariant repair mutant fails the covariance receipt; broken-cycle
  mutant fails the conservation receipt; unsigned action mutant fails the
  A5 receipts; planted label violation is detected by the verifier; fixed
  seeds give identical receipts; sector action equivariance and stability
  invariance under a declared generator).

## 12. Boundaries

Exploratory and non-evidential throughout. No physical particle claim: a
defect class is a finite combinatorial invariant of a Z/6 link
configuration on the committed carrier, not a particle, and the (q, t, d)
label is a declared convention of section 8, not a committed
identification. The repair rule and both stability probes are declared
conventions of this lane. No instrument is frozen or armed, no decision
rule exists, and any future evidential use requires its own immutable
preregistration under the campaign rules.
