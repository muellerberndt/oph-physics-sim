# Family readout v2 for the Z/6 defect census (lane D2)

Status: exploratory, non-evidential, design-only. No instrument is frozen or
armed, no decision rule exists, and nothing here is a physical particle
claim. This document is written and fixed before the v2 readout and verifier
code exist. The census pipeline of lane C6 (commit a37f771) is consumed
read-only; the exact census streams are reused so the class sets match.

Predecessor record: `oph_fpe/defects/DESIGN.md` (lane C6). The C6 readout
`total_chord_holonomy_with_canonical_quotients.v1` emits labels on the CRT
diagonal only: its image is exactly the committed six-element descent
kernel, the descent check is structurally vacuous, and row occupancy is a
six-valued comparison. This lane supplies a v2 readout whose triality and
duality coordinates are characters on carrier structures distinct from the
total holonomy, so the matter-grammar checks become live.

## 1. Corpus findings (step 1, read-only mining of RER)

The committed corpus fixes the meaning of the label coordinates at the
level of the matter table. The defining passages:

* `Lean/Screen/GlobalFormCharacterDescent.lean` (lines 133 to 141, doc of
  `exterior_rows_integer_congruence`): "The same ten rows as exact integer
  congruences on the committed charge and bidegree columns:
  `2t + 3d + q = 0 mod 6` with `t` the color exterior degree, `d` the weak
  exterior degree, and `q` the committed integer charge `q = 6Y` of the
  row."
* `Lean/Screen/QuantumMatterIntegration.lean` (lines 28 to 35, doc and
  body of `componentWeight`): "Central weight canonically read from an
  exterior component row. Color triality is its color exterior degree
  modulo three, weak duality is its weak degree modulo two, and the first
  coordinate is the frozen integer charge modulo six."
* `Lean/Screen/ExteriorComponentBridge.lean`: `colorDegree` counts modes
  `0,1,2` of the five-mode carrier, `weakDegree` counts modes `3,4`
  (lines 40 to 46); `component_charge_binding` (lines 113 to 119): "The
  frozen integer charge is the additive exterior weight `-2c+3w`",
  `charge i = -2 * c + 3 * w` on the row bidegree `(c, w)`.
* `Lean/Screen/Z6Descent.lean` (line 51): "A realized weight:
  `(q mod 6, triality, duality)` under `q = 6Y`."
* `paper/deriving_the_particle_zoo_from_observer_consistency.tex` (exterior
  substrate passage, lines 1556 to 1564): "For the supplied carrier
  \(V=\mathbb C^3\oplus\mathbb C^2\), the exterior basis is indexed by all
  32 subsets of the five coordinate modes. ... On every row, the integer
  charge is \(-2c+3w\), fermionic parity is \(c+w\bmod2\), and charge
  conjugation is complement in the three color and two weak modes."
* `paper/deriving_standard_model_gauge_structure_from_observer_overlap_consistency.tex`
  (descent passage, lines 280 to 284): "On the finite weight lattice of
  hypercharge, triality, and duality labels in the \(q=6Y\) normalization,
  a label is fixed by every element of the diagonal \(\mathbb Z_6\) kernel
  exactly when \(2t+3d+q\equiv 0 \pmod 6\)."
* `Lean/Screen/FermionSectorAssembly.lean` (header): "The fermion carrier
  is one complex amplitude per row of the committed ten-component exterior
  table (register row PR-59): ... each row carries the committed integer
  charge column (`q = 6Y`) and the committed chirality grading through the
  index dictionary of `MatterGrammarIndexBridge`." The module consumes the
  ten-row table; it defines no readout on seam configurations.
* `Lean/Screen/NeutralCurrentDictionary.lean` (header): "A weak slot is a
  row of the committed table together with a weak component index; its
  third-isospin entry in sixths is `t3Six` ..., its hypercharge entry in
  sixths is the committed integer charge column." Table-level again.
* `Lean/Screen/ChargedCurrentDictionary.lean` (header): "the per-row
  charged coupling dictionary on the weak slots of the committed exterior
  component table." Table-level again.
* Quarantined row values: `oph_fpe/defects/z6_matter_grammar_verifier.py`
  carries the committed charge column `(-2, 3, -4, 1, 6, 4, -1, -6, 2, -3)`
  and bidegree column `((1,0), (0,1), (2,0), (1,1), (0,2), (1,2), (2,1),
  (3,0), (2,2), (3,1))`, with row weights
  `(charge mod 6, color degree mod 3, weak degree mod 2)`.

Verdict on the step-1 question: the corpus defines triality and duality as
characters on structures distinct from the total Z/6 holonomy. Triality is
the occupancy count of a three-element mode family modulo three; duality is
the occupancy count of a two-element mode family modulo two; the charge is
the weighted combination `q = -2c + 3w`. None of the corpus modules define
a map from a Z/6 seam sector on the icosahedral carrier to `(t, d)`: the
five-mode exterior carrier is a distinct object, and the current
dictionaries and the fermion assembly consume the finished ten-row table.

## 2. Decision-tree resolution

### 2a. What the corpus fixes, and the v1 verdict

Fixed by the corpus: the label lattice `W = Z/6 x Z/3 x Z/2`, the shape of
the coordinates (t counts a 3-family occupancy mod 3, d counts a 2-family
occupancy mod 2, q is a weighted charge), the congruence
`2t + 3d + q = 0 mod 6` as the descent criterion, the ten row weights, and
the blocked control label `(1, 0, 0)`.

Derived consequence, recorded here with its derivation: on the committed
table the charge binding forces every row weight onto the CRT diagonal.
From `q = -2c + 3w`: `q mod 3 = (-2c) mod 3 = c mod 3 = t` and
`q mod 2 = (3w) mod 2 = w mod 2 = d`. The ten committed rows therefore
carry exactly six distinct labels, all diagonal, with row multiplicity
profile: rows (0, 5) share `(4, 1, 0)`, rows (1, 9) share `(3, 0, 1)`,
rows (2, 8) share `(2, 2, 0)`, rows (4, 7) share `(0, 0, 0)`, row 3
carries `(1, 1, 1)`, row 6 carries `(5, 2, 1)`. The committed descent
kernel (six labels with `2t + 3d + q = 0`) coincides with the CRT diagonal
as a set.

Verdict on v1: the v1 diagonal readout agrees with the committed row
labels pointwise (every committed row label is diagonal, as derived above)
and contradicts the committed definitions as definitions. In the corpus,
t and d are independent characters and the diagonal landing of the table
is a theorem of the charge binding; a readout that imposes
`t = q mod 3, d = q mod 2` by construction turns that theorem, and the
descent congruence with it, into tautologies. The v2 readout removes the
imposition so the census can fail the congruence.

### 2b. Carrier realization: declared convention (underdetermined case)

The carrier-level realization is a convention of this lane, disclosed as
such. Every ingredient is derived from the committed base-carrier tables
(`oph_fpe/em/base_carrier.py` through `base_carrier_spec()`), with
deterministic construction and fail-closed receipts.

Characters on sector space. The sector space is `(Z/6)^19` in chord
holonomy coordinates `h_c` (C6 classification receipt). Every group
character of sector space is a coefficient vector on the chords; a seam
functional `sum_e v(e) A(e)` is gauge-invariant exactly when `v` has zero
port boundary, and the fundamental cycles span that space. The v2
coordinates are declared as chord-coefficient characters, with the
coefficient vectors drawn from two canonical carrier structures:

* Triality structure: the face-rainbow 3-coloring. A coloring
  `col: seams -> {0, 1, 2}` such that the three seams of every oriented
  face carry three distinct colors; equivalently a proper 3-edge-coloring
  (Tait coloring) of the dual dodecahedral graph, whose vertices are the
  twenty faces. The canonical coloring `col_0` is the lexicographically
  minimal such coloring in committed seam order (deterministic
  backtracking, smallest feasible color first). Receipts: the rainbow
  condition on all twenty faces; color class sizes (10, 10, 10);
  byte-determinism. On the committed tables
  `col_0 = (0,1,1,0,2,2,2,0,1,2,1,0,0,0,1,1,1,2,2,2,0,2,0,2,1,0,1,1,0,2)`.
* Duality structure: the antipodal pairing. The antipodal port map is
  derived from committed adjacency as the unique port at graph distance
  three (uniqueness receipted per port; on committed indices the map is
  `p -> 11 - p`). It induces a fixed-point-free involution on seams with
  fifteen orbits (receipted). The involution is a graph automorphism and
  is not orientation-preserving, hence not an element of the derived
  rotation group (both receipted).

Declared readout `tait_antipodal_family_readout.v2`, on a class `k` with
chord holonomies `(h_c)`:

* `q(k) = sum_c h_c mod 6`: the total chord holonomy, unchanged from v1.
  The Z/6 coordinate keeps v1 comparability; the v2 change replaces the
  two quotient coordinates with independent characters.
* `t(k) = sum_c col_0(chord_c) * h_c mod 3`: color-index weights on
  chords. This realizes the "sum over color-1 seams minus sum over
  color-2 seams" shape: weights `(0, +1, -1) mod 3 = (0, 1, 2)` are the
  color indices. The restriction to chords is forced by gauge invariance:
  the all-seam colored sum has nonzero port boundary (section 2c mutant).
  Under `col_0` the chord color counts are (5, 6, 8).
* `d(k) = sum_{c : antipodal(chord_c) is a chord} h_c mod 2`: the
  paired-chord indicator. This realizes the "holonomy differences across
  antipodal pairs" shape modulo two: for a pair with both members chords,
  `h_c - h_{c*} = h_c + h_{c*} mod 2`, and summing over the seven
  both-chord pairs gives the indicator with support fourteen. Pairs
  meeting the tree contribute gauge-dependent seam data and are excluded;
  the fifteen pairs split as seven both-chord, five mixed, three
  both-tree on the committed tree.

The readout depends on three declared objects: the committed spanning tree
(inherited from C6/C4), the canonical coloring `col_0`, and the derived
antipodal pairing. The tree dependence matches v1 (the v1 total is also a
chord-set functional). All three are pinned above.

### 2c. Well-definedness receipts (fail-closed, exact)

1. Gauge invariance, exact over generators, not sampled. Each coordinate
   is a chord-coefficient character; its seam weight vector is the integer
   combination `v = sum_c w_c * cycle_c` of fundamental cycles. The
   receipt computes the port boundary of `v` at all twelve ports over the
   integers and requires zero; gauge moves are Z-linear combinations of
   unit port moves, so zero boundary is invariance for every configuration
   and every gauge move. A functional form of the receipt evaluates the
   seam-level formula on `A + d(delta_p)` for all twelve unit port moves
   and requires equality. Verified at design time: boundary of the q, t,
   d weight vectors is zero at all twelve ports.
2. Gauge-variance mutant (first mutation guard). The naive all-seam
   colored sum, weight vector `v(e) = col_0(e)` on all thirty seams, has
   nonzero port boundary modulo three (design-time value: nonzero at nine
   of twelve ports). The invariance receipt rejects it; the test suite
   asserts the rejection.
3. A5 behavior. Theorem, receipted by exact rank computation over the
   sixty derived rotations: the fixed subspace of the character space
   under the induced sector action is zero over GF(3) (signed action) and
   over GF(2). Consequence: no nonzero A5-invariant triality or duality
   character on sector space exists, so no A5-equivariant assignment rule
   into a fixed character is available, for this construction or any
   other. The zero statement is scoped to characters (the dual action);
   the A5-fixed sector space itself is one dimensional over GF(2), so
   paraphrases that drop the character scoping are wrong. Coloring-independence fails as well: the transport identity
   `t_{sigma . col}(sigma_* k) = t_{col}(k)` fails on sampled pairs
   (design-time diagnostic: 25 mismatches of 40; the unoriented coloring
   does not track the orientation signs of holonomy transport). The
   chosen option is therefore the quotient: the declared A5-quotient
   object is the per-class orbit label multiset, the multiset of v2
   labels over the distinct sectors of the class's A5 orbit under the
   fixed convention. Receipt: the multiset computed from any orbit member
   is identical (exact on a declared generator and on samples). Per-class
   labels under the fixed convention are recorded as convention-relative
   data; every lattice table is reported both per class and per orbit
   multiset. This disclosure supersedes nothing in C6, which records the
   same non-constancy caveat for its labels.
4. Determinism: fixed seeds and the deterministic coloring give
   byte-identical receipts (double-run test).

### 2d. Structural non-vacuity receipt

Exact subgroup computation, census-independent: the image of the label
homomorphism `(Z/6)^19 -> W` is the subgroup generated by the nineteen
unit-chord labels `(1, col_0(chord_c), paired(chord_c))`. Closure
enumeration at design time gives the full 36-element lattice. Therefore:

* the image is not contained in the six-element CRT diagonal / descent
  kernel: the descent congruence is a live check with lattice baseline
  one sixth;
* explicit off-diagonal witness: the unit sector on chord index 0 (seam
  5) has label `(1, 2, 1)` with center character
  `2*2 + 3*1 + 1 = 8 = 2 mod 6`, nonzero;
* the blocked control label `(1, 0, 0)` lies in the image: the control
  check is live, a nonzero occupancy is a measurement rather than a
  structural impossibility, and the verifier's ability to report it is
  established by a planted-class test;
* vacuity detector (second mutation guard): a readout whose image lies in
  the diagonal is flagged; the v1 reimplementation (all-ones weights in
  both slots, `t = q mod 3`, `d = q mod 2`) has image exactly the
  diagonal and must trigger the detector. If the v2 construction itself
  triggered the detector, the finding would be reported and the
  comparison stopped; the design-time image computation shows it does
  not.

### 2e. Pinned comparison plan (fixed before the census rerun)

The verifier consumes the finished v2-labeled census and reports; nothing
feeds back. Streams are the exact C6 declared streams: `uniform_iid`
seed 20260820 N = 160, `sparse_pair` seed 20260821 N = 80. Tables:

1. Full 36-point lattice occupancy: per label, defect-class count and
   total multiplicity, over all realized classes; and per label, the
   count of A5 orbits whose full-orbit label multiset contains the label,
   with the orbit multiset table alongside.
2. Committed-row occupancy: the ten rows with weights computed from the
   quarantined charge and bidegree columns; per row, realization, class
   count, multiplicity; per-parity-sector counts for the two committed
   survivor masks. Pinned in advance: the ten rows carry six distinct
   labels (section 2a), so the lattice partitions as 6 committed-row
   labels + 30 complement labels; the lane brief's 26-point complement
   presupposes ten distinct row labels, and the measured collapse to six
   is recorded with the 30-point complement used alongside the 10-row
   table.
3. Baselines for the committed-versus-complement mass: uniform over the
   36-point lattice (committed-set mass 6/36 = 1/6) and uniform over the
   realized label set (committed-set mass = |realized and committed| /
   |realized|), reported for class counts and for multiplicity weights.
4. Descent-congruence pass fraction, by class and by multiplicity, with
   lattice baseline one sixth; the vacuity detector result on the
   realized labels (expected: not diagonal, check live).
5. Control label `(1, 0, 0)`: occupancy count and multiplicity, plus the
   planted-class detection receipt in the test suite.
6. Cross-tabulation label x energy x depth-2 stability, for the depth-2
   stable classes (neutral_escapable false; 22 classes in the C6 census)
   and for all realized classes (170 in the C6 census).
7. Multiplicity structure among depth-2 stable classes sharing a label:
   counts only. The word "generation" appears in this lane only inside a
   quoted hypothesis, and this document quotes none.

## 3. Module and test map

* `oph_fpe/defects/z6_family_readout.py`: antipodal derivation, canonical
  coloring, chord weights, the v2 label map, orbit label multisets, and
  the receipts of sections 2c and 2d (gauge invariance exact over ports,
  mutant rejection, A5 fixed-space ranks, image subgroup and vacuity
  detector, off-diagonal witness). Imports the committed C6 modules
  read-only.
* `oph_fpe/defects/z6_matter_grammar_verifier_v2.py`: the section 2e
  tables on a finished v2-labeled census; committed row values imported
  from the quarantined v1 verifier module; the `__main__` CLI reruns the
  exact C6 census streams, attaches v2 labels, runs the verifier, prints
  one canonical receipt JSON to stdout and its SHA-256 to stderr, and
  accepts `--out PATH` for a byte-identical file copy. No writes under
  `data/` (concurrent lane ownership).
* `tests/test_z6_family_readout.py`: mutation guards and receipts:
  gauge-variance mutant rejected by the invariance receipt; v1
  reimplementation triggers the vacuity detector and the v2 weights do
  not; planted class carrying `(1, 0, 0)` reported in occupancy; receipt
  byte-stability under rerun; rainbow and class-size coloring receipts;
  antipodal uniqueness, involution, and non-membership in the rotation
  group; A5 fixed-space zero; orbit multiset stability across orbit
  members; label invariance under sampled gauge moves and unit port
  moves.

## 4. Boundaries

Exploratory and non-evidential throughout. Defect classes are finite
combinatorial invariants of Z/6 link configurations on the committed
carrier; the v2 label is a declared convention of section 2b, disclosed,
with the corpus fixing only the label-lattice shape and the descent
grammar. Numbers only; no physical particle claim; no instrument is
frozen or armed; any future evidential use requires its own immutable
preregistration under the campaign rules.
