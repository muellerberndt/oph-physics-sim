# W/Z/H Numerical Source-Closure Backend

## Purpose

`oph_fpe.bosons` is the heavy numerical half of the OPH W/Z/H source lane. It
does not define the D10 or D11 carriers and does not select their coefficients.
Those theorem objects remain canonical in the sibling
`reverse-engineering-reality/code/particles/` tree.

The backend consumes hash-pinned paper-side artifacts and computes four kinds
of numerical evidence:

1. a finite source-clock spectral gap and its perturbation interval;
2. piecewise RG and matching transport with a stability/error bound;
3. determinants of declared low-dimensional BRST mixing blocks;
4. Rouché-contour zero counts and the readout
   `s_B = (M_B - i Gamma_B/2)^2`.

## Observer-like system boundary

The paper-side carriers must describe observer-like self-reading patches with
bounded local state, typed ports or boundaries, readback, persistent records,
feedback or repair moves, and public evidence. The simulator consumes their
receipts. It must not manufacture a graph or path table around desired W/Z/H
coefficients.

## Run the diagnostic control

```bash
python3 -m oph_fpe.bosons \
  --config configs/bosons/wzh_source_closure_diagnostic_v1.yml \
  --out runs/bosons/wzh_source_closure_diagnostic_v1
```

The tracked fixture uses synthetic dimensionless matrices to test the solver.
It deliberately has null expected hashes, missing D10/D11 certificates, an
unverified source clock, synthetic RG coefficients, and unverified BRST
identities. Its expected result is:

```text
overall_status = diagnostic_backend_source_packets_incomplete
promotion_allowed = false
```

Numerical zeros in this fixture are control receipts, not boson masses.

## How an actual prediction run differs

A certificate-candidate config must:

- set `claim_level: certificate_candidate`;
- pin every source artifact with its predeclared `sha256:` digest;
- provide a strict source-root interval certificate on one branch;
- provide trusted D10 QT1--QT5 and D11 source-character certificates;
- replace the toy clock Hamiltonian with the source electromagnetic, electron,
  cesium nuclear, and atomic hyperfine packet;
- replace affine RG controls with the frozen physical beta, threshold,
  matching, scheme, and truncation packet;
- replace toy matrices with source BRST-complete W, photon-Z, and Higgs
  inverse two-point blocks;
- declare each pole coordinate as either `GeV_squared` or
  `dimensionless_E_star_squared`. The latter is converted only after the
  verified clock receipt supplies (E_\star); `dimensionless_control` can
  never emit a GeV readout;
- provide Ward/Slavnov--Taylor, Nielsen, sheet, residue, uncertainty, and
  refinement receipts;
- freeze the prospective claim manifest before target comparison.

Only the conjunction of all promotion gates can produce physical
`(M_W,Gamma_W,M_Z,Gamma_Z,M_H,Gamma_H)` intervals.

## Final source-emission boundary

The generic mathematical implication stack is complete; this backend is not
waiting for a new stochastic simulation method. Explicit model-extension
counterfamilies show that the structural OPH branch alone does not select the
D10 repair character, D11 split, absolute scale, or physical pole kernel. A
formal path table is not sufficient either, because any finite polynomial can
be encoded one monomial per weighted path.

The remaining inputs are four independently generated source packets:

- `C_clk`: factorized electromagnetic, electron, nuclear, and atomic clock
  Hamiltonian data;
- `C_10`: non-vacuous D10 transitions, normalized path measure, quotient,
  response map, exhaustive enumeration, and rigidity certificate;
- `C_11`: D11 split-character and rigidity carrier;
- `P_pole`: BRST-complete W, neutral photon/Z, and Higgs two-point kernels.

Given those inputs, the existing numerical lane can certify clock refinement,
RG/matching stability, Rouché pole isolation, mass/width readout, and pole
displacement bounds. Runtime target separation remains distinct from
historical blindness, which requires disclosure and a prospective frozen
claim.

## Files

- `oph_fpe/bosons/source_clock.py`
- `oph_fpe/bosons/rg_transport.py`
- `oph_fpe/bosons/brst_blocks.py`
- `oph_fpe/bosons/pole_enclosure.py`
- `oph_fpe/bosons/pipeline.py`
- `schemas/bosons/wzh_source_closure_run.schema.json`
- `schemas/bosons/wzh_source_closure_receipt.schema.json`
- `configs/bosons/wzh_source_closure_diagnostic_v1.yml`

Run bundles contain the frozen config, source hashes, the complete receipt, and
the bundle manifest. They should be imported back into
`reverse-engineering-reality` only after independent verification.
