# Pro Handoff: P/N Scale Bridge and OPH-FPE Claim-Boundary Fixes

Date: 2026-06-11

This handoff is self-contained. The Pro model should not rely on local paths outside the attached package. All referenced files are relative to this handoff directory or the included zip.

## Task

Plan the simulator fixes and next execution steps for OPH-FPE after the clarified P/N/G result:

> OPH's novelty is the separation of fixed dimensionless observer geometry from dimensional scale selection. The local P closure and global N closure fix exact dimensionless invariants. The local edge-area law then forces `G_geom = ell_star^2` because P cancels. To express this in SI units, an independent scale bridge supplies `B_ell = Lambda_star N_star`. Once that branch coordinate is fixed, `ell_star^2` and `G_SI` are no longer adjustable. The simulator must therefore report P/N invariants and scale-bridge eligibility separately, and must not label any P,N-only computation as a dimensionful G derivation.

Please return a concrete patch plan: files to edit, new report schema, compatibility strategy for old JSON keys, tests to add/update, and next run/verification commands.

## Attached Package

The zip sibling to this file contains:

- `source_notes/pasted_text_pn_scaling_theorem.txt`
- `source_notes/pasted_text_scale_bridge_branch_locator.txt`
- `source_notes/compact_proof_of_oph.tex`
- `package_root/oph-physics-sim/...`

The simulator checkout included here is a focused code/data snapshot assembled from the existing June 10 handoff package plus base files from the submodule Git object store. It is not a full repo checkout.

## Core Theorem

The two OPH closures are:

```text
P_star = a_cell / ell_star^2

N_star = A_dS / (4 ell_star^2)
       = 3 pi / (Lambda_star ell_star^2)
```

Therefore:

```text
Lambda_star ell_star^2 = 3 pi / N_star

Lambda_star a_cell = 3 pi P_star / N_star
```

Define:

```text
B_ell := Lambda_star N_star
```

Then:

```text
B_ell ell_star^2 = 3 pi
B_ell a_cell = 3 pi P_star
```

These are exact dimensionless OPH geometry invariants.

## Scaling Boundary

`P_star` and `N_star` are dimensionless. Under a global length rescaling:

```text
ell_star^2 -> lambda^2 ell_star^2
a_cell -> lambda^2 a_cell
Lambda_star -> lambda^-2 Lambda_star
```

`P_star` and `N_star` are unchanged, while:

```text
B_ell = Lambda_star N_star -> lambda^-2 B_ell
```

So P and N alone cannot determine:

```text
B_ell in m^-2
ell_star^2 in m^2
G_SI
```

This is a theorem boundary, not an algebra gap.

## Scale Bridge

The dimensionful bridge is:

```text
B_ell = Lambda_star N_star
ell_star^2 = 3 pi / B_ell
```

where `Lambda_star` must be an independent observer-facing de Sitter curvature coordinate or an equivalent scale certificate. It must not be backed out from Newton's constant.

Target branch certificate:

```text
B_ell =
3.6078739146803215760518414801601725476877083072171853821242070198789871988886
x 10^70 m^-2

ell_star^2 =
2.61228030237427777887347769215451001461202676866
x 10^-70 m^2

G_SI =
6.6742999959105279999999999999999999999999999999986
x 10^-11 m^3 kg^-1 s^-2
```

Observed de Sitter branch-locator example:

```text
Lambda_obs = 3 Omega_Lambda H0^2 / c^2

Using H0 = 67.4 km/s/Mpc and Omega_Lambda ~= 0.685:
Lambda_obs ~= 1.09091050283809 x 10^-52 m^-2

Then target N_star = B_ell / Lambda_obs
~= 3.3072134747022262700115238431483979537 x 10^122
```

This is compatible with the rough display `N_scr ~= 3.31 x 10^122`, but the rounded value is not enough for high-precision `G`.

## Local Gravity Cancellation

The local OPH cell has:

```text
a_cell = P_star ell_star^2
ellbar_shared = P_star / 4
```

Area law:

```text
G_geom = a_cell / (4 ellbar_shared)
       = P_star ell_star^2 / (4(P_star/4))
       = ell_star^2
```

P cancels from G. P's role is structural: it ties the local cell-area and edge-entropy readings to the same observer-cell unit.

SI display:

```text
G_SI = c^3 ell_star^2 / hbar
```

`c` and `h` are exact SI defining constants, so once `ell_star^2` is fixed by the scale branch, `G_SI` is forced.

## Current Simulator State In Included Data

The package includes current reports from:

```text
package_root/oph-physics-sim/runs/stage318_measurement_pack_clean_two_seed_cmb_h3_20260610/
```

Important gate values:

```text
screen_capacity_observed_branch_available = true
screen_capacity_finite_fixed_point_solved = false
repair_scale_closure_numeric_match = true
repair_scale_24_rounds_derived = false
finite_collar_boltzmann_physical_certificate = false
finite_collar_cmb_projection_physical_k = false
```

The current simulator is already mostly disciplined:

- P is used as cap area / entropy normalization, not as a dimension-forcing input.
- Finite patch counts remain regulators, not cosmic horizon capacity.
- Screen-capacity reports emit `Lambda*l_P^2 = 3*pi/N_CRC`.
- `N_CRC_fixed_point_solved_from_finite_simulator` is false.
- Physical CMB/Boltzmann promotion is blocked by physical k/a calibration, finite input, CDM-limit, and likelihood gates.

## Main Risk To Fix

The repair-scale lane currently has language and keys that can be overread as `P -> N`:

```text
capacity_predicted_from_local_P
Lambda_lP2_predicted_from_local_P
N_CRC_predicted_from_P
N_CRC(P)
mean predicted N_CRC from P
```

After the scaling theorem, this must be reframed as:

```text
capacity_implied_by_declared_repair_depth_ansatz
Lambda_lP2_implied_by_declared_repair_depth_ansatz
relative_error_ansatz_capacity_vs_declared_N_CRC
```

Compatibility can preserve old keys for one release, but reports and markdown must say clearly that this is not a derivation of N from P and not a dimensionful scale derivation.

## Relevant Code Hotspots

All paths are relative to `package_root/oph-physics-sim/` in the package.

Core P/scale files:

```text
oph_fpe/constants/oph_pixel.py
oph_fpe/core/pixel_scale.py
oph_fpe/cosmology/oph_constants.py
```

Scale/capacity reports:

```text
oph_fpe/cosmology/screen_capacity.py
oph_fpe/cosmology/repair_scale_closure.py
oph_fpe/scale/scale_compressed_repair.py
docs/repair_rounds.txt
```

CMB and physical-readiness gates:

```text
oph_fpe/cosmology/physical_cmb_contract.py
oph_fpe/cosmology/physical_cmb_prediction.py
oph_fpe/cosmology/finite_collar_boltzmann_bundle.py
oph_fpe/cosmology/finite_collar_projection.py
```

Summaries / downstream wording:

```text
oph_fpe/cosmology/comparable_data.py
oph_fpe/cosmology/camb_adapter.py
oph_fpe/measurement_pack.py
oph_fpe/bulk/proof_certificate.py
oph_fpe/cli.py
```

Tests included:

```text
tests/test_measurement_pack.py
tests/test_comparable_data.py
tests/test_bulk_proof_certificate.py
tests/test_pixel_scale.py
tests/test_screen_capacity.py
tests/test_physical_cmb_contract.py
tests/test_physical_cmb_prediction.py
```

## Recommended New Report

Add:

```text
oph_fpe/cosmology/scale_bridge.py
```

Suggested output files:

```text
oph_scale_bridge_report.json
oph_scale_bridge_report.md
```

Suggested CLI command:

```text
python -m oph_fpe.cli scale-bridge-report --out runs/<out>
```

Optional arguments:

```text
--p-value <float>
--n-star <float>
--lambda-star-m2 <float>
--b-ell-m2 <float>
--mode none|direct-b-ell|de-sitter-curvature
```

Report schema:

```json
{
  "mode": "oph_scale_bridge_v0",
  "dimensionless_invariants": {
    "P_definition": "P_star = a_cell / ell_star^2",
    "N_definition": "N_star = 3*pi/(Lambda_star*ell_star^2)",
    "Lambda_ell_star_squared": "...",
    "Lambda_a_cell": "...",
    "B_ell_ell_star_squared": "3*pi",
    "B_ell_a_cell": "3*pi*P_star"
  },
  "scale_theorem": {
    "P_N_determine_dimensionful_scale": false,
    "rescaling_argument": "P and N are invariant under global length rescaling; B_ell scales as length^-2."
  },
  "scale_bridge": {
    "source": "none|direct_B_ell|independent_de_sitter_curvature_branch",
    "N_star": null,
    "Lambda_star_m^-2": null,
    "B_ell_m^-2": null,
    "ell_star_squared_m2": null,
    "G_SI": null
  },
  "gravity_readout": {
    "G_geom_formula": "a_cell/(4*ellbar_shared) = ell_star^2",
    "P_cancels_from_G": true,
    "P_role": "structural cell-area / edge-entropy identity"
  },
  "readiness_gates": {
    "P_closure_available": true,
    "N_capacity_branch_declared": true,
    "N_fixed_point_solved_from_finite_simulator": false,
    "independent_scale_bridge_supplied": false,
    "dimensionful_G_SI_eligible": false,
    "finite_simulator_derived_G_SI": false
  },
  "claim_boundary": "P and N fix exact dimensionless OPH geometry. A dimensionful SI scale requires an independent scale bridge."
}
```

If `B_ell` is provided directly, compute:

```text
ell_star_squared_m2 = 3*pi/B_ell
G_SI = c^3 * ell_star_squared_m2 / hbar
```

If `Lambda_star` and `N_star` are provided:

```text
B_ell = Lambda_star * N_star
ell_star_squared_m2 = 3*pi/B_ell
G_SI = c^3 * ell_star_squared_m2 / hbar
```

Do not mark `finite_simulator_derived_G_SI` true from any of this. The scale bridge can be independently supplied or observed-branch-located, but it is not finite-simulator-derived unless a future finite proof exists.

## Recommended Edits

1. `screen_capacity.py`
   - Keep `Lambda_lP2 = 3*pi/N_CRC`.
   - Add explicit fields that this is dimensionless.
   - Add a gate like `dimensionful_scale_bridge_supplied = false`.
   - Say the report does not determine `Lambda` in `m^-2`, `ell_star^2` in `m^2`, or `G_SI`.

2. `repair_scale_closure.py`
   - Add compatibility-preserving aliases:
     - new: `capacity_implied_by_declared_repair_depth_ansatz`
     - old: `capacity_predicted_from_local_P`
   - Same for `Lambda_lP2`.
   - Rewrite markdown lines from “N_CRC predicted from P” to “N_CRC implied by declared repair-depth ansatz”.
   - Strengthen claim boundary:
     “not a derivation of N from P, not a dimensionful SI scale bridge.”

3. `scale_compressed_repair.py`
   - Rename display strings and report keys where possible:
     - `N_CRC_predicted_from_P` should be compatibility-only.
     - Add `N_CRC_implied_by_repair_depth_ansatz`.

4. `comparable_data.py`
   - Replace summary wording:
     - “mean predicted N_CRC from P”
     - with “mean N_CRC implied by repair-depth ansatz”.

5. `camb_adapter.py` and `proof_certificate.py`
   - Avoid propagating `N_CRC_predicted_from_P` as if it were theorem-side derivation.
   - Preserve old fields only as compatibility metadata.

6. `physical_cmb_contract.py`
   - Add or reserve source label:
     `OPH_independent_scale_bridge_supplied`
   - Do not let theorem-side P/N constants pass physical k/a calibration.

7. `measurement_pack.py`
   - Add scale-bridge claims when report is present:
     - `scale_bridge_dimensionful_G_eligible`
     - `scale_bridge_finite_simulator_derived_G_SI`
   - Default false when no report.

8. `docs/repair_rounds.txt`
   - Update status language to say:
     `N_CRC = |g'(P)|^-48` is a declared repair-depth ansatz consistency relation, not a P-only derivation of N.

## Tests To Add Or Update

Add a new test module:

```text
tests/test_scale_bridge.py
```

Required cases:

1. No scale bridge supplied:
   - dimensionless invariants emitted
   - `independent_scale_bridge_supplied = false`
   - `dimensionful_G_SI_eligible = false`
   - `ell_star_squared_m2` absent or null
   - `G_SI` absent or null

2. Direct `B_ell` supplied:
   - `B_ell_m^-2` recorded
   - `ell_star_squared_m2 = 3*pi/B_ell`
   - `G_SI = c^3*ell_star_squared_m2/hbar`
   - `P_cancels_from_G = true`
   - `dimensionful_G_SI_eligible = true`
   - `finite_simulator_derived_G_SI = false`

3. De Sitter curvature branch supplied:
   - `B_ell = Lambda_star*N_star`
   - same downstream calculations as direct B.
   - source label is independent/observed branch locator, not finite derivation.

4. Scaling theorem:
   - rescaling leaves P and N invariant
   - B scales as lambda^-2

Update existing tests:

```text
tests/test_measurement_pack.py
tests/test_comparable_data.py
tests/test_screen_capacity.py
tests/test_physical_cmb_contract.py
```

They should assert the stronger no-overclaim wording/gates.

## Recommended Verification Commands

After implementing:

```bash
python3 -m pytest -q tests/test_scale_bridge.py tests/test_screen_capacity.py tests/test_measurement_pack.py tests/test_comparable_data.py tests/test_physical_cmb_contract.py tests/test_physical_cmb_prediction.py
```

Then regenerate a small focused report set:

```bash
python3 -m oph_fpe.cli scale-bridge-report --out runs/stageXXX_scale_bridge_no_bridge
python3 -m oph_fpe.cli scale-bridge-report --out runs/stageXXX_scale_bridge_direct_b --b-ell-m2 3.6078739146803216e70
python3 -m oph_fpe.cli screen-capacity-report --out runs/stageXXX_screen_capacity
python3 -m oph_fpe.cli repair-scale-closure --out runs/stageXXX_repair_scale
```

Then export/refresh measurement pack if available in the current repo workflow.

## Desired Pro Output

Please return:

1. A prioritized patch plan.
2. Exact schema additions and compatibility aliases.
3. A list of files and tests to modify.
4. Any risky ambiguity to resolve before code edits.
5. A compact next-run plan for the simulator after these claim-boundary fixes.

