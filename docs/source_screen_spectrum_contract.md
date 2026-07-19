# Source screen spectrum contract

The canonical contract is `scr330-radial-v2`. It separates source artifacts
from diagnostic sky comparisons. A packet carries quotient screen geometry,
the primitive collar release law and energy, infinitesimal reserve generator,
conformal precision, physical angular mode basis, radial lift, null-space
report, and transfer firewall on one dependency graph.

The source graph fails if it is empty or if any E4 ancestor is a measurement,
residual, transfer output, likelihood, posterior, fit, observed TT/TE/EE row,
or data-calibrated proxy. A passed source packet cannot claim physical
temperature or polarization spectra. E5 transfer is a distinct downstream
receipt and never feeds back into E4 ancestry.

The amplitude reducer pools total weight and total quadratic release energy
before division by the retained mode count; shard-local amplitudes remain
diagnostics. The tilt reducer consumes an emitted infinitesimal full-collar
generator density. The canonical clock requires the full-collar derivative
`P/24`, orientation half `theta=P/48`, and its source evidence bundle. A finite
one-step survival probability converted as `-log(u)/log(b)` is a finite-step
survival exponent, not the infinitesimal generator.

The source-family radial lift is one-dimensional only under the declared
dilation cocycle. The unrestricted radial operator emits singular values,
right-null basis, finite-window error, quadrature and stability diagnostics,
and a forward residual. The `SOURCE_DILATION` and `RADIAL_TOMOGRAPHY` branches
can satisfy their respective source gates. `PRIOR_CONTINUATION` is useful as a
regularized diagnostic but can never emit a source-derived E4 promotion.

The ten schema-enforced receipt identifiers are:

- `SCR330_SOURCE_SHELL_EMBEDDING_RECEIPT`
- `SCR330_PHYSICAL_MODE_BASIS_RECEIPT`
- `SCR330_RADIAL_DILATION_INTERTWINER_RECEIPT`
- `SCR330_THIN_SHELL_MELLIN_LIFT_RECEIPT`
- `SCR330_FINITE_WINDOW_KERNEL_RECEIPT`
- `SCR330_RADIAL_NULL_REPORT`
- `SCR330_RADIAL_FORWARD_RESIDUAL_RECEIPT`
- `SCR330_RADIAL_TOMOGRAPHY_RECEIPT`
- `SCR330_RADIAL_PROMOTION_RECEIPT`
- `SCR330_TRANSFER_FIREWALL_RECEIPT`

Every receipt uses the exact schema at
`schemas/cosmology/source_screen_spectrum_receipt.schema.json`, accepts only
literal booleans for pass fields, records blockers, and preserves the E4/E5
firewall. A true receipt always requires its own recomputed evidence payload;
`--claimed-pass`, an empty payload, or a placeholder all-zero digest cannot
promote a row. Generate a canonical JSON packet with:

```bash
python3 -m oph_fpe.cli scr330-radial-receipt \
  --source-dag path/to/source_dag.json \
  --receipt SCR330_RADIAL_NULL_REPORT --claim-tier E3 \
  --out runs/scr330_radial_receipt.json
```
