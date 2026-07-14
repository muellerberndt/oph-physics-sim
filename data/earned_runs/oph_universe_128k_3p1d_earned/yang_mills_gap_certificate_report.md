# Yang-Mills Gap Certificate Lane

- schema: `oph_yang_mills_gap_certificate_v0`
- gauge group: `SU(2)`
- finite diagnostic receipt: `True`
- finite transfer gap estimate: `0.12204558417161115`
- finite gap floor estimate: `0.12204558417161115`
- Yang-Mills mass-gap reproduced: `False`
- Clay receipt: `False`

## Promotion Status

- finite_nonabelian_regulator: `pass`
- finite_positive_gap_floor: `pass`
- continuum_certificate: `pending`
- os_reconstruction: `pending`
- yang_mills_identification: `conditional`
- yang_mills_mass_gap: `not_promoted`

## Blockers

- missing continuum certificate field: support_visible_extraction_receipt
- missing continuum certificate field: renormalized_schwinger_convergence_receipt
- missing continuum certificate field: reflection_positivity_receipt
- missing continuum certificate field: euclidean_covariance_locality_receipt
- missing continuum certificate field: nontriviality_receipt
- missing continuum certificate field: transfer_intertwiner_convergence_receipt
- Clay/Yang-Mills promotion remains disabled in this simulator lane

## Claim Boundary

Finite SU(2) compact-simple nonabelian lattice diagnostic. A positive finite diagnostic receipt means this run emitted nonabelian Wilson-lattice data, deterministic replay, a finite transfer-gap proxy, and a finite reflection-Gram proxy. It is not a reproduction of the Yang-Mills mass gap. The Clay/Jaffe-Witten claim remains closed until the support-visible compact-gauge continuum certificate supplies Schwinger convergence, reflection positivity, Euclidean covariance/locality, nontriviality, and transfer/intertwiner convergence.
