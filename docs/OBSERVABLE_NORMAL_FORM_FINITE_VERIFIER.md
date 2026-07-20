# Finite observable-normal-form verifier

`oph_fpe.quotient.observable_normal_form` implements two exact, finite theorem
clients:

- `verify_observation_determined_normal_form(...)` checks observation
  preservation, equality of computational terminals and the declared
  consistent set, identification of consistent states modulo a supplied
  quotient, and the stronger agreement of normal endpoints reached from
  different sources carrying the same protected observation.
- `recognize_conditional_resampling_kernel(...)` checks an exact rational
  Markov table against R1 fiber support, R2 equal rows within an observation
  fiber, R3 weighted detailed balance, and the independently recomputed
  weighted-fiber formula.

The first client is an executable instance of the machine-checked theorem
`boundaryIdentifiesModulo_iff_observerEndpointUniqueModulo`. It closes an
important gap in the older finite repair exhaustor: a unique endpoint for each
source does not imply a common endpoint for two distinct sources with the same
observer-visible boundary.

The second client uses the machine-checked finite algebraic core of
`kernel_eq_conditionalResamplingKernel_iff_recognition`. Inputs use `int` or
`Fraction`; floats and booleans are rejected rather than tolerance checked.
The simulator additionally requires the supplied positive weights to sum
exactly to one, matching the probability-space hypothesis in the manuscript.

## Claim boundary

Both APIs are conditional finite-table checks. They do not authenticate that:

- the supplied state enumeration exhausts an external simulator run;
- the supplied transition table exhausts every physically allowed repair;
- a kernel producer was independent of the target fiber-average formula;
- the observation, consistency predicate, quotient, or dynamics has a physical
  source; or
- a finite result has a stable refinement or continuum limit.

Every returned report therefore fixes these fields to `False`:

- `physical_promotion`
- `scale_authorized`
- `demo_assumption_accepted_as_evidence`

Conditional resampling also fixes representative selection, spectral-gap,
convergence-rate, and external-kernel-provenance receipts to `False`. Averaging
over a fiber is not selection of one state from that fiber and is not a mixing
rate for another repair dynamics.

The normal-form completeness flags default to `False`. A caller must declare
them explicitly, and the report still records
`external_completeness_authenticated = False`. Dependencies on scheduler,
worker, presentation-only, or downstream target fields fail the theorem
application firewall.

## Relationship to physical promotion

The output is not registered as an emergence-ladder receipt. A future
production adapter would need a closed on-disk manifest that hash-binds the
enumerated states, transition relation, protected observation, consistency
audit, quotient map, and independent producer/checker identities. Only that
adapter could discharge the finite inventory/provenance obligations. The
normal-form theorem would remain a necessary mathematical step, never a source
of geometry, gravity, a clock, Standard-Model content, or scale authorization.

## Tests

Run:

```bash
python -m pytest -q tests/test_observable_normal_form.py
```

The tests cover the checked two-bit positive example, the coarse-observation
counterexample, same-source branching, observation leakage, incomplete-table
defaults, target/presentation firewalls, a reachable cycle, exact R1--R3
recognition, independent failures of R1/R2/R3, probability normalization, and
float rejection.

