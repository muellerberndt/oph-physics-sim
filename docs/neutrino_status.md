# Neutrino Status Contract

`oph-physics-sim` currently has no source-derived OPH neutrino mass prediction.
Neutrino outputs are divided into three non-interchangeable branches:

1. `oph_neutrino_mass_status` is fail-closed. Its masses and mass sum are null,
   `available` is false, and `public_promotion_allowed` is false.
2. `conventional_camb_baseline` is the solver reference
   `sum_mnu_0.06eV_one_massive_two_massless`. It may be propagated through
   CAMB/CLASS and compressed cosmology checks, but it never counts as an OPH
   prediction.
3. `historical_rejected_weighted_cycle_benchmark` is hidden by default. It can
   be included only with `--include-rejected-weighted-cycle-benchmark`; even
   then it remains retrospective, target-informed, rejected by the declared
   NuFIT 6.1 compatibility gate, and permanently non-promotable.

The machine contract is
`schemas/cosmology/neutrino_status.schema.json`. Generate a default report with:

```bash
oph-fpe oph-cnb-neutrinos --out <run>/neutrinos
```

The explicit historical audit form is:

```bash
oph-fpe oph-cnb-neutrinos \
  --include-rejected-weighted-cycle-benchmark \
  --out <run>/neutrinos
```

The default H0/S8 calculator likewise uses the conventional 0.06 eV input and
labels it as such. No report may map the rejected weighted-cycle row into an
OPH mass receipt, a physical CMB receipt, or a public prediction flag.
