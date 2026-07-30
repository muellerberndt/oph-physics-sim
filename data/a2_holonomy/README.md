# A2 holonomy and twelve-port current audit

The stored report separates the exact compact Lie-type classifier from the
unbuilt source-current producer.

The classifier uses the carrier-derived twelve-dimensional port module, its
one-dimensional fixed space, the A1 compact response premise, the A2
endogenous inner-holonomy premise, and the classical list of compact simple
Lie-algebra dimensions below twelve. It leaves one branch:

```text
centre dimension 1
simple-factor dimensions 3 + 8
Lie type u(1) + su(2) + su(3)
```

The source audit fails closed. The released response artifacts do not contain
ordered two-sided response histories, reconstructed infinitesimal generators,
an exact commutator table, or closed overlap words implemented by that same
current. The report therefore sets the physical current receipt to `false`.

Rebuild the report from the repository root:

```bash
python3 -m oph_fpe.gauge.a2_holonomy_selector \
  --carrier-manifest tests/fixtures/echosahedral_federation_reference.json \
  --out data/a2_holonomy/a2_holonomy_current_selector_report.json
python3 -m pytest -q tests/test_a2_holonomy_selector.py
```

The scalar seam-equalizer control is stored separately under
`data/repair_closure/`. It proves that those repair maps generate
`so(11)` at Lie closure and cannot be relabelled as the desired
twelve-dimensional current.
