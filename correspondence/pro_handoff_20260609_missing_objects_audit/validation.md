# Validation

Validated on 2026-06-09 after applying the Pro audit fixes.

```text
python3 tools/check_bundle_reproducibility.py
BUNDLE_IMPORTS_OK
```

Clean-unzip full test run:

```text
python3 -m pytest -q
291 passed in 28.11s
```

Local full test run:

```text
python3 -m pytest -q
291 passed in 27.29s
```

Current comparable-data headline after the receipt split:

```text
top-level bulk_3d_established: false
theorem-assisted H3 object preview: true
object H3 nonboundary population: false
strict neutral 3D bulk: false
finite certificate compiler-ready: true
theorem-grade finite inputs: false
proxy certificate: true
physical CMB prediction: false
```

