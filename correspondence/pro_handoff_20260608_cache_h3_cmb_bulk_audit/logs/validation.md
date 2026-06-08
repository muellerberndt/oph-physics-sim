# Validation Log

Commands run before packaging:

```bash
python3 -m pytest -q \
  tests/test_modular_response_h3.py::test_geometry_cache_persists_cap_transport_map \
  tests/test_modular_response_h3.py::test_object_transition_kernel_is_signed_and_writes_controls \
  tests/test_modular_response_h3.py::test_modular_response_kernel_shape_and_summary
```

Result:

```text
3 passed
```

```bash
python3 -m pytest -q tests/test_bw_array.py::test_bw_array_writes_bw_report
```

Result:

```text
1 passed
```

Two-pass persistent geometry-cache smoke:

```text
first  {"memory_hits": 138, "mode": "finite_screen_geometry_cache", "persistent_cache_disk_hits": 0, "persistent_cache_disk_misses": 6, "persistent_cache_disk_writes": 6, "persistent_cache_enabled": true, "transport_map_count": 6}
second {"memory_hits": 138, "mode": "finite_screen_geometry_cache", "persistent_cache_disk_hits": 6, "persistent_cache_disk_misses": 0, "persistent_cache_disk_writes": 0, "persistent_cache_enabled": true, "transport_map_count": 6}
```

```bash
python3 tools/check_bundle_reproducibility.py
```

Result:

```text
HANDOFF_IMPORTS_OK
```

Process check:

```bash
ps -axo pid,etime,pcpu,pmem,command | rg 'oph_fpe|run-bw|h3-refit|cache_smoke|pytest' || true
```

Result: no active OPH simulation jobs were found, only the process-check command itself.
