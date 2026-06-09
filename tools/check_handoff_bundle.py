from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_IMPORTS = [
    "oph_fpe.claims",
    "oph_fpe.constants.oph_pixel",
    "oph_fpe.bulk.cap_normals",
    "oph_fpe.bulk.cap_geometry",
    "oph_fpe.bulk.cap_profile_geometry",
    "oph_fpe.bulk.h3_chart",
    "oph_fpe.bulk.h3_response_fit",
    "oph_fpe.bulk.modular_probe",
    "oph_fpe.bulk.modular_response_kernel",
    "oph_fpe.bulk.observer_reconstruction",
    "oph_fpe.bulk.record_to_h3",
    "oph_fpe.bulk.proof_certificate",
    "oph_fpe.cosmology.selector_elimination",
    "oph_fpe.cosmology.finite_certificates",
    "oph_fpe.cosmology.camb_adapter",
    "oph_fpe.cosmology.oph_cmb_adapter",
    "oph_fpe.cosmology.boltzmann_inputs",
    "oph_fpe.cosmology.comparable_data",
    "oph_fpe.defects.array_s3_holonomy",
    "oph_fpe.defects.controlled_assay",
    "oph_fpe.observers.objects",
]


def main() -> int:
    failed: list[tuple[str, str]] = []
    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - exercised as a command.
            failed.append((name, repr(exc)))

    if failed:
        for name, exc in failed:
            print(f"IMPORT_FAIL {name}: {exc}")
        return 1

    print("BUNDLE_IMPORTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
