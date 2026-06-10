from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_IMPORTS = [
    "oph_fpe.claims",
    "oph_universe.arrow.entropy",
    "oph_universe.arrow.scenarios",
    "oph_fpe.constants.oph_pixel",
    "oph_fpe.evidence.hashes",
    "oph_fpe.experiments",
    "oph_fpe.bulk.cap_normals",
    "oph_fpe.bulk.cap_geometry",
    "oph_fpe.bulk.cap_profile_geometry",
    "oph_fpe.bulk.h3_chart",
    "oph_fpe.bulk.h3_response_fit",
    "oph_fpe.bulk.modular_probe",
    "oph_fpe.bulk.modular_response_kernel",
    "oph_fpe.bulk.neutral_bulk",
    "oph_fpe.bulk.observer_reconstruction",
    "oph_fpe.bulk.record_to_h3",
    "oph_fpe.bulk.proof_certificate",
    "oph_fpe.cosmology.selector_elimination",
    "oph_fpe.cosmology.finite_certificates",
    "oph_fpe.cosmology.ba_kernel",
    "oph_fpe.cosmology.camb_adapter",
    "oph_fpe.cosmology.oph_cmb_adapter",
    "oph_fpe.cosmology.boltzmann_inputs",
    "oph_fpe.cosmology.comparable_data",
    "oph_fpe.cosmology.finite_repair_transition_clock",
    "oph_fpe.cosmology.physical_cmb_contract",
    "oph_fpe.cosmology.physical_cmb_prediction",
    "oph_fpe.defects.array_s3_holonomy",
    "oph_fpe.defects.controlled_assay",
    "oph_fpe.observers.objects",
]

REQUIRED_PATHS = [
    "experiments/arrow_configs/faithful_record_chain.yaml",
    "experiments/arrow_configs/hidden_export_sweep.yaml",
    "experiments/arrow_configs/fake_past_sweep.yaml",
    "experiments/arrow_configs/janus_neck.yaml",
    "experiments/arrow_configs/record_reversal.yaml",
    "experiments/arrow_configs/coarse_grain_refinement.yaml",
]


def main() -> int:
    failed: list[tuple[str, str]] = []
    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - exercised as a command.
            failed.append((name, repr(exc)))

    missing_paths = [path for path in REQUIRED_PATHS if not (REPO_ROOT / path).exists()]

    if failed:
        for name, exc in failed:
            print(f"IMPORT_FAIL {name}: {exc}")
    if missing_paths:
        for path in missing_paths:
            print(f"MISSING_PATH {path}")
    if failed or missing_paths:
        return 1

    print("BUNDLE_IMPORTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
