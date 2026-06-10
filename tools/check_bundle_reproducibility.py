from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_IMPORTS = [
    "oph_fpe.claims",
    "oph_fpe.experiments",
    "oph_fpe.measurement_pack",
    "oph_fpe.constants.oph_pixel",
    "oph_fpe.bulk.cap_normals",
    "oph_fpe.bulk.cap_geometry",
    "oph_fpe.bulk.cap_profile_geometry",
    "oph_fpe.bulk.h3_chart",
    "oph_fpe.bulk.h3_response_fit",
    "oph_fpe.bulk.h3_refit",
    "oph_fpe.bulk.modular_probe",
    "oph_fpe.bulk.modular_response_kernel",
    "oph_fpe.bulk.observer_reconstruction",
    "oph_fpe.bulk.record_to_h3",
    "oph_fpe.bulk.proof_certificate",
    "oph_fpe.cosmology.cmb_compare",
    "oph_fpe.cosmology.cl_postprocess",
    "oph_fpe.cosmology.freezeout",
    "oph_fpe.cosmology.sync_gap",
    "oph_fpe.cosmology.adiabaticity",
    "oph_fpe.cosmology.parent_collar_ladder",
    "oph_fpe.cosmology.repair_scale_closure",
    "oph_fpe.cosmology.camb_adapter",
    "oph_fpe.cosmology.oph_cmb_adapter",
    "oph_fpe.cosmology.comparable_data",
    "oph_fpe.cosmology.unique_predictions",
    "oph_fpe.cosmology.shape_projection",
    "oph_fpe.cosmology.shape_certificates",
    "oph_fpe.cmb_fossil",
    "oph_fpe.cmb_fossil.screen_covariance",
    "oph_fpe.cmb_fossil.primordial_bridge",
    "oph_fpe.cmb_fossil.report",
    "oph_fpe.defects.array_s3_holonomy",
    "oph_fpe.defects.controlled_assay",
    "oph_fpe.microphysics.three_way_vertex",
    "oph_fpe.microphysics.dodeca_cell",
    "oph_fpe.microphysics.loop_particles",
    "oph_fpe.scale.scale_compressed_repair",
    "oph_fpe.scale.shape_substrate",
]


def main() -> int:
    failed: list[tuple[str, str]] = []
    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - command-level check
            failed.append((name, repr(exc)))

    if failed:
        for name, exc in failed:
            print(f"IMPORT_FAIL {name}: {exc}")
        return 1

    print("BUNDLE_IMPORTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
