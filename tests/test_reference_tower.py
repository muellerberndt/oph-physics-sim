from __future__ import annotations

import unittest
import json
from pathlib import Path

import numpy as np

from oph_fpe.scale.reference_tower import (
    EdgeCenterFactor,
    PresentationCircuit,
    PresentationGate,
    ReferenceTower,
    RegulatorStage,
    TowerError,
    TowerStage,
    certify_positive_transfer,
    issue361_certificate_report,
    orthogonal_projection_from_partition,
    random_unitary,
    reference_tower_certificate,
    repair_generator,
    transfer_matrix,
)


class ReferenceTowerTests(unittest.TestCase):
    def setUp(self) -> None:
        rho0 = np.diag([0.7, 0.3]).astype(np.complex128)
        rho1 = np.diag([0.6, 0.4]).astype(np.complex128)
        rho2 = np.diag([0.8, 0.2]).astype(np.complex128)
        self.tower = ReferenceTower(
            [
                TowerStage((rho0,), random_unitary(2, 11)),
                TowerStage((rho0, rho1), random_unitary(4, 12)),
                TowerStage((rho0, rho1, rho2), random_unitary(8, 13)),
            ]
        )

    def test_tower_identities(self) -> None:
        for coarse, fine in [(0, 1), (1, 2), (0, 2)]:
            defects = self.tower.verify_pair(coarse, fine)
            self.assertLess(max(defects.values()), 1e-8)

    def test_reference_state_has_zero_transport_error(self) -> None:
        fine = 2
        coarse = 0
        error = self.tower.transported_state_error(
            self.tower.stages[fine].physical_density, fine, coarse
        )
        self.assertLess(error, 1e-10)

    def test_positive_transfer(self) -> None:
        stationary = np.array([0.1, 0.2, 0.3, 0.4])
        p1 = orthogonal_projection_from_partition(
            4, [[0, 1], [2, 3]], stationary
        )
        p2 = orthogonal_projection_from_partition(
            4, [[0, 2], [1, 3]], stationary
        )
        generator = repair_generator([p1, p2], [1.0, 0.7])
        transfer = transfer_matrix(generator, 0.1)
        certificate = certify_positive_transfer(transfer)
        self.assertEqual(certificate["positive"], 1.0)
        self.assertLess(certificate["hermitian_defect"], 1e-10)
        self.assertGreater(certificate["lambda_min"], 0.0)
        self.assertLessEqual(certificate["lambda_max"], 1.0 + 1e-10)

    def test_issue361_metadata_objects_validate_public_manifest(self) -> None:
        stage = RegulatorStage(
            stage_id="r0",
            m=0,
            a_s=1.0,
            a_t=0.25,
            physical_volume=(1.0, 1.0),
            boundary_label="round_cap",
            phase_label="reference",
            cellulation_hash="sha256:cell",
            factor_ids=("f0",),
            presentation_circuit_hash="sha256:circuit",
        )
        factor = EdgeCenterFactor(
            factor_id="f0",
            support="cap:C0",
            resolution_shell=0,
            sector_labels=("trivial",),
            matrix_dimensions=(2,),
            detail_state_eigenvalues=(0.7, 0.3),
            symmetry_labels=("A5",),
            gauge_labels=("support-visible",),
        )
        gate = PresentationGate(
            support_cells=("C0",),
            shell_level=0,
            physical_support_diameter=1.0,
            unitary_certificate=True,
            matrix_hash="sha256:gate",
        )
        circuit = PresentationCircuit(
            circuit_id="W0",
            circuit_hash="sha256:circuit",
            gates=(gate,),
        )

        self.assertEqual(stage.factor_ids, ("f0",))
        self.assertEqual(factor.matrix_dimensions, (2,))
        self.assertEqual(circuit.gates[0].support_cells, ("C0",))

    def test_issue361_metadata_rejects_nonfaithful_or_unverified_entries(self) -> None:
        with self.assertRaises(TowerError):
            EdgeCenterFactor(
                factor_id="bad",
                support="cap:C0",
                resolution_shell=0,
                sector_labels=("trivial",),
                matrix_dimensions=(2,),
                detail_state_eigenvalues=(1.0, 0.0),
            )
        with self.assertRaises(TowerError):
            PresentationGate(
                support_cells=("C0",),
                shell_level=0,
                physical_support_diameter=1.0,
                unitary_certificate=False,
                matrix_hash="sha256:gate",
            )

    def test_issue361_report_keeps_continuum_gates_conditional(self) -> None:
        report = reference_tower_certificate(self.tower, run_id="issue361-test")

        status = report["promotion_status"]
        self.assertTrue(report["issue361_certificate_receipt"])
        self.assertFalse(report["continuum_claim_receipt"])
        self.assertEqual(status["finite_regulator"], "pass")
        self.assertEqual(status["continuum_correlations"], "fail")
        self.assertEqual(status["modular_bw"], "fail")
        self.assertEqual(status["lorentzian_unitarity"], "conditional")
        self.assertEqual(status["yang_mills_identification"], "conditional")
        self.assertIn("four-dimensional OS/gauge certificate", " ".join(status["reasons"]))

    def test_issue361_report_rejects_bad_tower_identity(self) -> None:
        report = reference_tower_certificate(self.tower, run_id="issue361-test")
        report["reference_tower"]["identity_defects"]["modular_compatibility"] = 1.0

        checked = issue361_certificate_report(report)

        self.assertEqual(checked["promotion_status"]["finite_regulator"], "fail")
        self.assertFalse(checked["issue361_certificate_receipt"])

    def test_issue361_schema_file_exposes_required_promotion_fields(self) -> None:
        schema = json.loads(Path("docs/oph_issue361_certificate_schema.json").read_text(encoding="utf-8"))

        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["properties"]["schema_version"]["const"], "1.0")
        regulator = schema["properties"]["regulator"]["properties"]
        self.assertIn("stage_id", regulator)
        self.assertIn("a_s", regulator)
        self.assertIn("factor_ids", regulator)
        promotion = schema["properties"]["promotion_status"]["properties"]
        self.assertEqual(
            set(["finite_regulator", "continuum_correlations", "modular_bw", "lorentzian_unitarity", "yang_mills_identification"]),
            set(promotion).intersection(
                {
                    "finite_regulator",
                    "continuum_correlations",
                    "modular_bw",
                    "lorentzian_unitarity",
                    "yang_mills_identification",
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
