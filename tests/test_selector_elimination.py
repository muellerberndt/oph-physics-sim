from __future__ import annotations

import csv
import math
from pathlib import Path

from oph_fpe.constants.oph_pixel import OPHPixelConstants
from oph_fpe.cosmology.selector_elimination import (
    ir_kernel,
    selector_elimination_report,
    write_selector_elimination_report,
)


def test_selector_elimination_derives_exact_targets_without_source_dir():
    report = selector_elimination_report()
    pixel = OPHPixelConstants()
    expected_eta = math.e * (pixel.P - pixel.phi)

    assert report["THEOREM_SIDE_SELECTOR_ELIMINATION_RECEIPT"] is True
    assert report["SOURCE_PACKET_AUDIT_RECEIPT"] is False
    assert report["selector_elimination"]["q_IR_selector_removed"] is True
    assert report["selector_elimination"]["ell_IR_selector_removed"] is True
    assert report["selector_elimination"]["eta_R_free_selector_removed"] is False
    assert report["selector_elimination"]["eta_R_reduced_to_repair_clock_certificate"] is True
    assert report["cmb_ir_kernel"]["q_IR"] == 0.25
    assert report["cmb_ir_kernel"]["ell_IR"] == 32.0
    assert report["cmb_ir_kernel"]["N_frz_proxy"] == 1089
    assert math.isclose(report["scalar_tilt"]["eta_R"], expected_eta)
    assert abs(report["scalar_tilt"]["n_s"] - 0.964841143031) < 2.0e-12
    assert report["finite_lattice_derived"] is False
    assert report["physical_cmb_prediction"] is False


def test_selector_elimination_audits_v15_source_packet(tmp_path: Path):
    source = tmp_path / "cmb7"
    source.mkdir()
    (source / "OPH-CMB-Selector-Elimination-v1.5.md").write_text("v1.5", encoding="utf-8")
    _write_status_csv(source / "selector_elimination_status_v1_5.csv")
    _write_kernel_csv(source / "exact_ir_kernel_values_v1_5.csv", [2, 3, 32, 220])

    report = write_selector_elimination_report(source, tmp_path / "out")

    assert (tmp_path / "out" / "oph_cmb_selector_elimination_report.json").exists()
    assert (tmp_path / "out" / "oph_cmb_selector_elimination_report.md").exists()
    assert (tmp_path / "out" / "exact_ir_kernel_values_v1_5.csv").exists()
    assert report["SOURCE_PACKET_AUDIT_RECEIPT"] is True
    assert report["source_status_audit"]["q_ir_selector_removed"] is True
    assert report["source_status_audit"]["ell_ir_selector_removed"] is True
    assert report["source_status_audit"]["eta_r_reduced_to_repair_clock_certificate"] is True
    assert report["exact_ir_kernel_csv_audit"]["passed"] is True
    assert report["exact_ir_kernel_csv_audit"]["max_abs_error"] <= 5.0e-13


def _write_status_csv(path: Path) -> None:
    rows = [
        {
            "old_selector": "S1: eta_R = e alpha sqrt(pi)",
            "v1_5_status": "not free, but still requires the repair-clock normalization theorem/certificate",
            "replacement": "eta_R=kappa_rep alpha sqrt(pi)",
            "what_is_closed": "dependence on alpha",
            "what_remains": "derive kappa_rep=e",
        },
        {
            "old_selector": "S2: q_IR = 1/4 from four equipotent sectors",
            "v1_5_status": "removed as selector",
            "replacement": "affine zero-mode quarter reserve",
            "what_is_closed": "1/(1+3)",
            "what_remains": "validate freezeout branch",
        },
        {
            "old_selector": "S3: ell_IR = 32",
            "v1_5_status": "removed as selector",
            "replacement": "visible covariance rank",
            "what_is_closed": "F+V+1=33",
            "what_remains": "finite-register noncollapse",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_kernel_csv(path: Path, ells: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ell", "F_IR_exact_q1over4_L32"])
        writer.writeheader()
        for ell in ells:
            writer.writerow({"ell": ell, "F_IR_exact_q1over4_L32": float(ir_kernel(float(ell)))})
