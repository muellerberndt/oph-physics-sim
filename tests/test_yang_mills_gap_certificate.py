import json
from pathlib import Path

from oph_fpe.gauge.yang_mills_gap import (
    FINITE_GAP_RECEIPT,
    yang_mills_gap_certificate_report,
    write_yang_mills_gap_certificate_report,
)


def test_yang_mills_gap_lane_emits_finite_su2_but_keeps_mass_gap_closed() -> None:
    report = yang_mills_gap_certificate_report(
        lattice_size=2,
        sweeps=6,
        seed=1234,
        refinement_lattice_sizes=(2,),
    )

    assert report["schema"] == "oph_yang_mills_gap_certificate_v0"
    assert report["gauge_group"]["name"] == "SU(2)"
    assert report["gauge_group"]["compact"] is True
    assert report["gauge_group"]["simple"] is True
    assert report["gauge_group"]["nonabelian"] is True
    assert report["lattice_gauge_stage"]["four_dimensional_wilson_lattice"] is True
    assert report[FINITE_GAP_RECEIPT] is True
    assert report["finite_nonabelian_gauge_gap_diagnostic_receipt"] is True
    assert report["finite_transfer_gap_diagnostic"]["finite_transfer_gap_proxy_receipt"] is True
    assert report["finite_lattice_diagnostics"]["canonical_serial_chain_replay_receipt"] is True
    assert report["YANG_MILLS_GAP_REPRODUCED_RECEIPT"] is False
    assert report["CLAY_YANG_MILLS_GAP_RECEIPT"] is False
    assert report["promotion_status"]["yang_mills_mass_gap"] == "not_promoted"
    assert "support_visible_extraction_receipt" in report["continuum_certificate"]["missing"]


def test_yang_mills_gap_lane_records_but_does_not_promote_external_continuum_fields() -> None:
    report = yang_mills_gap_certificate_report(
        lattice_size=2,
        sweeps=6,
        seed=1234,
        refinement_lattice_sizes=(2,),
        continuum_certificate={
            "support_visible_extraction_receipt": True,
            "renormalized_schwinger_convergence_receipt": True,
            "reflection_positivity_receipt": True,
            "euclidean_covariance_locality_receipt": True,
            "nontriviality_receipt": True,
            "transfer_intertwiner_convergence_receipt": True,
            "external_certificate_hash": "sha256:test",
        },
    )

    assert report["continuum_certificate"]["candidate_complete"] is True
    assert report["continuum_certificate"]["continuum_certificate_receipt"] is False
    assert report["continuum_certificate"]["trusted_external_verification"] is False
    assert report["YANG_MILLS_GAP_REPRODUCED_RECEIPT"] is False
    assert report["CLAY_YANG_MILLS_GAP_RECEIPT"] is False
    assert "Clay/Yang-Mills promotion remains disabled" in " ".join(report["promotion_status"]["reasons"])


def test_write_yang_mills_gap_certificate_exports_json_markdown_and_csv(tmp_path: Path) -> None:
    report = write_yang_mills_gap_certificate_report(
        tmp_path / "ym" / "yang_mills_gap_certificate_report.json",
        lattice_size=2,
        sweeps=6,
        seed=1234,
        refinement_lattice_sizes=(2,),
    )

    report_path = Path(report["report_path"])
    assert report_path.exists()
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["schema"] == "oph_yang_mills_gap_certificate_v0"
    assert report_path.with_suffix(".md").exists()
    assert report_path.with_name("yang_mills_gap_certificate_report_plaquette_trace.csv").exists()
    assert report_path.with_name("yang_mills_gap_certificate_report_wilson_loop_trace.csv").exists()
    assert report_path.with_name("yang_mills_gap_certificate_report_refinement_gap.csv").exists()
    assert report_path.with_name("yang_mills_gap_certificate_report_promotion_gates.csv").exists()
