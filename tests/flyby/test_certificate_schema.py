import json

from oph_fpe.flyby import STATUS_ENUM, write_closeflyby_public_certificates


def test_public_certificates_are_fail_closed(tmp_path) -> None:
    result = write_closeflyby_public_certificates(tmp_path)
    assert result["certificate_count"] == 12
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "closeflyby_summary.md").exists()

    for cert_path in tmp_path.glob("*.closeflyby.json"):
        cert = json.loads(cert_path.read_text(encoding="utf-8"))
        for key in (
            "certificate_id",
            "flyby_id",
            "public_row",
            "anderson_comparator",
            "raw_od_receipt",
            "models",
            "results",
            "tests",
            "status",
            "hashes",
        ):
            assert key in cert
        assert cert["status"] in STATUS_ENUM
        assert cert["status"] == "OD_REPLAY_PENDING"
        assert not cert["status"].startswith("SOLVED")
        assert cert["results"]["A_hist_mm_s"] is None
        assert cert["results"]["A_full_mm_s"] is None
        assert cert["results"]["A_proj_mm_s"] is None


def test_near_certificate_records_raw_manifest(tmp_path) -> None:
    write_closeflyby_public_certificates(tmp_path)
    cert = json.loads((tmp_path / "NEAR.closeflyby.json").read_text(encoding="utf-8"))
    assert cert["raw_od_receipt"]["receipt_status"] == "RAW_TRACKING_AVAILABLE_REPLAY_PENDING"
    assert cert["raw_od_receipt"]["tracking_files"]
    assert cert["anderson_comparator"]["prediction_mm_s"] > 13.0
