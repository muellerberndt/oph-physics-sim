from oph_fpe.flyby import synthetic_projection_validation, write_synthetic_closeflyby_validation


def test_synthetic_projection_validates_pipeline_without_historical_closure() -> None:
    validation = synthetic_projection_validation()
    cert = validation["certificate"]
    assert cert["status"] == "SYNTHETIC_PIPELINE_VALIDATED"
    assert abs(validation["A_proj_mm_s"] - (validation["A_hist_mm_s"] - validation["A_full_mm_s"])) < 1.0e-12
    assert "cannot close any historical flyby row" in " ".join(cert["nonclaims"])


def test_synthetic_validation_writer(tmp_path) -> None:
    result = write_synthetic_closeflyby_validation(tmp_path)
    assert result["status"] == "SYNTHETIC_PIPELINE_VALIDATED"
    assert (tmp_path / "synthetic_closeflyby_certificate.json").exists()
    assert (tmp_path / "synthetic_projection_validation.json").exists()
    assert (tmp_path / "projection_decomposition.csv").exists()
