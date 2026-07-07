from copy import deepcopy

from oph_fpe.gallium import (
    build_template_certificate,
    compute_mdg_forward_comparison,
    load_json,
    validate_certificate,
    validate_no_target_leak,
    write_ga71_template_bundle,
)
from oph_fpe.gallium.ga71_td import DEFAULT_MANIFEST


def test_mdg_forward_comparison_is_diagnostic() -> None:
    result = compute_mdg_forward_comparison()
    assert abs(result["kappa_51Cr"] - 4.50 / 5.28) < 1.0e-12
    assert abs(result["kappa_37Ar"] - 5.24 / 6.32) < 1.0e-12
    best_inner = next(row for row in result["rows"] if row["run_id"] == "BEST_Cr_R1")
    assert abs(best_inner["corrected_R"] - 1.021) < 0.002
    assert result["all_source_weighted_mean"]["covariance_model"] == "diagonal_diagnostic"
    assert result["status"] == "COMPARISON_ONLY_NOT_SOURCE_CERTIFICATE"


def test_no_target_leak_template_passes_structure_only() -> None:
    manifest = load_json(DEFAULT_MANIFEST)
    passed, dag = validate_no_target_leak(manifest)
    assert passed
    assert dag["protected_failures"] == []


def test_no_target_leak_fails_if_gallium_ratio_feeds_density() -> None:
    manifest = load_json(DEFAULT_MANIFEST)
    tainted = deepcopy(manifest)
    tainted["nodes"].append(
        {
            "id": "GALLEX_R",
            "type": "comparison_only",
            "path_or_reference": "forbidden target",
            "sha256": "x",
            "parents": [],
            "measurement_use_flag": "FORBIDDEN_TARGET",
            "forbidden_target_ancestor": True,
        }
    )
    for node in tainted["nodes"]:
        if node["id"] == "rho_TD_71":
            node["parents"].append("GALLEX_R")
    passed, dag = validate_no_target_leak(tainted)
    assert not passed
    assert "rho_TD_71" in dag["protected_failures"]


def test_template_certificate_is_not_promotion_valid() -> None:
    cert = build_template_certificate()
    validation = validate_certificate(cert)
    assert cert["status"] == "TEMPLATE_NOT_VALID_FOR_PROMOTION"
    assert validation["valid_json_shape"]
    assert not validation["promotion_valid"]
    assert "SOURCE_NUCLEAR_SELECTOR" in validation["pending_gates"]


def test_write_template_bundle(tmp_path) -> None:
    report = write_ga71_template_bundle(tmp_path)
    assert report["status"] == "TEMPLATE_NOT_VALID_FOR_PROMOTION"
    assert not report["promotion_valid"]
    assert (tmp_path / "GA71_TD_SOURCE_CERTIFICATE.template.json").exists()
    assert (tmp_path / "GA71_TD_SOURCE_CERTIFICATE.schema.json").exists()
    assert (tmp_path / "no_target_leak.template.json").exists()
    assert (tmp_path / "mdg_forward_compare.diagnostic.json").exists()
