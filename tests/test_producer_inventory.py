from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPOSITORY_ROOT / "tools" / "build_producer_inventory.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "build_producer_inventory", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase0_producer_inventory_is_complete_and_current():
    generator = _load_generator()

    assert generator.validate_inventory() == []
    assert generator.OUTPUT_PATH.read_text(encoding="utf-8") == (
        generator.render_inventory()
    )

    rows = {row.row_id: row for row in generator.ROWS}
    assert {
        "WZH_V1_DECLARATIONS",
        "E1_NULL_STRESS",
        "E3_SMALL_BALL_AREA",
        "E5_LAMBDA_CONSTANCY",
        "H2_MEASURED_OVERLAP",
        "H2_STRICT_NEUTRAL",
        "H2_HASH_TOKEN_LEGACY",
        "POSITIVE_GEOMETRY_PLUGIN",
        "K1_GAUGE_COVARIANT_MISMATCH",
    } <= set(rows)
    for row in rows.values():
        if row.producer is None:
            assert row.producer_status in {
                "DECLARED_INPUT_NONPROMOTING",
                "NO_PRODUCER_NOT_TESTABLE",
            }
        if row.public_status == "NOT_TESTABLE":
            assert row.producer_status in {
                "NO_PRODUCER_NOT_TESTABLE",
                "INVENTORY_ONLY_NOT_TESTABLE",
                "PRODUCED_NONPROMOTING",
            }
