"""Public viewer reference custody and path confinement."""
import json
from pathlib import Path
import pytest
from oph_defects.bundle_research import bundle, check, referenced_paths
from oph_defects.serve import Handler, VIEWER
from oph_defects.reference_repository import theory_repository


def test_reference_bundle_is_exact_and_rejects_tampering(tmp_path):
    src=tmp_path/'source';view=tmp_path/'viewer';src.mkdir();view.mkdir()
    (src/'entry.json').write_text('{"value":3}\n')
    (view/'app.js').write_text("fetch('/rl/entry.json')")
    data=bundle(src,view)
    assert len(data['files'])==1 and check(view)==data
    (view/'research/entry.json').write_text('{"value":4}\n')
    with pytest.raises(ValueError,match='content changed'):check(view)


def test_reference_menu_and_symlink_cannot_escape(tmp_path):
    src=tmp_path/'source';view=tmp_path/'viewer';src.mkdir();view.mkdir()
    (view/'app.js').write_text("fetch('/rl/../outside.json')")
    with pytest.raises(ValueError,match='escapes'):referenced_paths(view)
    (view/'app.js').write_text("fetch('/rl/alias.json')")
    outside=tmp_path/'outside.json';outside.write_text('{}')
    (src/'alias.json').symlink_to(outside)
    with pytest.raises(ValueError,match='escapes'):bundle(src,view)


def test_run_routes_confine_siblings_and_encoded_traversal(tmp_path):
    run=tmp_path/'run';run.mkdir()
    sibling=tmp_path/'run-secret';sibling.mkdir();(sibling/'payload.json').write_text('{}')
    handler=object.__new__(Handler);handler.runs={'safe':run}
    assert Path(handler.translate_path('/runs/safe/record.json'))==run/'record.json'
    for path in ('/runs/safe/../run-secret/payload.json','/runs/safe/%2e%2e/run-secret/payload.json'):
        assert Path(handler.translate_path(path))==VIEWER/'missing'
    assert Path(handler.translate_path('/lab/Linvalid.npz'))==VIEWER/'missing'
    assert Path(handler.translate_path('/rl/../readout.js'))==VIEWER/'missing'


def test_explicit_theory_checkout_is_respected(monkeypatch,tmp_path):
    monkeypatch.setenv('OPH_THEORY_REPOSITORY',str(tmp_path))
    assert theory_repository()==tmp_path.resolve()
