"""Births, sampling and own-clock checks at the exporter boundary."""
import numpy as np
import pytest

from oph_defects.run_evidence import catalogue_census, audit_observer, passive_register_control


def cat(classes, *, seeded=0):
    n = len(classes)
    return {'cls': np.array(classes), 'tier': np.ones(n),
            'birth_sweep': np.r_[np.full(seeded, -1), np.zeros(n-seeded)],
            'birth_depth': np.r_[np.zeros(seeded), np.ones(n-seeded)],
            'death_sweep': np.full(n, -1), 'hops': np.zeros(n),
            'parent1': np.full(n, -1), 'parent2': np.full(n, -1), 'cause': np.zeros(n)}


def test_prepared_defects_are_not_pair_production():
    c = cat([1, 5, 1, 5, 3, 3], seeded=2)
    s = catalogue_census(c, complete=True)['tiers']['screen']
    assert s['pair_birth_events'] == {'0 -> 1 + 5': 1, '0 -> 2 + 4': 0, '0 -> 3 + 3': 1}
    assert s['classes']['1']['born']['prepared'] == 1
    assert s['classes']['1']['born']['conjugate_pair'] == 1


def test_view_rows_cannot_be_inflated_into_pair_events():
    s = catalogue_census(cat([1, 1, 3]), complete=False)['tiers']['screen']
    assert s['pair_birth_events'] is None
    assert s['classes']['1']['ever'] == 2


def test_complete_unpaired_births_fail():
    with pytest.raises(ValueError, match='unmatched'):
        catalogue_census(cat([1, 3]), complete=True)


def test_signed_holonomy_is_not_silently_relabelled():
    with pytest.raises(ValueError, match='Z6'):
        catalogue_census(cat([255, 1]), complete=True)


def test_clock_replay_counts_holds_and_uses_causal_cut():
    x = np.zeros(12, dtype=int); x[0] = 2
    x1 = x.copy(); x1[0] -= 1; x1[1] += 1
    x2 = x1.copy(); x2[1] -= 1
    ob = {'carrier': np.array(42), 'patch_cells': np.array([42, 43]),
          'patch_seams': np.array([[0, 1, 100], [1, 12, 101]]),
          'loads_initial': x, 'loads_final': x2,
          'tape_seam': np.array([100, 101, 100]), 'tape_current': np.array([1, 1, 0]),
          'loads_by_tick': np.array([x, x1, x2, x2]),
          'tape_depth': np.array([1, 3, 4]), 'cut_depth': np.array([0, 1, 2, 3, 4, -1]),
          'cut_ticks': np.array([0, 1, 1, 2, 3, 3]), 'frame': np.eye(3)}
    a = audit_observer(ob, {})
    assert a['incident_ticks'] == 3 and a['hold_ticks'] == 1
    assert a['final_loads_match'] and a['load_row_mismatches'] == 0
    assert a['cut_tick_mismatches'] == 0
    ob['cut_ticks'][2] = 2
    assert audit_observer(ob, {})['cut_tick_mismatches'] == 1


def test_register_can_change_without_changing_native_observations():
    c = passive_register_control()
    assert c['all_load_current_contrast_total_depth_records_equal']
    assert c['register_on_identity_count'] > 0
    assert c['register_off_identity_count'] == 0
