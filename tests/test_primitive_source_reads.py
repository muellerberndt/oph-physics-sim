"""Native binding, independent replay and adversarial primitive-read checks."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oph_fpe.bulk import primitive_source_reads as produce
from oph_fpe.bulk import verify_primitive_source_reads_independent as check


@pytest.fixture(scope="module")
def packet():
    return produce.produce()


def reseal(value):
    value["sha256"] = hashlib.sha256(check.canonical({k:v for k,v in value.items() if k != "sha256"})).hexdigest()
    return value


def test_native_driver_binding_and_complete_replay(packet):
    receipt = check.verify(packet, produce.ROOT)
    assert receipt["source_bytes_checked"] is True
    assert [r["commits"] for r in receipt["cases"]] == [8, 16, 24, 24, 48, 96, 24]
    assert [r["ideal_real_read_algebra"]["observation_rank"] for r in receipt["cases"]] == [17, 19, 23, 23, 23, 23, 23]
    assert receipt["quantum"]["certified_nonzero_unitary_entries"] == 144
    assert F(receipt["quantum"]["equal_input_intensities_output_gap_lower_bound"]) > 0
    assert receipt["M1_derived"] is False


def test_partial_word_rank_is_computed_from_its_actual_projection(packet):
    # Construct the operator on a basis, then eliminate; a full-sweep constant
    # is incorrect before the word has visited every seam.
    receipt = check.verify(packet)
    for raw, result in zip(packet['cases'], receipt['cases']):
        size = 12*raw['carriers']
        matrix = [[F(int(i == j)) for j in range(size)] for i in range(size)]
        budget = (len(raw['seams'])+15)//16
        for attempt in range(raw['cycles']*budget):
            _, a, b = raw['seams'][raw['order'][attempt % len(raw['order'])]]
            mean = [(u+v)/2 for u,v in zip(matrix[a],matrix[b])]
            matrix[a],matrix[b] = list(mean),list(mean)
        assert result['terminal_real_projection_rank'] == check.rank(matrix)
        assert result['full_sweep_real_projection_rank'] == 6*raw['carriers']
    assert [r['terminal_real_projection_rank'] for r in receipt['cases'][:3]] == [40,32,24]


@pytest.mark.parametrize("path,value", [
    (("scope", "M1_derived"), True),
    (("cases", 0, "carriers"), True),
    (("cases", 0, "commits"), 0),
    (("cases", 0, "noops"), 1),
    (("cases", 0, "native_log_sha256"), "sha256:"+"0"*64),
    (("cases", 2, "final_hex", 0), "0x0.0p+0"),
    (("cases", 4, "order"), []),
    (("cases", 4, "seams", 0, 1), 0),
    (("cases", 1, "initial_hex", 0), "nan"),
    (("cases", 2, "observer_records", 0, 3), 11),
    (("cases", 2, "observer_records", 0, 4, 0), "0x0.0p+0"),
    (("cases", 2, "observer_records"), []),
    (("quantum", "unitary_hex", 0, 0), ["0x0.0p+0", "0x0.0p+0"]),
    (("quantum", "phase_output_hex", 1), "0x1.0000000000000p-1"),
    (("quantum", "laplacian", 0, 0), 4),
    (("quantum", "uniform_snapshot_feedback", "native_next_port"), 0),
    (("quantum", "uniform_snapshot_feedback", "stabilizer_permutation"), list(range(12))),
    (("instruments", "threshold_output_hex", 1, 0), "0x1.0000000000000p-1"),
    (("instruments", "phase_lift_coherence_hex", 0, 0), "0x0.0p+0"),
    (("instruments", "twirl_output_sparse"), []),
    (("instruments", "dephased_output_sparse"), []),
    (("instruments", "next_read_hex", 0), "0x0.0p+0"),
    (("instruments", "extensions_selected_by_source"), True),
    (("instruments", "measured_recurrence", 2, "steps", 1, "positive_slot_entries"), 0),
    (("instruments", "measured_recurrence", 2, "steps", 2, "slot_support_sha256"), "0"*64),
    (("instruments", "measured_recurrence", 2, "coarse_transition_hex", 0), "0x0.0p+0"),
    (("instruments", "measured_recurrence"), []),
    (("instruments", "measured_recurrence", 0, "erasure", "cycle_flow"), [0]*48),
    (("instruments", "measured_recurrence", 0, "erasure", "inverse_flow_hex"), ["0x0.0p+0"]*48),
    (("instruments", "measured_recurrence", 0, "erasure", "first_outputs_hex", 0, 0), "0x0.0p+0"),
    (("instruments", "retained_flags"), []),
    (("instruments", "retained_flags", 2, "flags", 0), True),
    (("instruments", "retained_flags", 1, "branch_digest"), "0"*64),
    (("instruments", "retained_flags", 1, "recovered_digest"), "0"*64),
    (("instruments", "deterministic_archive"), []),
    (("instruments", "deterministic_archive", 2, "chord_differences"), []),
    (("instruments", "deterministic_archive", 2, "tree_seams"), [0]*15),
    (("instruments", "deterministic_archive", 2, "means", 0), "0"),
    (("instruments", "deterministic_archive", 2, "initial", 0), "0"),
    (("instruments", "deterministic_archive", 2, "recovered_sha256"), "0"*64),
    (("cases",), []),
])
def test_resealed_forgery(packet, path, value):
    bad = deepcopy(packet)
    target = bad
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises((ValueError, KeyError)):
        check.verify(reseal(bad))


def test_smallest_propagator_entries_cannot_be_forged_to_zero(packet):
    bad = deepcopy(packet)
    matrix = bad["quantum"]["unitary_hex"]
    i, j = min(((i,j) for i in range(12) for j in range(12)),
               key=lambda pair: sum(abs(float.fromhex(v)) for v in matrix[pair[0]][pair[1]]))
    matrix[i][j] = ["0x0.0p+0", "0x0.0p+0"]
    with pytest.raises(ValueError, match="Taylor"):
        check.verify(reseal(bad))


def test_normalization_recovers_an_unobserved_twelfth_coordinate():
    # Arbitrary matching diagnostic, not an exported registered federation.
    mate = {i:i+12 for i in range(12)} | {i+12:i for i in range(12)}
    mate.update({i:i+12 for i in range(24,36)})
    mate.update({i+12:i for i in range(24,36)})
    result = check.read_algebra(4, mate, set(range(11)))
    assert 23 not in result["directly_observed_slots"]
    assert 23 in result["recoverable_slots_with_normalization"]
    assert result["observation_rank"] == 22


def test_two_hidden_coordinates_are_an_actual_observation_kernel():
    n = 4
    # A full captured source topology, with every permitted operation included.
    row = produce.case(n, 16, 1)
    mate = {a:b for _,a,b in row["seams"]} | {b:a for _,a,b in row["seams"]}
    result = check.read_algebra(n, mate, set(range(12*n)))
    p, q = result["hidden_pair"]
    assert p//12 == q//12 != 0
    assert result["hidden_dimension"] == 21
    assert result["global_hash_distinguishes"] is True


def test_verifier_runs_without_any_simulator_or_numerical_import(packet, tmp_path):
    artifact = tmp_path/"packet.json"
    artifact.write_bytes(check.canonical(packet))
    script = '''
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'oph_fpe','numpy','scipy'}:
            raise AssertionError('forbidden import: '+fullname)
sys.meta_path.insert(0, Block())
module=runpy.run_path(sys.argv[1],run_name='independent')
print(module['verify'](module['strict_load'](sys.argv[2]))['M1_derived'])
'''
    result = subprocess.run([sys.executable, "-c", script, str(Path(check.__file__)), str(artifact)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("raw", ['{"x":0,"x":1}', '{"x":NaN}', '{"x":1.0}', '{}'])
def test_real_cli_rejects_trash(raw, tmp_path):
    artifact = tmp_path/"bad.json"
    artifact.write_text(raw, encoding="ascii")
    result = subprocess.run([sys.executable, str(Path(check.__file__)), str(artifact)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert '"packet_sha256"' not in result.stdout


def test_source_custody_rejects_real_file_tampering(packet, tmp_path):
    for name in packet["source"]["files"]:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((produce.ROOT/name).read_bytes())
    target = tmp_path/"oph_fpe/core/echosahedral_dynamics.py"
    target.write_bytes(target.read_bytes()+b"\n# changed source\n")
    with pytest.raises(ValueError, match="source custody"):
        check.verify(packet, tmp_path)


def test_skipped_pairs_are_not_committed_record_parents(packet):
    # A valid uniform preparation at the same operation interface, independent
    # of the Gaussian seed. Every attempt is a no-op, so no committed writer
    # can appear in the actual record graph even though the ideal word visits it.
    row = deepcopy(packet["cases"][2])
    row["initial_hex"] = row["final_hex"] = [(1/12).hex()]*48
    row["commits"], row["noops"] = 0, 32
    row["cycle_counts"] = [[0, 2]]*16
    row["native_log_sha256"] = "sha256:"+hashlib.sha256(b"").hexdigest()
    port = 0
    values = [check.clean(1/12)]*12
    for record in row["observer_records"]:
        record[3], record[4] = port, [v.hex() for v in values]
        port = (port+1+abs(round(1e6*sum((j+1)*v for j,v in enumerate(values))))%11)%12
    result = check.case(row, (4,16,1))
    assert result["observer_payload"]["repair_to_record_edges"] == []
    assert result["maximum_version"] == 0


def test_native_threshold_has_an_exact_affinity_defect():
    from oph_fpe.bulk.physical_h3_kms_source_capture import _visible_pair_mean
    delta = 2.0**-51
    def first(x):
        mean = _visible_pair_mean(x, 1-x)
        return x if mean is None else mean
    assert first(.5+delta)-(first(.5)+first(.5+2*delta))/2 == delta


def test_native_driver_commits_the_instrument_primitive_return_value(monkeypatch):
    from oph_fpe.bulk import physical_h3_kms_source_capture as driver
    original = driver._visible_pair_mean
    # A mutation must reach actual ledger writes, not merely the no-op gate.
    monkeypatch.setattr(driver, '_visible_pair_mean',
                        lambda a,b: None if original(a,b) is None else .25)
    config = driver._normalize_config({'carrier_count':4,'cycles':4,'seed':1729})
    dynamics, _, _, final, events = driver._source_dynamics(config,driver._build_federation(config))
    assert dynamics['repair_event_count'] > 0
    assert dynamics['REPAIR_ORDER_REPLAY_EXACT_RECEIPT'] is False
    assert events
    for event in events:
        for write in event['write_set']:
            assert write['value'] == .25
            carrier = int(write['carrier_id'].rsplit('-',1)[1])
            assert final[carrier,write['port']] == .25


def test_terminal_lift_keeps_the_registered_arithmetic():
    import numpy as np
    from oph_fpe.bulk.physical_h3_kms_source_capture import _terminal_complex_lift
    values = np.array([[0, 1, -1, 1j, -1j, 1+1j, -1-1j, 2, 3, 4, 5, 6]], dtype=complex)
    visible = np.arange(12).reshape(1,12)/16
    expected = np.sqrt(np.maximum(visible,0))*np.exp(1j*np.angle(values))
    assert np.array_equal(_terminal_complex_lift(values,visible), expected)


def test_quantum_extensions_have_positive_choi_and_correct_partial_trace():
    import numpy as np
    from oph_fpe.bulk.primitive_source_instruments import pair_twirl
    size=4
    pairs=[(0,1),(2,3)]
    def choi(channel):
        blocks=[]
        for i in range(size):
            line=[]
            for j in range(size):
                unit=np.zeros((size,size),dtype=complex);unit[i,j]=1
                line.append(channel(unit))
            blocks.append(line)
        return np.block(blocks)
    for channel in (lambda x: pair_twirl(x,pairs),
                    lambda x: np.diag(np.diag(pair_twirl(x,pairs)))):
        matrix=choi(channel)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-13
        partial=np.einsum('iaja->ij',matrix.reshape(size,size,size,size))
        assert np.array_equal(partial,np.eye(size))
    # A positive trace-preserving map is not enough: this catches transpose.
    assert np.linalg.eigvalsh(choi(lambda x:x.T)).min() < -.9


@pytest.mark.parametrize('bad', [None, [], [(0,1)], [(0,1),(1,2)], [(0,1),(2,True)], [(0,1,2),(3,)]])
def test_channel_extension_rejects_incomplete_or_invalid_matchings(bad):
    import numpy as np
    from oph_fpe.bulk.primitive_source_instruments import pair_twirl
    with pytest.raises(ValueError):
        pair_twirl(np.eye(4),bad)


def test_same_diagonal_repairs_do_not_fix_future_reads(packet):
    result=check.verify(packet)['instruments']
    assert F(result['subsequent_read_gap_lower_bound'])>0
    assert result['native_threshold_rule_is_quantum_channel'] is False
    assert result['extensions_selected_by_source'] is False
    assert all(r['carrier_totals_are_Markov_state'] is False for r in result['measured_recurrence'])


def test_pair_iterators_are_not_consumed_before_channel_execution():
    import numpy as np
    from oph_fpe.bulk.primitive_source_instruments import pair_twirl
    matrix=np.diag([1,0,0,0])
    result=pair_twirl(matrix,iter([(0,1),(2,3)]))
    assert np.array_equal(result,np.diag([.5,.5,0,0]))
    with pytest.raises(ValueError):
        pair_twirl(np.empty((0,0)),[])


def test_all_flag_branches_recover_and_average_to_the_twirl():
    import itertools
    import numpy as np
    from oph_fpe.bulk.primitive_source_instruments import flagged_pair_repair, pair_twirl
    pairs=[(0,1),(2,3)]
    words=list(map(list,itertools.product((0,1),repeat=2)))
    # A complete matrix-unit basis tests the superoperators, including coherence.
    for i,j in itertools.product(range(4),repeat=2):
        unit=np.zeros((4,4),dtype=complex);unit[i,j]=1
        branches=[flagged_pair_repair(unit,pairs,w) for w in words]
        assert np.array_equal(sum(branches)/4,pair_twirl(unit,pairs))
        for word,branch in zip(words,branches):
            assert np.array_equal(flagged_pair_repair(branch,pairs,word),unit)


@pytest.mark.parametrize('flags', [None, [], [0], [0,2], [True,0], [0.,1], (0,1)])
def test_retained_flag_api_rejects_invalid_words(flags):
    import numpy as np
    from oph_fpe.bulk.primitive_source_instruments import flagged_pair_repair
    with pytest.raises(ValueError):
        flagged_pair_repair(np.eye(4),[(0,1),(2,3)],flags)


@pytest.mark.parametrize('n', [4,8,16])
def test_minimal_difference_archive_reconstructs_arbitrary_positive_inputs(n):
    import random
    from oph_fpe.bulk.primitive_source_archive import recover, tree_seams
    seams=produce.case(n,16,1)['seams']
    tree=tree_seams(n,seams)
    rng=random.Random(n)
    for _ in range(6):
        weights=[[rng.randrange(1,100) for _ in range(12)] for _ in range(n)]
        state=[F(w,sum(line)) for line in weights for w in line]
        means=list(state)
        for _,a,b in seams: means[a]=means[b]=(state[a]+state[b])/2
        chords={e:(state[a]-state[b])/2 for e,(_,a,b) in enumerate(seams) if e not in tree}
        assert len(chords)==5*n+1
        assert recover(n,seams,means,chords)==state
        missing=dict(chords);missing.pop(next(iter(missing)))
        with pytest.raises(ValueError): recover(n,seams,means,missing)
        bad=list(means);bad[0]+=1
        with pytest.raises(ValueError): recover(n,seams,bad,chords)
        with pytest.raises(ValueError): recover(n,seams,list(map(float,means)),chords)
        with pytest.raises(ValueError): recover(n,seams[:-1],means,chords)
