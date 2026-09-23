"""Independent primitive-source replay using only the Python standard library.

No capture, simulator dynamics, numpy, scipy, or producer is imported. Exact
ledger algebra and a rational Taylor enclosure are separate from IEEE replay.
"""
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import re


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False)+"\n").encode("ascii")


def equal(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise ValueError(label)


def keys(value, fields):
    if type(value) is not dict or set(value) != set(fields.split()):
        raise ValueError("fields: "+fields)


def strict_load(path):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate JSON key")
            result[k] = v
        return result
    def reject(value):
        raise ValueError("JSON number must be an integer: "+value)
    return json.loads(Path(path).read_text(encoding="ascii"), object_pairs_hook=pairs,
                      parse_float=reject, parse_constant=reject)


def hex_number(value):
    if type(value) is not str:
        raise ValueError("hex float type")
    result = float.fromhex(value)
    if not math.isfinite(result) or result.hex() != value:
        raise ValueError("noncanonical or nonfinite hex float")
    return result


def digest_native(value):
    # The existing capture hashes compact JSON without a trailing newline.
    return "sha256:"+hashlib.sha256(canonical(value)[:-1]).hexdigest()


def clean(value):
    rounded = round(value, 15)
    return 0.0 if rounded == 0 else rounded


def rank(rows):
    a = [list(map(F, row)) for row in rows]
    pivot = 0
    for col in range(len(a[0])):
        k = next((i for i in range(pivot, len(a)) if a[i][col]), None)
        if k is None:
            continue
        a[pivot], a[k] = a[k], a[pivot]
        scale = a[pivot][col]
        a[pivot] = [x/scale for x in a[pivot]]
        for i in range(pivot+1, len(a)):
            factor = a[i][col]
            if factor:
                a[i] = [x-factor*y for x, y in zip(a[i], a[pivot])]
        pivot += 1
        if pivot == len(a):
            break
    return pivot


def read_algebra(n, mate, activated, receivers=(0,)):
    """Maximal retained payload at receiver carriers, with initial samples."""
    size = 12*n
    own = {12*c+p for c in receivers for p in range(12)}
    observed = own | {mate[p] for p in own if p in activated}
    recoverable = set(observed)
    for carrier in range(n):
        ports = set(range(12*carrier, 12*(carrier+1)))
        if len(ports-observed) <= 1:
            recoverable.update(ports)
    # Normalized source ledger: x_(c,11)=1-sum_(p<11) x_(c,p).
    rows = []
    for slot in sorted(observed):
        row = [0]*(11*n)
        c, p = divmod(slot, 12)
        if p == 11:
            row[11*c:11*(c+1)] = [-1]*11
        else:
            row[11*c+p] = 1
        rows.append(row)
    dimension = rank(rows)
    hidden = next((sorted(set(range(12*c, 12*(c+1)))-observed)
                   for c in range(1, n)
                   if len(set(range(12*c, 12*(c+1)))-observed) >= 2), None)
    if hidden is None:
        raise ValueError("missing normalized indistinguishability witness")
    p, q = hidden[:2]
    left, right = [F(1, 12)]*size, [F(1, 12)]*size
    for v, sign in ((left, 1), (right, -1)):
        v[p] += sign*F(1, 48)
        v[q] -= sign*F(1, 48)
        if any(sum(v[12*c:12*(c+1)]) != 1 for c in range(n)) or min(v) <= 0:
            raise ValueError("inadmissible normalized witness")
    if any(left[i] != right[i] for i in observed) or left[p] == right[p]:
        raise ValueError("witness observation kernel")
    # Check the real endpoint means, not just the set of nominal input slots.
    transcript = lambda v: [v[i] for i in sorted(own)]+[(v[i]+v[mate[i]])/2
                                        for i in sorted(own) if i in activated]
    if transcript(left) != transcript(right):
        raise ValueError("witness receiver payload")
    # Global custody hashes distinguish these preparations: they are extra data.
    hashes_differ = hashlib.sha256(canonical(list(map(str, left)))).digest() != \
                    hashlib.sha256(canonical(list(map(str, right)))).digest()
    if not hashes_differ:
        raise ValueError("global metadata control")
    return {"directly_observed_slots": sorted(observed),
            "recoverable_slots_with_normalization": sorted(recoverable),
            "preparation_dimension": 11*n, "observation_rank": dimension,
            "hidden_dimension": 11*n-dimension, "hidden_pair": [p, q],
            "witness_values": [str(left[p]), str(right[p])],
            "equal_prior_best_binary_error": "1/2", "global_hash_distinguishes": True}


def case(row, spec):
    keys(row, "carriers cycles support_level seams order initial_hex final_hex commits noops native_log_sha256 cycle_counts observer_records")
    n, cycles, level = spec
    equal([row["carriers"], row["cycles"], row["support_level"]], list(spec), "case specification")
    size, count = 12*n, 6*n
    seams = row["seams"]
    if type(seams) is not list or len(seams) != count:
        raise ValueError("seam cardinality")
    mate = {}
    for index, seam in enumerate(seams):
        if type(seam) is not list or len(seam) != 3:
            raise ValueError("seam row")
        name, a, b = seam
        if type(a) is not int or type(b) is not int or not (0 <= a < size and 0 <= b < size):
            raise ValueError("seam endpoint")
        if a//12 == b//12 or a%12 != b%12 or a in mate or b in mate:
            raise ValueError("seams must partition all port slots")
        equal(name, f"seam-p{index//(n//2):02d}-{index%(n//2):06d}", "seam identity")
        equal(a%12, index//(n//2), "port order")
        mate[a], mate[b] = b, a
    equal(sorted(mate), list(range(size)), "all twelve ports covered")
    order = row["order"]
    if type(order) is not list or any(type(i) is not int for i in order):
        raise ValueError("order type")
    equal(sorted(order), list(range(count)), "complete scheduler permutation")
    if type(row["initial_hex"]) is not list or len(row["initial_hex"]) != size:
        raise ValueError("initial ledger size")
    x = list(map(hex_number, row["initial_hex"]))
    if min(x) < 0 or max(x) > 1 or any(abs(sum(x[12*c:12*(c+1)])-1) > 2**-40 for c in range(n)):
        raise ValueError("post-unitary preparation normalization")
    initial = list(x)
    budget = (count+15)//16
    versions, touched, activated = [0]*size, set(), set()
    log_hasher = hashlib.sha256()
    commits = noops = 0
    cycle_counts, snapshots, touched_at, committed_at = [], [], [], []
    for cycle in range(cycles):
        done = skipped = 0
        for offset in range(budget):
            index = order[(cycle*budget+offset)%count]
            name, a, b = seams[index]
            touched.add(index)
            u, v = x[a], x[b]
            if abs(u-v) <= 1e-15:
                skipped += 1
                continue
            old_a, old_b = versions[a], versions[b]
            if old_a != 0 or old_b != 0:
                raise ValueError("repeat committed write contradicts disjoint idempotence")
            mean = .5*(u+v)
            x[a] = x[b] = mean
            versions[a] += 1
            versions[b] += 1
            activated.update((a, b))
            reads = [{"carrier_id": f"carrier-{slot//12:05d}", "port": slot%12,
                      "version": version, "value": clean(value)}
                     for slot, version, value in ((a, old_a, u), (b, old_b, v))]
            writes = [{"carrier_id": r["carrier_id"], "port": r["port"],
                       "expected_version": r["version"], "committed_version": r["version"]+1,
                       "value": clean(mean)} for r in reads]
            material = {"cycle": cycle, "transaction_index": done, "seam_id": name,
                        "read_set": reads, "write_set": writes,
                        "mismatch_before": clean(abs(u-v)), "mismatch_after": 0.0,
                        "strict_descent": True, "update": "endpoint_arithmetic_mean"}
            event = {**material, "event_id": digest_native(material)}
            encoded = canonical(event)[:-1]
            log_hasher.update(len(encoded).to_bytes(8, "big"))
            log_hasher.update(encoded)
            done += 1
        cycle_counts.append([done, skipped])
        commits += done
        noops += skipped
        snapshots.append(list(x))
        touched_at.append(set(touched))
        committed_at.append(set(activated))
    equal(row["cycle_counts"], cycle_counts, "complete attempt accounting")
    equal([row["commits"], row["noops"]], [commits, noops], "native operation counts")
    equal(row["final_hex"], [v.hex() for v in x], "native terminal ledger")
    equal(row["native_log_sha256"], "sha256:"+log_hasher.hexdigest(), "complete native log replay")
    # Closed form is independent of order and repetition. Retain the actual
    # floating-point no-op threshold; do not silently replace it by equality.
    closed = list(initial)
    for index in touched:
        _, a, b = seams[index]
        if abs(initial[a]-initial[b]) > 1e-15:
            closed[a] = closed[b] = .5*(initial[a]+initial[b])
    equal([v.hex() for v in closed], row["final_hex"], "commuting closed form")
    # Reconstruct the numeric observer output from native source versions,
    # independent of the observer producer, event IDs, and global-state hash.
    receiver_neighbor = min(mate[p]//12 for p in range(12))
    receivers = (0, receiver_neighbor)
    schedule = [((j+1)*cycles)//4-1 for j in range(4)]
    records, read_rows, provenance = [], [], []
    next_port = 0
    seam_of = {slot: i for i, (_, a, b) in enumerate(seams) for slot in (a,b)}
    for sample in range(6):
        carrier = receivers[sample%2]
        cycle = schedule[sample%4]
        values = [clean(v) for v in snapshots[cycle][12*carrier:12*(carrier+1)]]
        records.append([sample, carrier, cycle, next_port, [v.hex() for v in values]])
        weighted = sum((p+1)*v for p,v in enumerate(values))
        next_port = (next_port+1+abs(int(round(weighted*1_000_000.0)))%11)%12
        parents = set()
        for p in range(12):
            slot = 12*carrier+p
            repaired = seam_of[slot] in touched_at[cycle]
            full_row = [F(0)]*size
            if repaired:
                full_row[slot] = full_row[mate[slot]] = F(1,2)
            if slot in committed_at[cycle]:
                parents.add(seam_of[slot])
            if not repaired:
                full_row[slot] = 1
            read_rows.append([full_row[12*c+p]-full_row[12*c+11]
                              for c in range(n) for p in range(11)])
        provenance.extend([i,sample] for i in sorted(parents))
    equal(row["observer_records"], records, "actual observer payload and feedback")
    return {"carriers": n, "cycles": cycles, "support_level": level,
            "slots": size, "seams": count, "attempts": cycles*budget,
            "commits": commits, "noops": noops, "touched_seams": len(touched),
            "saturation_cycle_count": (count+budget-1)//budget,
            "repair_read_after_write_edges": 0,
            "maximum_version": max(versions),
            "formal_real_transition_family_size": str(2**count),
            "terminal_real_projection_rank": size-len(touched),
            "full_sweep_real_projection_rank": count,
            "ideal_real_read_algebra": read_algebra(n, mate, {p for i in touched for p in seams[i][1:]}),
            "observer_payload": {"receiver_carriers": list(receivers), "records": 6,
                "ideal_normalized_read_rank": rank(read_rows),
                "repair_to_record_edges": provenance,
                "maximal_payload_algebra": read_algebra(n, mate, set(range(size)), receivers)},
            "native_noop_threshold": str(F(1e-15)),
            "native_committed_inverse_absolute_error_bound": str(F(1, 2**51)),
            "rounded_record_inverse_absolute_error_bound": str(F(1, 2**50)+F(3, 2*10**15))}


def multiply(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def quantum(row):
    keys(row, "laplacian step unitary_hex phase_ports phase_output_hex uniform_snapshot_feedback")
    L = row["laplacian"]
    if type(L) is not list or len(L) != 12 or any(type(r) is not list or len(r) != 12 for r in L):
        raise ValueError("C12 generator shape")
    if any(type(x) is not int for r in L for x in r):
        raise ValueError("integral generator")
    for i in range(12):
        if L[i][i] != 5 or sum(L[i]) != 0:
            raise ValueError("icosahedral Laplacian")
        for j in range(12):
            if L[i][j] != L[j][i] or (i != j and L[i][j] not in (-1, 0)):
                raise ValueError("undirected incidence")
        neighbors = [j for j in range(12) if L[i][j] == -1]
        if any(sum(L[j][k] == -1 for k in neighbors) != 2 for j in neighbors):
            raise ValueError("five-cycle link")
    distances = []
    for start in range(12):
        ds = {start: 0}
        queue = [start]
        for i in queue:
            for j in range(12):
                if L[i][j] == -1 and j not in ds:
                    ds[j] = ds[i]+1
                    queue.append(j)
        equal([list(ds.values()).count(k) for k in range(4)], [1, 5, 5, 1], "distance shells")
        distances.append([ds[i] for i in range(12)])
    equal(row["step"], "1/131072", "exact native step")
    t = F(row["step"])
    powers = [[[int(i == j) for j in range(12)] for i in range(12)]]
    for _ in range(8):
        powers.append(multiply(powers[-1], L))
    real = [[sum((-1)**(k//2)*t**k*powers[k][i][j]/math.factorial(k)
                 for k in range(0, 9, 2)) for j in range(12)] for i in range(12)]
    imag = [[sum((-1)**((k+1)//2)*t**k*powers[k][i][j]/math.factorial(k)
                 for k in range(1, 9, 2)) for j in range(12)] for i in range(12)]
    tail = 2*(10*t)**9/math.factorial(9)
    U = row["unitary_hex"]
    if type(U) is not list or len(U) != 12 or any(type(r) is not list or len(r) != 12 for r in U):
        raise ValueError("unitary shape")
    lower = []
    for i in range(12):
        for j in range(12):
            d = distances[i][j]
            first = abs(F(powers[d][i][j]))*t**d/math.factorial(d)
            remainder = 2*(10*t)**(d+1)/math.factorial(d+1)
            if first <= remainder:
                raise ValueError("uncertified nonzero propagator entry")
            lower.append(first-remainder)
            z = U[i][j]
            if type(z) is not list or len(z) != 2:
                raise ValueError("complex entry")
            # Relative numerical agreement cannot accept a forged zero in a
            # small distance-three entry. Exact nonzero proof uses the bound.
            tolerance = tail+(abs(real[i][j])+abs(imag[i][j]))*F(1, 2**36)
            for component, target in zip(z, (real[i][j], imag[i][j])):
                if abs(F(hex_number(component))-target) > tolerance:
                    raise ValueError("native propagation disagrees with rational Taylor enclosure")
    neighbor = next(j for j in range(12) if L[0][j] == -1)
    equal(row["phase_ports"], [0, neighbor], "phase witness ports")
    a, b, c, d = real[0][0], imag[0][0], real[0][neighbor], imag[0][neighbor]
    expected = [((a-d)**2+(b+c)**2)/2, ((a+d)**2+(b-c)**2)/2]
    if type(row["phase_output_hex"]) is not list or len(row["phase_output_hex"]) != 2:
        raise ValueError("phase output shape")
    actual = [F(hex_number(v)) for v in row["phase_output_hex"]]
    if any(abs(x-y) > F(1, 2**42) for x, y in zip(actual, expected)):
        raise ValueError("native phase witness")
    bound = 2*t-400*t*t  # trace-norm bound on the second commutator derivative
    if bound <= 0 or actual[1]-actual[0] < bound:
        raise ValueError("phase-sensitive intensity read")
    control = row["uniform_snapshot_feedback"]
    keys(control, "input_port native_next_port stabilizer_permutation")
    equal(control["input_port"], 0, "feedback input port")
    predicted = (1+6_500_000%11)%12
    equal(control["native_next_port"], predicted, "uniform native feedback")
    perm = control["stabilizer_permutation"]
    if type(perm) is not list or any(type(i) is not int for i in perm):
        raise ValueError("stabilizer permutation type")
    equal(sorted(perm), list(range(12)), "stabilizer permutation")
    if perm[0] != 0 or perm[predicted] == predicted:
        raise ValueError("missing feedback equivariance violation")
    if any(L[perm[i]][perm[j]] != L[i][j] for i in range(12) for j in range(12)):
        raise ValueError("stabilizer does not preserve incidence")
    if any(perm[perm[perm[perm[perm[i]]]]] != i for i in range(12)):
        raise ValueError("stabilizer must have order five")
    return {"dimension": 12, "generator_edges": 30, "generator_diameter": 3,
            "certified_nonzero_unitary_entries": 144, "step": str(t),
            "entry_modulus_lower_bound": str(min(lower)),
            "equal_input_intensities_output_gap_lower_bound": str(bound),
            "intensity_only_markov_closure": False,
            "uniform_feedback_equivariant": False,
            "analytic_reversible_port_algebra_complex_dimension": 144,
            "coordinate_ports_central_in_reversible_closure": False,
            "native_next_port": predicted, "rotated_next_port": perm[predicted]}


def cadd(a, b):
    return a[0]+b[0], a[1]+b[1]


def cmul(a, b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def read_sparse(rows, expected):
    if type(rows) is not list or any(type(r) is not list or len(r) != 4 for r in rows):
        raise ValueError("sparse density shape")
    equal([r[:2] for r in rows], [list(p) for p in sorted(expected)], "complete sparse density support")
    for i, j, re_part, im_part in rows:
        for actual, target in zip((re_part, im_part), expected[i, j]):
            if abs(F(hex_number(actual))-target) > F(1, 2**45):
                raise ValueError("Kraus channel output")


def instruments(row, cases, quantum_row):
    keys(row, "threshold_input_hex threshold_output_hex phase_lift_coherence_hex twirl_output_sparse dephased_output_sparse next_read_hex extensions_selected_by_source measured_recurrence retained_flags deterministic_archive")
    equal(row["extensions_selected_by_source"], False, "instrument selection boundary")
    delta = 2.0**-51
    points = [[.5+i*delta, .5-i*delta] for i in range(3)]
    outputs = [points[0], points[1], [.5, .5]]
    equal(row["threshold_input_hex"], [[v.hex() for v in p] for p in points], "affinity witness inputs")
    equal(row["threshold_output_hex"], [[v.hex() for v in p] for p in outputs], "native threshold affinity witness")
    if not (2*delta <= 1e-15 < 4*delta):
        raise ValueError("threshold witness straddling")
    n, size = cases[2]["carriers"], 12*cases[2]["carriers"]
    pairs = [(a,b) for _,a,b in cases[2]["seams"]]
    expected_lift = [math.sqrt(13)/(24*n)]*2+[7/(24*n), -7/(24*n)]
    lift = row["phase_lift_coherence_hex"]
    if type(lift) is not list or len(lift) != 4 or any(type(z) is not list or len(z) != 2 for z in lift):
        raise ValueError("phase lift witness shape")
    for z, target in zip(lift, expected_lift):
        if abs(hex_number(z[0])-target) > 2**-45 or abs(hex_number(z[1])) > 2**-45:
            raise ValueError("native phase lift ensemble witness")
    neighbor = quantum_row["phase_ports"][1]
    zero = (F(0),F(0))
    rho = {(i,i):(F(1,12*n),F(0)) for i in range(12,size)}
    rho[0,0] = rho[neighbor,neighbor] = (F(1,2*n),F(0))
    rho[0,neighbor],rho[neighbor,0] = (F(0),-F(1,2*n)),(F(0),F(1,2*n))
    original = dict(rho)
    words = [[0]*len(pairs),[1]*len(pairs)]
    words += [[int(i == j) for i in range(len(pairs))] for j in range(len(pairs))]
    words += [[i%2 for i in range(len(pairs))]]
    flags = row["retained_flags"]
    if type(flags) is not list or len(flags)!=len(words):
        raise ValueError("complete retained-flag control family")
    def native_digest(state):
        # The producer initializes these rational entries in binary64 before
        # applying exact permutations. Account for that conversion explicitly.
        data=[[i,j,str(F(float(z[0]))),str(F(float(z[1])))] for (i,j),z in sorted(state.items())]
        return hashlib.sha256(canonical(data)).hexdigest()
    for record,word in zip(flags,words):
        keys(record,"flags branch_digest recovered_digest")
        equal(record["flags"],word,"retained branch word")
        perm=list(range(size))
        for (a,b),flag in zip(pairs,word):
            if flag: perm[a],perm[b]=b,a
        if any(perm[perm[i]] != i for i in range(size)):
            raise ValueError("flagged branch inverse")
        branch={(perm[i],perm[j]):z for (i,j),z in original.items()}
        equal(record["branch_digest"],native_digest(branch),"executed flagged branch")
        equal(record["recovered_digest"],native_digest(original),"exact retained-flag recovery")
    # Independent exact Gaussian-rational Kraus action, including coherences.
    for a,b in pairs:
        swap = lambda i: b if i == a else a if i == b else i
        permuted = {(swap(i),swap(j)):z for (i,j),z in rho.items()}
        both = {p:tuple(v/2 for v in cadd(rho.get(p,zero),permuted.get(p,zero)))
                for p in rho.keys()|permuted.keys()}
        rho = {p:z for p,z in both.items() if z != zero}
    dephased = {p:z for p,z in rho.items() if p[0] == p[1]}
    read_sparse(row["twirl_output_sparse"], rho)
    read_sparse(row["dephased_output_sparse"], dephased)
    U = [[tuple(F(hex_number(v)) for v in z) for z in line] for line in quantum_row["unitary_hex"]]
    def next_probability(state):
        result = zero
        for (i,j), z in state.items():
            if i < 12 and j < 12:
                result = cadd(result,cmul(cmul(U[0][i],z),(U[0][j][0],-U[0][j][1])))
        if result[1] != 0:
            raise ValueError("nonreal channel probability")
        return result[0]
    expected = [next_probability(state) for state in (rho,dephased)]
    if type(row["next_read_hex"]) is not list or len(row["next_read_hex"]) != 2:
        raise ValueError("channel read shape")
    actual = [F(hex_number(v)) for v in row["next_read_hex"]]
    if any(abs(a-b) > F(1,2**44) for a,b in zip(actual,expected)):
        raise ValueError("subsequent native unitary reads")
    t=F(1,131072)
    gap=(2*t-400*t*t)/(8*n)
    if actual[1]-actual[0] <= gap:
        raise ValueError("missing channel extension ambiguity")
    recurrence = row["measured_recurrence"]
    if type(recurrence) is not list or len(recurrence) != 3:
        raise ValueError("recurrence family")
    archives = row["deterministic_archive"]
    if type(archives) is not list or len(archives) != 3:
        raise ValueError("complete deterministic archive controls")
    archive_results = [verify_archive(r,c) for r,c in zip(archives,(cases[2],cases[4],cases[5]))]
    return {"deterministic_archive":archive_results, "threshold_affinity_defect": str(F(1,2**51)),
            "phase_lift_ensemble_coherence_gap_lower_bound": str(F(1,8*n)),
            "ideal_repair_CPTP_extensions": 2,
            "retained_flag_controls":len(words), "retained_flag_bits_per_full_sweep":len(pairs),
            "all_flag_words_recover_input_analytically":True,
            "subsequent_read_gap_lower_bound": str(gap),
            "native_threshold_rule_is_quantum_channel": False,
            "phase_preserving_lift_is_quantum_channel": False,
            "extensions_selected_by_source": False,
            "measured_recurrence": [verify_recurrence(r,c,U) for r,c in zip(recurrence,(cases[2],cases[4],cases[5]))]}



def verify_archive(row, case_row):
    keys(row, "carriers tree_seams initial means chord_differences recovered_sha256")
    n,seams=case_row['carriers'],case_row['seams']
    equal(row['carriers'],n,'archive population')
    weights=[[1+((c+1)*(p+3))%17 for p in range(12)] for c in range(n)]
    state=[F(w,sum(line)) for line in weights for w in line]
    equal(row['initial'],list(map(str,state)),'exact archive preparation')
    means=list(state)
    for _,a,b in seams: means[a]=means[b]=(state[a]+state[b])/2
    equal(row['means'],list(map(str,means)),'deterministic ideal means')
    tree=row['tree_seams']
    if type(tree) is not list or len(tree)!=n-1 or any(type(e) is not int or not 0<=e<len(seams) for e in tree) or len(set(tree))!=len(tree):
        raise ValueError('tree seam indices')
    # Gaussian elimination is independent of the producer's leaf decoder.
    incidence=[[F(int(a//12==c)-int(b//12==c)) for _,a,b in seams] for c in range(n)]
    tree_matrix=[[line[e] for e in tree] for line in incidence[:-1]]
    if rank(tree_matrix)!=n-1: raise ValueError('archive is not a spanning tree')
    chords={e:(state[a]-state[b])/2 for e,(_,a,b) in enumerate(seams) if e not in tree}
    equal(row['chord_differences'],[[e,str(v)] for e,v in sorted(chords.items())],'complete retained differences')
    rhs=[1-sum(means[12*c:12*c+12])-sum(incidence[c][e]*v for e,v in chords.items()) for c in range(n)]
    matrix=[line+[b] for line,b in zip(tree_matrix,rhs)]
    for j in range(n-1):
        k=next(i for i in range(j,n-1) if matrix[i][j])
        matrix[j],matrix[k]=matrix[k],matrix[j]
        matrix[j]=[x/matrix[j][j] for x in matrix[j]]
        for i in range(n-1):
            if i!=j:
                coefficient=matrix[i][j]
                matrix[i]=[x-coefficient*y for x,y in zip(matrix[i],matrix[j])]
    differences=dict(chords)|{e:matrix[j][-1] for j,e in enumerate(tree)}
    if any(sum(incidence[c][e]*v for e,v in differences.items())!=1-sum(means[12*c:12*c+12]) for c in range(n)):
        raise ValueError('complete normalized reconstruction')
    recovered=list(means)
    for e,(_,a,b) in enumerate(seams):
        recovered[a]+=differences[e];recovered[b]-=differences[e]
    if recovered!=state: raise ValueError('archive failed exact inversion')
    equal(row['recovered_sha256'],hashlib.sha256(canonical(list(map(str,recovered)))).hexdigest(),'executed archive recovery')
    return {'carriers':n,'retained_real_coordinates':len(chords),
            'linear_archive_minimum':5*n+1,'exact_input_recovered':True,
            'native_record_interface':False}


def verify_recurrence(row, case_row, U):
    keys(row, "carriers steps coarse_target coarse_input_ports coarse_transition_hex erasure")
    n, size = case_row["carriers"], 12*case_row["carriers"]
    equal(row["carriers"], n, "recurrence population")
    mate = {a:b for _,a,b in case_row["seams"]}|{b:a for _,a,b in case_row["seams"]}
    graph = [{mate[12*c+p]//12 for p in range(12)} for c in range(n)]
    distances = []
    for c in range(n):
        ds = {c:0}; queue = [c]
        for v in queue:
            for w in sorted(graph[v]-ds.keys()):
                ds[w] = ds[v]+1; queue.append(w)
        if len(ds) != n:
            raise ValueError("disconnected carrier graph")
        distances.append([ds[d] for d in range(n)])
    one = [{12*c+p for c in (i//12,mate[i]//12) for p in range(12)} for i in range(size)]
    current = [{i} for i in range(size)]
    expected_steps = []
    for k in range(1,5):
        current = [set().union(*(current[j] for j in sources)) for sources in one]
        formula = [{12*d+q for d in range(n) for q in range(12)
                    if min(distances[i//12][d],distances[mate[i]//12][d]) <= k-1}
                   for i in range(size)]
        if current != formula:
            raise ValueError("slot propagation ball theorem")
        raw = bytes(int(j in current[i]) for i in range(size) for j in range(size))
        counts = [len(set().union(*(current[12*c+p] for p in range(12))))//12 for c in range(n)]
        if counts != [sum(d <= k for d in ds) for ds in distances]:
            raise ValueError("carrier propagation ball theorem")
        expected_steps.append({"step":k,"slot_support_sha256":hashlib.sha256(raw).hexdigest(),
                               "positive_slot_entries":sum(map(len,current)),"carrier_influence_counts":counts})
    equal(row["steps"], expected_steps, "executed measured recurrence support")
    target=mate[0]//12
    other=next(p for p in range(1,12) if mate[p]//12 != target)
    equal(row["coarse_target"], target, "coarse target")
    equal(row["coarse_input_ports"], [0,other], "coarse input ports")
    expected = [sum((U[p][q][0]**2+U[p][q][1]**2)/2 for p in range(12)
                    if mate[12*target+p]//12 == 0) for q in (0,other)]
    if type(row["coarse_transition_hex"]) is not list or len(row["coarse_transition_hex"]) != 2:
        raise ValueError("coarse transition shape")
    actual=[F(hex_number(v)) for v in row["coarse_transition_hex"]]
    if any(abs(a-b)>F(1,2**44) for a,b in zip(actual,expected)):
        raise ValueError("coarse transition probability")
    d=F(20,131072)
    lower=(1-2*d-10*d*d)/2
    if actual[0]-actual[1] < lower:
        raise ValueError("missing carrier lumpability obstruction")
    erasure = row["erasure"]
    keys(erasure, "cycle_flow inverse_flow_hex first_outputs_hex")
    flow = [0]*size
    carrier, port, visited = 0, 0, set()
    while carrier not in visited:
        visited.add(carrier)
        slot = 12*carrier+port
        flow[slot],flow[mate[slot]] = 1,-1
        carrier,port = mate[slot]//12,1-port
    if carrier != 0 or len(visited) != n:
        raise ValueError("alternating Hamiltonian circulation")
    equal(erasure["cycle_flow"],flow,"complete normalized erasure flow")
    if any(flow[i]+flow[mate[i]] for i in range(size)) or any(sum(flow[12*c:12*(c+1)]) for c in range(n)):
        raise ValueError("flow is not a normalized repair kernel")
    if type(erasure["inverse_flow_hex"]) is not list or len(erasure["inverse_flow_hex"]) != size:
        raise ValueError("inverse flow shape")
    inverse = [F(hex_number(v)) for v in erasure["inverse_flow_hex"]]
    B = [[z[0]**2+z[1]**2 for z in line] for line in U]
    bound=F(1,2**40)
    propagated=[sum(B[p][q]*inverse[12*c+q] for q in range(12)) for c in range(n) for p in range(12)]
    if max(abs(a-b) for a,b in zip(propagated,flow)) > bound or max(map(abs,inverse))>2:
        raise ValueError("native Born inverse circulation residual")
    if any(abs(sum(inverse[12*c:12*(c+1)]))>bound for c in range(n)):
        raise ValueError("normalized inverse circulation")
    preps=[[F(1,12)+sign*v/48 for v in inverse] for sign in (1,-1)]
    if min(min(v) for v in preps) < F(1,24):
        raise ValueError("nonpositive erasure preparation")
    outputs=erasure["first_outputs_hex"]
    if type(outputs) is not list or len(outputs)!=2 or any(type(v) is not list or len(v)!=size for v in outputs):
        raise ValueError("erasure output shape")
    actual=[[F(hex_number(v)) for v in line] for line in outputs]
    for state,result in zip(preps,actual):
        local=[sum(B[p][q]*state[12*c+q] for q in range(12)) for c in range(n) for p in range(12)]
        expected=[(local[i]+local[mate[i]])/2 for i in range(size)]
        if max(abs(a-b) for a,b in zip(expected,result))>bound:
            raise ValueError("first measured-cycle erasure replay")
    if max(abs(a-b) for a,b in zip(*actual))>bound:
        raise ValueError("first cycle failed to erase the intervention")
    # Rank on normalized preparations: the antisymmetric pair coordinates
    # obey the connected carrier incidence equations, of rank N-1.
    incidence=[[0]*(6*n) for _ in range(n)]
    for e,(_,a,b) in enumerate(case_row["seams"]):
        incidence[a//12][e],incidence[b//12][e]=1,-1
    equal(rank(incidence),n-1,"connected incidence rank")
    return {"carriers":n,"carrier_graph_diameter":max(map(max,distances)),
            "steps":expected_steps,"carrier_totals_are_Markov_state":False,
            "coarse_transition_gap_lower_bound":str(lower),
            "normalized_first_cycle_kernel_dimension":5*n+1,
            "normalized_first_cycle_image_dimension":6*n-1,
            "inverse_and_replay_numerical_residual_bound":str(bound),
            "exact_real_erasure_is_analytic":True}


def verify(packet, source_root=None):
    keys(packet, "schema source scope cases quantum instruments sha256")
    equal(packet["schema"], "oph.primitive-source-reads.v2", "schema")
    equal(packet["scope"], {"driver": "registered_all_port_capture", "complete_A1_A3": False,
        "M1_derived": False, "record_channel": "local_numeric_ledger_payload",
        "global_custody_hash_is_local_readout": False}, "scope")
    equal(packet["sha256"], hashlib.sha256(canonical({k:v for k,v in packet.items() if k != "sha256"})).hexdigest(), "packet digest")
    source = packet["source"]
    keys(source, "repository revision files")
    equal(source["repository"], "https://github.com/muellerberndt/oph-physics-sim", "source repository")
    if type(source["revision"]) is not str or not re.fullmatch("[0-9a-f]{40}", source["revision"]):
        raise ValueError("source revision")
    files = source["files"]
    required = {"oph_fpe/bulk/primitive_source_reads.py", "oph_fpe/bulk/primitive_source_instruments.py", "oph_fpe/bulk/primitive_source_archive.py",
                "oph_fpe/bulk/physical_h3_kms_source_capture.py",
                "oph_fpe/bulk/verify_primitive_source_reads_independent.py", "oph_fpe/core/echosahedral_dynamics.py"}
    if type(files) is not dict or not required <= set(files):
        raise ValueError("source closure")
    for name, value in files.items():
        if not re.fullmatch(r"oph_[a-z_]+/[A-Za-z0-9_/]+\.py", name) or ".." in name:
            raise ValueError("source path")
        if type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value):
            raise ValueError("source hash")
        if source_root is not None:
            equal(hashlib.sha256((Path(source_root)/name).read_bytes()).hexdigest(), value, "live source custody")
    specs = ((4, 4, 1), (4, 8, 1), (4, 16, 1), (4, 32, 1), (8, 16, 1), (16, 16, 1), (4, 16, 2))
    if type(packet["cases"]) is not list or len(packet["cases"]) != len(specs):
        raise ValueError("complete case family")
    rows = [case(row, spec) for row, spec in zip(packet["cases"], specs)]
    # The execution prefix, rather than the driver's rescaled snapshot times,
    # is the compared object. Support refinement does not change this driver.
    for i in (1, 2, 3, 6):
        for field in ("seams", "order", "initial_hex"):
            equal(packet["cases"][i][field], packet["cases"][0][field], "fixed-population source inputs")
    equal(packet["cases"][2]["final_hex"], packet["cases"][3]["final_hex"], "repair saturation")
    equal(packet["cases"][2]["final_hex"], packet["cases"][6]["final_hex"], "support-regulator independence")
    for i in range(3):
        equal(packet["cases"][i+1]["cycle_counts"][:specs[i][1]], packet["cases"][i]["cycle_counts"], "actual execution prefix")
    return {"schema": 1, "packet_sha256": packet["sha256"], "cases": rows,
            "quantum": quantum(packet["quantum"]),
            "instruments": instruments(packet["instruments"], packet["cases"], packet["quantum"]), "M1_derived": False,
            "source_bytes_checked": source_root is not None,
            "history_extension_is_spatial_refinement": False}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    print(canonical(verify(strict_load(args.packet), args.source_root)).decode("ascii"), end="")
