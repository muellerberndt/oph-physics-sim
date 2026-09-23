"""Instrument tests at the registered operation boundary and explicit extensions.

The pair mean and terminal lift are the functions used by the native driver.
The CPTP controls below are alternative mathematical extensions of its ideal
diagonal repairs. Neither is substituted into the registered source driver.
"""
import hashlib

import numpy as np

from oph_fpe.bulk.primitive_source_archive import archive_control

from oph_fpe.bulk.physical_h3_kms_source_capture import (
    _terminal_complex_lift, _visible_pair_mean,
)
from oph_fpe.core.echosahedral_dynamics import (
    LocalRecurrentCarrierState, propagate_local_recurrent_carriers,
)


def validated_matching(matrix, pairs):
    """Validate a finite matrix and materialize its complete slot matching."""
    rho = np.array(matrix, dtype=complex, copy=True)
    if rho.ndim != 2 or rho.shape[0] < 2 or rho.shape[0] != rho.shape[1] or not np.all(np.isfinite(rho)):
        raise ValueError("finite square matrix required")
    size = len(rho)
    try:
        pairs = [tuple(pair) for pair in pairs]
    except TypeError as exc:
        raise ValueError("slot-pair iterable required") from exc
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("two endpoints per seam required")
    slots = [p for pair in pairs for p in pair]
    if any(type(p) is not int for p in slots) or sorted(slots) != list(range(size)):
        raise ValueError("disjoint exhaustive slot matching required")
    return rho, pairs


def flagged_pair_repair(matrix, pairs, flags):
    """Apply one recorded unitary branch; the flag is extra retained state."""
    rho, pairs = validated_matching(matrix, pairs)
    if type(flags) is not list or len(flags) != len(pairs) or any(type(x) is not int or x not in (0,1) for x in flags):
        raise ValueError("one exact binary flag per seam required")
    perm = np.arange(len(rho))
    for (a,b), flag in zip(pairs,flags):
        if flag:
            perm[a],perm[b]=b,a
    return rho[np.ix_(perm,perm)]


def pair_twirl(matrix, pairs):
    """Compose (rho + S rho S*)/2 for disjoint, exhaustive slot transpositions."""
    rho,pairs=validated_matching(matrix,pairs)
    size=len(rho)
    for a, b in pairs:
        perm = np.arange(size)
        perm[a], perm[b] = b, a
        rho = (rho+rho[np.ix_(perm, perm)])/2
    return rho


def sparse(matrix):
    return [[i, j, float(z.real).hex(), float(z.imag).hex()]
            for i, row in enumerate(matrix) for j, z in enumerate(row) if z != 0]


def instrument_control(cases):
    # Exactly representable threshold-affinity witness. The native primitive
    # is executed, not replaced by an independently guessed repair formula.
    delta = 2.0**-51
    points = [[.5+i*delta, .5-i*delta] for i in range(3)]
    outputs = []
    for a, b in points:
        mean = _visible_pair_mean(a, b)
        outputs.append([a, b] if mean is None else [mean, mean])
    row = cases[2]
    n, size = row["carriers"], 12*row["carriers"]
    pairs = [(a, b) for _, a, b in row["seams"]]
    # Four per-carrier normalized preparations at the post-unitary cut.
    # Their equally weighted block-diagonal density encodings have equal
    # ensemble averages. Only carrier zero changes; other carriers are fixed.
    coherence = []
    for a, b in ((1, 0), (0, 1), (1/np.sqrt(2), 1/np.sqrt(2)),
                 (1/np.sqrt(2), -1/np.sqrt(2))):
        amplitudes = np.ones((n, 12), dtype=complex)/np.sqrt(12)
        amplitudes[0] = 0
        amplitudes[0, :2] = a, b
        visible = (abs(amplitudes)**2).reshape(-1)
        for left, right in pairs:
            mean = _visible_pair_mean(float(visible[left]), float(visible[right]))
            if mean is not None:
                visible[left] = visible[right] = mean
        lift = _terminal_complex_lift(amplitudes, visible.reshape(n, 12))
        z = lift[0, 0]*lift[0, 1].conjugate()/n
        coherence.append([float(z.real).hex(), float(z.imag).hex()])

    unitary = propagate_local_recurrent_carriers(
        LocalRecurrentCarrierState(np.eye(12, dtype=complex), np.zeros(12)),
        intrinsic_step=2.0**-17).amplitudes.T
    # Choose the same adjacent pair as the native phase control.
    from oph_fpe.core.echosahedral_dynamics import reference_icosahedral_coupling
    neighbor = int(np.flatnonzero(reference_icosahedral_coupling()[0] == -1)[0])
    rho = np.eye(size, dtype=complex)/(12*n)
    rho[:12, :12] = 0
    rho[0, 0] = rho[neighbor, neighbor] = 1/(2*n)
    rho[0, neighbor], rho[neighbor, 0] = -1j/(2*n), 1j/(2*n)
    flags = [[0]*len(pairs),[1]*len(pairs)]
    flags += [[int(i == j) for i in range(len(pairs))] for j in range(len(pairs))]
    flags += [[i%2 for i in range(len(pairs))]]
    retained = []
    for word in flags:
        branch=flagged_pair_repair(rho,pairs,word)
        restored=flagged_pair_repair(branch,pairs,word)
        retained.append({"flags":word,"branch_digest":matrix_digest(branch),
                         "recovered_digest":matrix_digest(restored)})
    coherent = pair_twirl(rho, pairs)
    measured = np.diag(np.diag(coherent))
    next_reads = [float((unitary@state[:12, :12]@unitary.conj().T)[0, 0].real)
                  for state in (coherent, measured)]
    return {"threshold_input_hex": [[v.hex() for v in p] for p in points],
            "threshold_output_hex": [[v.hex() for v in p] for p in outputs],
            "phase_lift_coherence_hex": coherence,
            "twirl_output_sparse": sparse(coherent),
            "dephased_output_sparse": sparse(measured),
            "next_read_hex": [v.hex() for v in next_reads],
            "extensions_selected_by_source": False, "retained_flags":retained,
            "deterministic_archive": [archive_control(c) for c in (cases[2], cases[4], cases[5])],
            "measured_recurrence": [recurrence_control(c, unitary) for c in (cases[2], cases[4], cases[5])]}


def recurrence_control(row, unitary):
    """Execute the explicit dephased extension; no metric or success predicate."""
    n = row["carriers"]
    size = 12*n
    mate = np.zeros(size, dtype=int)
    for _, a, b in row["seams"]:
        mate[a], mate[b] = b, a
    born = abs(unitary)**2
    local = np.kron(np.eye(n), born)
    transition = (local+local[mate])/2
    current = np.eye(size)
    steps = []
    for step in range(1, 5):
        current = transition@current
        support = (current > 0).astype(np.uint8)
        carrier_support = support.reshape(n, 12, n, 12).any(axis=(1, 3))
        steps.append({"step": step, "slot_support_sha256": hashlib.sha256(support.tobytes()).hexdigest(),
                      "positive_slot_entries": int(support.sum()),
                      "carrier_influence_counts": carrier_support.sum(axis=1).astype(int).tolist()})
    target = int(mate[0]//12)
    other = next(p for p in range(1, 12) if mate[p]//12 != target)
    coarse = transition.reshape(n, 12, n, 12).sum(axis=1)
    # The 0/1 matchings form the registered carrier cycle. Its circulation
    # survives normalization but is erased by the first full repair sweep.
    flow = np.zeros(size, dtype=int)
    carrier, port = 0, 0
    visited = set()
    while carrier not in visited:
        visited.add(carrier)
        slot = 12*carrier+port
        flow[slot], flow[mate[slot]] = 1, -1
        carrier, port = int(mate[slot]//12), 1-port
    if carrier != 0 or len(visited) != n:
        raise ValueError("registered alternating carrier cycle")
    inverse = np.linalg.solve(born, flow.reshape(n,12).T).T.reshape(-1)
    preparations = [np.ones(size)/12+sign*inverse/48 for sign in (1,-1)]
    terminal = [transition@x for x in preparations]
    return {"carriers": n, "steps": steps, "coarse_target": target,
            "coarse_input_ports": [0, other],
            "coarse_transition_hex": [float(coarse[target, 0, p]).hex() for p in (0, other)],
            "erasure": {"cycle_flow":flow.tolist(),
                        "inverse_flow_hex":[float(v).hex() for v in inverse],
                        "first_outputs_hex":[[float(v).hex() for v in x] for x in terminal]}}


def matrix_digest(matrix):
    """Exact numeric hash; a signed zero is the same zero matrix entry."""
    from fractions import Fraction
    import json
    rows=[[i,j,str(Fraction(float(z.real))),str(Fraction(float(z.imag)))]
          for i,line in enumerate(matrix) for j,z in enumerate(line) if z != 0]
    return hashlib.sha256((json.dumps(rows,separators=(",",":"))+"\n").encode("ascii")).hexdigest()
