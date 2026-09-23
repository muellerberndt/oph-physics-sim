"""Exact classical completion of ideal pair means with retained differences.

This is an explicit enlarged record interface, not a native source record.
The spanning tree is chosen from the supplied seams, without geometric input.
"""
from fractions import Fraction as F
import hashlib
import json


def tree_seams(n, seams):
    if type(n) is not int or n < 2 or type(seams) is not list:
        raise ValueError("finite carrier matching required")
    if any(type(edge) not in (list, tuple) or len(edge) != 3 for edge in seams):
        raise ValueError("indexed slot seams required")
    slots = [i for _, a, b in seams for i in (a, b)]
    if any(type(i) is not int for i in slots) or sorted(slots) != list(range(12*n)):
        raise ValueError("complete disjoint matching required")
    if any(a//12 == b//12 for _, a, b in seams):
        raise ValueError("inter-carrier seams required")
    parent = list(range(n))
    def root(c):
        while parent[c] != c:
            c = parent[c]
        return c
    tree = []
    for e, (_, a, b) in enumerate(seams):
        ca, cb = root(a//12), root(b//12)
        if ca != cb:
            parent[ca] = cb
            tree.append(e)
    if len(tree) != n-1:
        raise ValueError("connected carrier graph required")
    return tree


def recover(n, seams, means, chords):
    """Recover normalized inputs from ideal means and non-tree differences."""
    tree = tree_seams(n, seams)
    expected = set(range(len(seams)))-set(tree)
    if type(chords) is not dict or type(means) is not list or set(chords) != expected or len(means) != 12*n:
        raise ValueError("complete chord archive and slot means required")
    if any(type(e) is not int for e in chords) or any(type(x) not in (int, F) for x in means+list(chords.values())):
        raise ValueError("exact rational archive required")
    if any(means[a] != means[b] for _, a, b in seams):
        raise ValueError("paired means required")
    residual = [F(1)-sum(means[12*c:12*c+12]) for c in range(n)]
    differences = dict(chords)
    for e, value in chords.items():
        _, a, b = seams[e]
        residual[a//12] -= value
        residual[b//12] += value
    pending = set(tree)
    while pending:
        degree = [0]*n
        for e in pending:
            _, a, b = seams[e]
            degree[a//12] += 1
            degree[b//12] += 1
        leaf = next(c for c in range(n) if degree[c] == 1)
        e = next(e for e in sorted(pending) if leaf in (seams[e][1]//12, seams[e][2]//12))
        _, a, b = seams[e]
        value = residual[leaf] if a//12 == leaf else -residual[leaf]
        differences[e] = value
        residual[a//12] -= value
        residual[b//12] += value
        pending.remove(e)
    if any(residual):
        raise ValueError("inconsistent normalized archive")
    result = list(means)
    for e, (_, a, b) in enumerate(seams):
        result[a] += differences[e]
        result[b] -= differences[e]
    if min(result) < 0:
        raise ValueError("archive does not describe positive populations")
    return result


def archive_control(row):
    n, seams = row['carriers'], row['seams']
    weights = [[1+((c+1)*(p+3))%17 for p in range(12)] for c in range(n)]
    state = [F(w, sum(line)) for line in weights for w in line]
    means = list(state)
    for _, a, b in seams:
        means[a] = means[b] = (state[a]+state[b])/2
    tree = tree_seams(n, seams)
    chords = {e:(state[a]-state[b])/2 for e, (_, a, b) in enumerate(seams) if e not in tree}
    restored = recover(n, seams, means, chords)
    digest = hashlib.sha256((json.dumps(list(map(str, restored)), separators=(',', ':'))+'\n').encode()).hexdigest()
    return {'carriers':n, 'tree_seams':tree, 'initial':list(map(str,state)),
            'means':list(map(str,means)),
            'chord_differences':[[e,str(v)] for e,v in sorted(chords.items())],
            'recovered_sha256':digest}
