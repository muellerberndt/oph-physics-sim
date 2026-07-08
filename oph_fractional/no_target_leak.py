from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

from .receipts import fail, pass_report


def no_target_leak_audit(
    *,
    source_nodes: Iterable[str],
    target_nodes: Iterable[str],
    edges: Iterable[tuple[str, str]],
) -> dict:
    graph: dict[str, list[str]] = defaultdict(list)
    for left, right in edges:
        graph[str(left)].append(str(right))
    sources = {str(node) for node in source_nodes}
    targets = {str(node) for node in target_nodes}
    leaks: list[dict[str, str]] = []
    for target in targets:
        queue = deque([(target, target)])
        seen = {target}
        while queue:
            node, path = queue.popleft()
            if node in sources and node != target:
                leaks.append({"target": target, "source": node, "path": path})
                continue
            for nxt in graph.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, f"{path}->{nxt}"))
    if leaks:
        return fail("TARGET_LEAK_DETECTED", details={"leaks": leaks})
    return pass_report(receipts={"NO_TARGET_LEAK": True})
