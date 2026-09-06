"""
detection.py
------------
Deadlock DETECTION (as opposed to avoidance -- this is used when the system
does NOT run Banker's Algorithm up front, and we must periodically check
whether a deadlock has already occurred).

Two algorithms are provided:

1. cycle_detect(rag) / find_cycle(wfg)
   For single-instance resource types: a deadlock exists iff the
   Resource Allocation Graph's wait-for graph contains a cycle.
   (Necessary AND sufficient condition when every resource type has
   exactly one instance.)

2. detect_deadlock_multi_instance(available, allocation, request, ...)
   For resource types with multiple instances: a generalization of the
   Banker's safety algorithm ("reduction" method, CLRS/Silberschatz).
   A process whose request can currently be satisfied is assumed to run
   to completion and release its resources; any processes that can never
   have their request satisfied are deadlocked.
"""

import networkx as nx
from .rag import build_wait_for_graph


def find_cycle(wfg):
    """
    Returns one cycle (list of process names, in cycle order) if the
    wait-for graph has one, else None. Uses networkx's cycle finder.
    """
    try:
        cycle_edges = nx.find_cycle(wfg, orientation="original")
        nodes = [u for u, v, _ in cycle_edges]
        return nodes
    except nx.NetworkXNoCycle:
        return None


def expand_cycle_to_rag_path(rag, process_cycle):
    """
    Given a cycle expressed purely in process names (from the wait-for graph),
    expand it into the full alternating Process -> Resource -> Process ... path
    as it actually appears in the RAG, so it can be highlighted on the RAG drawing.
    """
    if not process_cycle:
        return None
    full_path = []
    n = len(process_cycle)
    for k in range(n):
        p, p_next = process_cycle[k], process_cycle[(k + 1) % n]
        full_path.append(p)
        # find the resource r such that p -request-> r -assignment-> p_next
        for r in rag.successors(p):
            if rag.nodes[r].get("kind") != "resource":
                continue
            if rag[p][r].get("kind") != "request":
                continue
            if rag.has_edge(r, p_next) and rag[r][p_next].get("kind") == "assignment":
                full_path.append(r)
                break
    return full_path


def cycle_detect(rag):
    """
    Convenience wrapper: build the wait-for graph from a RAG and look for a cycle.
    Returns (deadlocked: bool, cycle: list[str] or None, wait_for_graph).
    `cycle` is expanded to include the resource nodes on the path (Process ->
    Resource -> Process ...) so it can be passed directly to draw_rag's
    highlight_cycle parameter.
    Valid as an exact deadlock test when all resources are single-instance.
    """
    wfg = build_wait_for_graph(rag)
    process_cycle = find_cycle(wfg)
    full_cycle = expand_cycle_to_rag_path(rag, process_cycle) if process_cycle else None
    return (process_cycle is not None), full_cycle, wfg


def detect_deadlock_multi_instance(available, allocation, request, process_names=None):
    """
    General deadlock detection algorithm for multi-instance resource types.

    available: list[int]        length m
    allocation: list[list[int]] n x m -- currently held by each process
    request:    list[list[int]] n x m -- currently outstanding request by each process
                (NOT the declared max -- just what it's asking for RIGHT NOW)

    Returns (deadlocked_processes: list[str], safe_finish_order: list[str])
    Any process not in the finish order is deadlocked.
    """
    n = len(allocation)
    m = len(available)
    process_names = process_names or [f"P{i}" for i in range(n)]

    work = available[:]
    finish = [all(v == 0 for v in allocation[i]) for i in range(n)]  # trivially finished if holding nothing... but
    # Standard algorithm: finish[i] starts False for all processes that hold or request resources.
    finish = [False] * n

    finish_order = []
    progress = True
    while progress:
        progress = False
        for i in range(n):
            if finish[i]:
                continue
            if all(request[i][j] <= work[j] for j in range(m)):
                # Process i's outstanding request CAN be satisfied -> it can finish
                work = [work[j] + allocation[i][j] for j in range(m)]
                finish[i] = True
                finish_order.append(process_names[i])
                progress = True

    deadlocked = [process_names[i] for i in range(n) if not finish[i]]
    return deadlocked, finish_order
