"""
Quick sanity tests for Deadlock Guardian.
Run with:  python -m pytest tests/ -v   (from project root)
or simply: python tests/test_system.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.resource_system import ResourceSystem
from src.rag import build_rag, build_wait_for_graph
from src.detection import find_cycle, cycle_detect, detect_deadlock_multi_instance
from src.victim_selection import ProcessProfile, compute_costs, select_victim


def test_bankers_classic_safe_example():
    # Classic textbook example (Silberschatz): 5 processes, 3 resource types
    total = [10, 5, 7]
    allocation = [
        [0, 1, 0],
        [2, 0, 0],
        [3, 0, 2],
        [2, 1, 1],
        [0, 0, 2],
    ]
    max_claim = [
        [7, 5, 3],
        [3, 2, 2],
        [9, 0, 2],
        [2, 2, 2],
        [4, 3, 3],
    ]
    rs = ResourceSystem(total, allocation, max_claim)
    assert rs.available == [3, 3, 2]
    safe, seq = rs.is_safe_state()
    assert safe, "Classic example should be in a safe state"
    print("Safe sequence:", seq)


def test_bankers_rejects_unsafe_request():
    total = [10, 5, 7]
    allocation = [
        [0, 1, 0],
        [2, 0, 0],
        [3, 0, 2],
        [2, 1, 1],
        [0, 0, 2],
    ]
    max_claim = [
        [7, 5, 3],
        [3, 2, 2],
        [9, 0, 2],
        [2, 2, 2],
        [4, 3, 3],
    ]
    rs = ResourceSystem(total, allocation, max_claim)
    # P1 requesting (3,3,0) exceeds available -> should be denied immediately
    granted, reason, seq = rs.request_resources(1, [3, 3, 0])
    assert not granted
    print("Correctly denied:", reason)

    # P1 requesting (1,0,2) is a known-safe request from the textbook example
    granted, reason, seq = rs.request_resources(1, [1, 0, 2])
    assert granted
    print("Correctly granted:", reason, seq)


def test_cycle_detection_single_instance_deadlock():
    # P0 holds R0, wants R1. P1 holds R1, wants R0. Classic 2-process deadlock.
    allocation = [[1, 0], [0, 1]]
    request = [[0, 1], [1, 0]]
    rag = build_rag(allocation, request)
    deadlocked, cycle, wfg = cycle_detect(rag)
    assert deadlocked
    print("Cycle found:", cycle)


def test_cycle_detection_no_deadlock():
    # P0 holds R0, wants R1. R1 is free (no one holds it) -> no deadlock.
    allocation = [[1, 0], [0, 0]]
    request = [[0, 1], [0, 0]]
    rag = build_rag(allocation, request)
    deadlocked, cycle, wfg = cycle_detect(rag)
    assert not deadlocked


def test_multi_instance_detection():
    # P0, P2 can finish immediately (empty request) and release their holdings.
    # Even after that, P1/P3/P4 each ask for more than will ever be available
    # -> they are genuinely deadlocked.
    available = [0, 0, 0]
    allocation = [
        [0, 1, 0],   # P0
        [2, 0, 0],   # P1
        [3, 0, 3],   # P2
        [2, 1, 1],   # P3
        [0, 0, 2],   # P4
    ]
    request = [
        [0, 0, 0],   # P0 - none, can finish
        [4, 0, 0],   # P1 - needs 4 of R0, unsatisfiable
        [0, 0, 0],   # P2 - none, can finish
        [0, 5, 0],   # P3 - needs 5 of R1, unsatisfiable
        [0, 0, 5],   # P4 - needs 5 of R2, unsatisfiable
    ]
    deadlocked, order = detect_deadlock_multi_instance(available, allocation, request)
    assert set(deadlocked) == {"P1", "P3", "P4"}
    assert set(order) == {"P0", "P2"}
    print("Deadlocked:", deadlocked, "Finished:", order)


def test_victim_selection_prefers_cheap_process():
    profiles = [
        ProcessProfile("P1", cpu_time_used=100, priority=9, resources_held=2, progress=0.9, interactive=True),
        ProcessProfile("P3", cpu_time_used=2, priority=1, resources_held=5, progress=0.05, interactive=False),
        ProcessProfile("P4", cpu_time_used=50, priority=5, resources_held=3, progress=0.5, interactive=False),
    ]
    victim, cost, ranked = select_victim(profiles)
    print("Victim ranking:", [(n, round(c, 3)) for n, c, _ in ranked])
    # P3 has low cpu time, low priority, low progress -> should be cheapest
    assert victim == "P3"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}\n")
    print(f"All {len(tests)} tests passed.")
