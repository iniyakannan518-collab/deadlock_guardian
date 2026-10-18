"""
run_backend_demo.py
-------------------
One-command live demo of the Deadlock Guardian BACKEND for the project review.

Run from the project root:
    python run_backend_demo.py            # prints everything, pauses between steps
    python run_backend_demo.py --fast     # no pauses
    python run_backend_demo.py --save     # also saves RAG images into images/

Steps shown:
    1. Banker's Algorithm  - safe state, granted request, denied request
    2. Deadlock detection  - single-instance (cycle) and multi-instance (matrix)
    3. Victim selection    - cost ranking + automatic recovery
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.resource_system import ResourceSystem
from src.rag import build_rag, draw_rag
from src.detection import cycle_detect, detect_deadlock_multi_instance
from src.victim_selection import ProcessProfile, compute_costs, recover_until_safe

FAST = "--fast" in sys.argv
SAVE = "--save" in sys.argv


def banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def pause():
    if not FAST:
        input("\n[press Enter to continue] ")


def show_matrix(label, matrix, names=None):
    print(f"  {label}:")
    for i, row in enumerate(matrix):
        tag = names[i] if names else f"P{i}"
        print(f"    {tag}: {row}")


# ----------------------------------------------------------------------
def step1_bankers():
    banner("STEP 1 - BANKER'S ALGORITHM (Deadlock Avoidance)")
    total = [10, 5, 7]
    allocation = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
    max_claim = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
    rs = ResourceSystem(total, allocation, max_claim)

    print("  Total resources :", total)
    show_matrix("Allocation", rs.allocation)
    show_matrix("Max claim", rs.max_claim)
    show_matrix("Need (max - allocation)", rs.need())
    print("  Available       :", rs.available)

    safe, seq = rs.is_safe_state()
    print(f"\n  Is the current state safe?  {safe}")
    print(f"  Safe sequence               : {' -> '.join(seq)}")

    print("\n  --- P1 requests [1, 0, 2] ---")
    granted, reason, seq = rs.request_resources(1, [1, 0, 2], verbose=False)
    print(f"  Granted: {granted} | {reason}")
    print(f"  Safe sequence after grant   : {' -> '.join(seq)}")

    print("\n  --- P4 requests [3, 3, 0] (more than available) ---")
    granted, reason, _ = rs.request_resources(4, [3, 3, 0], verbose=False)
    print(f"  Granted: {granted} | {reason}")

    print("\n  --- P0 requests [0, 2, 0] (would make the state UNSAFE) ---")
    granted, reason, _ = rs.request_resources(0, [0, 2, 0], verbose=False)
    print(f"  Granted: {granted} | {reason}")
    print("  State unchanged, available still:", rs.available)


# ----------------------------------------------------------------------
def step2_detection():
    banner("STEP 2 - DEADLOCK DETECTION")

    print("  [A] Single-instance resources -> cycle in wait-for graph")
    allocation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    request = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    rag = build_rag(allocation, request)
    deadlocked, cycle, wfg = cycle_detect(rag)
    print("      P0 holds R0 wants R1 | P1 holds R1 wants R2 | P2 holds R2 wants R0")
    print(f"      Deadlock?  {deadlocked}")
    print(f"      Cycle   :  {' -> '.join(cycle)} -> {cycle[0]}")
    print(f"      Wait-for graph edges: {list(wfg.edges())}")
    if SAVE:
        os.makedirs("images", exist_ok=True)
        draw_rag(rag, title="RAG - Deadlock (cycle highlighted)",
                 highlight_cycle=cycle, save_path="images/demo_rag_deadlock.png")
        plt.close("all")
        print("      Saved images/demo_rag_deadlock.png")

    print("\n  [B] Multi-instance resources -> matrix reduction")
    available = [0, 0, 0]
    allocation = [[0, 1, 0], [2, 0, 0], [3, 0, 3], [2, 1, 1], [0, 0, 2]]
    request = [[0, 0, 0], [4, 0, 0], [0, 0, 0], [0, 5, 0], [0, 0, 5]]
    dead, order = detect_deadlock_multi_instance(available, allocation, request)
    print(f"      Processes that can finish : {order}")
    print(f"      DEADLOCKED processes      : {dead}")


# ----------------------------------------------------------------------
def step3_recovery():
    banner("STEP 3 - COST-BASED VICTIM SELECTION (Recovery)")
    profiles = [
        ProcessProfile("P0", cpu_time_used=120, priority=8, resources_held=1, progress=0.85, interactive=True),
        ProcessProfile("P1", cpu_time_used=5, priority=2, resources_held=1, progress=0.10),
        ProcessProfile("P2", cpu_time_used=40, priority=4, resources_held=1, progress=0.45),
    ]
    print("  Deadlocked processes and their profiles:")
    for p in profiles:
        print(f"    {p.name}: cpu={p.cpu_time_used}s priority={p.priority} "
              f"progress={int(p.progress * 100)}% interactive={p.interactive}")

    print("\n  Cost ranking (lowest cost = cheapest victim):")
    for name, total, breakdown in compute_costs(profiles):
        parts = ", ".join(f"{k}={v:+.2f}" for k, v in breakdown.items() if abs(v) > 1e-9)
        print(f"    {name}: total={total:.3f}   ({parts})")

    print("\n  Automatic recovery loop:")
    allocation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    request = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    rs = ResourceSystem([1, 1, 1], allocation, [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
                        process_names=["P0", "P1", "P2"])

    def detect_fn():
        # a process that was aborted holds nothing and requests nothing
        req = [r[:] if any(rs.allocation[i]) else [0, 0, 0] for i, r in enumerate(request)]
        d, c, _ = cycle_detect(build_rag(rs.allocation, req))
        return d, c

    aborted = recover_until_safe(["P0", "P1", "P2"], {p.name: p for p in profiles},
                                 rs, detect_fn, verbose=True)
    print(f"\n  Aborted (in order): {[(n, round(c, 3)) for n, c in aborted]}")
    print(f"  Final allocation  : {rs.allocation}")
    print(f"  Final available   : {rs.available}")


if __name__ == "__main__":
    print("DEADLOCK GUARDIAN - backend demo")
    step1_bankers(); pause()
    step2_detection(); pause()
    step3_recovery()
    print("\nDemo complete.")
