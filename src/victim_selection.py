"""
victim_selection.py
--------------------
Deadlock RECOVERY: once a deadlock has been detected among a set of
processes, the system must break the cycle by aborting/preempting one or
more processes. Naively picking a random victim is wasteful; this module
implements a weighted COST FUNCTION (as suggested in Silberschatz et al.,
"Operating System Concepts", ch. on Deadlocks) so the "cheapest" process
to kill is chosen first.

Cost factors considered (each normalized to [0, 1] before weighting so
no single raw unit -- e.g. resource COUNT vs CPU time -- dominates just
because of its scale):

    - cpu_time_used     : how much computation would be thrown away
    - priority          : higher-priority processes are costlier to kill
    - resources_held     : count of resource units held (more units held
                            => killing it frees more, but also destroys
                            more work -- counted as a cost, tunable)
    - progress           : how close to completion (0..1); near-complete
                            processes are expensive to kill
    - interactive        : bool; interactive/user-facing processes cost more
    - num_restarts       : how many times this process has already been
                            rolled back (used to avoid starvation --
                            repeatedly picking the same victim)

A LOWER total cost = BETTER victim (cheaper to sacrifice).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


DEFAULT_WEIGHTS = {
    "cpu_time_used": 0.30,
    "priority": 0.25,
    "resources_held": 0.15,
    "progress": 0.20,
    "interactive": 0.05,
    "num_restarts": -0.15,   # NEGATIVE weight: processes restarted often should
                             # become LESS likely to be picked again (avoid starvation)
}


@dataclass
class ProcessProfile:
    name: str
    cpu_time_used: float          # seconds of CPU time consumed so far
    priority: int                 # 1 (low) .. 10 (high)
    resources_held: int           # total resource UNITS currently held
    progress: float                # 0.0 .. 1.0, fraction of task complete
    interactive: bool = False     # True if user-facing / interactive
    num_restarts: int = 0         # times this process has been chosen as victim before


def _normalize(values: List[float]) -> List[float]:
    """Min-max normalize a list of values into [0, 1]. Constant lists -> all 0."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def compute_costs(candidates: List[ProcessProfile],
                   weights: Dict[str, float] = None) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    Compute a normalized weighted cost for every candidate process.
    Returns a list of (process_name, total_cost, breakdown_dict), SORTED
    ascending by cost (index 0 = cheapest = best victim to abort first).
    """
    weights = weights or DEFAULT_WEIGHTS
    if not candidates:
        return []

    cpu = _normalize([c.cpu_time_used for c in candidates])
    pr = _normalize([c.priority for c in candidates])
    res = _normalize([c.resources_held for c in candidates])
    prog = _normalize([c.progress for c in candidates])
    inter = [1.0 if c.interactive else 0.0 for c in candidates]
    restarts = _normalize([c.num_restarts for c in candidates])

    results = []
    for idx, c in enumerate(candidates):
        breakdown = {
            "cpu_time_used": cpu[idx] * weights["cpu_time_used"],
            "priority": pr[idx] * weights["priority"],
            "resources_held": res[idx] * weights["resources_held"],
            "progress": prog[idx] * weights["progress"],
            "interactive": inter[idx] * weights["interactive"],
            "num_restarts": restarts[idx] * weights["num_restarts"],
        }
        total = sum(breakdown.values())
        results.append((c.name, total, breakdown))

    results.sort(key=lambda t: t[1])
    return results


def select_victim(candidates: List[ProcessProfile], weights: Dict[str, float] = None):
    """
    Convenience: return just the single best (cheapest) victim's name,
    its cost, and the full ranked list -- for use in recovery routines.
    """
    ranked = compute_costs(candidates, weights)
    if not ranked:
        return None, None, []
    best_name, best_cost, _ = ranked[0]
    return best_name, best_cost, ranked


def recover_until_safe(deadlocked_names: List[str],
                        profiles: Dict[str, ProcessProfile],
                        resource_system,
                        detect_fn,
                        weights: Dict[str, float] = None,
                        verbose: bool = True):
    """
    Iteratively pick the cheapest victim among the currently-deadlocked
    processes, "abort" it (release all its held resources back to the
    system), then re-run detect_fn to see if the deadlock is resolved.
    Repeats until no deadlock remains or no candidates are left.

    resource_system: a ResourceSystem instance (its .allocation/.available
                      are mutated as victims are aborted)
    detect_fn:        a zero-arg callable returning (deadlocked: bool, cycle_or_list)
                      re-evaluated after each abort

    Returns list of (victim_name, cost) in the order they were aborted.
    """
    aborted = []
    remaining = set(deadlocked_names)

    while remaining:
        candidates = [profiles[n] for n in remaining if n in profiles]
        if not candidates:
            break
        victim_name, cost, ranked = select_victim(candidates, weights)
        if verbose:
            print(f"[Recovery] Ranked victims (cheapest first): "
                  f"{[(n, round(c, 3)) for n, c, _ in ranked]}")
            print(f"[Recovery] Aborting {victim_name} (cost={cost:.3f}) to break the cycle.")

        idx = resource_system.process_names.index(victim_name)
        held = resource_system.allocation[idx][:]
        resource_system.release_resources(idx, held)
        aborted.append((victim_name, cost))
        remaining.discard(victim_name)

        still_deadlocked, _ = detect_fn()
        if not still_deadlocked:
            if verbose:
                print("[Recovery] Deadlock resolved.")
            break

    return aborted
