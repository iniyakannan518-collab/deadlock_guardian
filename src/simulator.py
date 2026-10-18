"""
simulator.py
------------
Stress-test simulator: runs the SAME random workload twice and compares

    avoidance ON  : every request is checked with Banker's Algorithm
    avoidance OFF : requests are granted whenever resources are free; deadlocks
                    are found by matrix-reduction detection and broken by
                    cost-based victim selection

Each trial builds a random system (processes, multi-instance resources, max
claims), then repeatedly picks a random active process that either requests
resources (up to its remaining need) or, once fully allocated, finishes and
releases everything.

Run from the project root:
    python -m src.simulator                 # 300 trials, prints a summary table
    python -m src.simulator --trials 500 --save images/simulation_results.png
"""
import argparse
import random
from dataclasses import dataclass

from .resource_system import ResourceSystem
from .detection import detect_deadlock_multi_instance
from .victim_selection import ProcessProfile, compute_costs


@dataclass
class TrialResult:
    completed: int = 0       # processes that ran to completion
    aborted: int = 0         # processes killed to break deadlocks
    deadlocks: int = 0       # deadlock events detected
    denied: int = 0          # requests refused / left waiting
    steps: int = 0


def make_workload(rng, n_proc, n_res):
    """Random system: total units per resource and a max claim per process."""
    total = [rng.randint(3, 6) for _ in range(n_res)]
    max_claim = [[rng.randint(0, total[j]) for j in range(n_res)] for _ in range(n_proc)]
    for row in max_claim:                      # every process needs something
        if not any(row):
            row[rng.randrange(n_res)] = 1
    profiles = [
        ProcessProfile(f"P{i}", cpu_time_used=rng.randint(1, 100),
                       priority=rng.randint(1, 10), resources_held=0,
                       progress=0.0, interactive=rng.random() < 0.2)
        for i in range(n_proc)
    ]
    return total, max_claim, profiles


def _random_request(rng, need_row):
    req = [rng.randint(0, v) for v in need_row]
    if not any(req):                           # always ask for at least one unit
        j = rng.choice([k for k, v in enumerate(need_row) if v > 0])
        req[j] = 1
    return req


def run_trial(seed, avoidance, n_proc=5, n_res=3, max_steps=400):
    rng = random.Random(seed)
    total, max_claim, profiles = make_workload(rng, n_proc, n_res)
    rs = ResourceSystem(total, [[0] * n_res for _ in range(n_proc)], max_claim)
    res = TrialResult()

    active = set(range(n_proc))
    pending = {}                               # proc index -> blocked request

    for step in range(max_steps):
        if not active:
            break
        res.steps = step + 1
        i = rng.choice(sorted(active))
        need_row = rs.need()[i]

        # fully allocated -> finish and release everything
        if not any(need_row):
            rs.release_resources(i, rs.allocation[i][:])
            active.discard(i)
            pending.pop(i, None)
            res.completed += 1
            continue

        req = pending.get(i) or _random_request(rng, need_row)

        if avoidance:
            granted, _, _ = rs.request_resources(i, req, verbose=False)
            if not granted:
                res.denied += 1
        else:
            if all(req[j] <= rs.available[j] for j in range(n_res)):
                for j in range(n_res):
                    rs.available[j] -= req[j]
                    rs.allocation[i][j] += req[j]
                pending.pop(i, None)
            else:
                pending[i] = req
                res.denied += 1

        if not avoidance:
            _detect_and_recover(rs, pending, active, profiles, res)

    return res


def _deadlocked(rs, pending, active):
    """Names of processes that can never finish (matrix-reduction detection)."""
    m = rs.m
    request = [pending.get(i, [0] * m) if i in active else [0] * m for i in range(rs.n)]
    dead, _ = detect_deadlock_multi_instance(rs.available, rs.allocation, request,
                                             rs.process_names)
    return dead


def _detect_and_recover(rs, pending, active, profiles, res):
    """Detect a deadlock; abort the cheapest victim until it is broken."""
    counted = False
    while True:
        dead = _deadlocked(rs, pending, active)
        if not dead:
            return
        if not counted:                       # one event per detection, however many victims
            res.deadlocks += 1
            counted = True
        cands = []
        for name in dead:
            idx = rs.process_names.index(name)
            p = profiles[idx]
            frac = sum(rs.allocation[idx]) / max(1, sum(rs.max_claim[idx]))
            cands.append(ProcessProfile(name, p.cpu_time_used, p.priority,
                                        sum(rs.allocation[idx]), frac, p.interactive))
        victim = rs.process_names.index(compute_costs(cands)[0][0])
        rs.release_resources(victim, rs.allocation[victim][:])
        active.discard(victim)
        pending.pop(victim, None)
        res.aborted += 1


def run_many(trials=300, **kw):
    on, off = [], []
    for seed in range(trials):
        on.append(run_trial(seed, avoidance=True, **kw))
        off.append(run_trial(seed, avoidance=False, **kw))
    return on, off


def summarize(results):
    n = len(results)
    return {
        "trials": n,
        "trials_with_deadlock": sum(1 for r in results if r.deadlocks > 0),
        "deadlock_events": sum(r.deadlocks for r in results),
        "processes_completed": sum(r.completed for r in results),
        "processes_aborted": sum(r.aborted for r in results),
        "requests_denied_or_waiting": sum(r.denied for r in results),
    }


def print_table(on_sum, off_sum):
    rows = [
        ("Trials run", "trials"),
        ("Trials that hit a deadlock", "trials_with_deadlock"),
        ("Deadlock events", "deadlock_events"),
        ("Processes completed", "processes_completed"),
        ("Processes aborted (victims)", "processes_aborted"),
        ("Requests denied / left waiting", "requests_denied_or_waiting"),
    ]
    print(f"\n{'Metric':36s}{'Avoidance ON':>16s}{'Avoidance OFF':>16s}")
    print("-" * 68)
    for label, key in rows:
        print(f"{label:36s}{on_sum[key]:>16d}{off_sum[key]:>16d}")


def save_chart(on_sum, off_sum, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Deadlock events", "Processes aborted", "Processes completed"]
    keys = ["deadlock_events", "processes_aborted", "processes_completed"]
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    b1 = ax.bar([i - w / 2 for i in x], [on_sum[k] for k in keys], w,
                label="Avoidance ON (Banker's)", color="#2E5CE6")
    b2 = ax.bar([i + w / 2 for i in x], [off_sum[k] for k in keys], w,
                label="Avoidance OFF (detect + recover)", color="#E8930C")
    ax.bar_label(b1, padding=3)
    ax.bar_label(b2, padding=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count over all trials")
    ax.set_title(f"Same workload, two strategies ({on_sum['trials']} trials)",
                 fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--save", default=None, help="path for the comparison chart PNG")
    args = ap.parse_args()
    on, off = run_many(args.trials)
    s_on, s_off = summarize(on), summarize(off)
    print_table(s_on, s_off)
    if args.save:
        save_chart(s_on, s_off, args.save)
        print(f"\nSaved chart to {args.save}")
