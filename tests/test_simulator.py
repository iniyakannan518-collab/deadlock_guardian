"""Tests for the stress-test simulator. Run: python tests/test_simulator.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulator import run_trial, run_many, summarize


def test_same_seed_gives_same_result():
    a = run_trial(7, avoidance=False)
    b = run_trial(7, avoidance=False)
    assert a == b


def test_avoidance_never_deadlocks_and_all_finish():
    on, _ = run_many(100)
    s = summarize(on)
    assert s["deadlock_events"] == 0
    assert s["processes_aborted"] == 0
    assert s["processes_completed"] == 100 * 5


def test_without_avoidance_deadlocks_happen_and_are_recovered():
    _, off = run_many(100)
    s = summarize(off)
    assert s["deadlock_events"] > 0
    assert s["processes_aborted"] > 0
    # every process ends up either completed or aborted (no one stuck forever)
    assert s["processes_completed"] + s["processes_aborted"] == 100 * 5


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
    print(f"All {len(tests)} tests passed.")
