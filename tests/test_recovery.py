
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.resource_system import ResourceSystem
from src.rag import build_rag
from src.detection import cycle_detect
from src.victim_selection import ProcessProfile, recover_until_safe


def test_recovery_releases_resources_and_resolves_cycle():
    # P0 holds R0 and requests R1.
    # P1 holds R1 and requests R0.
    process_names = ["P0", "P1"]
    resource_names = ["R0", "R1"]

    allocation = [
        [1, 0],
        [0, 1],
    ]

    request = [
        [0, 1],
        [1, 0],
    ]

    profiles = {
        "P0": ProcessProfile(
            "P0", cpu_time_used=10, priority=2,
            resources_held=1, progress=0.2
        ),
        "P1": ProcessProfile(
            "P1", cpu_time_used=80, priority=9,
            resources_held=1, progress=0.9
        ),
    }

    resource_system = ResourceSystem(
        total=[1, 1],
        allocation=allocation,
        max_claim=[
            [1, 1],
            [1, 1],
        ],
        process_names=process_names,
        resource_names=resource_names,
    )

    # Verify the initial deadlock.
    rag = build_rag(
        resource_system.allocation,
        request,
        process_names=process_names,
        resource_names=resource_names,
    )
    deadlocked, _, _ = cycle_detect(rag)
    assert deadlocked, "Test setup must contain a deadlock"

    # Rebuild the graph after each victim's resources are released.
    def detect_current_deadlock():
        current_rag = build_rag(
            resource_system.allocation,
            request,
            process_names=process_names,
            resource_names=resource_names,
        )
        is_deadlocked, _, _ = cycle_detect(current_rag)
        return is_deadlocked, None

    aborted = recover_until_safe(
        deadlocked_names=process_names,
        profiles=profiles,
        resource_system=resource_system,
        detect_fn=detect_current_deadlock,
        verbose=True,
    )

    assert len(aborted) >= 1
    assert len(aborted) <= 2
    assert sum(resource_system.available) >= 1
    print("Recovery test passed:", aborted)


if __name__ == "__main__":
    test_recovery_releases_resources_and_resolves_cycle()
    print("PASSED: recovery test")