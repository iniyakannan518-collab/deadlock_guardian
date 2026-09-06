"""
resource_system.py
-------------------
Core data structures for the Deadlock Guardian system, plus the
Banker's Algorithm for deadlock AVOIDANCE.

A ResourceSystem tracks, for a set of processes P0..Pn-1 and resource
types R0..Rm-1:
    - total[j]       : total instances of resource j in the system
    - available[j]    : instances of resource j currently free
    - allocation[i][j]: instances of resource j currently held by process i
    - max_claim[i][j] : maximum instances of resource j process i may ever need
    - need[i][j]      : max_claim[i][j] - allocation[i][j]  (derived)

This class is deliberately independent of the graph/visualization code
so it can be unit tested in isolation.
"""

from copy import deepcopy


class ResourceSystem:
    def __init__(self, total, allocation, max_claim, process_names=None, resource_names=None):
        """
        total:      list[int]         length m  -- total units of each resource
        allocation: list[list[int]]   n x m      -- units held by each process
        max_claim:  list[list[int]]   n x m      -- max units each process may request
        """
        self.n = len(allocation)          # number of processes
        self.m = len(total)               # number of resource types

        self.total = list(total)
        self.allocation = [row[:] for row in allocation]
        self.max_claim = [row[:] for row in max_claim]

        self.process_names = process_names or [f"P{i}" for i in range(self.n)]
        self.resource_names = resource_names or [f"R{j}" for j in range(self.m)]

        allocated_per_resource = [sum(self.allocation[i][j] for i in range(self.n))
                                   for j in range(self.m)]
        self.available = [self.total[j] - allocated_per_resource[j] for j in range(self.m)]

        self._validate()

    # ------------------------------------------------------------------
    def _validate(self):
        for j in range(self.m):
            if self.available[j] < 0:
                raise ValueError(
                    f"Resource {self.resource_names[j]} over-allocated: "
                    f"total={self.total[j]} but allocated={self.total[j] - self.available[j]}"
                )
        for i in range(self.n):
            for j in range(self.m):
                if self.max_claim[i][j] < self.allocation[i][j]:
                    raise ValueError(
                        f"{self.process_names[i]} allocation exceeds its max_claim "
                        f"for {self.resource_names[j]}"
                    )

    def need(self):
        """Return the n x m NEED matrix (max_claim - allocation)."""
        return [[self.max_claim[i][j] - self.allocation[i][j] for j in range(self.m)]
                for i in range(self.n)]

    def snapshot(self):
        """Deep copy of the current state, useful for simulation logging."""
        return {
            "available": self.available[:],
            "allocation": deepcopy(self.allocation),
            "max_claim": deepcopy(self.max_claim),
            "need": self.need(),
        }

    # ------------------------------------------------------------------
    # Banker's Algorithm
    # ------------------------------------------------------------------
    def is_safe_state(self, available=None, allocation=None, need=None, verbose=False):
        """
        Classic Banker's safety algorithm.
        Returns (is_safe: bool, safe_sequence: list[str] or None).
        Can be run against a hypothetical state (used by request_resources)
        or the system's current state (defaults).
        """
        available = available[:] if available is not None else self.available[:]
        allocation = deepcopy(allocation) if allocation is not None else deepcopy(self.allocation)
        need = deepcopy(need) if need is not None else self.need()

        n = self.n
        finish = [False] * n
        safe_sequence = []
        work = available[:]

        progress = True
        while progress and len(safe_sequence) < n:
            progress = False
            for i in range(n):
                if finish[i]:
                    continue
                if all(need[i][j] <= work[j] for j in range(self.m)):
                    # Process i can finish; it releases its resources back
                    work = [work[j] + allocation[i][j] for j in range(self.m)]
                    finish[i] = True
                    safe_sequence.append(self.process_names[i])
                    progress = True
                    if verbose:
                        print(f"  {self.process_names[i]} can finish -> work becomes {work}")

        is_safe = all(finish)
        return is_safe, (safe_sequence if is_safe else None)

    def request_resources(self, proc_index, request, verbose=True):
        """
        Attempt to grant `request` (list[int], length m) to process proc_index
        using the Banker's Algorithm. The request is only granted if:
          1. request <= need[proc_index]         (doesn't exceed declared max)
          2. request <= available                (resources actually exist)
          3. the resulting state is SAFE          (Banker's check)

        Returns (granted: bool, reason: str, safe_sequence or None).
        On success, self.allocation / self.available are updated in place.
        On failure, state is left unchanged (request is rolled back).
        """
        need = self.need()
        name = self.process_names[proc_index]

        # Rule 1
        if any(request[j] > need[proc_index][j] for j in range(self.m)):
            return False, f"{name} requested more than its declared maximum need.", None

        # Rule 2
        if any(request[j] > self.available[j] for j in range(self.m)):
            return False, f"{name} must wait: resources not currently available.", None

        # Rule 3 -- tentatively grant, then test safety
        trial_available = [self.available[j] - request[j] for j in range(self.m)]
        trial_allocation = deepcopy(self.allocation)
        trial_allocation[proc_index] = [trial_allocation[proc_index][j] + request[j]
                                         for j in range(self.m)]
        trial_need = [[self.max_claim[i][j] - trial_allocation[i][j] for j in range(self.m)]
                      for i in range(self.n)]

        safe, sequence = self.is_safe_state(trial_available, trial_allocation, trial_need, verbose=False)

        if not safe:
            return False, f"Granting {name}'s request would lead to an UNSAFE state. Request denied.", None

        # Commit
        self.available = trial_available
        self.allocation = trial_allocation
        if verbose:
            print(f"[Banker's] Granted {request} to {name}. Safe sequence: {sequence}")
        return True, f"Request granted. System remains in a safe state.", sequence

    def release_resources(self, proc_index, release):
        """Process voluntarily releases resources (e.g. finished / partial release)."""
        name = self.process_names[proc_index]
        for j in range(self.m):
            if release[j] > self.allocation[proc_index][j]:
                raise ValueError(f"{name} cannot release more than it holds of {self.resource_names[j]}")
        for j in range(self.m):
            self.allocation[proc_index][j] -= release[j]
            self.available[j] += release[j]

    def __repr__(self):
        return (f"ResourceSystem(n={self.n}, m={self.m}, "
                f"available={self.available}, allocation={self.allocation})")
