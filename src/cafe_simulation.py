
import threading
import time
import random
from collections import defaultdict


class CafeResourceSystem:
    """Manage café equipment using Banker's Algorithm."""

    def __init__(self, total_resources):
        self.total = total_resources.copy()
        self.available = total_resources.copy()
        self.allocation = {}
        self.max_claim = {}
        self.priority = {}
        self.lock = threading.RLock()

    def register_order(self, order_id, max_claim, priority=1):
        """Register an order and its maximum resource requirements."""
        with self.lock:
            if order_id in self.allocation:
                raise ValueError(f"Order {order_id} is already registered.")

            for res, count in max_claim.items():
                if res not in self.total:
                    raise ValueError(f"Unknown resource: {res}")
                if not isinstance(count, int) or count < 0:
                    raise ValueError(f"Invalid maximum claim for {res}.")
                if count > self.total[res]:
                    raise ValueError(
                        f"Maximum claim for {res} exceeds total resources."
                    )

            self.allocation[order_id] = {
                res: 0 for res in self.total
            }
            self.max_claim[order_id] = {
                res: max_claim.get(res, 0) for res in self.total
            }
            self.priority[order_id] = priority

    def request_resources(self, order_id, request):
        """Grant a request only if it is valid and leaves a safe state."""
        with self.lock:
            if order_id not in self.allocation:
                print(f"Order {order_id} is not registered.")
                return False

            # Validate the request before changing any resource counts.
            for res, count in request.items():
                if res not in self.total:
                    print(f"Unknown resource: {res}")
                    return False

                if not isinstance(count, int) or isinstance(count, bool):
                    print(f"Invalid resource count for {res}: {count}")
                    return False

                if count < 0:
                    print(f"Negative resource request for {res} is invalid.")
                    return False

                remaining_need = (
                    self.max_claim[order_id][res]
                    - self.allocation[order_id][res]
                )

                if count > remaining_need:
                    print(
                        f"Order {order_id} exceeded its maximum claim "
                        f"for {res}. Remaining need: {remaining_need}"
                    )
                    return False

                if count > self.available[res]:
                    print(
                        f"Resources unavailable for Order {order_id}: {res}"
                    )
                    return False

            # Temporarily allocate the requested resources.
            for res, count in request.items():
                self.available[res] -= count
                self.allocation[order_id][res] += count

            is_safe, safe_sequence = self._is_safe_state()

            if is_safe:
                print(
                    f"[Safe State] Granted request for Order {order_id}: "
                    f"{request} | Safe Sequence: {safe_sequence}"
                )
                return True

            # Roll back if the resulting state is unsafe.
            for res, count in request.items():
                self.available[res] += count
                self.allocation[order_id][res] -= count

            print(
                f"[Unsafe State] Request for Order {order_id} "
                "denied to prevent deadlock."
            )
            return False

    def _is_safe_state(self):
        """Return (is_safe, safe_sequence) using Banker's safety algorithm."""
        with self.lock:
            work = self.available.copy()
            finish = {order_id: False for order_id in self.allocation}
            safe_sequence = []

            # Consider higher-priority orders first; registration order
            # breaks ties because dictionaries preserve insertion order.
            order_ids = sorted(
                self.allocation,
                key=lambda oid: -self.priority.get(oid, 1)
            )

            while True:
                progress = False

                for order_id in order_ids:
                    if finish[order_id]:
                        continue

                    need = {
                        res: (
                            self.max_claim[order_id][res]
                            - self.allocation[order_id][res]
                        )
                        for res in self.total
                    }

                    if all(need[res] <= work[res] for res in self.total):
                        for res in self.total:
                            work[res] += self.allocation[order_id][res]

                        finish[order_id] = True
                        safe_sequence.append(order_id)
                        progress = True

                if not progress:
                    break

            is_safe = all(finish.values())
            return is_safe, safe_sequence

    def release_resources(self, order_id):
        """Release an order's allocated resources and remove its record."""
        with self.lock:
            if order_id not in self.allocation:
                return False

            for res, count in self.allocation[order_id].items():
                self.available[res] += count

            del self.allocation[order_id]
            del self.max_claim[order_id]
            del self.priority[order_id]

            print(f"[Release] Resources released for Order {order_id}.")
            return True


class DeadlockDetector:
    """Build a wait-for graph and detect cycles."""

    def __init__(self, resource_system):
        self.rs = resource_system
        self.waiting = {}

    def add_waiting_order(self, order_id, request):
        self.waiting[order_id] = request.copy()

    def remove_waiting_order(self, order_id):
        self.waiting.pop(order_id, None)

    def build_wait_for_graph(self):
        """Create edges from waiting orders to orders holding needed resources."""
        graph = defaultdict(set)

        with self.rs.lock:
            for waiting_order, request in self.waiting.items():
                if waiting_order not in self.rs.allocation:
                    continue

                for res, count in request.items():
                    if res not in self.rs.total or count <= self.rs.available[res]:
                        continue

                    for holder, allocated in self.rs.allocation.items():
                        if holder == waiting_order:
                            continue

                        if allocated.get(res, 0) > 0:
                            graph[waiting_order].add(holder)

        return graph

    def detect_deadlock(self):
        """Return a cycle if one is found, otherwise return None."""
        graph = self.build_wait_for_graph()
        visiting = set()
        visited = set()
        path = []

        def visit(node):
            if node in visiting:
                start = path.index(node)
                return path[start:] + [node]

            if node in visited:
                return None

            visiting.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                cycle = visit(neighbor)
                if cycle:
                    return cycle

            path.pop()
            visiting.remove(node)
            visited.add(node)
            return None

        for node in list(graph):
            cycle = visit(node)
            if cycle:
                return cycle

        return None


class RecoveryCoordinator:
    """Select a recovery victim using priority and restart cost."""

    def __init__(self, resource_system):
        self.rs = resource_system
        self.num_restarts = defaultdict(int)

    def select_victim_and_recover(self, involved_orders=None):
        with self.rs.lock:
            candidates = [
                oid for oid in self.rs.allocation
                if involved_orders is None or oid in involved_orders
            ]

            if not candidates:
                print("[Recovery] No eligible victim found.")
                return None

            def recovery_cost(order_id):
                priority = self.rs.priority.get(order_id, 1)
                allocated_units = sum(self.rs.allocation[order_id].values())
                restart_count = self.num_restarts[order_id]

                # Lower-priority, resource-heavy, frequently restarted
                # orders are generally preferred victims.
                return (
                    priority * 10
                    + allocated_units
                    + restart_count * 5
                )

            victim_id = min(candidates, key=recovery_cost)
            self.num_restarts[victim_id] += 1

        print(f"[Recovery] Selected Order {victim_id} as victim.")
        self.rs.release_resources(victim_id)
        return victim_id


def run_customer_order(order_id, rs, detector, recovery, request, priority):
    """Simple demonstration of a customer order requesting café equipment."""
    rs.register_order(
        order_id,
        max_claim=request,
        priority=priority
    )

    detector.add_waiting_order(order_id, request)

    while order_id in rs.allocation:
        granted = rs.request_resources(order_id, request)

        if granted:
            detector.remove_waiting_order(order_id)
            print(f"Order {order_id} is preparing the café order...")
            time.sleep(random.uniform(0.5, 1.5))

            rs.release_resources(order_id)
            detector.remove_waiting_order(order_id)
            print(f"Order {order_id} completed.")
            return

        cycle = detector.detect_deadlock()

        if cycle:
            print(f"[Deadlock Detected] Cycle: {cycle}")
            recovery.select_victim_and_recover(set(cycle[:-1]))
            detector.remove_waiting_order(order_id)
            return

        # This demo does not block on a condition variable; it retries.
        time.sleep(0.2)

    detector.remove_waiting_order(order_id)


def main():
    print("=== Cafe Deadlock Guardian Simulation ===")

    total_resources = {
        "coffee_machine": 1,
        "oven": 1,
        "blender": 1,
    }

    rs = CafeResourceSystem(total_resources)
    detector = DeadlockDetector(rs)
    recovery = RecoveryCoordinator(rs)

    # Each order's maximum claim must be within the total resources.
    orders = [
        (1, {"coffee_machine": 1}, 10),
        (2, {"blender": 1}, 1),
    ]

    threads = []

    for order_id, request, priority in orders:
        thread = threading.Thread(
            target=run_customer_order,
            args=(order_id, rs, detector, recovery, request, priority)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("\n=== Simulation Finished ===")
    print("Available resources:", rs.available)


if __name__ == "__main__":
    main()
