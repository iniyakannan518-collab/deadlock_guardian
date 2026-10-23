
from src.cafe_simulation import CafeResourceSystem


def test_vip_order_is_first_in_safe_sequence():
    resources = {
        "espresso_machine": 1,
        "grinder": 1,
    }

    rs = CafeResourceSystem(resources)

    # Register normal order first
    rs.register_order(
        1,
        {"espresso_machine": 1, "grinder": 1},
        priority=1,
    )

    # Register VIP order second
    rs.register_order(
        2,
        {"espresso_machine": 1, "grinder": 1},
        priority=10,
    )

    is_safe, sequence = rs._is_safe_state()

    assert is_safe is True
    assert sequence == [2, 1]


def test_equal_priority_preserves_registration_order():
    resources = {
        "espresso_machine": 1,
        "grinder": 1,
    }

    rs = CafeResourceSystem(resources)

    rs.register_order(
        1,
        {"espresso_machine": 1, "grinder": 1},
        priority=1,
    )

    rs.register_order(
        2,
        {"espresso_machine": 1, "grinder": 1},
        priority=1,
    )

    is_safe, sequence = rs._is_safe_state()

    assert is_safe is True
    assert sequence == [1, 2]
