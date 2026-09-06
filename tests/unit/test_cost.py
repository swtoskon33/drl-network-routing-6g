"""Eq. (1) to (3) checked by hand.

Each test states the number the paper's formula gives and compares it with what the code
computes, so a change to the cost model has to justify itself against the equations
rather than against whatever it produced last time.
"""
import math

import pytest

from drl_routing.routing.cost import (
    ARRIVAL_RATE_HZ,
    LATENCY_BUDGET_MS,
    SUBBAND_COUNT,
    T_PROC_MS,
    TTI_MS,
    Link,
    Network,
    collision_probability,
    queueing_delay_ms,
    transmission_time_ms,
    transport_block_bytes,
)


def _link(src, dst, sinr_db=20.0, bler=0.001):
    return Link(src=src, dst=dst, kind="backhaul", distance_m=100.0,
                los=True, sinr_db=sinr_db, bler=bler)


@pytest.mark.unit
def test_tti_and_processing_follow_numerology_three():
    """Section III-A: the slot is 1/2^u ms and Tproc is four of them."""
    assert TTI_MS == pytest.approx(0.125)
    assert T_PROC_MS == pytest.approx(0.5)


@pytest.mark.unit
def test_transmission_time_is_slots_rounded_up():
    """Eq. (1): Ttrans = ceil(pkt / TB) x TTI."""
    link = _link(0, 1)
    tb = transport_block_bytes(link)
    expected = math.ceil(100_000 / tb) * TTI_MS
    assert transmission_time_ms(link) == pytest.approx(expected)


@pytest.mark.unit
def test_a_better_link_carries_more_per_slot():
    assert transport_block_bytes(_link(0, 1, sinr_db=30.0)) > \
        transport_block_bytes(_link(0, 1, sinr_db=5.0))


@pytest.mark.unit
def test_delay_matches_equation_two():
    """Eq. (2): Tque + (n+2)/2 x Tproc + the transmission time of every hop."""
    net = Network([_link(0, 1), _link(1, 2)])
    path = [0, 1, 2]
    relays = 1

    transmission = sum(transmission_time_ms(net.link(u, v))
                       for u, v in ((0, 1), (1, 2)))
    queue = queueing_delay_ms(40, transmission_time_ms(net.link(0, 1)),
                              net.graph.degree(0))
    expected = queue + (relays + 2) / 2 * T_PROC_MS + transmission

    assert net.path_delay_ms(path, active_ues=40) == pytest.approx(expected)


@pytest.mark.unit
def test_reliability_is_the_product_over_hops():
    """Eq. (3): P = product of ps(i) x (1 - pc(i))."""
    net = Network([_link(0, 1, bler=0.01), _link(1, 2, bler=0.05)])
    expected = 1.0
    for u, v in ((0, 1), (1, 2)):
        pc = collision_probability(40, net.graph.degree(u))
        expected *= (1.0 - net.link(u, v).bler) * (1.0 - pc)
    assert net.path_reliability([0, 1, 2], active_ues=40) == pytest.approx(expected)


@pytest.mark.unit
def test_more_hops_cost_more_delay_and_less_reliability():
    net = Network([_link(0, 1), _link(1, 2), _link(2, 3)])
    assert net.path_delay_ms([0, 1, 2, 3]) > net.path_delay_ms([0, 1])
    assert net.path_reliability([0, 1, 2, 3]) < net.path_reliability([0, 1])


@pytest.mark.unit
def test_scheduled_transmission_does_not_collide():
    """Section III-C: with the donor coordinating subbands, pc is zero. That is what
    makes Problem 1 an exact shortest path."""
    assert collision_probability(40, 6) == 0.0
    assert collision_probability(1000, 8) == 0.0


@pytest.mark.unit
def test_unscheduled_collision_rises_with_contention():
    """Under the DRL framework each node decides locally and nothing coordinates the
    subband choices, so collisions appear -- with the neighbours in range and with the
    traffic they carry."""
    assert collision_probability(40, 8, scheduled=False) > \
        collision_probability(40, 2, scheduled=False)
    assert collision_probability(100, 6, scheduled=False) > \
        collision_probability(20, 6, scheduled=False)
    assert collision_probability(40, 0, scheduled=False) == 0.0


@pytest.mark.unit
def test_a_single_transmitter_collides_at_the_subband_rate():
    """One neighbour, always transmitting: the chance it picks the same subband."""
    load = min(1000 * ARRIVAL_RATE_HZ / 1000.0 * TTI_MS, 1.0)
    assert load == pytest.approx(1.0)
    assert collision_probability(1000, 1, scheduled=False) == \
        pytest.approx(1.0 / SUBBAND_COUNT)


@pytest.mark.unit
def test_queueing_grows_with_the_ue_count():
    """Fig. 5(a): latency rises with the number of UEs, through the queue."""
    light = queueing_delay_ms(20, 1.5, 6)
    heavy = queueing_delay_ms(100, 1.5, 6)
    assert heavy > light


@pytest.mark.unit
def test_targets_are_the_3gpp_ones():
    net = Network([_link(0, 1)])
    within_budget, reliable = net.meets_targets([0, 1])
    assert LATENCY_BUDGET_MS == pytest.approx(5.0)
    assert isinstance(within_budget, bool) and isinstance(reliable, bool)
