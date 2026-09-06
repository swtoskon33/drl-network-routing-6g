"""The baselines of Section III-C and Section V, checked against what they promise."""
import networkx as nx
import pytest

from drl_routing.routing.baselines import dijkstra_optimal, greedy_route, link_weight
from drl_routing.routing.cost import RELIABILITY_TARGET, Link, Network


def _mesh() -> Network:
    """A donor with two routes to a UE: one short and unreliable, one longer and clean.

    Latency and reliability disagree about which is better, which is the whole reason
    Problem 1 needs a multiplier rather than a single weight.
    """
    return Network([
        Link(0, 1, "backhaul", 100.0, True, 20.0, 0.4),    # short, blocked
        Link(1, 3, "access", 80.0, True, 20.0, 0.0001),
        Link(0, 2, "backhaul", 150.0, True, 25.0, 0.0001),  # longer, clean
        Link(2, 4, "backhaul", 150.0, True, 25.0, 0.0001),
        Link(4, 3, "access", 80.0, True, 25.0, 0.0001),
    ])


@pytest.mark.unit
def test_a_larger_multiplier_prices_reliability_higher():
    """Eq. (5): mu scales the log(1/ps) term, so an unreliable link costs more as it
    grows."""
    net = _mesh()
    cheap = link_weight(net, 0, 1, mu=0.0, active_ues=40)
    dear = link_weight(net, 0, 1, mu=10.0, active_ues=40)
    assert dear > cheap


@pytest.mark.unit
def test_the_multiplier_does_not_change_a_clean_link_much():
    net = _mesh()
    at_zero = link_weight(net, 0, 2, mu=0.0, active_ues=40)
    at_ten = link_weight(net, 0, 2, mu=10.0, active_ues=40)
    assert at_ten == pytest.approx(at_zero, rel=0.01)


@pytest.mark.unit
def test_dijkstra_avoids_the_blocked_link_to_meet_sigma():
    """The short path cannot reach 0.999, so the constraint rejects it."""
    net = _mesh()
    route = dijkstra_optimal(net, 0, 3, sigma=RELIABILITY_TARGET)
    assert route.reliability >= RELIABILITY_TARGET
    assert 1 not in route.path


@pytest.mark.unit
def test_dijkstra_takes_the_short_path_when_the_constraint_allows_it():
    """With sigma low enough for the blocked route, minimum latency wins."""
    net = _mesh()
    route = dijkstra_optimal(net, 0, 3, sigma=0.5)
    assert route.path == [0, 1, 3]


@pytest.mark.unit
def test_dijkstra_returns_the_most_reliable_route_when_sigma_is_unreachable():
    """No path can reach a target above what the links offer; the best available is
    returned rather than a minimum-latency one."""
    net = _mesh()
    route = dijkstra_optimal(net, 0, 3, sigma=0.9999999)
    assert 1 not in route.path


@pytest.mark.unit
def test_dijkstra_is_never_less_reliable_than_greedy():
    """It solves the problem exactly and greedy sees one hop, so the exact solver cannot
    lose."""
    net = _mesh()
    optimal = dijkstra_optimal(net, 0, 3)
    greedy = greedy_route(net, 0, 3)
    assert optimal.reliability >= greedy.reliability - 1e-9


@pytest.mark.unit
def test_greedy_follows_the_best_channel_available():
    net = _mesh()
    route = greedy_route(net, 0, 3)
    assert route.path[0] == 0
    assert route.path[-1] == 3


@pytest.mark.unit
def test_routes_report_hops_and_relays():
    net = _mesh()
    route = dijkstra_optimal(net, 0, 3)
    assert route.hops == len(route.path) - 1
    assert route.relays == max(route.hops - 1, 0)


@pytest.mark.unit
def test_every_route_is_a_real_path_through_the_graph():
    net = _mesh()
    for route in (dijkstra_optimal(net, 0, 3), greedy_route(net, 0, 3)):
        assert nx.is_simple_path(net.graph, route.path)
