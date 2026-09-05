"""Tests for the topology model and the scenario traces it produces."""
import networkx as nx
import pytest

from drl_routing.baselines.dijkstra_iab import (
    dijkstra_optimal,
    greedy_path,
    path_delay_ms,
    path_reliability,
)


def _mesh() -> nx.Graph:
    """A donor, two routes to it, and a UE. The short route runs through a blocked link,
    so latency and reliability disagree about which path is better."""
    g = nx.Graph()
    g.add_edge(0, 1, delay_ms=5.0, pb=0.5)      # short but blocked
    g.add_edge(1, 3, delay_ms=1.0, pb=0.0001)
    g.add_edge(0, 2, delay_ms=5.0, pb=0.0001)   # longer, clean
    g.add_edge(2, 4, delay_ms=5.0, pb=0.0001)
    g.add_edge(4, 3, delay_ms=1.0, pb=0.0001)
    return g


@pytest.mark.unit
def test_reliability_is_the_product_over_links():
    g = _mesh()
    assert path_reliability(g, [0, 1, 3]) == pytest.approx(0.5 * 0.9999, rel=1e-3)


@pytest.mark.unit
def test_delay_grows_with_hop_count():
    g = _mesh()
    assert path_delay_ms(g, [0, 2, 4, 3]) > path_delay_ms(g, [0, 1, 3])


@pytest.mark.unit
def test_dijkstra_avoids_a_blocked_link_to_meet_the_target():
    """The whole point of the Lagrangian weight: the shorter path is rejected because it
    cannot reach 0.999, even though it has fewer hops."""
    g = _mesh()
    path, mu = dijkstra_optimal(g, 0, 3, sigma=0.999)
    assert path_reliability(g, path) >= 0.999
    assert 1 not in path
    assert mu > 0


@pytest.mark.unit
def test_greedy_follows_the_best_link_not_the_best_route():
    g = _mesh()
    path = greedy_path(g, 0, 3)
    assert path[0] == 0 and path[-1] == 3
