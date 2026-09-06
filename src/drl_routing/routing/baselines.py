"""The optimal routing problem, Eq. (4) to (7) and Algorithm 1.

Phase 3. The donor knows the whole graph and the condition of every link, so it can solve
Problem 1 exactly: the minimum-latency path subject to a reliability floor.

    minimise  Tdelay(q)   subject to  P(q) >= sigma

The constraint is folded into the objective with a Lagrange multiplier (Eq. 5), which
turns the problem into a shortest path under the link weight

    c(i, mu) = Tproc/2 + Ttrans(i) + mu * log(1 / ps(i))

and leaves finding mu. Lemma 1 says the optimum is where P(q*(mu)) meets sigma, and
Algorithm 1 bisects to get there.

This is the paper's expert: it needs a current view of every link in the network, which
is what the learned policy does without.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import networkx as nx

from drl_routing.routing.cost import (
    RELIABILITY_TARGET,
    T_PROC_MS,
    Network,
    collision_probability,
    transmission_time_ms,
)

MU_MAX = 100.0
BISECTION_ITERATIONS = 40
BISECTION_TOLERANCE = 1e-3


@dataclass(frozen=True)
class Route:
    """One source-destination decision and what it costs."""

    source: int
    destination: int
    algorithm: str
    path: list[int]
    delay_ms: float
    reliability: float
    mu: float | None = None

    @property
    def hops(self) -> int:
        return len(self.path) - 1

    @property
    def relays(self) -> int:
        return max(self.hops - 1, 0)


def link_weight(net: Network, u: int, v: int, mu: float, active_ues: int) -> float:
    """c(i, mu) from Eq. (5).

    Half the processing time, the transmission time, and the reliability term weighted by
    mu. The reliability term uses the same ps(i)(1 - pc(i)) that Eq. (3) multiplies, so
    the weight prices exactly what the constraint measures.
    """
    link = net.link(u, v)
    pc = collision_probability(active_ues, net.graph.degree(u))
    success = link.success_probability * (1.0 - pc)
    return (0.5 * T_PROC_MS
            + transmission_time_ms(link)
            + mu * math.log(1.0 / max(success, 1e-12)))


def shortest_path_at(net: Network, source: int, destination: int, mu: float,
                     active_ues: int) -> list[int]:
    """The optimal path for a given multiplier, which is Dijkstra on c(i, mu)."""
    return nx.shortest_path(
        net.graph, source, destination,
        weight=lambda u, v, _data: link_weight(net, u, v, mu, active_ues))


def dijkstra_optimal(net: Network, source: int, destination: int,
                     sigma: float = RELIABILITY_TARGET,
                     active_ues: int = 40) -> Route:
    """Solve Problem 1 by bisecting mu, Algorithm 1.

    A small mu buys latency at the cost of reliability and a large one the reverse, so
    the search keeps the smallest mu whose path still clears sigma. Only feasible paths
    are kept: bisection passes through infeasible values on its way, and returning
    whichever came last hands back a route the constraint rejects.

    When no path can reach sigma -- which happens when every route to a UE crosses a
    blocked link -- the most reliable path available is returned, with mu at its maximum
    to record that the constraint could not be met.
    """
    low, high = 0.0, MU_MAX
    best_path: list[int] | None = None
    best_mu = MU_MAX

    for _ in range(BISECTION_ITERATIONS):
        mu = (low + high) / 2.0
        path = shortest_path_at(net, source, destination, mu, active_ues)
        if net.path_reliability(path, active_ues) >= sigma:
            best_path, best_mu = path, mu
            high = mu          # feasible: try to buy back some latency
        else:
            low = mu           # infeasible: weight reliability more heavily
        if high - low < BISECTION_TOLERANCE:
            break

    if best_path is None:
        best_path = shortest_path_at(net, source, destination, MU_MAX, active_ues)
        best_mu = MU_MAX

    return Route(source=source, destination=destination, algorithm="dijkstra",
                 path=best_path,
                 delay_ms=net.path_delay_ms(best_path, active_ues),
                 reliability=net.path_reliability(best_path, active_ues),
                 mu=best_mu)


def greedy_route(net: Network, source: int, destination: int,
                 sigma: float = RELIABILITY_TARGET,
                 active_ues: int = 40, max_hops: int = 12) -> Route:
    """The semi-persistent scheduling baseline of Section V.

    Each node forwards to the neighbour with the best channel and keeps that choice while
    the URLLC requirement holds, only reconsidering when it stops holding. It sees one
    hop ahead: the link it picks is the best link, not necessarily a step on the best
    route.

    Progress toward the donor is required, since a purely channel-driven choice would
    otherwise walk between two good links forever.
    """
    hops_to_destination = nx.single_source_shortest_path_length(net.graph, destination)
    path = [source]
    node = source
    visited = {source}

    while node != destination and len(path) <= max_hops:
        candidates = [n for n in net.graph.neighbors(node)
                      if n not in visited
                      and hops_to_destination.get(n, math.inf)
                      < hops_to_destination.get(node, math.inf)]
        if not candidates:
            break

        # the best channel among the neighbours that make progress
        node = max(candidates, key=lambda n: net.link(path[-1], n).success_probability)
        visited.add(node)
        path.append(node)

    reached = path[-1] == destination
    return Route(source=source, destination=destination, algorithm="greedy",
                 path=path,
                 delay_ms=net.path_delay_ms(path, active_ues) if reached else float("nan"),
                 reliability=net.path_reliability(path, active_ues) if reached else 0.0)


def route_all(net: Network, ue_ids: list[int], donor: int = 0,
              sigma: float = RELIABILITY_TARGET,
              active_ues: int = 40) -> list[Route]:
    """Both baselines over every UE, on the same graph and the same load."""
    routes: list[Route] = []
    for ue in ue_ids:
        if not nx.has_path(net.graph, donor, ue):
            continue
        routes.append(dijkstra_optimal(net, donor, ue, sigma, active_ues))
        routes.append(greedy_route(net, donor, ue, sigma, active_ues))
    return routes


def path_is_valid(net: Network, path: list[int]) -> bool:
    """Every consecutive pair is a link that exists."""
    return all((u, v) in net.links for u, v in pairwise(path))
