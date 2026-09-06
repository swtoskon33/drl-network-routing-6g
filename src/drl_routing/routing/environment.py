"""The routing decision as an MDP, Section IV of the paper.

Phase 4. A packet sits at a node and picks its next hop from what that node can see: its
neighbours, their channels, and what the route through each of them costs. It has no view
of the graph -- that belongs to the donor, and the exact solver in baselines.py is what
having it buys.

State is Eq. (8), reward is Eq. (11). The reward is paid once per transmission rather
than once per hop: Eq. (11) scores a delivery, and charging it every step makes each
extra hop profitable whenever the path is inside the latency budget.

Collisions are live here. Section III-C sets pc to zero for the routing problem because
the donor schedules the subbands; under the DRL framework each node decides locally and
gets no such guarantee.
"""
from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np

from drl_routing.routing.baselines import dijkstra_optimal
from drl_routing.routing.cost import (
    LATENCY_BUDGET_MS,
    T_PROC_MS,
    Network,
    collision_probability,
    queueing_delay_ms,
    transmission_time_ms,
)

MAX_DEGREE = 16           # padded neighbour slots; the densest node in the measured
                          # topology has fifteen links, and truncating the list drops
                          # neighbours a route may need
MAX_STEPS = 12            # abandon the packet after this many hops
FEATURES_PER_NEIGHBOUR = 5

PSI_D = 1.0               # weight on the latency term of Eq. (11)
PSI_R = 0.5               # weight on retransmissions
HOP_COST = 0.1            # so a shorter route is preferred among equals
ARRIVAL_BONUS = 10.0      # scaled by how much of the packet survived
TIMEOUT_PENALTY = 10.0


class RoutingEnvironment:
    """One packet walking from a UE to the donor, one hop per step."""

    def __init__(self, net: Network, donor: int = 0, active_ues: int = 40,
                 sigma: float = 0.999):
        self.net = net
        self.donor = donor
        self.active_ues = active_ues
        self.neighbours = {n: sorted(net.graph.neighbors(n))[:MAX_DEGREE]
                           for n in net.graph.nodes}
        self.configured = self._configured_routes(sigma)
        self.route_cost = self._route_costs()
        self.state_dim = MAX_DEGREE * FEATURES_PER_NEIGHBOUR + 2
        self.action_dim = MAX_DEGREE
        self.reset(next(iter(net.graph.nodes)))

    # -- what the donor configures, Section IV --

    def _configured_routes(self, sigma: float) -> dict[int, int]:
        """The next hop each node takes on the donor's solution to Problem 1.

        Section IV: the IAB nodes are fixed, so the donor solves the routing problem
        centrally and configures a routing table at each node. The agent starts from that
        table and re-selects neighbours as it learns.
        """
        routes: dict[int, int] = {}
        for node in self.net.graph.nodes:
            if node == self.donor:
                continue
            if not nx.has_path(self.net.graph, self.donor, node):
                continue
            route = dijkstra_optimal(self.net, self.donor, node, sigma, self.active_ues)
            # the path runs donor -> node, so the hop toward the donor is the one before
            if len(route.path) >= 2:
                routes[node] = route.path[-2]
        return routes

    def _route_costs(self) -> dict[int, tuple[float, float]]:
        """T_n,delay and P_n for every node: what reaching the donor through it costs.

        Eq. (8) puts these in the state for each neighbour. Nodes are ordered by the
        length of their configured route rather than by hop count, since a route that
        avoids blocked links may be longer than the shortest one and its next hop would
        otherwise be uncomputed when its turn came.
        """
        def route_length(node: int) -> int:
            steps, seen = 0, set()
            while node != self.donor and node not in seen:
                seen.add(node)
                node = self.configured.get(node)
                if node is None:
                    return len(self.net.graph) + 1
                steps += 1
            return steps

        cost: dict[int, tuple[float, float]] = {self.donor: (0.0, 1.0)}
        for node in sorted(self.net.graph.nodes, key=route_length):
            if node == self.donor:
                continue
            nxt = self.configured.get(node)
            if nxt is None or nxt not in cost:
                cost[node] = (50.0, 0.0)
                continue
            delay_next, rel_next = cost[nxt]
            link = self.net.link(node, nxt)
            pc = collision_probability(self.active_ues, self.net.graph.degree(node),
                                       scheduled=False)
            cost[node] = (delay_next + transmission_time_ms(link),
                          rel_next * link.success_probability * (1.0 - pc))
        return cost

    def default_action(self, node: int) -> int | None:
        """The slot the configured route would take from this node."""
        nxt = self.configured.get(node)
        nbrs = self.neighbours[node]
        return nbrs.index(nxt) if nxt in nbrs else None

    # -- episode --

    def reset(self, source: int) -> np.ndarray:
        self.node = source
        self.steps = 0
        self.cum_transmission_ms = 0.0
        self.cum_log_reliability = 0.0
        self.retransmissions = 0
        self.visited = {source}
        self.source_degree = self.net.graph.degree(source)
        self.first_service_ms = 0.0
        return self.state()

    def action_mask(self) -> np.ndarray:
        """Slots pointing at a neighbour this packet has not already been to.

        Padding slots have to be masked or the agent learns that standing still costs
        less than moving. Revisits have to be masked too: a cycle returns to a state the
        critic has already valued, and it treats the loop as a source of value that never
        has to be paid for.
        """
        nbrs = self.neighbours[self.node]
        mask = np.zeros(MAX_DEGREE, dtype=bool)
        unvisited = [i for i, nb in enumerate(nbrs) if nb not in self.visited]
        for i in (unvisited if unvisited else range(len(nbrs))):
            mask[i] = True
        return mask

    def total_delay_ms(self) -> float:
        """Eq. (2) for the path walked so far.

        The queue forms at the source, where the packet waits for its first slot, so the
        service time that sets the wait is the first link's -- recorded when it is taken
        rather than reconstructed afterwards from where the packet has got to.
        """
        hops = self.steps
        if hops == 0:
            return 0.0
        relays = hops - 1
        queue = queueing_delay_ms(self.active_ues, self.first_service_ms,
                                  self.source_degree)
        return queue + (relays + 2) / 2 * T_PROC_MS + self.cum_transmission_ms

    def state(self) -> np.ndarray:
        """Eq. (8): per neighbour, its channel and the cost of the route through it.

        The route figures include the link that reaches the neighbour. Reporting the
        neighbour's own downstream cost alone makes the donor look perfect from anywhere
        adjacent to it -- its downstream reliability is one because it is the destination
        -- while the state of the link leading there sits in a separate feature the
        policy is free to ignore.
        """
        feats: list[float] = []
        nbrs = self.neighbours[self.node]
        degree = self.net.graph.degree(self.node)
        pc = collision_probability(self.active_ues, degree, scheduled=False)

        for i in range(MAX_DEGREE):
            if i < len(nbrs):
                nb = nbrs[i]
                link = self.net.link(self.node, nb)
                route_delay, route_rel = self.route_cost.get(nb, (50.0, 0.0))
                reach_delay = route_delay + transmission_time_ms(link)
                reach_rel = route_rel * link.success_probability * (1.0 - pc)
                feats += [1.0,
                          transmission_time_ms(link) / 5.0,
                          link.bler,
                          reach_delay / 20.0,
                          reach_rel]
            else:
                feats += [0.0, 1.0, 1.0, 1.0, 0.0]

        feats += [self.cum_transmission_ms / 20.0, math.exp(self.cum_log_reliability)]
        return np.array(feats, dtype=np.float32)

    def step(self, action: int):
        nbrs = self.neighbours[self.node]
        self.steps += 1

        if action >= len(nbrs):
            return self.state(), -1.0, self.steps >= MAX_STEPS, {"invalid": True}

        nxt = nbrs[action]
        link = self.net.link(self.node, nxt)
        pc = collision_probability(self.active_ues, self.net.graph.degree(self.node),
                                   scheduled=False)
        success = link.success_probability * (1.0 - pc)

        if self.steps == 1:
            self.first_service_ms = transmission_time_ms(link)
        self.cum_transmission_ms += transmission_time_ms(link)
        self.cum_log_reliability += math.log(max(success, 1e-12))
        self.visited.add(self.node)
        self.node = nxt

        # the environment answers with an ACK or a NACK, which is what Eq. (11) reads
        if random.random() > success:
            self.retransmissions += 1

        reward = -HOP_COST
        done = self.node == self.donor or self.steps >= MAX_STEPS

        if done:
            if self.node == self.donor:
                delay = self.total_delay_ms()
                reliability = math.exp(self.cum_log_reliability)
                # Eq. (11): the latency term against the budget, weighted by whether the
                # packet survived, less the retransmissions it needed
                latency_term = (LATENCY_BUDGET_MS - delay) / max(delay, 1e-6) + 1.0
                reward += PSI_D * latency_term * reliability
                reward += ARRIVAL_BONUS * reliability
                reward -= PSI_R * self.retransmissions
            else:
                reward -= TIMEOUT_PENALTY

        info = {"reached": self.node == self.donor,
                "reliability": math.exp(self.cum_log_reliability),
                "delay_ms": self.total_delay_ms()}
        return self.state(), reward, done, info

    def walk(self, path: list[int]) -> tuple[float, dict]:
        """Play a chosen path through the environment and return what it earns.

        For checking by hand that a good route pays better than a bad one, before any
        network is trained on it.
        """
        self.reset(path[0])
        total = 0.0
        info: dict = {}
        for nxt in path[1:]:
            slot = self.neighbours[self.node].index(nxt)
            _, reward, done, info = self.step(slot)
            total += reward
            if done:
                break
        return total, info
