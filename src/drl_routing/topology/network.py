"""Network topology: a graph of nodes and capacitated links.

Routing decisions are made on this graph. Each link carries a propagation delay and a
capacity; utilisation above capacity is what turns a shortest path into a congested one,
which is the whole reason a learned policy might beat a static one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import networkx as nx


@dataclass
class Link:
    """One directed link: how long it takes to traverse and how much it can carry."""

    source: int
    target: int
    delay_ms: float
    capacity_mbps: float
    load_mbps: float = 0.0

    @property
    def utilisation(self) -> float:
        return self.load_mbps / self.capacity_mbps if self.capacity_mbps else 1.0

    @property
    def effective_delay_ms(self) -> float:
        """Delay under load.

        Queueing delay grows sharply as a link approaches capacity, so the cost of a link
        is not its propagation delay alone. This uses the standard M/M/1 form, capped
        because a saturated link is unusable rather than infinitely slow.
        """
        u = min(self.utilisation, 0.99)
        return self.delay_ms / (1.0 - u)


@dataclass
class Network:
    """A topology plus the current load on every link."""

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    links: dict[tuple[int, int], Link] = field(default_factory=dict)

    def add_link(self, source: int, target: int, delay_ms: float,
                 capacity_mbps: float, bidirectional: bool = True) -> None:
        for s, t in ([(source, target), (target, source)] if bidirectional
                     else [(source, target)]):
            self.links[(s, t)] = Link(s, t, delay_ms, capacity_mbps)
            self.graph.add_edge(s, t, delay_ms=delay_ms, capacity_mbps=capacity_mbps)

    @property
    def nodes(self) -> list[int]:
        return sorted(self.graph.nodes)

    def neighbours(self, node: int) -> list[int]:
        return sorted(self.graph.successors(node))

    def reset_load(self) -> None:
        for link in self.links.values():
            link.load_mbps = 0.0

    def apply_load(self, path: list[int], demand_mbps: float) -> None:
        """Add a flow's demand to every link along its path."""
        for s, t in pairwise(path):
            self.links[(s, t)].load_mbps += demand_mbps

    def path_delay_ms(self, path: list[int]) -> float:
        """Total delay along a path under the current load."""
        return sum(self.links[(s, t)].effective_delay_ms for s, t in pairwise(path))

    def path_is_valid(self, path: list[int]) -> bool:
        return all((s, t) in self.links for s, t in pairwise(path))

    def max_utilisation(self) -> float:
        return max((link.utilisation for link in self.links.values()), default=0.0)


def build_topology(name: str = "abilene") -> Network:
    """Build one of the reference topologies.

    abilene: the 11-node US research backbone, a standard routing benchmark.
    grid:    a 4x4 mesh, where many equal-cost paths exist and load balancing matters.
    """
    net = Network()
    if name == "abilene":
        edges = [
            (0, 1, 6.0), (1, 2, 8.0), (2, 3, 5.0), (3, 4, 7.0), (4, 5, 6.0),
            (5, 6, 9.0), (6, 7, 4.0), (7, 8, 6.0), (8, 9, 5.0), (9, 10, 7.0),
            (0, 4, 12.0), (1, 6, 11.0), (2, 8, 13.0), (3, 9, 10.0), (5, 10, 8.0),
        ]
        for s, t, d in edges:
            net.add_link(s, t, delay_ms=d, capacity_mbps=1000.0)
    elif name == "grid":
        side = 4
        for r in range(side):
            for col in range(side):
                node = r * side + col
                if col + 1 < side:
                    net.add_link(node, node + 1, delay_ms=5.0, capacity_mbps=1000.0)
                if r + 1 < side:
                    net.add_link(node, node + side, delay_ms=5.0, capacity_mbps=1000.0)
    else:
        raise ValueError(f"unknown topology: {name}")
    return net
