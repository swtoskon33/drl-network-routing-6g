"""Link scheduling environment for a multi-hop mmWave mesh.

The second problem in this repository. Routing decides where a packet goes; scheduling
decides which links transmit in a given slot, and at what power. They interact through
interference: two links active at once degrade each other, so activating everything is
worse than activating a well-chosen subset.

Follows Gahtan, Cohen, Bronstein and Kedar, "Using Deep Reinforcement Learning for
mmWave Real-Time Scheduling", NoF 2023. Packets are routed on shortest paths and move
one hop per slot; a packet forwarded into a full buffer is dropped, and dropped packets
are what the agent is trained to avoid.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx
import numpy as np

BUFFER_CAPACITY = 650            # packets, per the paper's evaluation setup
LINK_CAPACITY_RANGE = (115, 125)  # packets per slot, sampled per link
SLOT_MS = 10.0                    # the near-RT control loop budget


@dataclass
class SchedulingConfig:
    """Everything the environment needs, with the paper's values as defaults."""

    buffer_capacity: int = BUFFER_CAPACITY
    capacity_range: tuple[int, int] = LINK_CAPACITY_RANGE
    interference_level: float = 0.6   # 0 to 1; at 1 only one link can transmit
    initial_packets: int = 2304       # the 10-link uniform workload
    max_slots: int = 200
    seed: int = 42


@dataclass
class Packet:
    """One packet, tracked by the hop it still has to make."""

    source: int
    destination: int
    path: list[int]
    position: int = 0

    @property
    def current(self) -> int:
        return self.path[self.position]

    @property
    def next_hop(self) -> int | None:
        return self.path[self.position + 1] if self.position + 1 < len(self.path) else None

    @property
    def delivered(self) -> bool:
        return self.position == len(self.path) - 1


class MmWaveMesh:
    """The topology, its link capacities, and the interference between links."""

    def __init__(self, graph: nx.DiGraph, config: SchedulingConfig):
        self.graph = graph
        self.config = config
        rng = random.Random(config.seed)
        self.links = sorted(graph.edges())
        self.capacity = {
            link: rng.randint(*config.capacity_range) for link in self.links
        }
        self.interference = self._interference_matrix(rng)

    def _interference_matrix(self, rng: random.Random) -> np.ndarray:
        """How much each active link degrades every other.

        Links that share a node interfere most, since the antennas point at overlapping
        space; links further apart interfere less. The interference level scales the
        whole matrix -- at 1.0 a single active neighbour saturates a link, which is the
        paper's definition of 100% interference.
        """
        n = len(self.links)
        matrix = np.zeros((n, n), dtype=np.float32)
        for i, (a, b) in enumerate(self.links):
            for j, (c, d) in enumerate(self.links):
                if i == j:
                    continue
                shared = len({a, b} & {c, d})
                base = 1.0 if shared == 2 else (0.5 if shared == 1 else 0.1)
                matrix[i, j] = base * rng.uniform(0.8, 1.2)

        # Scale so that the interference level means what the paper says it means: at
        # 100%, one active neighbour saturates a link and only one can transmit; below
        # that, a well-chosen subset can run together. Normalising by the row sum keeps
        # that true whether the topology has 10 links or 96, which an unnormalised
        # matrix does not -- on the large mesh every link would sit at zero capacity.
        row_sums = matrix.sum(axis=0, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return matrix / row_sums * self.config.interference_level

    def effective_capacity(self, powers: np.ndarray) -> np.ndarray:
        """Capacity of every link given the power assigned to each.

        Eq. (1) of the paper: the received power of a link is reduced by the sum of the
        interference from every other active link, and capacity follows from Shannon.
        """
        nominal = np.array([self.capacity[link] for link in self.links], dtype=np.float32)
        received = powers.copy()
        degradation = self.interference.T @ powers
        effective = np.clip(received - degradation, 0.0, None)
        return nominal * effective


def build_mesh(size: str = "small", config: SchedulingConfig | None = None) -> MmWaveMesh:
    """One of the three topology sizes the paper evaluates.

    small:  4 nodes, 10 links
    medium: 19 nodes, 48 links
    large:  37 nodes, 96 links

    The paper's topologies come from Ceragon and are not published, so these are
    generated at the same scale rather than copied.
    """
    config = config or SchedulingConfig()
    rng = random.Random(config.seed)
    spec = {"small": (4, 10), "medium": (19, 48), "large": (37, 96)}[size]
    nodes, target_links = spec

    g = nx.DiGraph()
    g.add_nodes_from(range(nodes))
    # a connected backbone first, so every node can reach every other
    for n in range(1, nodes):
        parent = rng.randrange(n)
        g.add_edge(parent, n)
        g.add_edge(n, parent)
    # then extra links until the topology has the size we want
    while g.number_of_edges() < target_links:
        a, b = rng.sample(range(nodes), 2)
        if not g.has_edge(a, b):
            g.add_edge(a, b)
    return MmWaveMesh(g, config)
