"""Latency and reliability of a multi-hop path, Eq. (1) to (3) of the paper.

Phase 2. The topology and its per-link block error rates come from the ns-3 run; this
turns a path through them into the two numbers the routing problem is stated in: how long
the packet takes and how likely it is to arrive.

Nothing here decides a route. Eq. (4) onwards is Phase 3.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import networkx as nx

# --- Table I and Section III-A ----------------------------------------------------

NUMEROLOGY = 3
TTI_MS = 1.0 / (2 ** NUMEROLOGY)      # slot length at numerology u is 1/2^u ms
T_PROC_MS = 4 * TTI_MS                # Tproc = 4 x TTI, Section III-A

BANDWIDTH_HZ = 100e6
FILE_BYTES = 0.1e6                    # FTP model 3: the file the UE transfers
# The latency the paper reports is per packet, not per file. A 0.1 Mbyte file over a
# 100 MHz carrier takes twelve slots on a good link and two hundred on a marginal one; a
# three-hop route would spend more than the whole 5 ms budget on transmission alone. The
# file is segmented at the MAC layer and the routing decision is made for a packet, which
# is what Eq. (1) sizes with ceil(pkt/TB).
PACKET_BYTES = 8000.0                 # one MAC packet
ARRIVAL_RATE_HZ = 100.0 / 3.0         # Poisson, mean 100/3 per second per UE
DL_UL_RATIO = 4.0                     # Section V

LATENCY_BUDGET_MS = 5.0               # tau: the VR/AR target
RELIABILITY_TARGET = 0.999            # sigma

# Subbands available for the TDMA scheduler to assign. Two nodes close enough to
# interfere and unlucky enough to pick the same one in the same slot collide.
SUBBAND_COUNT = 12

# The lowest modulation and coding scheme in TS 38.214 (QPSK, rate 0.12) needs about
# -6 dB and carries roughly 0.23 bit per hertz. Below that threshold a link is not
# slow, it is absent.
MIN_USABLE_SINR_DB = -6.0
MIN_SPECTRAL_EFFICIENCY = 0.23


@dataclass(frozen=True)
class Link:
    """One measured link from the ns-3 run."""

    src: int
    dst: int
    kind: str
    distance_m: float
    los: bool
    sinr_db: float
    bler: float

    @property
    def success_probability(self) -> float:
        """ps(i) = 1 - pb(i), the probability a transmission gets through."""
        return max(1.0 - self.bler, 1e-9)

    @property
    def usable(self) -> bool:
        """Whether link adaptation can carry anything over this link at all.

        The lowest MCS in TS 38.214 needs about -6 dB. Below that the scheduler has
        nothing to fall back on, and treating the link as merely slow rather than absent
        produces transmission times of tens of milliseconds -- a route through a link the
        BLER model has already written off as unusable.
        """
        return self.sinr_db >= MIN_USABLE_SINR_DB

    @property
    def spectral_efficiency(self) -> float:
        """Shannon capacity per hertz at this link's SINR.

        Capped at the rate the highest MCS delivers, since a 33 dB link cannot be
        credited with throughput no modulation scheme offers, and floored at the lowest
        MCS so a usable link always carries something.
        """
        sinr_linear = 10 ** (self.sinr_db / 10.0)
        return min(max(math.log2(1.0 + sinr_linear), MIN_SPECTRAL_EFFICIENCY), 7.4)


def transport_block_bytes(link: Link) -> float:
    """Bytes carried in one slot on this link.

    TS 38.214 sizes the transport block from the resource elements in a slot and the MCS
    the SINR supports. Approximated here as bandwidth x spectral efficiency x slot
    length, which is the same quantity without the quantisation the standard applies.
    """
    bits_per_slot = BANDWIDTH_HZ * link.spectral_efficiency * (TTI_MS / 1000.0)
    return max(bits_per_slot / 8.0, 1.0)


def transmission_time_ms(link: Link, packet_bytes: float = PACKET_BYTES) -> float:
    """Ttrans = ceil(pkt / TB) x TTI, Eq. (1)."""
    return math.ceil(packet_bytes / transport_block_bytes(link)) * TTI_MS


def collision_probability(active_ues: int, neighbours: int,
                          scheduled: bool = True) -> float:
    """pc: two nearby nodes transmitting on the same subband in the same slot.

    Section III-C is explicit about the routing problem: the donor has the whole picture,
    the edge nodes are far enough apart not to interfere, and pc is zero. That is what
    makes Problem 1 solvable exactly -- with random collisions on every hop the optimum
    would not be a shortest path at all.

    pc matters where the paper says it does: under the DRL framework of Section IV, where
    each node schedules from local information and nothing coordinates the subband
    choices. Pass scheduled=False for that case.

    The unscheduled form: each neighbour transmitting in a slot picks a subband at
    random, and a collision is at least one of them landing on the same one.
    """
    if scheduled or neighbours <= 0 or active_ues <= 0:
        return 0.0
    load_per_node = min(active_ues * ARRIVAL_RATE_HZ / 1000.0 * TTI_MS, 1.0)
    expected_transmitters = neighbours * load_per_node
    return 1.0 - (1.0 - 1.0 / SUBBAND_COUNT) ** expected_transmitters


def queueing_delay_ms(active_ues: int, service_time_ms: float,
                      node_degree: int = 6) -> float:
    """Tque: how long a packet waits for its slot.

    Fig. 5(a) rises with the UE count because more UEs contend for the same slots. The
    contention is local, though: a UE queues behind the traffic its own IAB node carries,
    not behind every packet in the deployment. Charging the whole load to one server puts
    utilisation above one at forty UEs and the wait alone past the 5 ms budget.

    An M/D/1 waiting time on the share of the load this node sees.
    """
    if active_ues <= 0:
        return 0.0
    # a node serves roughly its share of the UEs, so the load divides across the mesh
    local_ues = max(active_ues / max(node_degree, 1), 1.0)
    arrivals_per_ms = local_ues * ARRIVAL_RATE_HZ / 1000.0
    utilisation = min(arrivals_per_ms * service_time_ms, 0.95)
    return (utilisation * service_time_ms) / (2 * (1 - utilisation))


class Network:
    """The measured topology, and the cost of paths through it."""

    def __init__(self, links: list[Link], donor: int = 0, scheduled: bool = True):
        self.donor = donor
        # scheduled: the donor coordinates subbands, so pc = 0 (Section III-C). The
        # learned policy of Section IV decides locally and does not get that guarantee.
        self.scheduled = scheduled
        self.links = {(link.src, link.dst): link for link in links}
        self.links.update({(link.dst, link.src): link for link in links})
        self.graph = nx.Graph()
        for link in links:
            if link.usable:
                self.graph.add_edge(link.src, link.dst,
                                    bler=link.bler, sinr_db=link.sinr_db)

    @classmethod
    def from_csv(cls, path: str | Path, donor: int = 0,
                 scheduled: bool = True) -> Network:
        links = []
        with open(path) as f:
            rows = [row for row in csv.reader(f) if not row[0].startswith("#")]
        header = rows[0]
        for row in rows[1:]:
            record = dict(zip(header, row))
            links.append(Link(
                src=int(record["src"]),
                dst=int(record["dst"]),
                kind=record["kind"],
                distance_m=float(record["distance_m"]),
                los=record["los"] == "1",
                sinr_db=float(record["sinr_db"]),
                bler=float(record["bler"]),
            ))
        return cls(links, donor, scheduled)

    def link(self, u: int, v: int) -> Link:
        return self.links[(u, v)]

    def path_delay_ms(self, path: list[int], active_ues: int = 40) -> float:
        """Eq. (2): Tdelay = Tque + (n+2)/2 x Tproc + sum of transmission times.

        n is the number of relays, so a path of h hops has h-1 of them. The processing
        term is the source and destination stacks plus half of it at each relay, which is
        what the CU/DU split buys.
        """
        hops = len(path) - 1
        if hops <= 0:
            return 0.0
        relays = hops - 1
        transmission = sum(transmission_time_ms(self.link(u, v)) for u, v in pairwise(path))
        # the queue forms at the first hop, where the packet waits for its slot
        service = transmission_time_ms(self.link(path[0], path[1]))
        return (queueing_delay_ms(active_ues, service, self.graph.degree(path[0]))
                + (relays + 2) / 2 * T_PROC_MS
                + transmission)

    def path_reliability(self, path: list[int], active_ues: int = 40) -> float:
        """Eq. (3): P = product over hops of ps(i) x (1 - pc(i))."""
        product = 1.0
        for u, v in pairwise(path):
            link = self.link(u, v)
            pc = collision_probability(active_ues, self.graph.degree(u),
                                       scheduled=self.scheduled)
            product *= link.success_probability * (1.0 - pc)
        return product

    def meets_targets(self, path: list[int], active_ues: int = 40) -> tuple[bool, bool]:
        """Whether a path is inside the latency budget and above the reliability floor."""
        return (self.path_delay_ms(path, active_ues) <= LATENCY_BUDGET_MS,
                self.path_reliability(path, active_ues) >= RELIABILITY_TARGET)
