"""The scheduling episode: packets move, buffers fill, and drops are what count.

An episode starts with a traffic matrix and runs until every packet has been delivered or
dropped. Each slot the agent assigns a power to every link; capacity follows from the
interference, packets move one hop, and a packet forwarded into a full buffer is lost.

Reward, from Section IV of the paper:

    R_t = -beta - alpha * D/P + M/P

D is the packets dropped this slot, M the packets that moved, P the packets in the system
before the slot. beta charges for every extra slot the delivery takes. alpha weights drops
against movement: the paper trains a drop-insensitive agent at alpha = 1 and a
drop-sensitive one at alpha = 10.
"""
from __future__ import annotations

import random
from collections import defaultdict

import networkx as nx
import numpy as np

from drl_routing.scheduling.mesh import MmWaveMesh, Packet

ALPHA_DROP_INSENSITIVE = 1.0
ALPHA_DROP_SENSITIVE = 10.0
BETA = 1.0


class SchedulingEnv:
    """One episode of link scheduling over a mmWave mesh."""

    def __init__(self, mesh: MmWaveMesh, alpha: float = ALPHA_DROP_SENSITIVE,
                 beta: float = BETA, workload: str = "uniform"):
        self.mesh = mesh
        self.config = mesh.config
        self.alpha = alpha
        self.beta = beta
        self.workload = workload
        self.rng = random.Random(self.config.seed)
        self.paths = dict(nx.all_pairs_shortest_path(mesh.graph))
        self.n_links = len(mesh.links)
        self.link_index = {link: i for i, link in enumerate(mesh.links)}
        # state: buffer load, buffer share of all traffic, and the interference matrix
        self.state_dim = self.n_links * 2 + self.n_links * self.n_links
        self.action_dim = self.n_links
        self.reset()

    def _source_destination(self) -> tuple[int, int]:
        """Pick a flow according to the workload.

        uniform:      every node sends and receives equally
        few_to_many:  10% of nodes send to 90%
        many_to_few:  90% send to 10%, the incast case
        """
        nodes = list(self.mesh.graph.nodes)
        few = max(1, len(nodes) // 10)
        if self.workload == "few_to_many":
            src = self.rng.choice(nodes[:few])
            dst = self.rng.choice(nodes[few:]) if len(nodes) > few else src
        elif self.workload == "many_to_few":
            src = self.rng.choice(nodes[few:]) if len(nodes) > few else nodes[0]
            dst = self.rng.choice(nodes[:few])
        else:
            src, dst = self.rng.sample(nodes, 2)
        return src, dst

    def reset(self) -> np.ndarray:
        self.buffers: dict[tuple[int, int], list[Packet]] = defaultdict(list)
        self.pending: list[Packet] = []
        self.slot = 0
        self.delivered = 0
        self.dropped = 0

        for _ in range(self.config.initial_packets):
            src, dst = self._source_destination()
            path = self.paths.get(src, {}).get(dst)
            if not path or len(path) < 2:
                continue
            packet = Packet(source=src, destination=dst, path=path)
            link = (path[0], path[1])
            if len(self.buffers[link]) < self.config.buffer_capacity:
                self.buffers[link].append(packet)
            else:
                # the source buffer is full; the packet waits outside the system rather
                # than counting as a drop, since the agent had no chance to prevent it
                self.pending.append(packet)
        return self._state()

    @property
    def packets_in_system(self) -> int:
        return sum(len(q) for q in self.buffers.values())

    def _state(self) -> np.ndarray:
        """Buffer occupancy, each buffer's share of all traffic, and the interference."""
        total = max(self.packets_in_system, 1)
        load = np.zeros(self.n_links, dtype=np.float32)
        share = np.zeros(self.n_links, dtype=np.float32)
        for link, queue in self.buffers.items():
            if link in self.link_index:
                i = self.link_index[link]
                load[i] = len(queue) / self.config.buffer_capacity
                share[i] = len(queue) / total
        return np.concatenate([load, share, self.mesh.interference.flatten()])

    def step(self, powers: np.ndarray):
        """Activate links at the given powers, move one hop, and score the slot."""
        before = self.packets_in_system
        capacities = self.mesh.effective_capacity(np.clip(powers, 0.0, 1.0))

        moved = 0
        dropped = 0
        # collect moves first, apply after, so a packet cannot cross two hops in a slot
        arrivals: list[tuple[tuple[int, int], Packet]] = []

        for i, link in enumerate(self.mesh.links):
            queue = self.buffers.get(link)
            if not queue:
                continue
            allowed = int(capacities[i])
            for packet in queue[:allowed]:
                packet.position += 1
                moved += 1
                if packet.delivered:
                    self.delivered += 1
                else:
                    arrivals.append(((packet.current, packet.next_hop), packet))
            del queue[:allowed]

        for link, packet in arrivals:
            if len(self.buffers[link]) < self.config.buffer_capacity:
                self.buffers[link].append(packet)
            else:
                dropped += 1

        # admit packets that were waiting for room at their source
        still_waiting = []
        for packet in self.pending:
            link = (packet.path[0], packet.path[1])
            if len(self.buffers[link]) < self.config.buffer_capacity:
                self.buffers[link].append(packet)
            else:
                still_waiting.append(packet)
        self.pending = still_waiting

        self.dropped += dropped
        self.slot += 1

        p = max(before, 1)
        reward = -self.beta - self.alpha * (dropped / p) + (moved / p)
        done = ((self.packets_in_system == 0 and not self.pending)
                or self.slot >= self.config.max_slots)
        info = {"moved": moved, "dropped": dropped, "remaining": self.packets_in_system}
        return self._state(), reward, done, info

    def summary(self) -> dict:
        total = self.delivered + self.dropped
        return {
            "slots": self.slot,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "goodput": self.delivered / total if total else 0.0,
        }
