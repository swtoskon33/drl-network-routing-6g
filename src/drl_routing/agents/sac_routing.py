"""Discrete Soft Actor-Critic for IAB routing (Christodoulou 2019 variant),
matching the RL setup of Yin, Roy, Cao (TCOMM 2022): each IAB node/UE picks
its next hop using only local neighbour information (delay, BLER, hop-count
to donor), trained to jointly minimise latency and maximise reliability
(the paper's SAC vs A2C/greedy comparison, simplified to single-agent SAC
choosing next hop toward the donor).

Reads topology.csv (same file the Dijkstra baseline uses), trains a policy,
then evaluates every UE's learned route against dijkstra_vs_greedy.csv.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

MAX_DEGREE = 7          # padded neighbour slots; the densest IAB node has 7 links
MAX_STEPS = 10          # abandon the packet after this many hops

# 3GPP URLLC targets the paper works to (Section V): VR/AR traffic, a 5 ms latency
# budget and 0.999 success probability. Numerology 3 puts the TTI at 1/8 ms, and Eq. (2)
# charges half the processing time per relay on top of the direct-transmission cost.
TTI_MS = 0.125
T_PROC_MS = 4 * TTI_MS          # Tproc = 4 x TTI
LATENCY_BUDGET_MS = 5.0         # tau
RELIABILITY_TARGET = 0.999      # sigma

# Reward weights from Eq. (11): psi_d on the latency term, psi_r on retransmissions.
PSI_D = 1.0
PSI_R = 0.5

# Traffic model from Section V: FTP model 3, 0.1 Mbyte packets arriving as a Poisson
# process with mean 100/3 per second, DL/UL ratio 4:1. The transmission block size for
# the MCS the scheduler picks follows TS 38.214; Ttrans = ceil(pkt/TB) * TTI.
PACKET_BYTES = 0.1e6
TB_BYTES = 100000.0                     # transport block at 100 MHz, numerology 3,
                                        # 256QAM: about 100 kbyte per slot
UE_ARRIVAL_RATE = 100.0 / 3.0           # packets per second per UE
DEFAULT_UE_COUNT = 40                   # the fixed-topology case in Fig. 6 and 7


def transmission_time_ms(packet_bytes: float = PACKET_BYTES) -> float:
    """Ttrans = ceil(pkt / TB) * TTI, per Eq. (1)."""
    return math.ceil(packet_bytes / TB_BYTES) * TTI_MS


def queueing_delay_ms(active_ues: int, hops: int) -> float:
    """Queueing delay under TDMA, which is what makes latency grow with the UE count.

    Section V ties the rise in Fig. 5(a) to Tque: more UEs means more contention for the
    same slots. An M/D/1 waiting time on the aggregate arrival rate captures that
    behaviour -- the delay climbs slowly while the link is lightly loaded and sharply as
    utilisation approaches one.
    """
    service_ms = transmission_time_ms()
    arrivals_per_ms = active_ues * UE_ARRIVAL_RATE / 1000.0
    utilisation = min(arrivals_per_ms * service_ms, 0.95)
    return (utilisation * service_ms) / (2 * (1 - utilisation))


def load_topology(path: str) -> nx.Graph:
    g = nx.Graph()
    with open(path) as f:
        for row in csv.DictReader(f):
            g.add_edge(int(row["src"]), int(row["dst"]),
                       delay_ms=float(row["delay_ms"]), pb=float(row["pb"]))
    return g


class IabRoutingEnv:
    """Single-agent env: place a packet at a UE, choose next hops until the
    donor is reached (or MAX_STEPS runs out). Local state only."""

    def __init__(self, g: nx.Graph, donor: int = 0,
                 active_ues: int = DEFAULT_UE_COUNT):
        self.g = g
        self.donor = donor
        self.active_ues = active_ues
        self.hop_to_donor = nx.single_source_shortest_path_length(g, donor)
        self.neighbours = {n: sorted(g.neighbors(n))[:MAX_DEGREE] for n in g.nodes}
        self.default_next_hop = self._preconfigured_routes(self._bisect_mu())
        self.route_quality = self._route_quality()
        self.state_dim = MAX_DEGREE * 5 + 2   # per-neighbour [exists, delay, pb, downstream delay, downstream reliability]
        self.action_dim = MAX_DEGREE
        self.reset(random.choice(list(g.nodes)))

    def _bisect_mu(self, tolerance: float = 1e-3, iterations: int = 40) -> float:
        """Algorithm 1: bisect the Lagrange multiplier until the configured routes meet
        the reliability target.

        A fixed mu understates the cost of an unreliable link. At mu = 1 the penalty for
        a blocked link is log(1/0.5) = 0.69, cheaper than one extra hop, so the
        configured route runs straight through blocked links and the agent -- which
        learns its state and its warm start from that route -- inherits the preference.
        The Dijkstra baseline bisects to mu* around 3.8 on this topology.
        """
        low, high = 0.0, 100.0
        best = high
        for _ in range(iterations):
            mu = (low + high) / 2
            routes = self._preconfigured_routes(mu)
            worst = min((self._route_reliability(node, routes)
                         for node in self.g.nodes if node != self.donor), default=0.0)
            if worst >= RELIABILITY_TARGET:
                best = mu
                high = mu
            else:
                low = mu
            if high - low < tolerance:
                break
        return best

    def _route_reliability(self, node: int, routes: dict) -> float:
        """Success probability along a configured route, Eq. (3) with pc = 0."""
        product = 1.0
        seen = set()
        while node != self.donor:
            if node in seen or node not in routes:
                return 0.0
            seen.add(node)
            nxt = routes[node]
            product *= max(1.0 - self.g[node][nxt]["pb"], 1e-9)
            node = nxt
        return product

    def _preconfigured_routes(self, mu: float) -> dict:
        """The Dijkstra solution to Problem 1, used as the default routing setup.

        Section IV: the IAB nodes are fixed, so the donor solves the optimal routing
        problem centrally and configures each node's routing table. The agent starts from
        that table and re-selects neighbours as it learns. Link weight is c(i, mu) from
        Eq. (5): half the processing time, the transmission delay, and mu*log(1/ps).
        """
        weighted = nx.Graph()
        for u, v, data in self.g.edges(data=True):
            ps = max(1.0 - data["pb"], 1e-9)
            weighted.add_edge(u, v, weight=0.5 * T_PROC_MS + data["delay_ms"]
                              + mu * math.log(1 / ps))
        paths = nx.single_source_dijkstra_path(weighted, self.donor, weight="weight")
        return {node: path[-2] for node, path in paths.items() if len(path) >= 2}

    def _route_quality(self) -> dict:
        """Delay and success probability from each node to the donor on the configured
        route. This is the T_delay and P that Eq. (8) puts in the state for every
        neighbour."""
        quality = {self.donor: (0.0, 1.0)}

        def route_length(node: int) -> int:
            """How many hops the configured route takes, which is not the same as the
            shortest hop count: the route avoids blocked links and may go the long way
            round. Ordering by hop count leaves a node's next hop uncomputed when the
            route is longer, and its downstream reliability then falls back to zero --
            the agent sees a valid route reported as unusable."""
            steps, seen = 0, set()
            while node != self.donor and node not in seen and steps <= len(self.g):
                seen.add(node)
                node = self.default_next_hop.get(node)
                if node is None:
                    return len(self.g) + 1
                steps += 1
            return steps

        for node in sorted(self.g.nodes, key=route_length):
            if node == self.donor:
                continue
            nxt = self.default_next_hop.get(node)
            if nxt is None or nxt not in quality:
                quality[node] = (50.0, 0.0)
                continue
            d_next, rel_next = quality[nxt]
            link = self.g[node][nxt]
            quality[node] = (d_next + link["delay_ms"],
                             rel_next * max(1.0 - link["pb"], 1e-9))
        return quality

    def default_action(self, node: int):
        """Which action slot the pre-configured route takes from this node."""
        nxt = self.default_next_hop.get(node)
        nbrs = self.neighbours[node]
        return nbrs.index(nxt) if nxt in nbrs else None

    def reset(self, start_node: int):
        self.node = start_node
        self.previous = None
        self.cum_delay = 0.0
        self.cum_logrel = 0.0
        self.steps = 0
        return self._state()

    def action_mask(self) -> np.ndarray:
        """Which action slots correspond to a real neighbour.

        UEs have a single link and the densest IAB node has seven, so most slots are
        padding for most nodes. Penalising an invalid choice is not enough: the agent
        learns that standing still is cheaper than exploring, which is exactly what it
        did -- every UE looped on itself for ten steps. Masking removes the option.
        """
        nbrs = self.neighbours[self.node]
        mask = np.zeros(MAX_DEGREE, dtype=bool)
        mask[:len(nbrs)] = True
        # forbid stepping straight back where we came from: without this the policy
        # oscillates between two nodes, since the shaping it gains going one way it
        # loses coming back and the net gradient is zero
        if self.previous is not None and len(nbrs) > 1:
            for i, nb in enumerate(nbrs):
                if nb == self.previous:
                    mask[i] = False
        return mask

    def _state(self) -> np.ndarray:
        """State per Eq. (8): for each neighbour, the link quality *and* what that
        neighbour can offer downstream.

        The paper's state carries T_delay and P for every neighbouring node -- the
        latency and success probability of reaching the donor through it, not just the
        cost of the next link. Without that the agent cannot tell the donor apart from a
        dead-end UE: both are simply neighbours with a delay and a block error rate. That
        is exactly what happened here, and the policy stayed uniform.

        The donor computes these downstream figures on the configured route and hands
        them to each node, which is the same central configuration the routing tables
        already rely on.
        """
        feats = []
        nbrs = self.neighbours[self.node]
        for i in range(MAX_DEGREE):
            if i < len(nbrs):
                nb = nbrs[i]
                d = self.g[self.node][nb]["delay_ms"]
                pb = self.g[self.node][nb]["pb"]
                downstream_delay, downstream_rel = self.route_quality[nb]
                feats += [1.0, d / 15.0, pb,
                          downstream_delay / 50.0, downstream_rel]
            else:
                feats += [0.0, 1.0, 1.0, 1.0, 0.0]
        feats += [self.cum_delay / 50.0, self.cum_logrel / 5.0]
        return np.array(feats, dtype=np.float32)

    def total_delay_ms(self) -> float:
        """Eq. (2): Tdelay = Tque + (n+2)/2 * Tproc + sum of transmission times.

        With immediate scheduling Tque is zero and n is the number of relays.
        """
        relays = max(self.steps - 1, 0)
        hops = self.steps
        t_que = queueing_delay_ms(self.active_ues, hops)
        t_trans = hops * transmission_time_ms()
        return t_que + (relays + 2) / 2 * T_PROC_MS + t_trans

    def step(self, action: int):
        nbrs = self.neighbours[self.node]
        self.steps += 1
        if action >= len(nbrs):
            return self._state(), -2.0, self.steps >= MAX_STEPS, {}

        prev = self.node
        self.previous = prev
        nxt = nbrs[action]
        d = self.g[self.node][nxt]["delay_ms"]
        pb = self.g[self.node][nxt]["pb"]
        ps = max(1.0 - pb, 1e-9)
        self.cum_delay += d
        self.cum_logrel += math.log(ps)
        self.node = nxt

        done = self.node == self.donor or self.steps >= MAX_STEPS

        # Reward per Eq. (11). The outcome o is 0 on a successful transmission and 1 on a
        # NACK; here the link's block error rate decides it stochastically, so the agent
        # feels unreliable links rather than being told about them. K_trans counts the
        # transmissions the packet needed.
        outcome = 1 if random.random() < pb else 0
        transmissions = 2 if outcome else 1
        delay = self.total_delay_ms()
        latency_term = ((LATENCY_BUDGET_MS - delay) / (delay ** outcome if outcome else 1.0)
                        + (-1) ** outcome)
        reward = PSI_D * latency_term - PSI_R * (transmissions - 1)

        # Shaping toward the donor. The paper does not need this: its agents sit on
        # the configured route from the start, so progress is implicit. Ours has to
        # discover the donor, and without a gradient on hop count it wanders.
        #
        # The weight matters more than it looks. At a flat bonus per hop the agent learns
        # "get closer" and nothing else: for UE 25 it took 25-4-3-2-1-0, five hops through
        # three blocked links, over the configured 25-4-5-6-1-7-0, six hops all clean. A
        # blocked link has to cost more than the hop it saves, so the bonus is scaled by
        # the reliability of the link taken to earn it.
        before = self.hop_to_donor.get(prev, MAX_STEPS)
        after = self.hop_to_donor.get(self.node, MAX_STEPS)
        progress = before - after
        reward += 5.0 * progress * (ps ** 2 if progress > 0 else 1.0)

        if self.node == self.donor:
            reward += 30.0
        elif self.steps >= MAX_STEPS:
            reward -= 20.0

        return self._state(), reward, done, {}


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Actor network from Table II: (128, 512, 512, 128)
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 512), nn.ReLU(),
                                 nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, 128), nn.ReLU(),
                                 nn.Linear(128, action_dim))

    def forward(self, s):
        return self.net(s)   # logits

    def act(self, s, deterministic=False, mask=None):
        logits = self.forward(s)
        if mask is not None:
            # invalid slots get -inf so they carry no probability at all
            logits = logits.masked_fill(~mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            a = torch.argmax(probs, dim=-1)
        else:
            a = torch.multinomial(probs, 1).squeeze(-1)
        log_probs = torch.log(probs + 1e-8)
        if mask is not None:
            log_probs = log_probs.masked_fill(~mask, 0.0)
            probs = probs.masked_fill(~mask, 0.0)
        return a, probs, log_probs


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Critic network from Table II: (256, 1024, 1024, 256). The paper notes the
        # critic is deliberately larger than the actor, since predicting values for every
        # action is the harder learning problem.
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 1024), nn.ReLU(),
                                 nn.Linear(1024, 1024), nn.ReLU(),
                                 nn.Linear(1024, 256), nn.ReLU(),
                                 nn.Linear(256, action_dim))

    def forward(self, s):
        return self.net(s)   # Q(s, .) for every discrete action


@dataclass
class ReplayBuffer:
    capacity: int = 20000
    data: list = field(default_factory=list)

    def push(self, *transition):
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        s, a, r, s2, d, m, m2 = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32),
                torch.tensor(np.array(m), dtype=torch.bool),
                torch.tensor(np.array(m2), dtype=torch.bool))

    def __len__(self):
        return len(self.data)


def _device():
    """Apple MPS or CUDA when available.

    The Table II networks (critic 256-1024-1024-256, batch 1024) are slow enough on CPU
    that a full run takes the better part of an hour on a laptop.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_sac(g: nx.Graph, donor: int, ue_ids: list[int], episodes: int = 3000,
              gamma: float = 0.99, lr: float = 1e-3, batch_size: int = 1024):
    env = IabRoutingEnv(g, donor)
    dev = _device()
    print(f'training on {dev}')
    actor = Actor(env.state_dim, env.action_dim).to(dev)
    q1, q2 = (Critic(env.state_dim, env.action_dim).to(dev),
              Critic(env.state_dim, env.action_dim).to(dev))
    q1_t, q2_t = (Critic(env.state_dim, env.action_dim).to(dev),
                  Critic(env.state_dim, env.action_dim).to(dev))
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())

    log_alpha = torch.zeros(1, requires_grad=True, device=dev)
    # Target entropy is set against the number of actions actually available, not the
    # padded action space. Most nodes have far fewer real neighbours than MAX_DEGREE, so
    # aiming at log(MAX_DEGREE) asks the policy to stay uniform over choices that do not
    # exist -- alpha then grows without bound and the entropy term swamps the reward. The
    # symptom is a policy that solves the task early and then unlearns it: 36/36 UEs
    # reaching the donor at 1000 episodes, 0/36 at 4000, uniform logits at 12000.
    mean_degree = sum(len(n) for n in env.neighbours.values()) / len(env.neighbours)
    target_entropy = 0.6 * math.log(max(mean_degree, 2.0))
    MAX_LOG_ALPHA = math.log(0.2)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    alpha_opt = torch.optim.Adam([log_alpha], lr=lr)

    buf = ReplayBuffer()
    tau = 0.005

    for ep in range(episodes):
        ue = random.choice(ue_ids)
        s = env.reset(ue)
        done = False
        while not done:
            mask = torch.tensor(env.action_mask()).unsqueeze(0).to(dev)
            # The agent starts on the configured route and takes over gradually. Without
            # this it never completes an episode, so the buffer holds no transition that
            # reached the donor and there is nothing to learn from.
            follow = random.random() < max(0.0, 1.0 - ep / (0.4 * episodes))
            default = env.default_action(env.node) if follow else None
            if default is not None:
                a = default
            else:
                with torch.no_grad():
                    act, _, _ = actor.act(torch.tensor(s).unsqueeze(0).to(dev), mask=mask)
                a = act.item()
            mask_now = env.action_mask()
            s2, r, done, _ = env.step(a)
            mask_next = env.action_mask()
            buf.push(s, a, r, s2, float(done), mask_now, mask_next)
            s = s2

            if len(buf) >= batch_size:
                S, A, R, S2, D, M, M2 = buf.sample(batch_size)
                S, A, R, S2, D = S.to(dev), A.to(dev), R.to(dev), S2.to(dev), D.to(dev)
                M, M2 = M.to(dev), M2.to(dev)
                alpha = log_alpha.exp()

                with torch.no_grad():
                    _, probs2, logp2 = actor.act(S2, mask=M2)
                    q1_next, q2_next = q1_t(S2), q2_t(S2)
                    min_q_next = torch.min(q1_next, q2_next)
                    v_next = (probs2 * (min_q_next - alpha * logp2)).sum(-1)
                    target_q = R + gamma * (1 - D) * v_next

                q1_pred = q1(S).gather(1, A.unsqueeze(-1)).squeeze(-1)
                q2_pred = q2(S).gather(1, A.unsqueeze(-1)).squeeze(-1)
                q_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
                q_opt.zero_grad(); q_loss.backward(); q_opt.step()

                _, probs, logp = actor.act(S, mask=M)
                q1_val, q2_val = q1(S), q2(S)
                min_q = torch.min(q1_val, q2_val)
                actor_loss = (probs * (alpha.detach() * logp - min_q)).sum(-1).mean()
                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

                entropy = -(probs * logp).sum(-1).mean()
                alpha_loss = -(log_alpha * (target_entropy - entropy).detach())
                alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
                with torch.no_grad():
                    log_alpha.clamp_(max=MAX_LOG_ALPHA)

                for t, s_ in ((q1_t, q1), (q2_t, q2)):
                    for tp, sp in zip(t.parameters(), s_.parameters()):
                        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

        if (ep + 1) % 500 == 0:
            print(f"episode {ep+1}/{episodes}")

    return env, actor


def evaluate(env: IabRoutingEnv, actor: Actor, ue_ids: list[int]):
    rows = []
    for ue in ue_ids:
        s = env.reset(ue)
        path = [ue]
        done = False
        steps = 0
        while not done and steps < MAX_STEPS:
            dev = next(actor.parameters()).device
            mask = torch.tensor(env.action_mask()).unsqueeze(0).to(dev)
            with torch.no_grad():
                a, _, _ = actor.act(torch.tensor(s).unsqueeze(0).to(dev),
                                    deterministic=True, mask=mask)
            s, _r, done, _ = env.step(a.item())
            path.append(env.node)
            steps += 1
        reached = env.node == env.donor
        rows.append({
            "ue": ue, "algo": "sac", "hops": len(path) - 1,
            "delay_ms": env.total_delay_ms() if reached else float("nan"),
            "reliability": math.exp(env.cum_logrel) if reached else 0.0,
            "path": "-".join(map(str, path)),
            "meets_latency": int(reached and env.total_delay_ms() <= LATENCY_BUDGET_MS),
            "meets_reliability": int(reached and math.exp(env.cum_logrel) >= RELIABILITY_TARGET),
        })
    return rows


if __name__ == "__main__":
    import argparse
    import statistics as st

    ap = argparse.ArgumentParser()
    ap.add_argument("topology_csv")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    g = load_topology(args.topology_csv)
    ue_ids = [n for n in g.nodes if n >= 19]

    env, actor = train_sac(g, donor=0, ue_ids=ue_ids, episodes=args.episodes)
    results = evaluate(env, actor, ue_ids)

    print(f"\n{'ue':>4} {'hops':>5} {'delay_ms':>10} {'reliability':>12}")
    for r in results:
        print(f"{r['ue']:>4} {r['hops']:>5} {r['delay_ms']:>10.3f} {r['reliability']:>12.4f}")

    valid = [r for r in results if not math.isnan(r["delay_ms"])]
    reached = len(valid)
    if valid:
        print(f"\n[sac] reached donor: {reached}/{len(ue_ids)} UEs | "
              f"mean delay={st.mean(r['delay_ms'] for r in valid):.3f} ms | "
              f"mean reliability={st.mean(r['reliability'] for r in valid):.4f}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ue", "algo", "hops", "delay_ms", "reliability", "path",
                                    "meets_latency", "meets_reliability"])
            w.writeheader()
            w.writerows(results)
        print(f"Wrote {args.out}")
