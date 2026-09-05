"""Dijkstra baseline for the IAB routing problem (Yin, Roy, Cao, TCOMM 2022).

Implements Eq. (4)-(7): find the min-latency path subject to a reliability
constraint, via the Lagrangian relaxation c(i, mu) = 0.5*Tproc + Ttrans(i) +
mu * log(1/ps(i)), solved with Dijkstra for a fixed mu, and Algorithm 1
(bisection) to find the mu* that satisfies P(q*(mu)) = sigma.

Also implements a "greedy" baseline: shortest-hop path, tie-broken by link
quality, as a stand-in for the paper's Semi-Persistent-Scheduling-style
greedy algorithm (Section V).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from itertools import pairwise

import networkx as nx

from drl_routing.agents.sac_routing import (
    queueing_delay_ms,
    transmission_time_ms,
)

TTI_MS = 0.125          # numerology 3: TTI = 1 / 2^3 ms
T_PROC_MS = 4 * TTI_MS   # Tproc = 4 * TTI (paper, Sec III-A)


def load_topology(path: str) -> nx.Graph:
    """Build an undirected graph from topology.csv (src,dst,delay_ms,pb,capacity_mbps)."""
    g = nx.Graph()
    with open(path) as f:
        for row in csv.DictReader(f):
            g.add_edge(
                int(row["src"]), int(row["dst"]),
                delay_ms=float(row["delay_ms"]),
                pb=float(row["pb"]),
            )
    return g


def path_delay_ms(g: nx.Graph, path: list[int], active_ues: int = 40) -> float:
    """Total latency of a path per Eq. (2).

    Tdelay = Tque + (n+2)/2 * Tproc + sum of transmission times, where n is the number
    of relays. The transmission time per hop is ceil(pkt/TB) * TTI from Eq. (1), not the
    scheduling delay stored on the link -- the agent uses the same model, and comparing
    the two against different latency definitions would measure nothing.
    """
    hops = len(path) - 1
    relays = max(hops - 1, 0)
    t_que = queueing_delay_ms(active_ues, hops)
    t_trans = hops * transmission_time_ms()
    return t_que + (relays + 2) / 2 * T_PROC_MS + t_trans


def path_reliability(g: nx.Graph, path: list[int]) -> float:
    """P(q) = product of (1 - pb) over links in the path, Eq. (3) with pc=0."""
    p = 1.0
    for u, v in pairwise(path):
        p *= (1.0 - g[u][v]["pb"])
    return p


def _weighted_graph(g: nx.Graph, mu: float) -> nx.Graph:
    """Attach the Lagrangian edge weight c(i, mu) for a given mu (Eq. 5)."""
    h = g.copy()
    for u, v, data in h.edges(data=True):
        pb = data["pb"]
        ps = max(1.0 - pb, 1e-9)
        data["weight"] = 0.5 * T_PROC_MS + data["delay_ms"] + mu * math.log(1.0 / ps)
    return h


def dijkstra_optimal(g: nx.Graph, src: int, dst: int, sigma: float,
                      mu_max: float = 50.0, iters: int = 40) -> tuple[list[int], float]:
    """Algorithm 1 (bisection on mu) + Dijkstra with weight c(i, mu).

    Returns (path, mu*). Falls back to the min-hop path if even mu=0 cannot
    reach sigma (i.e. the constraint is infeasible on this graph).
    """
    mu_low, mu_high = 0.0, mu_max

    # Keep the best feasible path found, not the last one tried. Bisection walks through
    # infeasible values of mu on its way to mu*, and each of those yields a path that
    # misses sigma; returning whichever came last hands back a route the constraint
    # rejects even when a valid one exists.
    best_path: list[int] | None = None
    best_mu = mu_max

    for _ in range(iters):
        mu = (mu_low + mu_high) / 2.0
        path = nx.shortest_path(_weighted_graph(g, mu), src, dst, weight="weight")
        if path_reliability(g, path) >= sigma:
            best_path, best_mu = path, mu
            mu_high = mu       # constraint met: push back toward minimum latency
        else:
            mu_low = mu        # more weight on reliability

    if best_path is None:
        # sigma is unreachable on this graph; fall back to the most reliable route there
        # is, so the caller gets the best available rather than a min-latency one
        best_path = nx.shortest_path(
            g, src, dst, weight=lambda u, v, d: math.log(1.0 / max(1.0 - d["pb"], 1e-9)))
        best_mu = mu_max

    return best_path, best_mu


def greedy_path(g: nx.Graph, src: int, dst: int) -> list[int]:
    """Baseline: shortest-hop path, tie-broken by best (lowest pb) links —
    a simple stand-in for the paper's SPS-style greedy algorithm."""
    return nx.shortest_path(g, src, dst, weight=lambda u, v, d: d["pb"] + 1e-3)


@dataclass
class RouteResult:
    ue: int
    algo: str
    path: list[int]
    delay_ms: float
    reliability: float
    hops: int


def compare(topology_csv: str, donor: int = 0, ue_ids: list[int] | None = None,
            sigma: float = 0.999) -> list[RouteResult]:
    g = load_topology(topology_csv)
    if ue_ids is None:
        # a UE is a leaf of the mesh: one link, and not the donor. The old test used a
        # literal id range, which only held for the eighteen-node scenario.
        ue_ids = [n for n in g.nodes if g.degree(n) == 1 and n != donor]

    results: list[RouteResult] = []
    for ue in sorted(ue_ids):
        if not nx.has_path(g, donor, ue):
            continue

        opt_path, _mu = dijkstra_optimal(g, donor, ue, sigma)
        results.append(RouteResult(ue, "dijkstra_optimal", opt_path,
                                    path_delay_ms(g, opt_path), path_reliability(g, opt_path),
                                    len(opt_path) - 1))

        gpath = greedy_path(g, donor, ue)
        results.append(RouteResult(ue, "greedy", gpath,
                                    path_delay_ms(g, gpath), path_reliability(g, gpath),
                                    len(gpath) - 1))
    return results


if __name__ == "__main__":
    import argparse
    import statistics as st

    ap = argparse.ArgumentParser()
    ap.add_argument("topology_csv")
    ap.add_argument("--sigma", type=float, default=0.999, help="reliability target (paper: 0.999 for URLLC/VR)")
    ap.add_argument("--out", default=None, help="optional CSV to write per-UE results")
    args = ap.parse_args()

    results = compare(args.topology_csv, sigma=args.sigma)

    print(f"{'ue':>4} {'algo':>17} {'hops':>5} {'delay_ms':>10} {'reliability':>12}")
    for r in results:
        print(f"{r.ue:>4} {r.algo:>17} {r.hops:>5} {r.delay_ms:>10.3f} {r.reliability:>12.4f}")

    for algo in ("dijkstra_optimal", "greedy"):
        subset = [r for r in results if r.algo == algo]
        mean_delay = st.mean(r.delay_ms for r in subset)
        mean_rel = st.mean(r.reliability for r in subset)
        met_sigma = sum(1 for r in subset if r.reliability >= args.sigma)
        print(f"\n[{algo}] mean delay={mean_delay:.3f} ms | mean reliability={mean_rel:.4f} "
              f"| meets sigma={args.sigma}: {met_sigma}/{len(subset)} UEs")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ue", "algo", "hops", "delay_ms", "reliability", "path"])
            for r in results:
                w.writerow([r.ue, r.algo, r.hops, r.delay_ms, r.reliability, "-".join(map(str, r.path))])
        print(f"\nWrote {args.out}")
