"""Run every routing policy over the measured topology and report the comparison.

    python scripts/run_routing.py --episodes 6000

Reads data/phase1/links.csv, which the ns-3 scenario writes once with a fixed seed.
Writes the comparison and the routes each policy chose to data/phase4.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from drl_routing.routing.baselines import dijkstra_optimal, greedy_route
from drl_routing.routing.cost import Network
from drl_routing.routing.environment import RoutingEnvironment
from drl_routing.routing.sac import evaluate, train

LINKS = "data/phase1/links.csv"
OUT = Path("data/phase4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument("--links", default=LINKS)
    parser.add_argument("--donor", type=int, default=0)
    parser.add_argument("--active-ues", type=int, default=40)
    args = parser.parse_args()

    # scheduled=False: the agent decides locally, so collisions are live. The exact
    # solver is scored under the same conditions, or the comparison measures nothing.
    net = Network.from_csv(args.links, donor=args.donor, scheduled=False)
    ues = [n for n in net.graph.nodes if net.graph.degree(n) == 1]
    print(f"{net.graph.number_of_nodes()} nodes, {net.graph.number_of_edges()} links, "
          f"{len(ues)} UEs")

    results = {}
    for name, solve in (("dijkstra", dijkstra_optimal), ("greedy", greedy_route)):
        routes = [solve(net, args.donor, ue, active_ues=args.active_ues) for ue in ues]
        arrived = [r for r in routes if r.reliability > 0]
        results[name] = {
            "reached": len(arrived),
            "reliability": st.mean(r.reliability for r in arrived),
            "delay_ms": st.mean(r.delay_ms for r in arrived),
            "hops": st.mean(r.hops for r in arrived),
        }

    env = RoutingEnvironment(net, donor=args.donor, active_ues=args.active_ues)
    actor, _critic, returns = train(env, ues, episodes=args.episodes)
    rows = evaluate(env, actor, ues)
    arrived = [r for r in rows if r["reached"]]
    results["sac"] = {
        "reached": len(arrived),
        "reliability": st.mean(r["reliability"] for r in arrived),
        "delay_ms": st.mean(r["delay_ms"] for r in arrived),
        "hops": st.mean(r["hops"] for r in arrived),
    }

    print()
    for name, metrics in results.items():
        print(f"{name:9} reached {metrics['reached']:2}/{len(ues)}  "
              f"rel={metrics['reliability']:.4f}  "
              f"delay={metrics['delay_ms']:5.2f} ms  hops={metrics['hops']:.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(results, indent=2))
    (OUT / "sac_routes.json").write_text(json.dumps(rows, indent=2))
    (OUT / "returns.json").write_text(json.dumps(returns))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
