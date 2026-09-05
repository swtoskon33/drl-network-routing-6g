"""Run every routing policy across all twelve scenarios.

A policy evaluated on the deployment it trained on tells you it can memorise that
deployment. This trains an agent per scenario and evaluates it there, so the spread
across scenarios says whether the approach holds up as blockage, load and mesh size
change rather than whether one run got lucky.

    python scripts/evaluate_scenarios.py --episodes 4000
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from drl_routing.agents.sac_routing import evaluate as sac_evaluate
from drl_routing.agents.sac_routing import load_topology, train_sac
from drl_routing.baselines.dijkstra_iab import compare
from drl_routing.tracking import track

INDEX = Path("ns3-scenario/scenarios/index.json")
OUT = Path("docs/scenario_sweep.md")


def score(rows, algo: str) -> dict:
    """Mean delay and reliability over the UEs a policy actually delivered.

    Accepts either the RouteResult objects the baseline returns or the dicts the agent
    evaluation produces.
    """
    def field(row, name):
        return row.get(name) if isinstance(row, dict) else getattr(row, name)

    reached = [r for r in rows
               if field(r, "algo") == algo and (field(r, "reliability") or 0) > 0]
    if not reached:
        return {"reached": 0, "delay": float("nan"), "reliability": 0.0}
    return {
        "reached": len(reached),
        "delay": st.mean(float(field(r, "delay_ms")) for r in reached),
        "reliability": st.mean(float(field(r, "reliability")) for r in reached),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    scenarios = json.loads(INDEX.read_text())
    if args.limit:
        scenarios = scenarios[:args.limit]

    results = []
    for scenario in scenarios:
        topology = Path(scenario["path"]) / "topology.csv"
        graph = load_topology(str(topology))
        donor = 0
        ues = [n for n in graph.nodes if graph.degree(n) == 1 and n != donor]

        baseline = compare(str(topology), donor=donor, ue_ids=ues)
        env, actor = train_sac(graph, donor=donor, ue_ids=ues, episodes=args.episodes)
        sac_rows = sac_evaluate(env, actor, ues)

        row = {
            "scenario": scenario["name"],
            "links": graph.number_of_edges(),
            "ues": len(ues),
            "dijkstra": score(baseline, "dijkstra_optimal"),
            "greedy": score(baseline, "greedy"),
            "sac": score(sac_rows, "sac"),
        }
        results.append(row)
        print(f"  {row['scenario']}: links={row['links']:3} ues={row['ues']:3} "
              f"dijkstra_rel={row['dijkstra']['reliability']:.3f} "
              f"sac_rel={row['sac']['reliability']:.3f}")

        with track("routing-scenarios", row["scenario"]) as log:
            log(params={"scenario": row["scenario"], "links": row["links"],
                        "ues": row["ues"], "episodes": args.episodes},
                metrics={"dijkstra_reliability": row["dijkstra"]["reliability"],
                         "greedy_reliability": row["greedy"]["reliability"],
                         "sac_reliability": row["sac"]["reliability"],
                         "sac_reached": row["sac"]["reached"]})

    intro = (
        f"Every routing policy across {len(results)} ns-3 scenarios, differing in how "
        "often links are blocked, how many UEs are attached and how many rings the mesh "
        "has. One deployment shows whether a policy learned that deployment; the spread "
        "here shows whether the approach holds as the deployment changes."
    )
    lines = [
        "# Routing across scenarios", "", intro, "",
        "| Scenario | Links | UEs | Dijkstra | Greedy | SAC |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['scenario']} | {r['links']} | {r['ues']} "
                     f"| {r['dijkstra']['reliability']:.3f} "
                     f"| {r['greedy']['reliability']:.3f} "
                     f"| {r['sac']['reliability']:.3f} |")

    if results:
        d = st.mean(r["dijkstra"]["reliability"] for r in results)
        g = st.mean(r["greedy"]["reliability"] for r in results)
        s = st.mean(r["sac"]["reliability"] for r in results)
        spread = st.pstdev([r["sac"]["reliability"] for r in results])
        summary = (
            f"Mean reliability: Dijkstra {d:.3f}, greedy {g:.3f}, SAC {s:.3f}, with the "
            f"learned policy varying by {spread:.3f} across scenarios. Dijkstra sees the "
            "whole graph and solves the problem exactly, so it is the ceiling rather than "
            "a competitor; the comparison that means something is SAC against greedy, "
            "which works from the same local information."
        )
        lines += ["", summary, ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
