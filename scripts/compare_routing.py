"""Compare every routing policy on the same topology and the same UEs.

Three policies, one table:
  dijkstra - shortest path under the Lagrangian weight that trades latency against
             reliability. Has the whole graph, and solves the problem exactly.
  greedy   - always take the neighbour with the lowest block error rate. Local
             information only, no notion of distance to the donor.
  sac      - discrete Soft Actor-Critic, also local information only, trained to reach
             the donor while minimising delay and maximising reliability.

The comparison that matters is sac against greedy: both see only their neighbours, so the
difference is what the policy learned. Dijkstra is the ceiling -- it has global knowledge
neither of the others gets, and a learned local policy matching it would be the result
worth reporting.

    python scripts/compare_routing.py
"""
from __future__ import annotations

import csv
import math
import statistics as st
from pathlib import Path

TRACES = Path("ns3-scenario/traces")
BASELINE = TRACES / "dijkstra_vs_greedy.csv"
SAC = TRACES / "sac_results.csv"
OUT = Path("docs/routing_comparison.md")


def read(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def summarise(rows: list[dict], algo: str) -> dict:
    mine = [r for r in rows if r.get("algo") == algo]
    reached = [r for r in mine
               if r.get("delay_ms") not in ("", "nan") and not math.isnan(float(r["delay_ms"]))]
    if not reached:
        return {"algo": algo, "n": len(mine), "reached": 0}
    return {
        "algo": algo,
        "n": len(mine),
        "reached": len(reached),
        "hops": st.mean(float(r["hops"]) for r in reached),
        "delay": st.mean(float(r["delay_ms"]) for r in reached),
        "reliability": st.mean(float(r["reliability"]) for r in reached),
    }


def main() -> None:
    rows = read(BASELINE) + read(SAC)
    summaries = [summarise(rows, a) for a in ("dijkstra_optimal", "greedy", "sac")]
    summaries = [s for s in summaries if s["n"]]

    for s in summaries:
        if s["reached"]:
            print(f"  {s['algo']:9} reached {s['reached']}/{s['n']}  "
                  f"hops={s['hops']:.2f}  delay={s['delay']:.2f} ms  "
                  f"reliability={s['reliability']:.4f}")
        else:
            print(f"  {s['algo']:9} reached 0/{s['n']}")

    by = {s["algo"]: s for s in summaries}
    intro = (
        "All three routing policies on the same topology and the same UEs, from the ns-3 "
        "scenario. Dijkstra sees the whole graph and solves the problem exactly, so it is "
        "the ceiling rather than a competitor. Greedy and SAC both see only their "
        "immediate neighbours; the gap between those two is what the policy learned."
    )
    lines = [
        "# Routing policy comparison",
        "",
        intro,
        "",
        "| Policy | Information | Reached donor | Mean hops | Mean delay (ms) | Mean reliability |",
        "|--------|-------------|---------------|-----------|-----------------|------------------|",
    ]
    scope = {"dijkstra_optimal": "global graph", "greedy": "neighbours only",
             "sac": "neighbours only (learned)"}
    for s in summaries:
        if s["reached"]:
            lines.append(
                f"| {s['algo']} | {scope[s['algo']]} | {s['reached']}/{s['n']} "
                f"| {s['hops']:.2f} | {s['delay']:.2f} | {s['reliability']:.4f} |"
            )
        else:
            lines.append(f"| {s['algo']} | {scope[s['algo']]} | 0/{s['n']} | - | - | - |")

    finding = ""
    if "sac" in by and "greedy" in by and by["sac"].get("reached") and by["greedy"].get("reached"):
        d_sac, d_greedy = by["sac"]["delay"], by["greedy"]["delay"]
        gap = (d_greedy - d_sac) / d_greedy * 100
        verdict = ("beats" if gap > 0 else "loses to")
        finding = (
            f"On delay the learned policy {verdict} greedy by {abs(gap):.1f}% "
            f"({d_sac:.2f} ms against {d_greedy:.2f} ms), on the same local information. "
        )
        if "dijkstra_optimal" in by and by["dijkstra_optimal"].get("reached"):
            d_opt = by["dijkstra_optimal"]["delay"]
            over = (d_sac - d_opt) / d_opt * 100
            finding += (
                f"Against the global-knowledge optimum it is {over:+.1f}% "
                f"({d_sac:.2f} ms against {d_opt:.2f} ms). That gap is the price of "
                "deciding hop by hop without seeing the graph, and closing it is what a "
                "learned policy is for: Dijkstra needs a full, current view of every link, "
                "which a real IAB network does not hand to each node."
            )
    lines += ["", finding, ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
