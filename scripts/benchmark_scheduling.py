"""Benchmark the learned scheduler against RPMA on the same meshes.

The paper's central claim is about time: a forward pass costs the same regardless of
topology size, while a search over links and power levels does not, so only one of them
fits inside a 10 ms slot on a large mesh. This measures both, along with the goodput each
achieves, on the three topology sizes.

    python scripts/benchmark_scheduling.py
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np

from drl_routing.scheduling.aarl import SchedulingGymEnv, evaluate, train
from drl_routing.scheduling.env import SchedulingEnv
from drl_routing.scheduling.mesh import SchedulingConfig, build_mesh
from drl_routing.scheduling.rpma import run_rpma

OUT = Path("docs/scheduling_benchmark.md")
SLOT_BUDGET_MS = 10.0
PACKETS = {"small": 2304, "medium": 10812, "large": 45246}


def main(sizes=("small", "medium", "large"), steps: int = 60_000,
         interference: float = 0.6) -> None:
    rows = []
    for size in sizes:
        cfg = SchedulingConfig(interference_level=interference,
                               initial_packets=PACKETS[size])

        env = SchedulingEnv(build_mesh(size, cfg))
        rpma_summary, rpma_ms = run_rpma(env, random.Random(0))
        print(f"{size:7} rpma  goodput={rpma_summary['goodput']:.3f} "
              f"slots={rpma_summary['slots']:3} decision={rpma_ms:7.2f} ms")

        started = time.perf_counter()
        model, gym_env = train(size=size, alpha=10.0,
                               interference_level=interference, steps=steps)
        train_minutes = (time.perf_counter() - started) / 60
        aarl = evaluate(model, gym_env, episodes=3)
        print(f"{size:7} aarl  goodput={aarl['goodput']:.3f} "
              f"slots={aarl['slots']:5.1f} decision={aarl['decision_ms']:7.2f} ms "
              f"(trained {train_minutes:.1f} min)")

        rows.append({
            "size": size,
            "links": len(env.mesh.links),
            "rpma": (rpma_summary["goodput"], rpma_summary["slots"], rpma_ms),
            "aarl": (aarl["goodput"], aarl["slots"], aarl["decision_ms"]),
        })

    intro = (
        "Learned scheduling against the Residual Profit Maximizer on the same meshes and "
        f"the same traffic. The slot budget is {SLOT_BUDGET_MS:.0f} ms: a scheduler that "
        "cannot decide within it is not deployable, whatever its schedules look like."
    )
    lines = [
        "# Scheduling: learned policy versus RPMA",
        "",
        intro,
        "",
        "| Mesh | Links | RPMA goodput | AARL goodput | RPMA decision | AARL decision |",
        "|------|-------|--------------|--------------|---------------|---------------|",
    ]
    for r in rows:
        rg, _, rms = r["rpma"]
        ag, _, ams = r["aarl"]
        lines.append(f"| {r['size']} | {r['links']} | {rg:.3f} | {ag:.3f} "
                     f"| {rms:.2f} ms | {ams:.2f} ms |")

    if rows:
        small, large = rows[0], rows[-1]
        growth = large["rpma"][2] / small["rpma"][2] if small["rpma"][2] else float("inf")
        steady = large["aarl"][2] / small["aarl"][2] if small["aarl"][2] else float("inf")
        finding = (
            f"RPMA's decision time grows {growth:.0f}x between the smallest and largest "
            f"mesh; the learned policy's grows {steady:.1f}x. That is the whole argument: "
            "the search examines every link against every power level and every link "
            "already chosen, so it scales with the topology, while a forward pass through "
            "a fixed network does not care how many links it is scoring. Quality is the "
            "secondary question -- a better schedule computed after the slot has passed "
            "is not a schedule."
        )
        lines += ["", finding, ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--sizes", default="small,medium,large")
    args = ap.parse_args()
    main(sizes=tuple(args.sizes.split(",")), steps=args.steps)
