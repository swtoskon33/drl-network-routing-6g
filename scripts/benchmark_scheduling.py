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

from drl_routing.scheduling.aarl import evaluate, train
from drl_routing.scheduling.env import SchedulingEnv
from drl_routing.scheduling.mesh import SchedulingConfig, build_mesh
from drl_routing.scheduling.rpma import run_rpma
from drl_routing.tracking import track

OUT = Path("docs/scheduling_benchmark.md")
SLOT_BUDGET_MS = 10.0
PACKETS = {"small": 2304, "medium": 10812, "large": 45246}


def sweep_interference(size: str = "medium", steps: int = 20_000,
                       levels=(0.2, 0.4, 0.6, 0.8, 1.0)) -> list[dict]:
    """Goodput against interference level, the paper's Fig. 9.

    The level is what makes scheduling hard: at 20% almost everything can transmit
    together and the decision hardly matters, at 100% only a subset can and picking the
    wrong one costs the whole slot.
    """
    rows = []
    for level in levels:
        cfg = SchedulingConfig(interference_level=level, initial_packets=PACKETS[size])
        env = SchedulingEnv(build_mesh(size, cfg))
        rpma_summary, _ = run_rpma(env, random.Random(0))

        model, gym_env = train(size=size, alpha=10.0, interference_level=level,
                               steps=steps)
        aarl = evaluate(model, gym_env, episodes=3)
        rows.append({"level": level, "rpma": rpma_summary["goodput"],
                     "aarl": aarl["goodput"]})
        print(f"  interference {level:.1f}: rpma={rows[-1]['rpma']:.3f} "
              f"aarl={rows[-1]['aarl']:.3f}")
    return rows


def sweep_workloads(size: str = "medium", steps: int = 20_000) -> list[dict]:
    """The three traffic patterns the paper evaluates.

    uniform, few-to-many (10% of nodes sending to 90%) and many-to-few (the incast case,
    90% sending to 10%). Incast is the hard one: everything converges on a few buffers.
    """
    rows = []
    for workload in ("uniform", "few_to_many", "many_to_few"):
        cfg = SchedulingConfig(interference_level=0.6, initial_packets=PACKETS[size])
        env = SchedulingEnv(build_mesh(size, cfg), workload=workload)
        rpma_summary, _ = run_rpma(env, random.Random(0))

        model, gym_env = train(size=size, alpha=10.0, steps=steps)
        gym_env.workload = workload
        gym_env._build()
        aarl = evaluate(model, gym_env, episodes=3)
        rows.append({"workload": workload, "rpma": rpma_summary["goodput"],
                     "aarl": aarl["goodput"]})
        print(f"  {workload:12}: rpma={rows[-1]['rpma']:.3f} aarl={rows[-1]['aarl']:.3f}")
    return rows


def compare_alphas(size: str = "medium", steps: int = 20_000) -> list[dict]:
    """Drop-sensitive against drop-insensitive, the paper's AARL-DS and AARL-DI.

    alpha weights dropped packets against packets moved. At 1 the agent is largely
    indifferent to drops; at 10 it avoids them, which is what the paper finds works
    better on the larger meshes.
    """
    rows = []
    for label, alpha in (("drop-insensitive", 1.0), ("drop-sensitive", 10.0)):
        model, gym_env = train(size=size, alpha=alpha, steps=steps)
        result = evaluate(model, gym_env, episodes=3)
        rows.append({"agent": label, "alpha": alpha, "goodput": result["goodput"]})
        print(f"  {label:17} (alpha={alpha:4.1f}): goodput={result['goodput']:.3f}")
    return rows


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

        with track("scheduling", f"{size}-{steps}steps") as log:
            log(params={"mesh": size, "links": len(env.mesh.links),
                        "steps": steps, "interference": interference,
                        "packets": PACKETS[size], "alpha": 10.0},
                metrics={"rpma_goodput": rpma_summary["goodput"],
                         "rpma_decision_ms": rpma_ms,
                         "aarl_goodput": aarl["goodput"],
                         "aarl_decision_ms": aarl["decision_ms"],
                         "aarl_slots": aarl["slots"],
                         "train_minutes": train_minutes})

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

    print("\ninterference sweep:")
    interference_rows = sweep_interference(steps=max(steps // 3, 5000))
    lines += ["", "## Goodput against interference level", "",
              "The level is what makes the decision matter. At 20% nearly everything can",
              "transmit together; at 100% only a subset can, and the wrong subset costs",
              "the slot.", "",
              "| Interference | RPMA | AARL |", "|---|---|---|"]
    for r in interference_rows:
        lines.append(f"| {r['level']:.0%} | {r['rpma']:.3f} | {r['aarl']:.3f} |")

    print("\nworkloads:")
    workload_rows = sweep_workloads(steps=max(steps // 3, 5000))
    lines += ["", "## Goodput by traffic pattern", "",
              "Incast is the hard case: 90% of nodes sending to 10% converges everything",
              "on a few buffers, which is where drops come from.", "",
              "| Workload | RPMA | AARL |", "|---|---|---|"]
    for r in workload_rows:
        lines.append(f"| {r['workload'].replace('_', '-')} | {r['rpma']:.3f} "
                     f"| {r['aarl']:.3f} |")

    print("\ndrop sensitivity:")
    alpha_rows = compare_alphas(steps=max(steps // 3, 5000))
    lines += ["", "## Drop sensitivity", "",
              "alpha weights dropped packets against packets moved: at 1 the agent is",
              "largely indifferent to drops, at 10 it avoids them.", "",
              "| Agent | alpha | Goodput |", "|---|---|---|"]
    for r in alpha_rows:
        lines.append(f"| {r['agent']} | {r['alpha']:.0f} | {r['goodput']:.3f} |")

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
