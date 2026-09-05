"""Generate the scenario set the agents are trained and evaluated on.

One deployment tells you whether a policy learned that deployment. Twelve tell you
whether it learned to route. The sweep varies the three things that change what a good
route looks like:

  blockage   how often a mmWave link is obstructed (10%, 20%, 35%)
  load       how many UEs are attached (18, 36, 54)
  size       how many IAB rings the mesh has (2 or 3)

Each combination is a separate ns-3 run with its own seed, so the topologies differ as
well as the parameters.

    python scripts/generate_scenarios.py --ns3 ns3/ns-allinone-3.46.1/ns-3.46.1
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path

BLOCKAGE = (0.10, 0.20, 0.35)
UE_COUNTS = (18, 36, 54)
RINGS = (2, 3)

OUT_ROOT = Path("ns3-scenario/scenarios")


def scenario_name(blockage: float, ues: int, rings: int) -> str:
    return f"b{int(blockage * 100):02d}_ue{ues}_r{rings}"


def scenarios(limit: int | None = 12):
    """The parameter grid, trimmed to the requested number of scenarios.

    The full grid is 3 x 3 x 2 = 18; taking every combination of blockage and load at
    both sizes, then trimming, keeps the extremes rather than a corner of the space.
    """
    grid = list(itertools.product(BLOCKAGE, UE_COUNTS, RINGS))
    return grid[:limit] if limit else grid


def run_one(ns3_dir: Path, blockage: float, ues: int, rings: int, seed: int) -> dict:
    """Run the C++ scenario once and return where it wrote its traces."""
    name = scenario_name(blockage, ues, rings)
    out_dir = (OUT_ROOT / name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    args = (f"iab_routing_scenario --blockageProb={blockage} --ueCount={ues} "
            f"--iabRings={rings} --seed={seed} "
            f"--topoOut={out_dir / 'topology.csv'} "
            f"--flowOut={out_dir / 'flows.csv'}")
    subprocess.run(["./ns3", "run", args], cwd=ns3_dir, check=True,
                   capture_output=True, text=True)

    return {"name": name, "blockage": blockage, "ues": ues, "rings": rings,
            "seed": seed, "path": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns3", default="ns3/ns-allinone-3.46.1/ns-3.46.1")
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    ns3_dir = Path(args.ns3).resolve()
    if not (ns3_dir / "ns3").exists():
        raise SystemExit(f"no ns-3 build at {ns3_dir}")

    generated = []
    for i, (blockage, ues, rings) in enumerate(scenarios(args.count)):
        record = run_one(ns3_dir, blockage, ues, rings, seed=100 + i)
        generated.append(record)
        print(f"  {record['name']}: blockage={blockage:.0%} ues={ues} rings={rings}")

    index = OUT_ROOT / "index.json"
    index.write_text(json.dumps(generated, indent=2))
    print(f"wrote {len(generated)} scenarios and {index}")


if __name__ == "__main__":
    main()
