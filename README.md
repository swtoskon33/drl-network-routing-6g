# drl-network-routing

Routing for a 5G Integrated Access and Backhaul (IAB) network: a Dijkstra optimum with
global knowledge, a greedy baseline, and a Soft Actor-Critic agent that decides hop by
hop from local information alone. The topology and link conditions come from an ns-3
scenario in C++.

Implements the method of Yin, Roy and Cao, *Routing and Resource Allocation for IAB
Multi-Hop Network in 5G Advanced*, IEEE Transactions on Communications 70(10), 2022
([DOI](https://doi.org/10.1109/TCOMM.2022.3200673),
[open PDF](https://par.nsf.gov/servlets/purl/10359622)).

## The problem

An IAB network extends coverage by relaying user traffic through wireless backhaul nodes
to a donor with a fibre connection. Every extra hop costs latency and reliability, and
mmWave links fail suddenly when something blocks them. Problem 1 of the paper asks for
the minimum-latency path subject to a reliability floor:

```
    minimise  T_delay(q)     subject to  P(q) >= sigma
```

with a 5 ms latency budget and 0.999 success probability, the 3GPP targets for VR/AR
traffic.

## What is implemented

| Paper | Here |
|-------|------|
| Eq. (1), (2): latency model | `queueing_delay_ms`, `transmission_time_ms`, `total_delay_ms` |
| Eq. (3): reliability | product of per-link success probabilities, pc = 0 |
| Eq. (5): Lagrangian relaxation | link weight `Tproc/2 + Ttrans + mu*log(1/ps)` |
| Algorithm 1: bisection on mu | `_bisect_mu`, and the same in the Dijkstra baseline |
| Eq. (8): agent state | per-neighbour channel quality plus downstream latency and reliability |
| Eq. (11): reward | `psi_d * ((tau - T)/T^o + (-1)^o) - psi_r * (K - 1)` |
| Section IV: SAC on local information | `sac_routing.py`, discrete SAC with masked actions |
| Section IV: pre-configured routing | the agent warm-starts on the Dijkstra route |
| Table I: ns-3 parameters | 23 dBm, 100 MHz, 28 GHz, numerology 3, UMi |
| Table II: SAC hyperparameters | critic (256, 1024, 1024, 256), actor (128, 512, 512, 128), batch 1024, lr 1e-3 |
| Section IV-C: federated learning | not implemented |

## Architecture

### Routing: ns-3 to a hop decision

```
      ns-3 scenario (C++)
              |
   28 GHz link budget: UMi pathloss, 23 dBm,
   22 dB beam gain, 20% of links blocked
              |
              v
   topology.csv (links, delay, BLER) + flows.csv
              |
      +-------+--------+--------------+
      |                |              |
      v                v              v
   Dijkstra          greedy       SAC agent
   whole graph      neighbours    neighbours
   bisected mu*     lowest BLER   learned policy
      |                |              |
      +-------+--------+--------------+
              |
              v
   per-UE delay, reliability, hops
              |
              v
   docs/routing_comparison.md
```

### The SAC agent, one hop at a time

```
   node's neighbours
          |
   for each: link delay, BLER,
   and the delay and reliability of
   reaching the donor through it        <- Eq. (8)
          |
          v
   actor (128, 512, 512, 128)           <- Table II
          |
   mask out padding slots and the
   node we just came from
          |
          v
   next hop
          |
   reward: psi_d*((tau-T)/T^o + (-1)^o)
           - psi_r*(K-1)                <- Eq. (11)
          |
          v
   twin critics (256, 1024, 1024, 256) + entropy term
```

### Training: warm start, then hand over

```
   donor solves Problem 1 centrally
   (Dijkstra on c(i,mu), mu bisected to meet sigma)
                   |
                   v
   routing table configured at every node       <- Section IV
                   |
      +------------+------------+
      |                         |
   early episodes            later episodes
   follow the table          follow the policy
      |                         |
      +------------+------------+
                   |
        transitions into the replay buffer
                   |
                   v
        SAC update: twin critics, entropy
        target from real neighbour counts
```

### Scheduling: which links transmit this slot

```
   traffic matrix + interference matrix
                   |
                   v
   buffers per link, packets on shortest paths
                   |
                   v
   agent assigns a power to every link
                   |
   effective capacity = nominal x (power - interference)
                   |
                   v
   packets move one hop; a packet arriving at
   a full buffer is dropped
                   |
                   v
   reward: -beta - alpha*(dropped/P) + (moved/P)
```


## Running it

```
# build and run the ns-3 scenario (writes topology.csv and flows.csv)
cd ns3/ns-allinone-3.46.1/ns-3.46.1
cp ../../../ns3-scenario/iab_routing_scenario.cc scratch/
./ns3 build
./ns3 run "iab_routing_scenario --topoOut=../../../ns3-scenario/traces/topology.csv \\
                                --flowOut=../../../ns3-scenario/traces/flows.csv"

# baselines and agent
python src/drl_routing/baselines/dijkstra_iab.py ns3-scenario/traces/topology.csv \\
    --out ns3-scenario/traces/dijkstra_vs_greedy.csv
python src/drl_routing/agents/sac_routing.py ns3-scenario/traces/topology.csv \\
    --episodes 6000 --out ns3-scenario/traces/sac_results.csv
python scripts/compare_routing.py
```

Training picks up CUDA or Apple MPS when present. The Table II networks are large enough
that a CPU run takes the better part of an hour; on MPS it is a few minutes.

## Layout

```
ns3-scenario/
  iab_routing_scenario.cc    topology, mmWave link budget, traffic, CSV export
  traces/                    topology.csv, flows.csv, results

src/drl_routing/
  baselines/dijkstra_iab.py  Problem 1 via Eq. (5) and Algorithm 1, plus greedy
  agents/sac_routing.py      discrete SAC: environment, networks, training, evaluation
  topology/network.py        graph model with load-dependent delay

scripts/compare_routing.py   scores every policy on the same UEs
docs/routing_comparison.md   the comparison table
```

## Stack

ns-3.46.1 (C++), Python 3.11, PyTorch, NetworkX.

## Licence

MIT
