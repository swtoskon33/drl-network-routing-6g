# drl-network-routing-6g

Deep reinforcement learning for two decisions in a 5G millimetre-wave mesh, each
reproduced from a published paper, each measured against the non-learned method it has
to beat.

**Routing** — where does a packet go? An exact solver with the whole graph, a greedy
baseline, and a Soft Actor-Critic agent choosing hop by hop from what one node can see.
After Yin, Roy and Cao, *Routing and Resource Allocation for IAB Multi-Hop Network in 5G
Advanced*, IEEE Transactions on Communications 70(10), 2022
([DOI](https://doi.org/10.1109/TCOMM.2022.3200673),
[open PDF](https://par.nsf.gov/servlets/purl/10359622)).

**Scheduling** — which links transmit this slot, and at what power? Activating every
link is worse than activating a chosen subset, because they interfere. A PPO agent
against a combinatorial baseline, where the argument is about decision time as much as
quality. After Gahtan, Cohen, Bronstein and Kedar, *Using Deep Reinforcement Learning for
mmWave Real-Time Scheduling*, NoF 2023
([DOI](https://doi.org/10.1109/NoF58724.2023.10302794)).

## Results

15 UEs on the measured deployment, 40 active UEs setting the contention:

| Policy | Information | Reliability | Delay | Hops |
|--------|-------------|-------------|-------|------|
| Dijkstra, Eq. (5) and Algorithm 1 | whole graph | 0.646 | 1.53 ms | 2.80 |
| Greedy, the SPS extension | one hop ahead | 0.658 | 1.56 ms | 2.73 |
| SAC, Eq. (8) and (11) | neighbours only | **0.790** | 1.64 ms | 2.73 |

The learned policy is 22% more reliable than the exact solver while seeing far less of
the network, which is what Fig. 5(b) of the paper reports. Dijkstra minimises a cost
built from collision probabilities computed in advance; the agent is paid on the
acknowledgements it receives, so it learns which neighbours actually cost it
retransmissions.

Scheduling, over the three mesh sizes:

| Mesh | Links | RPMA goodput | AARL goodput | RPMA decision | AARL decision |
|------|-------|--------------|--------------|---------------|---------------|
| small | 10 | 1.000 | 1.000 | 0.85 ms | 0.11 ms |
| medium | 48 | 0.950 | 0.873 | 4.90 ms | 0.50 ms |
| large | 96 | 0.777 | 0.318 | 13.77 ms | 7.98 ms |

The decision time reproduces that paper's central claim: the search crosses the 10 ms
slot budget on the largest mesh, so its schedules arrive after the slot they were
computed for, while a forward pass does not care how many links it is scoring. On the
incast workload, where 90% of nodes send to 10%, the learned policy reaches 0.773
against the baseline's 0.460.

Details in docs/routing_results.md and docs/scheduling_benchmark.md.

## Routing

Four phases, each verified before the next was started.

**The deployment, measured.** An ns-3 scenario builds Fig. 4: a donor at the centre,
three hexagonal rings of six IAB nodes at 200 m, UEs dropped uniformly by area and
associated with their closest node. Every link goes through the real 3GPP TR 38.901
Urban Micro model -- the condition model decides LOS or NLOS, the loss model applies the
matching pathloss and shadowing, and 8x8 arrays at the nodes with 4x4 at the UEs supply
the 36 dB of beamforming gain without which nothing closes at 28 GHz over 200 m. Table I
parameters throughout. Written once with seed 42.

**What a path costs.** Eq. (1) to (3). The transport block follows from each link's
SINR, so a clean link carries a packet in fewer slots than a marginal one. Eq. (2)
charges the queue, half the processing time per relay, and the transmission time of
every hop. Eq. (3) multiplies the per-hop success probability by one minus the collision
probability.

**The exact answer.** Eq. (5) folds the reliability constraint into the objective with a
multiplier, making Problem 1 a shortest path under
`c(i, mu) = Tproc/2 + Ttrans(i) + mu*log(1/ps(i))`. Algorithm 1 bisects mu to where the
path meets sigma. Greedy is the semi-persistent scheduling baseline: best channel that
makes progress, one hop of foresight.

**The agent.** Discrete SAC over the state of Eq. (8) and the reward of Eq. (11), with
the networks and hyperparameters of Table II. Each node sees its neighbours and what the
route through each of them costs, and nothing else. It warm-starts on the route the
donor configured, as Section IV describes, and takes over as training proceeds.

## Reliability

The paper reports above 0.999 and we reach 0.79. The gap is one quantity: the collision
probability, which Eq. (3) pairs with the block error rate but the paper never
parameterises. It depends on how many subbands the scheduler has, and reliability tracks
that number directly:

| Subbands | pc at degree 6 | Mean route reliability |
|----------|----------------|------------------------|
| 12 | 0.083 | 0.613 |
| 24 | 0.042 | 0.786 |
| 48 | 0.021 | 0.887 |
| 96 | 0.010 | 0.942 |
| 192 | 0.005 | 0.971 |

Reaching 0.999 needs collisions to be almost absent, which a scheduler that avoids them
by design achieves and one drawing subbands at random does not. We use 12, plausible for
100 MHz at numerology 3, and report the sensitivity rather than tuning to the target.

Latency does match, at 1.5 to 1.6 ms per packet on routes of the same length.

Not implemented: UE mobility and the time-varying channel, the A2C baseline, and the
federated learning of Section IV-C.

## Running it

Build the topology once:

```
cd ns3/ns-allinone-3.46.1/ns-3.46.1
./ns3 configure --enable-modules=core,network,mobility,propagation,spectrum,antenna
cp ../../../ns3-scenario/iab_topology.cc scratch/
./ns3 build
./ns3 run "iab_topology --seed=42 --run=1 \\
    --positionsOut=../../../data/phase1/positions.csv \\
    --linksOut=../../../data/phase1/links.csv"
```

Then the policies:

```
python scripts/run_routing.py --episodes 6000
python scripts/benchmark_scheduling.py --sizes small,medium,large --steps 150000
```

Training picks up CUDA or Apple MPS when present; the Table II networks are slow enough
on CPU to matter.

Both papers fix their hyperparameters. To search them first, with ASHA stopping the
trials that fall behind:

```
pip install -e ".[tuning]"
python scripts/tune_scheduler.py --samples 40
```

Benchmark runs log their configuration and results to MLflow, so a number here traces
back to the run that produced it. CI lints and tests on every push, then runs a short
benchmark: enough to catch a change that breaks training, not enough to reproduce the
published numbers.

## Layout

```
ns3-scenario/iab_topology.cc     the deployment and the 3GPP channel model

src/drl_routing/
  routing/cost.py                Eq. (1) to (3): latency and reliability
  routing/baselines.py           Eq. (5) to (7) and Algorithm 1, plus greedy
  routing/environment.py         the MDP: state Eq. (8), reward Eq. (11)
  routing/sac.py                 discrete SAC, Table II
  scheduling/mesh.py             topology, capacities, interference
  scheduling/env.py              buffers, packet movement, drops
  scheduling/aarl.py             PPO agent and its gym wrapper
  scheduling/rpma.py             the combinatorial baseline
  tracking.py                    MLflow logging

scripts/run_routing.py           every routing policy on the same topology
scripts/benchmark_scheduling.py  RPMA against the learned scheduler
scripts/tune_scheduler.py        Ray Tune sweep

docs/routing_results.md          routing comparison and the subband sensitivity
docs/scheduling_benchmark.md     scheduling results and sweeps
```

## Stack

ns-3.46.1 (C++), Python 3.11, PyTorch, Stable-Baselines3, Gymnasium, NetworkX, Ray Tune,
MLflow, GitHub Actions.

## Licence

MIT
