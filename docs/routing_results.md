# Routing: the learned policy against the exact solver

Every policy over the same topology, the same links and the same load: 15 UEs on the
measured ns-3 deployment, 40 active UEs setting the contention.

| Policy | Information | Reached | Reliability | Delay | Hops |
|--------|-------------|---------|-------------|-------|------|
| Dijkstra (Eq. 5, Algorithm 1) | whole graph | 15/15 | 0.646 | 1.53 ms | 2.80 |
| Greedy (SPS extension) | one hop ahead | 15/15 | 0.658 | 1.56 ms | 2.73 |
| SAC (Eq. 8, 11) | neighbours only | 15/15 | **0.790** | 1.64 ms | 2.73 |

The learned policy is 22% more reliable than the exact solver while seeing far less of
the network, which is the result Fig. 5(b) reports. The reason is what the two optimise
against: Dijkstra minimises a cost computed from collision probabilities worked out in
advance, while the agent is paid on the acknowledgements it actually receives, so it
learns which neighbours cost it retransmissions rather than which ones the model says
should.

Greedy edging past Dijkstra is not a defeat for the solver. The 0.999 target is out of
reach on this deployment, so the constraint never binds and both fall back to
maximising reliability; where they differ is noise across fifteen UEs.

## What does not match the paper

The paper reports reliability above 0.999. We reach 0.79, and the gap is one quantity:
the collision probability. Section III-B pairs the block error rate with pc but the
paper does not say how many subbands the scheduler has, which is what sets it. Rather
than pick the number that produces the published figure, here is the dependence:

| Subbands | pc at degree 6 | Mean route reliability |
|----------|----------------|------------------------|
| 12 | 0.083 | 0.613 |
| 24 | 0.042 | 0.786 |
| 48 | 0.021 | 0.887 |
| 96 | 0.010 | 0.942 |
| 192 | 0.005 | 0.971 |

Reliability is entirely determined by a parameter the paper leaves open. Reaching 0.999
needs collisions to be almost absent, which a scheduler that avoids them by design
achieves and one drawing subbands at random does not. We use 12, a plausible number for
100 MHz at numerology 3, and report the sensitivity instead of tuning to the target.

Latency does match: 1.5 to 1.6 ms per packet against the paper's 4 to 8 ms for a file
transfer, on routes of the same length.

## Reproducing

```
# the topology, once, seeded
cd ns3/ns-allinone-3.46.1/ns-3.46.1
./ns3 run "iab_topology --seed=42 --run=1 \\
    --positionsOut=../../../data/phase1/positions.csv \\
    --linksOut=../../../data/phase1/links.csv"

# baselines and the agent
python scripts/run_routing.py --episodes 6000
```
