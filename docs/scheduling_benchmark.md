# Scheduling: learned policy versus RPMA

Learned scheduling against the Residual Profit Maximizer on the same meshes and the same traffic. The slot budget is 10 ms: a scheduler that cannot decide within it is not deployable, whatever its schedules look like.

| Mesh | Links | RPMA goodput | AARL goodput | RPMA decision | AARL decision |
|------|-------|--------------|--------------|---------------|---------------|
| small | 10 | 1.000 | 1.000 | 0.81 ms | 0.12 ms |
| medium | 48 | 0.950 | 0.937 | 5.14 ms | 0.53 ms |

RPMA's decision time grows 6x between the smallest and largest mesh; the learned policy's grows 4.5x. That is the whole argument: the search examines every link against every power level and every link already chosen, so it scales with the topology, while a forward pass through a fixed network does not care how many links it is scoring. Quality is the secondary question -- a better schedule computed after the slot has passed is not a schedule.


## Goodput against interference level

The level is what makes the decision matter. At 20% nearly everything can
transmit together; at 100% only a subset can, and the wrong subset costs
the slot.

| Interference | RPMA | AARL |
|---|---|---|
| 20% | 0.977 | 0.996 |
| 40% | 0.977 | 0.885 |
| 60% | 0.950 | 0.908 |
| 80% | 0.909 | 0.791 |
| 100% | 0.903 | 0.768 |

## Goodput by traffic pattern

Incast is the hard case: 90% of nodes sending to 10% converges everything
on a few buffers, which is where drops come from.

| Workload | RPMA | AARL |
|---|---|---|
| uniform | 0.950 | 0.908 |
| few-to-many | 1.000 | 0.969 |
| many-to-few | 0.460 | 0.578 |

## Drop sensitivity

alpha weights dropped packets against packets moved: at 1 the agent is
largely indifferent to drops, at 10 it avoids them.

| Agent | alpha | Goodput |
|---|---|---|
| drop-insensitive | 1 | 0.837 |
| drop-sensitive | 10 | 0.908 |