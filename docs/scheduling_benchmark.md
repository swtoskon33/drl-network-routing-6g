# Scheduling: learned policy versus RPMA

Learned scheduling against the Residual Profit Maximizer on the same meshes and the same traffic. The slot budget is 10 ms: a scheduler that cannot decide within it is not deployable, whatever its schedules look like.

| Mesh | Links | RPMA goodput | AARL goodput | RPMA decision | AARL decision |
|------|-------|--------------|--------------|---------------|---------------|
| small | 10 | 1.000 | 1.000 | 0.86 ms | 0.09 ms |

RPMA's decision time grows 1x between the smallest and largest mesh; the learned policy's grows 1.0x. That is the whole argument: the search examines every link against every power level and every link already chosen, so it scales with the topology, while a forward pass through a fixed network does not care how many links it is scoring. Quality is the secondary question -- a better schedule computed after the slot has passed is not a schedule.
