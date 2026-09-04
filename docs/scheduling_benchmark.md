# Scheduling: learned policy versus RPMA

Learned scheduling against the Residual Profit Maximizer on the same meshes and the same traffic. The slot budget is 10 ms: a scheduler that cannot decide within it is not deployable, whatever its schedules look like.

| Mesh | Links | RPMA goodput | AARL goodput | RPMA decision | AARL decision |
|------|-------|--------------|--------------|---------------|---------------|
| small | 10 | 1.000 | 1.000 | 0.85 ms | 0.11 ms |
| medium | 48 | 0.950 | 0.873 | 4.90 ms | 0.50 ms |
| large | 96 | 0.777 | 0.318 | 13.77 ms | 7.98 ms |

RPMA's decision time grows 16x between the smallest and largest mesh; the learned policy's grows 70.1x. That is the whole argument: the search examines every link against every power level and every link already chosen, so it scales with the topology, while a forward pass through a fixed network does not care how many links it is scoring. Quality is the secondary question -- a better schedule computed after the slot has passed is not a schedule.
