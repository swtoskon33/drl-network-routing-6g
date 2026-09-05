# Routing across scenarios

Every routing policy across 2 ns-3 scenarios, differing in how often links are blocked, how many UEs are attached and how many rings the mesh has. One deployment shows whether a policy learned that deployment; the spread here shows whether the approach holds as the deployment changes.

| Scenario | Links | UEs | Dijkstra | Greedy | SAC |
|---|---|---|---|---|---|
| b10_ue18_r2 | 27 | 12 | 0.708 | 0.875 | 0.562 |
| b10_ue18_r3 | 42 | 18 | 0.972 | 1.000 | 0.833 |

Mean reliability: Dijkstra 0.840, greedy 0.937, SAC 0.698, with the learned policy varying by 0.135 across scenarios. Dijkstra sees the whole graph and solves the problem exactly, so it is the ceiling rather than a competitor; the comparison that means something is SAC against greedy, which works from the same local information.
