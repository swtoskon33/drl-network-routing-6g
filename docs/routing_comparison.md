# Routing policy comparison

All three routing policies on the same topology and the same UEs, from the ns-3 scenario. Dijkstra sees the whole graph and solves the problem exactly, so it is the ceiling rather than a competitor. Greedy and SAC both see only their immediate neighbours; the gap between those two is what the policy learned.

| Policy | Information | Reached donor | Mean hops | Mean delay (ms) | Mean reliability |
|--------|-------------|---------------|-----------|-----------------|------------------|
| dijkstra_optimal | global graph | 36/36 | 3.50 | 11.26 | 0.9122 |
| greedy | neighbours only | 36/36 | 4.28 | 14.73 | 0.9306 |
| sac | neighbours only (learned) | 36/36 | 3.50 | 11.80 | 0.9102 |

On delay the learned policy beats greedy by 19.9% (11.80 ms against 14.73 ms), on the same local information. Against the global-knowledge optimum it is +4.7% (11.80 ms against 11.26 ms). That gap is the price of deciding hop by hop without seeing the graph, and closing it is what a learned policy is for: Dijkstra needs a full, current view of every link, which a real IAB network does not hand to each node.
