# Routing policy comparison

All three routing policies on the same topology and the same UEs, from the ns-3 scenario. Dijkstra sees the whole graph and solves the problem exactly, so it is the ceiling rather than a competitor. Greedy and SAC both see only their immediate neighbours; the gap between those two is what the policy learned.

| Policy | Information | Reached donor | Mean hops | Mean delay (ms) | Mean reliability |
|--------|-------------|---------------|-----------|-----------------|------------------|
| dijkstra_optimal | global graph | 36/36 | 3.83 | 1.70 | 0.8192 |
| greedy | neighbours only | 36/36 | 3.83 | 1.70 | 0.8192 |
| sac | neighbours only (learned) | 36/36 | 3.83 | 1.70 | 0.6161 |

On delay the learned policy loses to greedy by 0.0% (1.70 ms against 1.70 ms), on the same local information. Against the global-knowledge optimum it is +0.0% (1.70 ms against 1.70 ms). That gap is the price of deciding hop by hop without seeing the graph, and closing it is what a learned policy is for: Dijkstra needs a full, current view of every link, which a real IAB network does not hand to each node.
