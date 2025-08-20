import heapq
from typing import List, Optional, Tuple

# Helper class representing an edge in the residual graph
type_edge_comment = """
Edge stores: target node index, reverse edge index, remaining capacity, and per-unit cost.
"""
class Edge:
    def __init__(self, to: int, rev: int, cap: int, cost: int):
        self.to = to       # index of the node this edge points to
        self.rev = rev     # index of the reverse Edge in graph[to]
        self.cap = cap     # remaining capacity on this edge
        self.cost = cost   # cost per unit flow through this edge

# Implements successive shortest-path algorithm for min-cost max-flow
class MinCostMaxFlow:
    def __init__(self, N: int):
        self.N = N
        self.graph: List[List[Edge]] = [[] for _ in range(N)]

    def add_edge(self, frm: int, to: int, cap: int, cost: int = 0) -> None:
        """
        Add a directed edge frm -> to with capacity cap and cost per unit flow.
        Also adds a reverse edge to support residual graph updates.
        """
        fwd = Edge(to, len(self.graph[to]), cap, cost)
        bwd = Edge(frm, len(self.graph[frm]), 0, -cost)
        self.graph[frm].append(fwd)
        self.graph[to].append(bwd)

    def min_cost_flow(self, s: int, t: int, maxf: int) -> Tuple[int, int]:
        """
        Send up to maxf units of flow from s to t, minimizing total cost.
        Returns (flow_sent, total_cost).
        Uses potentials with Dijkstra to handle negative edge costs.
        """
        N = self.N
        prevv = [0] * N      # previous vertex on shortest path
        preve = [0] * N      # previous edge index on that path
        INF = float('inf')
        dist = [INF] * N     # distances in reduced cost graph
        potential = [0] * N  # potentials to ensure non-negative reduced costs
        flow = 0
        cost = 0

        while flow < maxf:
            # 1) Use Dijkstra to find shortest path w.r.t. reduced costs
            dist = [INF] * N
            dist[s] = 0
            pq = [(0, s)]
            while pq:
                d, v = heapq.heappop(pq)
                if d > dist[v]:
                    continue
                for i, e in enumerate(self.graph[v]):
                    if e.cap > 0:
                        nd = d + e.cost + potential[v] - potential[e.to]
                        if nd < dist[e.to]:
                            dist[e.to] = nd
                            prevv[e.to] = v
                            preve[e.to] = i
                            heapq.heappush(pq, (nd, e.to))
            if dist[t] == INF:
                break  # no more augmenting path

            # 2) Update potentials
            for v in range(N):
                if dist[v] < INF:
                    potential[v] += dist[v]

            # 3) Determine maximum flow we can push on this path
            d = maxf - flow
            v = t
            while v != s:
                d = min(d, self.graph[prevv[v]][preve[v]].cap)
                v = prevv[v]

            # 4) Augment flow along the path
            flow += d
            cost += d * potential[t]
            v = t
            while v != s:
                e = self.graph[prevv[v]][preve[v]]
                e.cap -= d
                self.graph[v][e.rev].cap += d
                v = prevv[v]

        return flow, cost

# Main solution function

def crowdedCampus(
    n: int,
    m: int,
    timePreferences: List[List[int]],
    proposedClasses: List[List[int]],
    minimumSatisfaction: int
) -> Optional[List[int]]:
    """
    Assign each student to exactly one class respecting class min/max sizes
    and ensuring at least minimumSatisfaction students receive a top-5 preferred time slot.
    Returns allocation list of length n or None if infeasible.
    """
    # Node indexing
    S = 0
    T = n + m + 1
    N = T + 1
    SS = N
    TT = N + 1
    mcmf = MinCostMaxFlow(N + 2)

    # 1) Source -> students
    for i in range(n):
        mcmf.add_edge(S, 1 + i, 1, 0)

    # 2) Students -> classes with cost -1 for top-5 match
    for i in range(n):
        prefs = timePreferences[i]
        for j in range(m):
            slot_j = proposedClasses[j][0]
            cost_edge = 0
            for k in range(5):
                if prefs[k] == slot_j:
                    cost_edge = -1
                    break
            mcmf.add_edge(1 + i, 1 + n + j, 1, cost_edge)

    # 3) Classes -> sink with capacity U - L; record demands for lower bounds
    demand = [0] * (N + 2)
    for j in range(m):
        L, U = proposedClasses[j][1], proposedClasses[j][2]
        mcmf.add_edge(1 + n + j, T, U - L, 0)
        demand[1 + n + j] -= L
        demand[T]           += L

    # 4) Connect super-source SS and super-sink TT to enforce demands
    totalDemand = 0
    for v in range(N):
        d = demand[v]
        if d > 0:
            mcmf.add_edge(SS, v, d, 0)
            totalDemand += d
        elif d < 0:
            mcmf.add_edge(v, TT, -d, 0)
    # allow circulation
    mcmf.add_edge(T, S, totalDemand, 0)

    # 5) Feasibility check: pure max-flow from SS to TT
    flow, _ = mcmf.min_cost_flow(SS, TT, totalDemand)
    if flow < totalDemand:
        return None

    # 6) Min-cost max-flow from S to T, sending n units
    flow, cost = mcmf.min_cost_flow(S, T, n)
    if flow < n or -cost < minimumSatisfaction:
        return None

    # 7) Extract allocation from saturated student->class edges
    allocation = [0] * n
    for i in range(n):
        for e in mcmf.graph[1 + i]:
            if n+1 <= e.to <= n+m and e.cap == 0:
                allocation[i] = e.to - (n+1)
                break
    return allocation

# === Test cases ===
if __name__ == "__main__":
    # Test 1: simple distinct assignment
    n, m = 3, 2
    timePrefs = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]] * 3
    props = [[0,1,2], [1,1,2]]
    assert set(crowdedCampus(n, m, timePrefs, props, 1)) == {0,1}

    # Test 2: insufficient capacity  None
    assert crowdedCampus(4, 2, [[0]*20]*4, [[0,2,2],[1,1,1]], 0) is None

    # Test 3: no top-5 possible  None
    assert crowdedCampus(4, 2, [list(range(20))]*4, [[5,0,4],[6,0,4]], 1) is None

    # Test 4: exact satisfaction

    # Test 5: forced class min with zero satisfaction requirement
    alloc = crowdedCampus(2, 1, [[10,0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]]*2, [[0,2,2]], 0)
    assert alloc == [0,0]

    print("All tests passed!")
