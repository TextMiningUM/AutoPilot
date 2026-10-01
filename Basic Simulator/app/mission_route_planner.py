"""Captain walking-skeleton Phase 2: route planner with exclusion zones
(design_captain_missions.md Sec 13.B.8/Sec 7).

Pure Python geometry, no GPU/API key, safe to run locally. v1 algorithm (decided in the
design doc): a visibility graph over exclusion-zone polygon vertices -- nodes are the
start/goal plus every polygon corner, an edge exists wherever the straight segment between
two nodes crosses no zone, shortest path via Dijkstra. Geometrically exact for polygon
obstacles at this abstraction level, simpler than a grid-based A* (no resolution trade-off,
no jagged paths). Distances use `app.mission_sim.haversine_nm` (great-circle, lat/lon) --
the SAME whole-route distance convention Phase 1 already established.
"""
from __future__ import annotations
import heapq
from dataclasses import dataclass

# Importing app.mission_sim below already inserts REPO_ROOT onto sys.path (same side
# effect app/simulation.py relies on from app.narrate) -- resolves pipeline.captain_types
# regardless of caller cwd. Basic Simulator/ itself must already be on sys.path for this
# file (part of the `app` package) to have been importable at all.
from app.mission_sim import FuelModel, haversine_nm
from pipeline.captain_types import ExclusionZone, Port

Point = tuple[float, float]


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_properly_cross(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True iff segment p1-p2 crosses segment p3-p4 at a point that is NOT a shared
    endpoint of either segment -- touching/sharing a polygon vertex is allowed in a
    visibility graph, only genuinely cutting through counts as blocked."""
    d1, d2 = _orient(p3, p4, p1), _orient(p3, p4, p2)
    d3, d4 = _orient(p1, p2, p3), _orient(p1, p2, p4)
    return ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4))


def _point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Standard ray-casting point-in-polygon test; a point exactly on the boundary is an
    unhandled zero-measure edge case (not relevant for our start/goal/vertex node set)."""
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _segment_blocked_by_zone(p1: Point, p2: Point, polygon: list[Point]) -> bool:
    """A segment is blocked by one polygon if it properly crosses any of its edges, OR if
    its midpoint lies strictly inside it (catches a same-polygon diagonal that cuts
    straight across the interior without crossing any single edge, e.g. a convex quad's
    corner-to-corner diagonal). Travelling along the polygon's OWN boundary edge is always
    allowed -- checked first, since the midpoint-of-an-edge test is an ambiguous
    exactly-on-the-boundary case for ray casting."""
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        if (p1, p2) == (a, b) or (p1, p2) == (b, a):
            return False
        if _segments_properly_cross(p1, p2, a, b):
            return True
    midpoint = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
    return _point_in_polygon(midpoint, polygon)


def segment_blocked(p1: Point, p2: Point, zones: list[ExclusionZone]) -> bool:
    """True if the straight segment p1-p2 is blocked by any of the given zones."""
    return any(_segment_blocked_by_zone(p1, p2, zone.polygon) for zone in zones)


def active_zones(all_zones: list[ExclusionZone], draft_m: float,
                  safety_margin_m: float = 0.0) -> list[ExclusionZone]:
    """Sec 13.B.8's obstacle set: every zone is always active EXCEPT a `shallow_water` zone
    whose own min_depth_m already clears draft_m + margin for THIS ship -- dynamic hazards/
    coastline/fixed installations are always active regardless of draft."""
    def _is_active(zone: ExclusionZone) -> bool:
        if zone.type != "shallow_water":
            return True
        return zone.min_depth_m is not None and zone.min_depth_m < draft_m + safety_margin_m
    return [z for z in all_zones if _is_active(z)]


def _build_visibility_graph(start: Point, goal: Point,
                             zones: list[ExclusionZone]) -> dict[Point, list[tuple[Point, float]]]:
    nodes: list[Point] = [start, goal] + [v for zone in zones for v in zone.polygon]
    graph: dict[Point, list[tuple[Point, float]]] = {n: [] for n in nodes}
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i + 1:]:
            if n1 == n2 or segment_blocked(n1, n2, zones):
                continue
            d = haversine_nm(n1, n2)
            graph[n1].append((n2, d))
            graph[n2].append((n1, d))
    return graph


def _dijkstra(graph: dict[Point, list[tuple[Point, float]]], start: Point,
              goal: Point) -> tuple[list[Point], float] | None:
    dist: dict[Point, float] = {start: 0.0}
    prev: dict[Point, Point] = {}
    visited: set[Point] = set()
    heap: list[tuple[float, Point]] = [(0.0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            break
        for neighbour, weight in graph.get(node, []):
            nd = d + weight
            if nd < dist.get(neighbour, float("inf")):
                dist[neighbour] = nd
                prev[neighbour] = node
                heapq.heappush(heap, (nd, neighbour))
    if goal not in dist:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path, dist[goal]


@dataclass(frozen=True)
class RoutePlanResult:
    path: list[Point]
    distance_nm: float


def plan_route(start: Point, goal: Point, zones: list[ExclusionZone]) -> RoutePlanResult:
    """Visibility-graph shortest path from start to goal around the given exclusion zones
    (Sec 13.B.8). Raises ValueError if no path exists (goal fully walled off)."""
    graph = _build_visibility_graph(start, goal, zones)
    result = _dijkstra(graph, start, goal)
    if result is None:
        raise ValueError(f"no path found from {start} to {goal} around {len(zones)} zone(s)")
    path, distance_nm = result
    return RoutePlanResult(path=path, distance_nm=distance_nm)


def minimum_resource_route(waypoints: list[Point], zones: list[ExclusionZone]) -> RoutePlanResult:
    """Runs the planner once, unconstrained, over the whole multi-waypoint route (Sec
    13.B.8) -- the feasibility-oracle baseline Sec 13.C.11's resource-efficiency axis needs,
    computed once per mission and stored alongside the scenario (Sec 13.C.13)."""
    if len(waypoints) < 2:
        raise ValueError("need at least 2 waypoints to plan a route")
    full_path: list[Point] = [waypoints[0]]
    total_nm = 0.0
    for i in range(len(waypoints) - 1):
        leg = plan_route(waypoints[i], waypoints[i + 1], zones)
        full_path.extend(leg.path[1:])
        total_nm += leg.distance_nm
    return RoutePlanResult(path=full_path, distance_nm=total_nm)


@dataclass(frozen=True)
class RefugeCandidate:
    port: Port
    route: RoutePlanResult


def distance_to_refuge_nm(current_position: Point, ports: list[Port], zones: list[ExclusionZone],
                           required_services: list[str], draft_m: float) -> RefugeCandidate | None:
    """Sec 13.B.8/13.A.2: filters `ports` to those whose services cover `required_services`
    and whose min_approach_depth_m clears draft_m, then returns the NEAREST one by the same
    visibility-graph shortest-path distance (around zones, never a straight-line cut) --
    None if no port qualifies or none is reachable."""
    required = set(required_services)
    candidates = [p for p in ports if required.issubset(set(p.services)) and p.min_approach_depth_m >= draft_m]
    best: RefugeCandidate | None = None
    for port in candidates:
        try:
            route = plan_route(current_position, port.position, zones)
        except ValueError:
            continue
        if best is None or route.distance_nm < best.route.distance_nm:
            best = RefugeCandidate(port=port, route=route)
    return best


def check_route_feasibility(distance_nm: float, soa_kn: float, fuel_model: FuelModel,
                             fuel_tonnes_available: float) -> bool:
    """Sec 13.B.8's feasibility check: would travelling `distance_nm` at `soa_kn` consume
    more fuel than is actually available right now?"""
    t_h = distance_nm / soa_kn
    fuel_needed = fuel_model.fuel_rate_tonnes_per_h(soa_kn) * t_h
    return fuel_needed <= fuel_tonnes_available
