"""Kinematics stepper: advances own-ship + targets in straight lines at their
current heading/speed, records a trajectory in the exact schema
evaluate_run.py expects, and exposes the manual controls from
Brain Storming/pilot_agents.ipynb (turn_left/turn_right/set_speed/
stop_vessel/speed_up/slow_down)."""
from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import replace

from app.missions import Mission, Vessel

# Matches evaluate_run.py's default collision_radius_m -- keep in sync.
COLLISION_RADIUS_M = 15.0


class Simulation:
    def __init__(self, mission: Mission):
        self.mission = mission
        self.t = 0.0
        self.own: Vessel = replace(mission.own_ship, name="own_ship")
        self.targets: list[Vessel] = [replace(t) for t in mission.targets]
        self.trajectory: list[dict] = []
        self.agent_log: list[dict] = []  # {t, narration, oow_decision}
        self._record()

    def _record(self) -> None:
        for v in [self.own] + self.targets:
            self.trajectory.append({
                "time": round(self.t, 2), "vehicle": v.name,
                "x": round(v.x, 2), "y": round(v.y, 2),
                "heading": round(v.heading, 2), "speed": round(v.speed, 3),
            })

    def step(self, dt: float = 10.0) -> None:
        for v in [self.own] + self.targets:
            h = math.radians(v.heading)
            v.x += v.speed * math.sin(h) * dt
            v.y += v.speed * math.cos(h) * dt
        self.t += dt
        self._record()

    # ── manual controls (own-ship only) ──────────────────────────────
    def turn_left(self, degrees: float = 10.0) -> None:
        self.own.heading = (self.own.heading - degrees) % 360

    def turn_right(self, degrees: float = 10.0) -> None:
        self.own.heading = (self.own.heading + degrees) % 360

    def set_speed(self, new_speed: float) -> None:
        self.own.speed = max(0.0, new_speed)

    def speed_up(self, delta: float = 2.0) -> None:
        self.set_speed(self.own.speed + delta)

    def slow_down(self, delta: float = 2.0) -> None:
        self.set_speed(self.own.speed - delta)

    def stop_vessel(self) -> None:
        self.own.speed = 0.0

    def apply_action(self, action: dict) -> None:
        act = (action or {}).get("action")
        if act == "turn_left":
            self.turn_left(action.get("degrees", 10.0))
        elif act == "turn_right":
            self.turn_right(action.get("degrees", 10.0))
        elif act == "speed_up":
            self.speed_up()
        elif act == "slow_down":
            self.slow_down()
        elif act == "stop":
            self.stop_vessel()
        # "hold_course" (or anything unrecognised) -> no-op by design

    def reached_goal(self, radius: float = 25.0) -> bool:
        gx, gy = self.mission.goal
        return math.hypot(self.own.x - gx, self.own.y - gy) <= radius

    def min_cpa_now(self) -> float:
        """Closest current range to any target -- for a quick live safety readout."""
        if not self.targets:
            return float("inf")
        return min(math.hypot(self.own.x - t.x, self.own.y - t.y) for t in self.targets)


def find_collision(trajectory: list[dict], radius: float = COLLISION_RADIUS_M) -> dict | None:
    """Scans a recorded trajectory for the first time any target comes within
    `radius` metres of own_ship. Returns {"time","vehicle","x","y","range_m"} for
    the earliest such moment, or None if the run never got that close."""
    by_time: dict[float, dict[str, dict]] = defaultdict(dict)
    for row in trajectory:
        by_time[row["time"]][row["vehicle"]] = row
    for t in sorted(by_time):
        own = by_time[t].get("own_ship")
        if not own:
            continue
        for name, row in by_time[t].items():
            if name == "own_ship":
                continue
            dist = math.hypot(own["x"] - row["x"], own["y"] - row["y"])
            if dist < radius:
                return {"time": t, "vehicle": name, "x": row["x"], "y": row["y"], "range_m": dist}
    return None


def project_scenario(mission: Mission, dt: float = 10.0, margin: float = 1.3,
                      max_steps: int = 400) -> list[dict]:
    """Straight-line projection of the WHOLE mission (no avoidance, no manual/agent
    intervention) -- lets you see the raw encounter geometry (the genuine collision
    course a scenario was built to create) at a glance, independent of the live,
    steerable Simulation instance."""
    sim = Simulation(mission)
    gx, gy = mission.goal
    straight_dist = math.hypot(gx - sim.own.x, gy - sim.own.y)
    total_time = (straight_dist / sim.own.speed if sim.own.speed > 0 else 0.0) * margin
    n_steps = min(max_steps, max(1, math.ceil(total_time / dt)))
    for _ in range(n_steps):
        sim.step(dt)
    return sim.trajectory
