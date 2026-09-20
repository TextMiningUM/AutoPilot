"""Kinematics stepper: advances own-ship + targets in straight lines at their
current heading/speed, records a trajectory in the exact schema
evaluate_run.py expects, and exposes the manual controls from
Brain Storming/pilot_agents.ipynb (turn_left/turn_right/set_speed/
stop_vessel/speed_up/slow_down)."""
from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, replace

from app.missions import Mission, Vessel
from app.narrate import cpa_tcpa, relative_bearing

# Matches evaluate_run.py's default collision_radius_m -- keep in sync.
COLLISION_RADIUS_M = 15.0


@dataclass
class VesselConstraints:
    """Own-ship's physical performance envelope -- fed by the sidebar's Ship
    performance/Mission/Simulation inputs (see streamlit_app.py's
    build_vessel_constraints()), never hardcoded elsewhere. Defaults here match
    those widgets' own `value=` so a Simulation built without an explicit
    constraints arg (e.g. in a script/test) still behaves sensibly.

    max_rudder_angle_deg is deliberately NOT enforced as a heading-delta limit
    yet -- only turn_rate_deg_s is actively applied by the kinematics layer
    below. It's carried here for a future, more precise model where rudder
    angle itself drives the achieved turn rate, and so it can be surfaced to
    the OOW agent's prompt (see app/agents.py) as an informational limit.
    """
    max_speed_mps: float = 10.0
    max_rudder_angle_deg: float = 30.0
    max_acceleration_mps2: float = 0.2
    max_deceleration_mps2: float = 0.2
    turn_rate_deg_s: float = 3.0
    cruise_speed_mps: float = 10.0
    min_cpa_m: float = 500.0
    time_step_s: float = 10.0


class Simulation:
    def __init__(self, mission: Mission, constraints: VesselConstraints | None = None):
        self.mission = mission
        self.constraints = constraints or VesselConstraints()
        self.t = 0.0
        self.own: Vessel = replace(mission.own_ship, name="own_ship")
        self.targets: list[Vessel] = [replace(t) for t in mission.targets]
        # Commanded targets the kinematics layer steers own-ship toward -- start
        # equal to the current state so nothing moves before the first manoeuvre
        # command (or the first CRUISING-status goal-tracking, see step()).
        self.target_heading: float = self.own.heading
        self.target_speed: float = self.own.speed
        self.status: str = "CRUISING"  # deterministic AVOIDING/CRUISING, see _update_behaviour_status()
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

    def _update_behaviour_status(self) -> None:
        """Deterministic (never LLM-decided) AVOIDING/CRUISING status machine -- purely
        descriptive (surfaced to the UI and the OOW agent's prompt context), never touches
        target_heading/target_speed itself. AVOIDING while any contact's CURRENT CPA is under
        constraints.min_cpa_m; CRUISING once every contact clears that threshold. Own-ship
        always just keeps whatever target_heading/target_speed the last manoeuvre command
        (agent or manual helm) set -- an automatic "resume goal course once clear" would
        silently overrule manual helm input, which defeats the point of manual control."""
        if self.targets:
            cpas = [cpa_tcpa(self.own.x, self.own.y, self.own.heading, self.own.speed,
                             t.x, t.y, t.heading, t.speed)[0] for t in self.targets]
            avoiding = any(cpa < self.constraints.min_cpa_m for cpa in cpas)
        else:
            avoiding = False
        self.status = "AVOIDING" if avoiding else "CRUISING"

    def _advance_own_kinematics(self, dt: float) -> None:
        """Rate-limits own-ship's heading/speed toward target_heading/target_speed
        by at most this step's turn-rate/acceleration allowance -- never jumps
        straight to the target unless the target is already closer than one
        step's limit (in which case it snaps there exactly, no overshoot)."""
        c = self.constraints
        max_turn = c.turn_rate_deg_s * dt
        diff = relative_bearing(self.own.heading, self.target_heading)  # shortest signed path, [-180, 180]
        if abs(diff) <= max_turn:
            self.own.heading = self.target_heading % 360
        else:
            self.own.heading = (self.own.heading + math.copysign(max_turn, diff)) % 360

        target_speed = max(0.0, min(self.target_speed, c.max_speed_mps))
        diff_s = target_speed - self.own.speed
        if diff_s > 0:
            self.own.speed = min(target_speed, self.own.speed + c.max_acceleration_mps2 * dt)
        elif diff_s < 0:
            self.own.speed = max(target_speed, self.own.speed - c.max_deceleration_mps2 * dt)

    def step(self, dt: float | None = None) -> None:
        dt = self.constraints.time_step_s if dt is None else dt
        self._update_behaviour_status()
        self._advance_own_kinematics(dt)
        for v in [self.own] + self.targets:
            h = math.radians(v.heading)
            v.x += v.speed * math.sin(h) * dt
            v.y += v.speed * math.cos(h) * dt
        self.t += dt
        self._record()

    def advance_straight(self, dt: float) -> None:
        """Position-only advance at each vehicle's CURRENT heading/speed -- no
        kinematics rate-limiting, no AVOIDING/CRUISING status/goal-tracking. This
        is the simulation's original step() behaviour, kept as its own method
        exclusively for project_scenario()'s "no avoidance" preview, which must
        keep showing the raw, un-steered collision geometry a scenario was built
        to create -- not autonomous goal-tracking once "cruising"."""
        for v in [self.own] + self.targets:
            h = math.radians(v.heading)
            v.x += v.speed * math.sin(h) * dt
            v.y += v.speed * math.cos(h) * dt
        self.t += dt
        self._record()

    # ── manual controls (own-ship only) ──────────────────────────────
    # These set the COMMANDED target, not the current state directly -- the
    # kinematics layer in step()/_advance_own_kinematics() rate-limits how fast
    # own-ship actually gets there. Turn commands are relative to the current
    # TARGET heading (not the current actual heading), so stacking manoeuvre
    # commands while a turn is still catching up extends the turn further,
    # instead of the second command being silently absorbed because the ship
    # hasn't visibly moved yet.
    def turn_left(self, degrees: float = 10.0) -> None:
        self.target_heading = (self.target_heading - degrees) % 360

    def turn_right(self, degrees: float = 10.0) -> None:
        self.target_heading = (self.target_heading + degrees) % 360

    def set_speed(self, new_speed: float) -> None:
        self.target_speed = max(0.0, min(new_speed, self.constraints.max_speed_mps))

    def speed_up(self, delta: float = 2.0) -> None:
        self.set_speed(self.target_speed + delta)

    def slow_down(self, delta: float = 2.0) -> None:
        self.set_speed(self.target_speed - delta)

    def stop_vessel(self) -> None:
        self.target_speed = 0.0

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
    steerable Simulation instance. Uses advance_straight() (own-ship's ORIGINAL
    heading/speed forever), never the kinematics/AVOIDING-CRUISING layer in step()
    -- this preview must keep showing the un-steered course, not autonomous
    goal-tracking once "cruising"."""
    sim = Simulation(mission)
    gx, gy = mission.goal
    straight_dist = math.hypot(gx - sim.own.x, gy - sim.own.y)
    total_time = (straight_dist / sim.own.speed if sim.own.speed > 0 else 0.0) * margin
    n_steps = min(max_steps, max(1, math.ceil(total_time / dt)))
    for _ in range(n_steps):
        sim.advance_straight(dt)
    return sim.trajectory

