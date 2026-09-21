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

    max_rudder_angle_deg caps how many degrees any SINGLE turn_left/turn_right command may
    request at once (see turn_left/turn_right below) -- independent of turn_rate_deg_s, which
    caps how fast own-ship swings toward whatever target_heading is currently commanded. A
    huge one-shot command (e.g. an LLM deciding "turn_left 103") previously still got
    accepted in full, just spread slowly over many steps by the turn-rate limit; capping the
    command itself is what actually keeps a single manoeuvre physically plausible.
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
        self.min_cpa_override_active: bool = False  # see _enforce_min_cpa()
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

    def _min_cpa_for(self, heading: float, speed: float) -> float:
        """Worst-case (smallest) projected CPA across all targets if own-ship immediately
        adopted the given heading/speed -- constant-velocity extrapolation, same model as
        cpa_tcpa() elsewhere. Used only to compare CANDIDATE manoeuvres in _enforce_min_cpa();
        the actual kinematics still rate-limits how fast own-ship gets there regardless of
        which candidate wins."""
        if not self.targets:
            return float("inf")
        return min(cpa_tcpa(self.own.x, self.own.y, heading, speed,
                            t.x, t.y, t.heading, t.speed)[0] for t in self.targets)

    def _enforce_min_cpa(self) -> None:
        """Mandatory min_cpa_m backstop -- min_cpa_m is no longer just advisory context handed
        to the OOW agent's prompt; if the CURRENTLY commanded target_heading/target_speed
        (whatever the agent, manual helm, or the goal-tracking logic last set) would leave the
        worst-case projected CPA below constraints.min_cpa_m, this overrides the command with
        whichever discrete emergency manoeuvre -- hard turn to starboard/port (at
        max_rudder_angle_deg, the full available helm), an emergency stop, or a turn+stop
        combination -- achieves the BEST projected CPA. Critically, it overrides to that best
        candidate even when it STILL can't reach min_cpa_m (a genuinely unavoidable
        close-quarters situation): the point is to always steer toward the safest available
        outcome, never to blindly accept a worse one just because the threshold is already lost.
        Never overrides a command that's already safe (min_cpa_m satisfied) or when there are no
        targets to be safe from."""
        self.min_cpa_override_active = False
        if not self.targets:
            return
        agent_cpa = self._min_cpa_for(self.target_heading, self.target_speed)
        if agent_cpa >= self.constraints.min_cpa_m:
            return

        c = self.constraints
        candidates = [(self.target_heading, self.target_speed)]
        for dh in (c.max_rudder_angle_deg, -c.max_rudder_angle_deg):
            hdg = (self.own.heading + dh) % 360
            candidates.append((hdg, self.target_speed))
            candidates.append((hdg, 0.0))
        candidates.append((self.own.heading, 0.0))

        best_heading, best_speed, best_cpa = self.target_heading, self.target_speed, agent_cpa
        for hdg, spd in candidates:
            cpa = self._min_cpa_for(hdg, spd)
            if cpa > best_cpa:
                best_cpa, best_heading, best_speed = cpa, hdg, spd

        if best_cpa > agent_cpa:
            self.target_heading, self.target_speed = best_heading, best_speed
            self.min_cpa_override_active = True

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
        self._enforce_min_cpa()
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
        degrees = max(0.0, min(degrees, self.constraints.max_rudder_angle_deg))
        self.target_heading = (self.target_heading - degrees) % 360

    def turn_right(self, degrees: float = 10.0) -> None:
        degrees = max(0.0, min(degrees, self.constraints.max_rudder_angle_deg))
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

