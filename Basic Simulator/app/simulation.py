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
from app.narrate import cpa_tcpa, relative_bearing, SPEED_CHANGE_INCREMENT_MPS
from app.units import nm_to_m
# pipeline.nomoto is the single shared home for this physics (also used by
# pipeline.oow_agent_spec.derive_risk_horizon_s_nomoto()) -- importing app.narrate above
# already inserted REPO_ROOT onto sys.path, so this resolves regardless of caller cwd.
from pipeline.nomoto import NomotoParams, NomotoState, advance as nomoto_advance

# Matches evaluate_run.py's default collision_radius_m -- keep in sync.
COLLISION_RADIUS_M = 15.0
# 0.1 NM: generous relative to one simulation step's own travel distance (~60-85m at
# Sawada-scale 12/8.4kt speeds and dt=10s) yet still small (~1.5%) relative to the 12NM
# transit -- a fixed 25m tolerance (tuned for the old, much slower/shorter missions) was
# tighter than a single step could reliably land inside, confirmed on Imazu01/v0_base:
# closest recorded sample was 48m, never <=25m, despite genuinely passing the goal.
GOAL_RADIUS_M = nm_to_m(0.1)


@dataclass
class VesselConstraints:
    """Own-ship's physical performance envelope -- fed by the sidebar's Ship
    performance/Mission/Simulation inputs (see streamlit_app.py's
    build_vessel_constraints()), never hardcoded elsewhere. Defaults here match
    those widgets' own `value=` so a Simulation built without an explicit
    constraints arg (e.g. in a script/test) still behaves sensibly.

    max_rudder_angle_deg is NO LONGER a per-command cap (2026-09-24 change) -- a single
    turn_left/turn_right command sets a persistent target_heading of any size, and
    turn_rate_deg_s (3 deg/s) is the ONLY physical limiter, exactly like a real vessel's
    continuous rudder command (matches Sawada et al.'s agent, which never one-shot-snaps
    to a new heading either). This field is now used only (a) as the reference angle
    derive_risk_horizon_s() sizes its manoeuvre-time estimate around, and (b) app/
    measurement.py's Check C, which became a >120 deg SANITY bound (a suspiciously large
    single order) rather than a physical-impossibility check.
    """
    max_speed_mps: float = 10.0
    max_rudder_angle_deg: float = 30.0
    # 0.01/0.02 kt/s (large commercial vessels accelerate/crash-stop far slower than the
    # old 0.2 m/s² -- that reached full speed in under a minute).
    max_acceleration_mps2: float = 0.005
    max_deceleration_mps2: float = 0.01
    turn_rate_deg_s: float = 3.0
    cruise_speed_mps: float = 10.0
    min_cpa_m: float = 500.0
    time_step_s: float = 10.0

    # "kinematics" (default) = the turn-rate slew model above, unchanged. "nomoto" =
    # Sawada et al. (2021)'s 2nd-order Nomoto + rudder-servo model (app/nomoto.py) --
    # opt-in only, own-ship heading dynamics ONLY (speed still uses max_acceleration_mps2/
    # max_deceleration_mps2 above either way). Any value other than "nomoto" falls back
    # to "kinematics", so an unrecognised/typo'd value never silently changes physics.
    kinematics_model: str = "kinematics"
    nomoto_K_per_s: float = 0.05
    nomoto_T_s: float = 50.0
    nomoto_T_E_s: float = 2.5
    nomoto_rudder_limit_deg: float = 10.0
    nomoto_autopilot_kp: float = 1.0
    nomoto_substep_s: float = 1.0


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
        # Only consulted when constraints.kinematics_model == "nomoto" -- see
        # _advance_own_kinematics_nomoto(). Kept even when unused so switching models
        # mid-run (e.g. a UI toggle) doesn't need a fresh Simulation instance.
        self._nomoto_state = NomotoState(rudder_deg=0.0, yaw_rate_deg_s=0.0, heading_deg=self.own.heading)
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
        if self.constraints.kinematics_model == "nomoto":
            self._advance_own_kinematics_nomoto(dt)
            return
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

    def _advance_own_kinematics_nomoto(self, dt: float) -> None:
        """Sawada et al. (2021)'s Nomoto + rudder-servo heading dynamics (app/nomoto.py),
        opt-in via constraints.kinematics_model=="nomoto". Speed still uses the SAME
        accel/decel rate-limiting as the legacy "kinematics" model above -- Nomoto only
        replaces the heading/rudder dynamics, not the speed dynamics."""
        c = self.constraints
        params = NomotoParams(K_per_s=c.nomoto_K_per_s, T_s=c.nomoto_T_s, T_E_s=c.nomoto_T_E_s,
                              rudder_limit_deg=c.nomoto_rudder_limit_deg,
                              autopilot_kp=c.nomoto_autopilot_kp)
        self._nomoto_state = nomoto_advance(self._nomoto_state, self.target_heading, params,
                                           dt, substep_s=c.nomoto_substep_s)
        self.own.heading = self._nomoto_state.heading_deg

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
    # hasn't visibly moved yet. No upper clamp here (2026-09-24) -- a turn_right 60
    # sets target_heading = current target + 60 and the sim keeps swinging toward it
    # across as many steps as turn_rate_deg_s needs; only a negative request is rejected.
    def turn_left(self, degrees: float = 10.0) -> None:
        degrees = max(0.0, degrees)
        self.target_heading = (self.target_heading - degrees) % 360

    def turn_right(self, degrees: float = 10.0) -> None:
        degrees = max(0.0, degrees)
        self.target_heading = (self.target_heading + degrees) % 360

    def set_speed(self, new_speed: float) -> None:
        self.target_speed = max(0.0, min(new_speed, self.constraints.max_speed_mps))

    def speed_up(self, delta: float = SPEED_CHANGE_INCREMENT_MPS) -> None:
        self.set_speed(self.target_speed + delta)

    def slow_down(self, delta: float = SPEED_CHANGE_INCREMENT_MPS) -> None:
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

    def reached_goal(self, radius: float = GOAL_RADIUS_M) -> bool:
        """Checks the closest approach WITHIN the last recorded step, not just the current
        instantaneous position -- Sawada-scale Imazu missions move ~60-85m per 10s step
        (more than this radius), so a fast vessel can pass within `radius` of the goal
        strictly BETWEEN two recorded samples without either endpoint's own instantaneous
        distance ever registering it. Same class of bug find_collision()/_segment_min_range
        already accounts for; reused here with the goal as a stationary "target".
        Confirmed on Imazu01/v0_base: closest recorded SAMPLE was 48m (never <=25m), yet the
        run overshot the goal and looped away instead of ever finishing -- interpolating the
        step it happened in shows the ship actually passed within radius."""
        goal = self.mission.goal
        own_rows = [r for r in self.trajectory if r["vehicle"] == "own_ship"]
        if len(own_rows) < 2:
            return math.hypot(self.own.x - goal[0], self.own.y - goal[1]) <= radius
        prev, cur = own_rows[-2], own_rows[-1]
        rng, _ = _segment_min_range(prev["time"], (prev["x"], prev["y"]), goal,
                                    cur["time"], (cur["x"], cur["y"]), goal)
        return rng <= radius

    def min_cpa_now(self) -> float:
        """Closest CURRENT range to any target -- a quick, cheap, instantaneous safety
        readout. Deliberately NOT predictive (no forward projection): an earlier version
        of this projected forward using each vessel's current heading/speed to catch a
        collision before it happened, but that could "predict" a collision that then never
        gets recorded at all -- the simulation loop would stop BEFORE the predicted moment,
        so the saved trajectory never shows the vessels actually getting close, while
        outcome still said "collision" (confirmed on Imazu01/bare_qwen: outcome=collision,
        but the saved trajectory's last recorded distance was 260m -- the loop broke 3
        steps early based on a projection that was never verified against what actually
        happened). See run_llm_scenario.py's loop: it now checks collision AFTER each step,
        against the trajectory actually recorded so far, using find_collision() below."""
        if not self.targets:
            return float("inf")
        return min(math.hypot(self.own.x - t.x, self.own.y - t.y) for t in self.targets)


def _segment_min_range(t0: float, o0: tuple[float, float], tg0: tuple[float, float],
                       t1: float, o1: tuple[float, float], tg1: tuple[float, float]
                       ) -> tuple[float, float]:
    """Closest own-ship/target range WITHIN [t0, t1], assuming each vessel moves in a
    straight line between its two recorded endpoints (true for this project's kinematics --
    step() moves every vessel at a constant heading/speed for the whole dt). Returns
    (min_range_m, time_of_min_range) -- reconstructing velocity from the position delta
    rather than trusting the recorded heading/speed fields avoids any ambiguity about
    whether a row's heading/speed reflects the value AT t0 or the (already rate-limited,
    already-applied) value used to reach t1."""
    dt = t1 - t0
    if dt <= 0:
        return math.hypot(o0[0] - tg0[0], o0[1] - tg0[1]), t0
    vox, voy = (o1[0] - o0[0]) / dt, (o1[1] - o0[1]) / dt
    vtx, vty = (tg1[0] - tg0[0]) / dt, (tg1[1] - tg0[1]) / dt
    dx, dy = tg0[0] - o0[0], tg0[1] - o0[1]
    dvx, dvy = vtx - vox, vty - voy
    rel_sq = dvx ** 2 + dvy ** 2
    if rel_sq < 1e-9:
        return math.hypot(dx, dy), t0
    # Standard CPA formula gives t* directly in SECONDS elapsed from t0 (not a [0,1]
    # fraction of the interval) since dvx/dvy are already true m/s velocities -- clamp to
    # the segment's own [0, dt] bound, not [0, 1].
    t_star = max(0.0, min(dt, -(dx * dvx + dy * dvy) / rel_sq))
    rng = math.hypot(dx + dvx * t_star, dy + dvy * t_star)
    return rng, t0 + t_star


def find_collision(trajectory: list[dict], radius: float = COLLISION_RADIUS_M) -> dict | None:
    """Scans a recorded trajectory for the first time any target comes within `radius`
    metres of own_ship -- checking not just the recorded sample instants but the CLOSEST
    APPROACH WITHIN each consecutive pair of samples (see _segment_min_range), since two
    fast-closing vessels (e.g. a 20 m/s head-on Imazu encounter) can pass each other
    entirely between two dt=10s samples without either endpoint ever recording a
    within-radius distance -- confirmed on Imazu01/v1_rag: sampled distances 260.8m (t=20s)
    then 79.2m (t=30s), yet evaluate_run.py's interpolated min_cpa_over_run() found 11.2m
    (an actual collision) strictly between those two samples. Returns
    {"time","vehicle","x","y","range_m"} for the earliest such moment, or None if the run
    never got that close."""
    by_time: dict[float, dict[str, dict]] = defaultdict(dict)
    for row in trajectory:
        by_time[row["time"]][row["vehicle"]] = row
    times = sorted(by_time)
    target_names = {name for row in by_time.values() for name in row if name != "own_ship"}
    for name in sorted(target_names):
        prev_t, prev_o, prev_tg = None, None, None
        for t in times:
            own_row, tgt_row = by_time[t].get("own_ship"), by_time[t].get(name)
            if not own_row or not tgt_row:
                continue
            o, tg = (own_row["x"], own_row["y"]), (tgt_row["x"], tgt_row["y"])
            if prev_t is not None:
                rng, rng_t = _segment_min_range(prev_t, prev_o, prev_tg, t, o, tg)
                if rng < radius:
                    frac = 0.0 if t == prev_t else (rng_t - prev_t) / (t - prev_t)
                    x = prev_o[0] + frac * (o[0] - prev_o[0])
                    y = prev_o[1] + frac * (o[1] - prev_o[1])
                    return {"time": rng_t, "vehicle": name, "x": x, "y": y, "range_m": rng}
            prev_t, prev_o, prev_tg = t, o, tg
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

