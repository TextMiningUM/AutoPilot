"""app/oracle_planner.py -- Stap 2 Step 2: a formalized oracle-planner.

Candidate plans (the 6 app.baselines.* decide() functions' own first-step turn choice,
PLUS a coarse turn-offset grid -- mirrors app/baselines/mpc.py's own candidate scan) are
each rolled forward under REAL Nomoto + rudder-servo physics (pipeline.nomoto, NOT the
legacy turn-rate slew mpc.py's own internal rollout still uses), THEN filtered by a HARD
COLREG rule-legality gate before cost-ranking the survivors -- mpc.py only ever applies a
SOFT cost term for clearance, never a hard "this candidate is COLREG-illegal, exclude it
outright" gate. Always rolls out under Nomoto regardless of the live
VesselConstraints.kinematics_model flag (this oracle exists specifically to label data
for a Nomoto-aware fine-tune, see repo memory's "Stap 2 plan" note) -- every
VesselConstraints instance always carries its nomoto_* fields (Sawada defaults) even when
kinematics_model=="kinematics", so this needs no special live-constraints wiring.

Lives under Basic Simulator/app/ (NOT pipeline/) because every direct dependency here is
an app.* module -- pipeline/ must never depend on Basic Simulator/app/, only the reverse
(app/simulation.py, app/baselines/ruletree.py etc. already import pipeline.nomoto/
pipeline.oow_agent_spec directly, the established, correct dependency direction).

Deliberately scoped to TURN decisions only, matching app/baselines/mpc.py's own scope --
none of the six baselines' internal rollout mechanism reasons about speed_up/slow_down/
stop either. A speed-action oracle is a possible future extension, not built here.

The rule-legality gate mirrors pipeline/track2/build_outcome_dpo.py's own
expected_acute_response() conservatism (a pure Rule-13 overtaking encounter has NO
mandated side and is left unconstrained) -- NOT app/baselines/ruletree.py's own
simplification, which always defaults to turn_right even for overtaking (a known,
documented oversimplification of that one baseline, not the ground truth to filter by).

Captain walking-skeleton Phase 9d (design_captain_missions.md Sec 13.A.1 point 5): `plan()`
gained an OPTIONAL `zones` parameter -- a candidate whose own rollout path crosses one of
these (already-projected-to-the-live-local-frame, see `app/captain_skeleton.py`) exclusion
zones is excluded by the SAME hard-gate mechanism as a COLREG-illegal candidate, not a
separate one. `zones=None` (every pre-existing caller, OOW/VHF missions included) makes
this a pure no-op -- zero behaviour change for anything that doesn't pass zones.
"""
from __future__ import annotations
import math
from dataclasses import replace

from app.baselines import DECISION_FUNCS
from app.baselines._rule_citation import infer_rule_citation
from app.baselines import ruletree
from app.mission_route_planner import segment_blocked
from app.missions import Mission, Vessel
from app.narrate import contact_line
from app.simulation import VesselConstraints
from pipeline.captain_types import ExclusionZone
from pipeline.nomoto import NomotoParams, NomotoState, advance as nomoto_advance
from pipeline.oow_agent_spec import bearing_and_range, derive_risk_horizon_s, real_risk, relative_bearing, risk_band

HORIZON_STEPS = 6
STEP_S = 10.0
HEADING_DEADBAND_DEG = 3.0
_GRID_OFFSETS_DEG = list(range(-90, 91, 15))
# Same weights as app/baselines/mpc.py's own cost function, reused verbatim for direct
# comparability between the two (see that module's W_EFFORT-vs-W_GOAL calibration note --
# the same reasoning applies unchanged here).
W_GOAL = 2.0
W_CLEARANCE = 5.0
W_EFFORT = 0.02

# classify_encounter()'s return strings -> classify_rules()'s role vocabulary -- same
# mapping as app/baselines/ruletree.py / _rule_citation.py.
_ENCOUNTER_TO_ROLE = {
    "head_on": "mutual",
    "crossing_target_on_starboard": "give_way",
    "crossing_target_on_port": "stand_on",
    "we_are_overtaking_target": "overtaking_give_way",
    "target_is_overtaking_us": "overtaking_stand_on",
}


def _decisive_contact(own: Vessel, targets: list[Vessel], constraints: VesselConstraints) -> dict | None:
    """The most urgent (soonest-TCPA) real-risk contact -- same selection rule as
    app/baselines/_rule_citation.py's decisive_contact(), reused here as a plain dict
    (not re-imported) since this module also needs risk_band() on the same inputs."""
    safe_distance_m, max_turn_deg = constraints.min_cpa_m, constraints.max_rudder_angle_deg
    risk_horizon_s = derive_risk_horizon_s(safe_distance_m, max_turn_deg, own.speed)
    contacts = [contact_line(own, t, safe_distance_m, max_turn_deg) for t in targets]
    at_risk = [c for c in contacts if real_risk(c["cpa_m"], c["tcpa_s"], safe_distance_m, risk_horizon_s)]
    at_risk.sort(key=lambda c: (c["tcpa_s"], c["cpa_m"]))
    return at_risk[0] if at_risk else None


def required_direction(own: Vessel, targets: list[Vessel], constraints: VesselConstraints) -> str | None:
    """'starboard'/'port'/'hold'/None (no hard constraint -- either an ambiguous
    encounter, or no real risk at all, so GOAL COURSE CHECK governs instead) for the
    CURRENT decisive contact. This is the oracle-planner's rule filter's entire ground
    truth -- see module docstring for why it deliberately does NOT reuse
    app/baselines/ruletree.py's own always-starboard-for-overtaking simplification."""
    decisive = _decisive_contact(own, targets, constraints)
    if decisive is None:
        return None
    safe_distance_m, max_turn_deg = constraints.min_cpa_m, constraints.max_rudder_angle_deg
    risk_horizon_s = derive_risk_horizon_s(safe_distance_m, max_turn_deg, own.speed)
    band = risk_band(decisive["cpa_m"], decisive["tcpa_s"], safe_distance_m, risk_horizon_s)
    role = _ENCOUNTER_TO_ROLE[decisive["encounter"]]
    if role in ("stand_on", "overtaking_stand_on"):
        if band == "acute" and decisive["stand_on_deadline_passed"]:
            # Rule 17(a)(ii)/(b): overdue emergency escalation -- turn AWAY from the contact.
            return "starboard" if decisive["rel_bearing_deg"] < 0 else "port"
        # Not yet past the deadline -- ANY turn (either direction) is the exact B_17c
        # "Stand-On Vessel Acted Too Early" compliance violation (confirmed the dominant
        # real failure mode on IMP13 in the 2026-09-29 audit) -- only holding is legal.
        return "hold" if band == "acute" else None
    if role in ("mutual", "give_way") and band == "acute":
        return "starboard"  # COLREG's mandatory default for these two roles.
    # "early" band (not yet mandatory) or overtaking_give_way (Rule 13 -- no mandated
    # side, matches build_outcome_dpo.py's own "either" ambiguity) -- unconstrained.
    return None


def _offset_direction(offset_deg: float) -> str:
    if abs(offset_deg) <= HEADING_DEADBAND_DEG:
        return "hold"
    return "starboard" if offset_deg > 0 else "port"


def _advance_straight(v: Vessel, dt: float) -> Vessel:
    h = math.radians(v.heading)
    return replace(v, x=v.x + v.speed * math.sin(h) * dt, y=v.y + v.speed * math.cos(h) * dt)


def _rollout_cost(mission: Mission, own0: Vessel, first_offset: float, targets: list[Vessel],
                  constraints: VesselConstraints, zones: list[ExclusionZone] | None = None
                  ) -> tuple[float, float, bool]:
    """(cost, min_clearance, zone_blocked) for committing to a target heading of
    own0.heading + first_offset as the first control, then letting ruletree.decide_action()
    adjust that target heading further for the rest of the horizon (same internal-base-
    policy pattern as app/baselines/mpc.py's own _rollout_cost) -- but own-ship's heading
    here is advanced via REAL Nomoto + rudder-servo dynamics (pipeline.nomoto.advance()),
    not mpc.py's legacy rate-limited-slew model. Lower cost is better. `zone_blocked`
    (Sec 13.A.1 point 5, Phase 9d) is True iff OWN-SHIP's own rolled-out path crosses any
    of `zones` at any step -- `zones=None`/empty is a no-op, always False."""
    params = NomotoParams(K_per_s=constraints.nomoto_K_per_s, T_s=constraints.nomoto_T_s,
                          T_E_s=constraints.nomoto_T_E_s, rudder_limit_deg=constraints.nomoto_rudder_limit_deg,
                          autopilot_kp=constraints.nomoto_autopilot_kp)
    state = NomotoState(rudder_deg=0.0, yaw_rate_deg_s=0.0, heading_deg=own0.heading)
    own = replace(own0)
    target_heading = (own0.heading + first_offset) % 360
    virt_targets = [replace(t) for t in targets]
    min_clearance = min((math.hypot(own.x - t.x, own.y - t.y) for t in virt_targets), default=float("inf"))
    goal_brg0, _ = bearing_and_range(own.x, own.y, mission.goal[0], mission.goal[1])
    deviations = [abs(relative_bearing(own.heading, goal_brg0)) / 180.0]
    zone_blocked = False

    for step in range(HORIZON_STEPS):
        state = nomoto_advance(state, target_heading, params, STEP_S, substep_s=constraints.nomoto_substep_s)
        own_before = own
        own = replace(own, heading=state.heading_deg)
        own = _advance_straight(own, STEP_S)
        if zones and segment_blocked((own_before.x, own_before.y), (own.x, own.y), zones):
            zone_blocked = True
        virt_targets = [_advance_straight(t, STEP_S) for t in virt_targets]
        if virt_targets:
            min_clearance = min(min_clearance, min(math.hypot(own.x - t.x, own.y - t.y) for t in virt_targets))
        if step < HORIZON_STEPS - 1:
            decision = ruletree.decide_action(mission, own, virt_targets, constraints)
            if decision["action"] == "turn_left":
                target_heading = (target_heading - decision["degrees"]) % 360
            elif decision["action"] == "turn_right":
                target_heading = (target_heading + decision["degrees"]) % 360
        goal_brg, _ = bearing_and_range(own.x, own.y, mission.goal[0], mission.goal[1])
        deviations.append(abs(relative_bearing(own.heading, goal_brg)) / 180.0)

    goal_deviation = sum(deviations) / len(deviations)
    safe_distance_m = constraints.min_cpa_m
    clearance_penalty = max(0.0, (safe_distance_m - min_clearance) / safe_distance_m) if targets else 0.0
    effort = abs(first_offset) / 180.0
    cost = W_GOAL * goal_deviation + W_CLEARANCE * clearance_penalty + W_EFFORT * effort
    return cost, min_clearance, zone_blocked


def _candidate_offsets(mission: Mission, own: Vessel, targets: list[Vessel],
                       constraints: VesselConstraints) -> list[float]:
    """The 6 baselines' own first-step turn choice (as a signed offset from own.heading)
    + the coarse grid, deduplicated. hold_course/speed-only baseline picks contribute an
    offset of 0.0 (this oracle is turn-scoped, see module docstring)."""
    offsets = set(_GRID_OFFSETS_DEG)
    for name, decide_fn in DECISION_FUNCS.items():
        decision, _debug = decide_fn(mission, own, targets, constraints)
        if decision["action"] == "turn_left":
            offsets.add(round(-decision["degrees"], 1))
        elif decision["action"] == "turn_right":
            offsets.add(round(decision["degrees"], 1))
        else:
            offsets.add(0.0)
    return sorted(offsets)


def plan(mission: Mission, own: Vessel, targets: list[Vessel], constraints: VesselConstraints,
        zones: list[ExclusionZone] | None = None) -> dict:
    """The oracle decision -- {"action","degrees","heading_deg","encounter_rule",
    "conduct_rule","reasoning","rejected_candidates"}. `rejected_candidates` (a list of
    {"offset_deg","cost","illegal"} for every candidate NOT chosen) is additive, meant
    for a future reflection-style critique trace (Stap 2 Step 3/5), never required by any
    existing consumer. `zones` (Sec 13.A.1 point 5, Phase 9d): already-projected-to-the-
    live-local-frame exclusion zones -- a candidate whose own rollout path crosses one is
    excluded by the SAME hard gate as a COLREG-illegal candidate; `zones=None` (every
    pre-existing caller) makes this entirely a no-op."""
    required = required_direction(own, targets, constraints)
    candidates = _candidate_offsets(mission, own, targets, constraints)
    scored = []
    for offset in candidates:
        cost, clearance, zone_blocked = _rollout_cost(mission, own, offset, targets, constraints, zones)
        colreg_illegal = required is not None and _offset_direction(offset) != required
        scored.append({"offset_deg": offset, "cost": cost, "clearance_m": clearance,
                      "illegal": colreg_illegal or zone_blocked, "zone_blocked": zone_blocked})

    legal = [s for s in scored if not s["illegal"]]
    # required_direction() (and an empty/no-op zones) always admits at least one candidate
    # (an exact-0.0 grid point for "hold", or the grid's own +/-90 extremes safely covering
    # "starboard"/"port" -- HEADING_DEADBAND_DEG=3 is far smaller than the grid's 15 deg
    # spacing) -- a genuinely boxed-in ship (every candidate zone-blocked too) is the one
    # realistic way this fallback could actually trigger; kept as a defensive last resort
    # either way, never silently returning no decision at all.
    pool = legal if legal else scored
    best = min(pool, key=lambda s: s["cost"])
    best_offset = best["offset_deg"]

    if abs(best_offset) <= HEADING_DEADBAND_DEG:
        action, degrees = "hold_course", None
    else:
        action = "turn_right" if best_offset > 0 else "turn_left"
        degrees = round(abs(best_offset), 1)

    encounter_rule, conduct_rule, decisive = infer_rule_citation(own, targets, constraints, action)
    n_rejected_illegal = sum(1 for s in scored if s["illegal"])
    if decisive is None:
        reasoning = (f"No contact meets real_risk() -- oracle rollout best offset "
                    f"{best_offset:.0f} deg (cost={best['cost']:.3f}).")
    else:
        reasoning = (f"Oracle: required_direction={required!r}, best LEGAL {HORIZON_STEPS}-step "
                    f"Nomoto rollout commits to a first-step offset of {best_offset:.0f} deg "
                    f"(cost={best['cost']:.3f}, min clearance {best['clearance_m']:.0f}m vs contact "
                    f"{decisive['name']!r}, {n_rejected_illegal}/{len(scored)} candidates excluded "
                    f"as rule-illegal/zone-blocked) -> action={action}.")

    return {
        "action": action, "degrees": degrees,
        "heading_deg": round((own.heading + best_offset) % 360, 1) if action != "hold_course" else None,
        "encounter_rule": encounter_rule, "conduct_rule": conduct_rule, "reasoning": reasoning,
        "rejected_candidates": [s for s in scored if s is not best],
        # Additive (Stap 2 Step 3): the same decisive-contact facts a Track-2 generator's
        # Fase B3 teacher payload needs (name/cpa_m/tcpa_s) -- None when no contact poses
        # a real risk, mirroring build_oow_scenarios_leo.py's own decisive_contact_name.
        "decisive_contact": ({"name": decisive["name"], "cpa_m": decisive["cpa_m"], "tcpa_s": decisive["tcpa_s"]}
                            if decisive is not None else None),
    }
