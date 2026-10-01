"""Generator: Data/Captain/Scenarios/generated/*.json -- procedurally generated Captain
missions (design_captain_missions.md Sec 13.C.9), beyond the 4 hand-authored walking-
skeleton scenarios. Same fictional stub_corridor_v1 region/ports (Data/Captain/Regions/)
and the exact Sec 13.C.13 scenario schema CaptainSkeleton.from_scenario_file() already
reads -- reused unchanged, no new mechanism to maintain, so every generated file is
immediately runnable via run_captain_scenario.py / the Captain Mission Streamlit page.

Deterministic from (seed, index): same args always produce byte-identical output.
Local-safe: pure JSON generation, no GPU/API key needed.

Run with:
    python generate_captain_missions.py --n 20 --seed 100
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.mission_route_planner import minimum_resource_route  # noqa: E402
from app.mission_sim import haversine_nm  # noqa: E402
from pipeline.captain_types import ExclusionZone  # noqa: E402

OUT_DIR = ROOT.parent / "Data" / "Captain" / "Scenarios" / "generated"

# Sec 13.C.9's "whole route templates held out" rule -- "long" is NEVER used for a
# training mission regardless of which events/severities get attached to it, the
# strongest single held-out guarantee. All templates share stub_corridor_v1's own
# region/ports (Data/Captain/Regions/stub_corridor_v1.json), so refuge/port lookups stay
# meaningful. "short"/"medium"/"long" are straight legs on one meridian (varying length
# only); "coastal_bend"/"zigzag" genuinely turn through several bearings -- real geometric
# variety for the mission-preview plot and for leg-bearing-dependent code
# (current_leg_bearing_deg, ambient-contact seeding) to actually be exercised.
ROUTE_TEMPLATES: dict[str, list[tuple[float, float]]] = {
    "short":        [(52.0, 3.0), (53.5, 3.0)],
    "medium":       [(52.0, 3.0), (54.0, 3.0), (56.0, 3.0)],
    "long":         [(52.0, 3.0), (54.0, 3.0), (56.0, 3.0), (58.5, 3.0)],
    "coastal_bend": [(52.0, 3.0), (53.5, 3.0), (54.5, 4.0), (56.0, 4.5)],
    "zigzag":       [(52.0, 3.0), (53.0, 3.8), (54.0, 3.0), (55.0, 3.8), (56.0, 3.0)],
}
HELD_OUT_TEMPLATES = {"long"}

ZONE_PROBABILITY = 0.4  # fraction of generated missions that get an exclusion zone to route around

EVENT_WEIGHTS = {
    "engine_failure": 0.30, "fog": 0.20, "whale_zone": 0.15,
    "distress_call": 0.20, "commercial_instruction": 0.15,
}
SEVERITIES = ["minor", "moderate", "serious", "catastrophic"]
SEVERITY_WEIGHTS = [0.35, 0.35, 0.20, 0.10]

VESSEL_NAMES = ["MV Example Trader", "MV Northern Carrier", "MV Atlantic Runner",
               "MV Pacific Voyager", "MV Southern Clipper", "MV Baltic Courier"]
WEATHER = ["calm seas, light wind", "moderate swell, overcast", "fresh breeze, scattered showers",
          "fog bank expected mid-route", "clear skies, following sea"]
ENGINE_FAULTS = ["main engine overheating", "turbocharger failure", "fuel injection fault",
                "propeller shaft vibration", "cooling-water pump failure"]
DISTRESS_RESOLUTIONS = ["survivors_recovered", "rescued_by_other_vessel", "false_alarm_stood_down"]
COMMERCIAL_RESOLUTIONS = ["company_accepts_refusal", "company_insists_citing_contract",
                          "company_withdraws_request"]
CONTEXT_FLAGS = {
    "engine_failure": ["deadline_pressure"], "fog": ["restricted_visibility"],
    "whale_zone": ["charted_hazard"], "distress_call": ["via_vhf"],
    "commercial_instruction": ["schedule_pressure"],
}

SHIP_PROFILE = "sawada2021"
ENGINE_MAX_SPEED_KN = 14.0  # sawada2021's own max_speed_kn (pipeline.nomoto.SHIP_PROFILES)


def _engine_failure_params(rnd: random.Random, severity: str, original_soa_kn: float) -> dict:
    cap_by_severity = {"minor": 10.0, "moderate": 8.0, "serious": 5.0, "catastrophic": 3.0}
    repair_by_severity = {"minor": (8.0, 16.0), "moderate": (16.0, 30.0),
                          "serious": (30.0, 50.0), "catastrophic": (50.0, 80.0)}
    capped = min(cap_by_severity[severity] + rnd.uniform(-1.0, 1.0), original_soa_kn - 1.0)
    capped = round(max(1.0, capped), 1)
    lo, hi = repair_by_severity[severity]
    return {"capped_speed_kn": capped, "repair_duration_h": round(rnd.uniform(lo, hi), 1),
           "fault": f"{rnd.choice(ENGINE_FAULTS)} -- Chief Engineer declares a {capped}kn safe-speed cap"}


def _fog_params(rnd: random.Random, severity: str) -> dict:
    speed_by_severity = {"minor": 8.0, "moderate": 6.0, "serious": 4.0, "catastrophic": 2.0}
    return {"safe_speed_kn": speed_by_severity[severity],
           "affected_distance_nm": round(rnd.uniform(15.0, 40.0), 1)}


def _whale_zone_params(rnd: random.Random, severity: str) -> dict:
    limit_by_severity = {"minor": 10.0, "moderate": 8.0, "serious": 6.0, "catastrophic": 4.0}
    return {"speed_limit_kn": limit_by_severity[severity],
           "affected_distance_nm": round(rnd.uniform(15.0, 35.0), 1)}


def _distress_call_params(rnd: random.Random, severity: str) -> dict:
    detour_by_severity = {"minor": (5.0, 10.0), "moderate": (10.0, 20.0),
                          "serious": (20.0, 35.0), "catastrophic": (35.0, 60.0)}
    lo, hi = detour_by_severity[severity]
    return {"detour_distance_nm": round(rnd.uniform(lo, hi), 1)}


def _commercial_instruction_params(rnd: random.Random, severity: str, original_soa_kn: float) -> dict:
    delta_by_severity = {"minor": 1.0, "moderate": 2.0, "serious": 4.0, "catastrophic": 6.0}
    demanded = original_soa_kn + delta_by_severity[severity] + rnd.uniform(-0.3, 0.3)
    return {"demanded_speed_kn": round(demanded, 1), "engine_max_speed_kn": ENGINE_MAX_SPEED_KN}


PARAM_BUILDERS = {
    "engine_failure": _engine_failure_params,
    "fog": lambda rnd, sev, soa: _fog_params(rnd, sev),
    "whale_zone": lambda rnd, sev, soa: _whale_zone_params(rnd, sev),
    "distress_call": lambda rnd, sev, soa: _distress_call_params(rnd, sev),
    "commercial_instruction": _commercial_instruction_params,
}


def _world_responder(rnd: random.Random, event_type: str) -> dict | None:
    """A fixed, scripted world-responder reply (Sec 13.A.2) -- only distress_call and
    commercial_instruction get one; the resolution TEXT is cosmetic/documentation only
    (app.captain_skeleton._handle_world_responder() doesn't branch on its value)."""
    if event_type == "distress_call":
        return {"type": "timed_resolution", "delay_s": round(rnd.uniform(1800, 7200)),
               "resolution": rnd.choice(DISTRESS_RESOLUTIONS)}
    if event_type == "commercial_instruction":
        return {"type": "timed_resolution", "delay_s": round(rnd.uniform(900, 3600)),
               "resolution": rnd.choice(COMMERCIAL_RESOLUTIONS)}
    return None


def _trigger_bucket(distance_nm: float, total_distance_nm: float) -> str:
    """Sec 13.C.9's bucketed trigger placement -- coarse early/mid/late leg, the dimension
    that (together with route_template_id/event_set/severity_set) makes the held-out key
    meaningful at all."""
    frac = distance_nm / total_distance_nm
    return "early" if frac < 1 / 3 else ("mid" if frac < 2 / 3 else "late")


def _zone_across_leg(rnd: random.Random, p1: tuple[float, float], p2: tuple[float, float],
                     leg_distance_nm: float, zone_id: str) -> ExclusionZone:
    """A square 'dynamic_hazard' zone centred on leg (p1, p2)'s own midpoint (Sec 7's
    storm-cell/piracy-corridor/whale-protection convention) -- half-width capped well
    below half the leg length, so the straight leg is genuinely blocked (the midpoint
    always lies inside its own zone) while both endpoints stay clear of it, guaranteeing
    `plan_route()` finds a real detour rather than raising 'no path found'."""
    mid_lat, mid_lon = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    half_width_nm = min(leg_distance_nm * 0.25, 20.0)
    half_lat = half_width_nm / 60.0  # ~60 nm per degree of latitude
    half_lon = half_width_nm / 60.0 / max(0.1, math.cos(math.radians(mid_lat)))
    polygon = [(mid_lat - half_lat, mid_lon - half_lon), (mid_lat - half_lat, mid_lon + half_lon),
              (mid_lat + half_lat, mid_lon + half_lon), (mid_lat + half_lat, mid_lon - half_lon)]
    return ExclusionZone(id=zone_id, type="dynamic_hazard", polygon=polygon)


def _zone_to_dict(zone: ExclusionZone) -> dict:
    """Serializes an ExclusionZone into the exact Sec 13.C.13 `route.exclusion_zones` row
    shape `pipeline.captain_types.zones_from_region()` already parses."""
    return {"id": zone.id, "type": zone.type, "polygon": [list(p) for p in zone.polygon],
           "speed_limit_kn": zone.speed_limit_kn, "min_depth_m": zone.min_depth_m}


def generate_mission(index: int, seed: int) -> dict:
    """One fully self-contained Sec 13.C.13 scenario file, deterministic from (seed, index)."""
    rnd = random.Random(f"{seed}::{index}")
    template_name = rnd.choice(list(ROUTE_TEMPLATES))
    intent_waypoints = ROUTE_TEMPLATES[template_name]
    held_out = template_name in HELD_OUT_TEMPLATES

    # With ZONE_PROBABILITY, one randomly-chosen leg gets a 'dynamic_hazard' zone placed
    # squarely across it -- minimum_resource_route() (Sec 13.B.8's own visibility-graph
    # planner) then bends ONLY that leg around it, exactly as if the OOW's own route
    # planning had already routed around a known hazard (Sec 7's stated convention) --
    # this is the first time any generated mission actually exercises that planner.
    zones: list[ExclusionZone] = []
    if len(intent_waypoints) >= 2 and rnd.random() < ZONE_PROBABILITY:
        leg_i = rnd.randrange(len(intent_waypoints) - 1)
        p1, p2 = intent_waypoints[leg_i], intent_waypoints[leg_i + 1]
        zones = [_zone_across_leg(rnd, p1, p2, haversine_nm(p1, p2), zone_id="hazard1")]

    planned = minimum_resource_route(intent_waypoints, zones)
    waypoints = planned.path
    distance_nm = planned.distance_nm

    original_soa_kn = round(rnd.uniform(10.0, 13.0), 1)
    nominal_time_h = distance_nm / original_soa_kn
    eta_deadline_h = round(nominal_time_h * rnd.uniform(1.15, 1.6), 1)
    fuel_tonnes = round(distance_nm * 3.5, 1)  # matches the hand-authored ~3.54 t/nm ratio

    # Sec 13.C.9's own density rule: ~1 brown envelope per 100nm, never the same type
    # twice in one mission (keeps speed-cap-stacking bookkeeping unambiguous for this v1
    # generator -- the skeleton itself DOES support stacking, see skeleton_fog_whale_zone_v1,
    # just not exercised here), triggers spaced into non-overlapping slots along the route.
    margin_nm = max(10.0, distance_nm * 0.08)
    min_gap_nm = max(15.0, distance_nm * 0.12)
    usable_span = distance_nm - 2 * margin_nm
    max_events_by_spacing = max(1, int(usable_span // min_gap_nm) + 1)
    n_events = max(1, min(4, round(distance_nm / 100.0) + rnd.randint(-1, 1), max_events_by_spacing))

    event_types = list(EVENT_WEIGHTS)
    rnd.shuffle(event_types)
    chosen_types = event_types[:n_events]

    slot_width = usable_span / n_events
    envelopes, buckets = [], []
    for i, event_type in enumerate(chosen_types):
        slot_lo = margin_nm + i * slot_width
        slot_hi = margin_nm + (i + 1) * slot_width
        trigger_nm = round(rnd.uniform(slot_lo, slot_hi), 1)
        severity = rnd.choices(SEVERITIES, weights=SEVERITY_WEIGHTS, k=1)[0]
        params = PARAM_BUILDERS[event_type](rnd, severity, original_soa_kn)
        envelopes.append({
            "event_id": f"ev{i + 1}", "type": event_type, "severity": severity,
            "context_flags": CONTEXT_FLAGS[event_type],
            "trigger": {"type": "distance_along_route_nm", "value": trigger_nm},
            "world_responder": _world_responder(rnd, event_type),
            "params": params,
        })
        buckets.append(_trigger_bucket(trigger_nm, distance_nm))

    mission_id = f"MSN-GEN-{seed:04d}-{index:04d}"
    zone_note = f", routed around zone {zones[0].id!r} on leg {leg_i}" if zones else ""
    return {
        "mission_id": mission_id,
        "seeds": {"mission_seed": seed * 10_000 + index, "responder_seed": seed * 10_000 + index},
        "held_out": held_out,
        "route_template": template_name,  # Sec 13.C.9's own held-out-key dimension, kept
                                          # explicit rather than re-inferred from geometry
                                          # (a zone can bend the stored path away from the
                                          # raw template's own waypoint list)
        "failure_category": "+".join(sorted(chosen_types)),
        "_comment": f"Procedurally generated (generate_captain_missions.py, seed={seed}, "
                   f"index={index}) -- route_template={template_name!r}{zone_note}, "
                   f"trigger_placement_buckets={buckets}. Same fictional stub_corridor_v1 "
                   f"region as the hand-authored skeleton scenarios, not for navigation use.",
        "mission_order": {
            "issued_by": "Fleet Operations Centre", "issued_at": "2026-10-01T00:00:00Z",
            "vessel": rnd.choice(VESSEL_NAMES), "t0_utc": "2026-10-01T00:00:00Z", "dt_mission_s": 600.0,
            "situation": {"weather_forecast": rnd.choice(WEATHER)},
            "goal": "Deliver cargo from the corridor's southern waypoint to its northern waypoint",
            "success_criteria": ["cargo delivered intact", "arrive within the stated ETA deadline"],
            "speed_of_advance_kn": original_soa_kn, "restricted_zones": [],
            "rules_of_conduct": "COLREG 1972 + company SMS",
            "admin_logistics": {
                "resources": {"fuel_tonnes": fuel_tonnes, "fuel_reserve_margin_pct": 15.0,
                              "draft_m": 7.5, "ship_profile": SHIP_PROFILE},
                "eta_deadline_h": eta_deadline_h,
            },
            "command_signal": {"reporting_interval_hours": 6.0},
        },
        "route": {"region_id": "stub_corridor_v1", "waypoints": [list(p) for p in waypoints],
                  "exclusion_zones": [_zone_to_dict(z) for z in zones]},
        "brown_envelopes": envelopes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate procedural Captain mission scenarios "
                                             "(design_captain_missions.md Sec 13.C.9).")
    ap.add_argument("--n", type=int, default=20, help="number of missions to generate")
    ap.add_argument("--seed", type=int, default=100,
                   help="base seed -- each mission is reproducible from (seed, index)")
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR),
                   help="output directory (default: Data/Captain/Scenarios/generated/)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_held_out = 0
    for i in range(args.n):
        mission = generate_mission(i, args.seed)
        n_held_out += mission["held_out"]
        (out_dir / f"{mission['mission_id']}.json").write_text(
            json.dumps(mission, indent=2), encoding="utf-8")
    print(f"Wrote {args.n} missions to {out_dir} ({n_held_out} held-out, "
         f"{args.n - n_held_out} usable for training).")


if __name__ == "__main__":
    main()
