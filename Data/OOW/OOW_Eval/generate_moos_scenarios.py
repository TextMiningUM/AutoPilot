#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates a set of MOOS-IvP multi-vehicle evaluation missions for testing a
COLREG-compliant collision-avoidance agent, structured like the Imazu-problem
benchmark family (single-target, two-target, three-target buckets of
increasing difficulty).

IMPORTANT — read before running against a real MOOS-IvP install:
This script produces .moos / .bhv / launch.sh files following standard
MOOS-IvP multi-community conventions (MOOSDB, uSimMarine, pMarinePID,
pHelmIvP, pLogger, pMarineViewer, uFldNodeBroker/uFldShoreBroker, pShare).
The vehicle-collision GEOMETRY (start positions/headings/speeds computed so
that, absent any avoidance, own-ship and target(s) genuinely collide) is
computed exactly below and is the most load-bearing, verified part of this
generator. The inter-process *sharing/bridging* block syntax (pShare /
uFldNodeBroker / uFldShoreBroker) varies somewhat across MOOS-IvP releases —
if multi-vehicle sharing doesn't come up cleanly on first launch, diff these
blocks against an official multi-vehicle example mission shipped with your
install (typically under moos-ivp/ivp/missions/ or moos-ivp-extend) and
adjust. BHV_AvoidCollision parameter names should also be checked with
`pHelmIvP --alist` / behavior --help against your installed version before
relying on the defaults given here.
"""
import json, math, os, shutil

OUT_ROOT = "/home/claude/moos_missions/missions"
if os.path.exists(OUT_ROOT):
    shutil.rmtree(OUT_ROOT)
os.makedirs(OUT_ROOT, exist_ok=True)

LAT_ORIGIN = 43.825300
LON_ORIGIN = -70.330400

# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def deg2rad(d):
    return d * math.pi / 180.0

def os_pos(t, v_os, heading_os_deg=0.0):
    """Own-ship position at time t if she runs straight (heading 0 = north)."""
    h = deg2rad(heading_os_deg)
    return (v_os * t * math.sin(h), v_os * t * math.cos(h))

def bearing_range_to_xy(bearing_deg, rng_m, origin=(0.0, 0.0)):
    """Convert a relative bearing (0=north/ahead, clockwise) + range from
    origin into (x,y) local meters."""
    b = deg2rad(bearing_deg)
    return (origin[0] + rng_m * math.sin(b), origin[1] + rng_m * math.cos(b))

def solve_intercept(ts_start, v_ts, v_os, heading_os_deg=0.0):
    """
    Given a target ship starting position and speed, and own-ship starting
    at (0,0) moving in a straight line at v_os along heading_os_deg, solve
    for the smallest positive time T at which the target ship, moving in a
    straight line at speed v_ts, could be at own-ship's position at time T
    (i.e., a genuine collision course if neither vessel avoids). Returns
    (T, heading_ts_deg). Raises ValueError if no real positive solution
    exists (e.g., target too slow to intercept from that geometry).
    """
    xs, ys = ts_start
    h = deg2rad(heading_os_deg)
    # own-ship position at time t: (v_os*t*sin h, v_os*t*cos h)
    # For heading_os=0 this reduces to (0, v_os t); keep general via rotation
    # by solving in a rotated frame aligned with heading_os, then rotating
    # the resulting TS heading back.
    # Rotate TS start into OS-heading-aligned frame (OS heading -> "north")
    xr = xs * math.cos(h) - ys * math.sin(h)
    yr = xs * math.sin(h) + ys * math.cos(h)
    # In rotated frame, OS moves along +y' at speed v_os: OS'(t) = (0, v_os t)
    a = (v_os ** 2 - v_ts ** 2)
    b = -2.0 * yr * v_os
    c = (xr ** 2 + yr ** 2)
    if abs(a) < 1e-9:
        # linear case
        if abs(b) < 1e-9:
            raise ValueError("No intercept solution (degenerate).")
        T = -c / b
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            raise ValueError("No real intercept solution for given speeds/geometry.")
        sq = math.sqrt(disc)
        T1 = (-b + sq) / (2 * a)
        T2 = (-b - sq) / (2 * a)
        cands = [t for t in (T1, T2) if t > 1.0]
        if not cands:
            raise ValueError("No positive intercept time.")
        T = min(cands)
    # OS position (rotated frame) at time T
    os_xr, os_yr = 0.0, v_os * T
    # heading from TS_start (rotated) to OS(T) (rotated), in rotated frame,
    # measured as compass bearing (0=+y', clockwise)
    dx, dy = os_xr - xr, os_yr - yr
    heading_rot = (math.degrees(math.atan2(dx, dy))) % 360.0
    # rotate back to world frame by adding heading_os_deg
    heading_ts = (heading_rot + heading_os_deg) % 360.0
    return T, heading_ts

def compute_target(bearing_from_os_deg, range_m, v_ts, v_os, heading_os_deg=0.0):
    """Full pipeline: given desired initial relative bearing/range of a
    target from own-ship, and target speed, compute target start position,
    target heading, and the resulting time-to-collision (if unavoided)."""
    ts_start = bearing_range_to_xy(bearing_from_os_deg, range_m)
    T, heading_ts = solve_intercept(ts_start, v_ts, v_os, heading_os_deg)
    return {
        "start_x": round(ts_start[0], 1),
        "start_y": round(ts_start[1], 1),
        "heading": round(heading_ts, 1),
        "speed": v_ts,
        "bearing_from_os_deg": bearing_from_os_deg,
        "range_m": range_m,
        "t_collision_s": round(T, 1),
    }

def same_line_target(range_ahead_m, v_ts, v_os, heading_os_deg=0.0, astern=False):
    """Overtaking-family construction: target on the SAME line as own-ship's
    track (dead ahead if astern=False, i.e. own-ship overtakes; dead astern
    if astern=True, i.e. own-ship is overtaken)."""
    sign = -1 if astern else 1
    x, y = bearing_range_to_xy(heading_os_deg if not astern else (heading_os_deg + 180) % 360,
                                range_ahead_m)
    heading_ts = heading_os_deg  # same direction of travel
    # time to collision if neither deviates (only meaningful if speeds differ appropriately)
    # relative closing speed along the line:
    if astern:
        closing = v_ts - v_os  # ts must be faster to catch up from behind
    else:
        closing = v_os - v_ts  # os must be faster to catch up to ts ahead
    t_coll = range_ahead_m / closing if closing > 0 else float("inf")
    return {
        "start_x": round(x, 1),
        "start_y": round(y, 1),
        "heading": round(heading_ts, 1),
        "speed": v_ts,
        "bearing_from_os_deg": heading_os_deg if not astern else (heading_os_deg + 180) % 360,
        "range_m": range_ahead_m,
        "t_collision_s": round(t_coll, 1),
    }

# ---------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------
V_OS = 2.5  # m/s, ~4.9 kn, own-ship nominal speed

scenarios = {}

def add_scenario(sid, name, rule_refs, os_role, description, pass_criteria,
                  targets, mission_len_m=2400):
    scenarios[sid] = {
        "id": sid,
        "name": name,
        "rule_refs": rule_refs,
        "own_ship_role": os_role,
        "description": description,
        "pass_criteria": pass_criteria,
        "v_os": V_OS,
        "mission_len_m": mission_len_m,
        "targets": targets,  # list of dicts from compute_target/same_line_target
    }

# --- Single-target scenarios (S01-S08) ---
add_scenario(
    "s01_head_on", "Head-on", ["Rule 14"], "mutual (both alter to starboard)",
    "Own-ship meets a single target nearly dead ahead on a reciprocal course.",
    ["Own-ship alters course to starboard (not port).",
     "Minimum CPA (closest point of approach) stays above the configured safe-distance threshold.",
     "Own-ship returns to her original track/heading after the target is past and clear."],
    [compute_target(bearing_from_os_deg=4, range_m=1100, v_ts=2.5, v_os=V_OS)],
)

add_scenario(
    "s02_crossing_stbd_fine", "Crossing - target fine on starboard bow",
    ["Rule 15", "Rule 16"], "give-way",
    "Target is on own-ship's starboard bow at a fine angle (~20 deg relative). Own-ship must give way.",
    ["Own-ship is give-way: she alters course (normally to starboard) and/or speed early.",
     "Own-ship avoids crossing ahead of the target if circumstances allow.",
     "Minimum CPA stays above the safe-distance threshold."],
    [compute_target(bearing_from_os_deg=20, range_m=900, v_ts=2.2, v_os=V_OS)],
)

add_scenario(
    "s03_crossing_stbd_broad", "Crossing - target broad on starboard bow",
    ["Rule 15", "Rule 16"], "give-way",
    "Target is broad on own-ship's starboard bow (~65 deg relative). Own-ship must give way.",
    ["Own-ship is give-way: early, substantial action to keep well clear.",
     "Minimum CPA stays above the safe-distance threshold."],
    [compute_target(bearing_from_os_deg=65, range_m=900, v_ts=2.6, v_os=V_OS)],
)

add_scenario(
    "s04_crossing_stbd_abeam", "Crossing - target near starboard beam",
    ["Rule 15", "Rule 16"], "give-way",
    "Target is nearly abeam on own-ship's starboard side (~90 deg relative). Own-ship must give way.",
    ["Own-ship is give-way and passes astern of the target.",
     "Minimum CPA stays above the safe-distance threshold."],
    [compute_target(bearing_from_os_deg=90, range_m=800, v_ts=3.2, v_os=V_OS)],
)

add_scenario(
    "s05_crossing_port_broad", "Crossing - target broad on port bow (own-ship stand-on)",
    ["Rule 15", "Rule 17"], "stand-on",
    "Target is broad on own-ship's port bow (~60 deg relative to port). The target does not "
    "give way (per the classic Imazu-style convention that targets run straight), so own-ship "
    "must hold course/speed under Rule 17(a)(i) and only act under Rule 17(a)(ii)/(b) if the "
    "target clearly fails to act and collision becomes otherwise unavoidable.",
    ["Own-ship holds course and speed for a meaningful initial period (does not manoeuvre prematurely).",
     "If the target never gives way, own-ship eventually takes late avoiding action under Rule 17(b) "
     "rather than colliding.",
     "Minimum CPA stays above the safe-distance threshold."],
    [compute_target(bearing_from_os_deg=300, range_m=900, v_ts=2.2, v_os=V_OS)],
)

add_scenario(
    "s06_crossing_port_fine", "Crossing - target fine on port bow (own-ship stand-on)",
    ["Rule 15", "Rule 17"], "stand-on",
    "Target is fine on own-ship's port bow (~20 deg relative to port). Own-ship is stand-on.",
    ["Own-ship holds course and speed initially.",
     "Own-ship does not alter to port toward the target if she must eventually act (Rule 17(c)).",
     "Minimum CPA stays above the safe-distance threshold."],
    [compute_target(bearing_from_os_deg=340, range_m=900, v_ts=2.2, v_os=V_OS)],
)

add_scenario(
    "s07_overtaking_os_gives_way", "Overtaking - own-ship overtakes a slower target",
    ["Rule 13"], "give-way (overtaking)",
    "A slower target is on own-ship's track ahead of her. Own-ship closes from astern and must "
    "keep clear as the overtaking vessel until finally past and clear.",
    ["Own-ship's initial approach is recognisably from more than 22.5 deg abaft the target's beam.",
     "Own-ship keeps clear of the target throughout the pass (does not cut in front).",
     "Minimum CPA stays above the safe-distance threshold."],
    [same_line_target(range_ahead_m=700, v_ts=1.0, v_os=V_OS, astern=False)],
)

add_scenario(
    "s08_overtaking_os_stands_on", "Overtaking - own-ship is overtaken (stand-on)",
    ["Rule 13", "Rule 17"], "stand-on (being overtaken)",
    "A faster target approaches from astern of own-ship and must keep clear as the overtaking vessel.",
    ["Own-ship holds course and speed while being overtaken, absent evidence the target is not "
     "keeping clear.",
     "Minimum CPA stays above the safe-distance threshold."],
    [same_line_target(range_ahead_m=600, v_ts=3.5, v_os=V_OS, astern=True)],
)

# --- Two-target scenarios (S09-S11) ---
add_scenario(
    "s09_double_crossing_squeeze", "Two-target: crossing from both sides",
    ["Rule 15", "Rule 16", "Rule 17", "Rule 8(d)"], "give-way to TS1 / stand-on to TS2",
    "TS1 crosses from own-ship's starboard bow (give-way) while TS2 simultaneously crosses "
    "from own-ship's port bow (stand-on). Own-ship must satisfy both encounters with one "
    "coherent manoeuvre plan.",
    ["Own-ship's manoeuvre for TS1 (starboard give-way) does not create a new close-quarters "
     "situation with TS2 (Rule 8(d)/8(c)).",
     "Minimum CPA to BOTH targets stays above the safe-distance threshold."],
    [compute_target(bearing_from_os_deg=45, range_m=900, v_ts=2.2, v_os=V_OS),
     compute_target(bearing_from_os_deg=315, range_m=900, v_ts=2.2, v_os=V_OS)],
)

add_scenario(
    "s10_headon_plus_crossing", "Two-target: head-on plus crossing from starboard",
    ["Rule 14", "Rule 15", "Rule 16", "Rule 8(d)"], "mutual with TS1 / give-way to TS2",
    "TS1 is head-on; TS2 crosses from own-ship's starboard bow at the same time. Both encounters "
    "call for a starboard alteration, testing whether a single coherent manoeuvre satisfies both.",
    ["Own-ship alters to starboard (satisfying both Rule 14 and Rule 15/16 simultaneously).",
     "Minimum CPA to BOTH targets stays above the safe-distance threshold."],
    [compute_target(bearing_from_os_deg=0, range_m=1100, v_ts=2.5, v_os=V_OS),
     compute_target(bearing_from_os_deg=60, range_m=900, v_ts=2.3, v_os=V_OS)],
)

add_scenario(
    "s11_overtake_plus_crossing", "Two-target: overtaking plus crossing from starboard",
    ["Rule 13", "Rule 15", "Rule 16"], "give-way to both (different rules)",
    "A slow target is ahead on own-ship's track (overtaking case) while a second target "
    "simultaneously crosses from own-ship's starboard bow (crossing case).",
    ["Own-ship keeps clear of the overtaken target (Rule 13) while also giving way to the "
     "crossing target (Rule 15/16) without conflating the two into a single miscategorised manoeuvre.",
     "Minimum CPA to BOTH targets stays above the safe-distance threshold."],
    [same_line_target(range_ahead_m=700, v_ts=1.0, v_os=V_OS, astern=False),
     compute_target(bearing_from_os_deg=50, range_m=900, v_ts=2.0, v_os=V_OS)],
)

# --- Three-target scenario (S12) ---
add_scenario(
    "s12_converging_cluster", "Three-target: converging cluster",
    ["Rule 14", "Rule 15", "Rule 16", "Rule 17", "Rule 8(d)"],
    "mixed give-way/stand-on across three simultaneous encounters",
    "TS1 crosses from starboard, TS2 crosses from port, and TS3 is head-on, all converging on "
    "own-ship at roughly the same time - the hardest single-agent test in this set, analogous "
    "to the higher-numbered multi-target Imazu cases.",
    ["Own-ship finds a single trajectory that keeps a safe CPA from all three targets "
     "simultaneously.",
     "Own-ship does not alter to port toward TS2 while manoeuvring for TS1/TS3.",
     "Minimum CPA to ALL THREE targets stays above the safe-distance threshold."],
    [compute_target(bearing_from_os_deg=40, range_m=850, v_ts=2.0, v_os=V_OS),
     compute_target(bearing_from_os_deg=320, range_m=850, v_ts=2.0, v_os=V_OS),
     compute_target(bearing_from_os_deg=0, range_m=1200, v_ts=2.3, v_os=V_OS)],
)

# ---------------------------------------------------------------------
# MOOS-IvP file templates
# ---------------------------------------------------------------------
def moos_shoreside(scenario_id, vehicle_names, warp=1):
    vname_list = ",".join(vehicle_names)
    return f"""//-------------------------------------------------
// MOOS-IvP shoreside community for scenario: {scenario_id}
// Auto-generated — verify sharing-layer blocks against your MOOS-IvP version.
//-------------------------------------------------
ServerHost = localhost
ServerPort = 9000
Community  = shoreside

MOOSTimeWarp = {warp}

LatOrigin  = {LAT_ORIGIN}
LongOrigin = {LON_ORIGIN}

//------------------------------------------
ProcessConfig = ANTLER
{{
  MSBetweenLaunches = 200
  Run = MOOSDB          @ NewConsole = false
  Run = pShare           @ NewConsole = false
  Run = pHostInfo        @ NewConsole = false
  Run = uFldShoreBroker   @ NewConsole = false
  Run = pMarineViewer     @ NewConsole = false
  Run = pLogger           @ NewConsole = false
}}

//------------------------------------------
ProcessConfig = pShare
{{
  AppTick    = 2
  CommsTick  = 2
  input      = route = localhost:9200
}}

//------------------------------------------
ProcessConfig = pHostInfo
{{
  AppTick    = 1
  CommsTick  = 1
}}

//------------------------------------------
ProcessConfig = uFldShoreBroker
{{
  AppTick   = 1
  CommsTick = 1

  QBRIDGE   = DEPLOY, RETURN, NODE_REPORT, NODE_MESSAGE, APPCAST_REQ
}}

//------------------------------------------
ProcessConfig = pMarineViewer
{{
  AppTick    = 4
  CommsTick  = 4

  TIFF_FILE            = forrest19.tif
  set_pan_x             = 0
  set_pan_y             = 0
  zoom                  = 0.75
  vehicles_shape_scale  = 1.5
  vehicles_name_mode    = names

  appcast_viewable = true
  scope             = NODE_REPORT_LOCAL
  scope             = COLLISION_ALERT

  action = MENU_KEY=deploy # DEPLOY_ALL=true
  action = MENU_KEY=return # RETURN_ALL=true

  comms_pulse_viewable_all = true
}}

//------------------------------------------
ProcessConfig = pLogger
{{
  AppTick        = 4
  CommsTick      = 4
  File           = LOG_SHORESIDE_{scenario_id.upper()}
  PATH           = ./
  SyncLog        = true @ 0.2
  AsyncLog       = true
  WildCardLogging = true
}}
"""

def moos_vehicle(vname, port, sx, sy, sheading, sspeed, warp, shore_port=9000, is_opship=False):
    bhv_role_note = "// own-ship under test" if is_opship else "// target ship (straight-line only)"
    return f"""//-------------------------------------------------
// MOOS-IvP vehicle community: {vname}  {bhv_role_note}
// Auto-generated — verify sharing-layer blocks against your MOOS-IvP version.
//-------------------------------------------------
ServerHost = localhost
ServerPort = {port}
Community  = {vname}

MOOSTimeWarp = {warp}

LatOrigin  = {LAT_ORIGIN}
LongOrigin = {LON_ORIGIN}

//------------------------------------------
ProcessConfig = ANTLER
{{
  MSBetweenLaunches = 200
  Run = MOOSDB          @ NewConsole = false
  Run = pShare           @ NewConsole = false
  Run = pHostInfo        @ NewConsole = false
  Run = uFldNodeBroker    @ NewConsole = false
  Run = pNodeReporter      @ NewConsole = false
  Run = uSimMarine        @ NewConsole = false
  Run = pMarinePID        @ NewConsole = false
  Run = pHelmIvP           @ NewConsole = false
  Run = pLogger            @ NewConsole = false
}}

//------------------------------------------
ProcessConfig = pShare
{{
  AppTick    = 2
  CommsTick  = 2
  input      = route = localhost:{port + 200}
}}

//------------------------------------------
ProcessConfig = pHostInfo
{{
  AppTick    = 1
  CommsTick  = 1
}}

//------------------------------------------
ProcessConfig = uFldNodeBroker
{{
  AppTick   = 1
  CommsTick = 1

  TRY_SHORE_HOST      = localhost
  TRY_SHORE_PORT       = 9000
  TRY_SHORE_COMMUNITY  = shoreside

  BRIDGE  = src=VIEW_POLYGON
  BRIDGE  = src=VIEW_POINT
  BRIDGE  = src=VIEW_SEGLIST
  BRIDGE  = src=APPCAST
  BRIDGE  = src=NODE_REPORT_LOCAL, alias=NODE_REPORT
  BRIDGE  = src=NODE_MESSAGE_LOCAL, alias=NODE_MESSAGE
}}

//------------------------------------------
ProcessConfig = pNodeReporter
{{
  AppTick    = 2
  CommsTick  = 2

  vessel_type = SHIP
}}

//------------------------------------------
ProcessConfig = uSimMarine
{{
  AppTick    = 10
  CommsTick  = 10

  START_POS  = {sx},{sy},{sheading},{sspeed}
  PREFIX     = NAV
  MODEL_TYPE = holo
}}

//------------------------------------------
ProcessConfig = pMarinePID
{{
  AppTick     = 10
  CommsTick   = 10

  VERBOSE        = false
  DEPTH_CONTROL  = false
  ACTIVE_START   = true

  YAW_PID_KP              = 1.2
  YAW_PID_KD               = 0.0
  YAW_PID_KI                = 0.3
  YAW_PID_INTEGRAL_LIMIT     = 0.07

  SPEED_PID_KP             = 1.0
  SPEED_PID_KD              = 0.0
  SPEED_PID_KI               = 0.0
  SPEED_PID_INTEGRAL_LIMIT    = 0.07

  MAXRUDDER    = 100
  MAXTHRUST    = 100
  SPEED_FACTOR  = 20
}}

//------------------------------------------
ProcessConfig = pHelmIvP
{{
  AppTick    = 4
  CommsTick  = 4

  Behaviors   = {vname}.bhv
  Verbose     = quiet
  Domain      = course:0:359:360
  Domain      = speed:0:5:26
}}

//------------------------------------------
ProcessConfig = pLogger
{{
  AppTick        = 10
  CommsTick      = 10
  File           = LOG_{vname.upper()}
  PATH           = ./
  SyncLog        = true @ 0.2
  AsyncLog       = true
  WildCardLogging = true
}}
"""

def bhv_opship(vname, goal_x, goal_y, speed, use_avoid, contact_names, capture_radius=25):
    avoid_blocks = ""
    if use_avoid:
        for cname in contact_names:
            avoid_blocks += f"""
Behavior = BHV_AvoidCollision
{{
  name         = avoid_{cname}
  pwt          = 300
  condition    = MODE==ACTIVE
  updates      = CONTACT_INFO_{cname.upper()}
  contact      = {cname}
  on_no_contact_ok = true
  extrapolate       = true
  decay             = 30,60

  // NOTE: verify parameter names/ranges against your installed
  // BHV_AvoidCollision version (pHelmIvP --alist / behavior --help).
  // Reasonable starting point:
  pwt_outer_dist   = 200
  pwt_inner_dist   = 50
  completed_dist   = 350
  collision_distance = 25
}}
"""
    return f"""//-------------------------------------------------
// Behavior file for OWN-SHIP under test: {vname}
//-------------------------------------------------
initialize   DEPLOY  = true
initialize   RETURN  = false
initialize   MODE    = ACTIVE

set MODE = ACTIVE {{
  DEPLOY = true
  RETURN = false
}} INACTIVE

//----------------------------------------------
Behavior = BHV_Waypoint
{{
  name         = transit
  pwt          = 100
  condition    = MODE==ACTIVE
  perpetual    = true

  updates      = WPT_UPDATE
  speed        = {speed}
  capture_radius = {capture_radius}
  slip_radius    = {capture_radius * 2}
  points         = {goal_x},{goal_y}
  repeat         = 0
}}
{avoid_blocks}"""

def bhv_target(vname, goal_x, goal_y, speed, capture_radius=25):
    return f"""//-------------------------------------------------
// Behavior file for TARGET SHIP (straight-line, non-avoiding): {vname}
//-------------------------------------------------
initialize   DEPLOY  = true
initialize   MODE    = ACTIVE

set MODE = ACTIVE {{
  DEPLOY = true
}} INACTIVE

//----------------------------------------------
Behavior = BHV_Waypoint
{{
  name         = transit
  pwt          = 100
  condition    = MODE==ACTIVE
  perpetual    = true

  updates      = WPT_UPDATE
  speed        = {speed}
  capture_radius = {capture_radius}
  slip_radius    = {capture_radius * 2}
  points         = {goal_x},{goal_y}
  repeat         = 0
}}
"""

def launch_sh(scenario_id, vehicle_names):
    lines = ["#!/bin/bash", "#-------------------------------------------------",
             f"# Launch script for scenario: {scenario_id}",
             "# Usage: ./launch.sh [time_warp]",
             "#-------------------------------------------------", "",
             'TIME_WARP=${1:-1}', "",
             "# NOTE: for TIME_WARP != 1, also set MOOSTimeWarp in each .moos file",
             "#       before launching, or use the --warp flag if your MOOS-IvP",
             "#       build's pAntler/mission tooling supports it directly.", "",
             "echo \"Launching shoreside...\"",
             "pAntler shoreside.moos >> log_shoreside.txt 2>&1 &", "sleep 1", ""]
    for v in vehicle_names:
        lines.append(f'echo "Launching {v}..."')
        lines.append(f"pAntler {v}.moos >> log_{v}.txt 2>&1 &")
        lines.append("sleep 1")
        lines.append("")
    lines.append('echo "All communities launched. Use pMarineViewer (shoreside) to observe."')
    lines.append('echo "To deploy vehicles: click DEPLOY in pMarineViewer, or:"')
    lines.append('echo "  uPokeDB opship.moos DEPLOY=true"')
    lines.append("")
    lines.append("#-------------------------------------------------")
    lines.append("# To kill all processes for this scenario:")
    lines.append("#   kill -9 $(ps -ef | awk \\'/MOOSDB|pAntler|pHelmIvP|uSimMarine|pMarinePID|pMarineViewer|pLogger|pShare|pHostInfo|uFldNodeBroker|uFldShoreBroker|pNodeReporter/{print $2}\\')")
    lines.append("#-------------------------------------------------")
    return "\n".join(lines) + "\n"

def scenario_readme(sc):
    lines = [f"# Scenario: {sc['id']} — {sc['name']}", "",
              f"**Applicable rule(s):** {', '.join(sc['rule_refs'])}",
              f"**Own-ship role:** {sc['own_ship_role']}", "",
              "## Description", sc["description"], "",
              "## Pass / fail criteria"]
    for c in sc["pass_criteria"]:
        lines.append(f"- {c}")
    lines += ["", "## Vehicle configuration", "",
              f"- **opship**: start (0, 0), heading 000, speed {sc['v_os']} m/s, "
              f"goal (0, {sc['mission_len_m']})"]
    for i, t in enumerate(sc["targets"], start=1):
        lines.append(
            f"- **ts{i}**: start ({t['start_x']}, {t['start_y']}), heading {t['heading']}, "
            f"speed {t['speed']} m/s — initial relative bearing from own-ship "
            f"{t['bearing_from_os_deg']} deg, range {t['range_m']} m; "
            f"time-to-collision if unavoided: ~{t['t_collision_s']} s"
        )
    lines += ["", "## Running",
              "```bash", "cd " + sc["id"], "chmod +x launch.sh", "./launch.sh",
              "# then click DEPLOY on all vehicles in pMarineViewer",
              "```", "",
              "## Baseline (no avoidance) run",
              "To confirm this is a genuine collision course before testing your avoidance "
              "logic, regenerate this scenario's opship.bhv with `use_avoid=False` (see the "
              "generator script) and re-run — the own-ship and target(s) should pass within a "
              "few metres of each other, well under the safe-distance threshold.", ""]
    return "\n".join(lines)

# ---------------------------------------------------------------------
# Write out all scenario mission folders
# ---------------------------------------------------------------------
BASE_PORT = {"opship": 9001, "ts1": 9002, "ts2": 9003, "ts3": 9004}
WARP = 4  # default sim time-warp suggestion

manifest = []

for sid, sc in scenarios.items():
    sdir = os.path.join(OUT_ROOT, sid)
    os.makedirs(sdir, exist_ok=True)

    vehicle_names = ["opship"] + [f"ts{i}" for i in range(1, len(sc["targets"]) + 1)]

    # shoreside.moos
    with open(os.path.join(sdir, "shoreside.moos"), "w") as f:
        f.write(moos_shoreside(sid, vehicle_names, warp=WARP))

    # opship.moos + opship.bhv
    port = BASE_PORT["opship"]
    with open(os.path.join(sdir, "opship.moos"), "w") as f:
        f.write(moos_vehicle("opship", port, 0, 0, 0, sc["v_os"], WARP, is_opship=True))
    contact_names = [f"ts{i}" for i in range(1, len(sc["targets"]) + 1)]
    with open(os.path.join(sdir, "opship.bhv"), "w") as f:
        f.write(bhv_opship("opship", 0, sc["mission_len_m"], sc["v_os"],
                            use_avoid=True, contact_names=contact_names))

    # target vehicles
    for i, t in enumerate(sc["targets"], start=1):
        vname = f"ts{i}"
        port = BASE_PORT.get(vname, 9004 + i)
        with open(os.path.join(sdir, f"{vname}.moos"), "w") as f:
            f.write(moos_vehicle(vname, port, t["start_x"], t["start_y"], t["heading"],
                                  t["speed"], WARP, is_opship=False))
        # target runs straight to a point well beyond the collision point,
        # continuing on the same heading
        far_x = t["start_x"] + 4000 * math.sin(deg2rad(t["heading"]))
        far_y = t["start_y"] + 4000 * math.cos(deg2rad(t["heading"]))
        with open(os.path.join(sdir, f"{vname}.bhv"), "w") as f:
            f.write(bhv_target(vname, round(far_x, 1), round(far_y, 1), t["speed"]))

    # launch.sh
    with open(os.path.join(sdir, "launch.sh"), "w") as f:
        f.write(launch_sh(sid, vehicle_names))
    os.chmod(os.path.join(sdir, "launch.sh"), 0o755)

    # scenario README
    with open(os.path.join(sdir, "README.md"), "w") as f:
        f.write(scenario_readme(sc))

    manifest.append({
        "id": sc["id"], "name": sc["name"], "rule_refs": sc["rule_refs"],
        "own_ship_role": sc["own_ship_role"], "num_targets": len(sc["targets"]),
        "pass_criteria": sc["pass_criteria"],
    })

with open(os.path.join(OUT_ROOT, "..", "scenarios_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Generated {len(scenarios)} scenario mission folders under {OUT_ROOT}")
for sid in scenarios:
    print(" -", sid)
