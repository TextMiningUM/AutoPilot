#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colreg_llm_bridge_paused.py — turn-based variant of colreg_llm_bridge.py.

Instead of a fixed wall-clock decision cadence, this version explicitly
PAUSES every vehicle's uSimMarine (own-ship AND every target ship) before
each LLM call, and RESUMES them only after the decision has been published.
This decouples LLM thinking time (which can be arbitrarily long — 1 second
or 1 minute) from simulated mission time: the world does not move while the
agent is thinking, because uSimMarine's kinematics update is frozen.

MECHANISM
    uSimMarine subscribes to USM_SIM_PAUSED (true/false). Publishing this
    into a vehicle's own MOOSDB freezes/unfreezes that vehicle's position,
    heading and speed updates. Since every vehicle runs its OWN uSimMarine
    inside its OWN MOOS community, this script opens a separate pymoos
    connection to every vehicle (own-ship + every target ship) purely to
    pause/resume them all in lockstep — pausing only own-ship while targets
    keep moving would hand the LLM a stale/wrong picture of the world.

    MOOSTimeWarp itself is untouched by this script — it's a static,
    launch-time setting shared by all communities in the mission and isn't
    what's being manipulated here. What's frozen is vehicle KINEMATICS,
    not MOOS's own clock; MOOSDB and other apps keep ticking in the
    background throughout, which is fine — only "does anything move"
    matters for this problem.

LOOP STRUCTURE (per decision cycle)
    1. Publish USM_SIM_PAUSED=true to every vehicle community.
    2. Snapshot own-ship + contact state (stable now, nothing is moving).
    3. Decide whether a decision is actually needed this cycle (a simple
       TCPA-based risk gate — see --risk-horizon — so you're not paying
       for an LLM call every cycle when nothing is nearby).
    4. If needed: call the LLM (may take arbitrarily long; the world is
       frozen throughout). Publish DESIRED_HEADING/DESIRED_SPEED to
       own-ship's MOOSDB. If not needed: do nothing (pMarinePID holds the
       last commanded heading/speed on own-ship already).
    5. Publish USM_SIM_PAUSED=false to every vehicle community.
    6. Sleep --step-duration seconds (wall-clock, scaled by whatever
       MOOSTimeWarp is set on the mission) to let the now-unpaused world
       advance, then go back to step 1.

INSTALL
    pip install pymoos anthropic --break-system-packages

USAGE
    python3 colreg_llm_bridge_paused.py \\
        --own-ship opship:9001 \\
        --other-vehicles ts1:9002,ts2:9003 \\
        --qa-dataset colreg_qa_500.json \\
        --step-duration 5 \\
        --risk-horizon 300 \\
        --default-speed 2.5

CAVEATS (same as colreg_llm_bridge.py, repeated because they still apply)
    - pymoos API details (comms.run signature, msg accessor names) vary a
      little by version; check yours if calls error.
    - NODE_REPORT field names should be verified against a live scope on
      your install before trusting the parser unmodified.
    - Not run against a live MOOS-IvP install in this environment — this
      is a verified-by-inspection scaffold, not a guaranteed drop-in.
    - USM_SIM_PAUSED is documented behaviour of uSimMarine; confirm your
      installed version still honours it (`uSimMarine --help` or check
      its source/docs) before relying on it for anything beyond a
      desktop test.
"""
import argparse
import json
import math
import re
import sys
import time

try:
    import pymoos
except ImportError:
    print("This script requires pymoos: pip install pymoos --break-system-packages",
          file=sys.stderr)
    raise

try:
    import anthropic
except ImportError:
    anthropic = None


# ---------------------------------------------------------------------
# Geometry helpers (identical to colreg_llm_bridge.py)
# ---------------------------------------------------------------------
def bearing_and_range(ox, oy, tx, ty):
    dx, dy = tx - ox, ty - oy
    rng = math.hypot(dx, dy)
    brg = math.degrees(math.atan2(dx, dy)) % 360.0
    return brg, rng

def relative_bearing(own_heading, true_bearing):
    return (true_bearing - own_heading + 540) % 360 - 180

def cpa_tcpa(ox, oy, ohdg, ospd, tx, ty, thdg, tspd):
    oh, th = math.radians(ohdg), math.radians(thdg)
    vox, voy = ospd * math.sin(oh), ospd * math.cos(oh)
    vtx, vty = tspd * math.sin(th), tspd * math.cos(th)
    dx, dy = tx - ox, ty - oy
    dvx, dvy = vtx - vox, vty - voy
    rel_speed_sq = dvx ** 2 + dvy ** 2
    if rel_speed_sq < 1e-6:
        return math.hypot(dx, dy), 0.0
    t_cpa = max(0.0, -(dx * dvx + dy * dvy) / rel_speed_sq)
    cx, cy = dx + dvx * t_cpa, dy + dvy * t_cpa
    return math.hypot(cx, cy), t_cpa

def classify_encounter(rel_brg, own_hdg, tgt_hdg):
    course_diff = (tgt_hdg - own_hdg + 540) % 360 - 180
    if abs(rel_brg) <= 6 and abs(abs(course_diff) - 180) <= 20:
        return "head_on", ["Rule 14"]
    if abs(rel_brg) > 112.5:
        return "overtaking_geometry", ["Rule 13"]
    if rel_brg > 0:
        return "crossing_target_on_starboard", ["Rule 15", "Rule 16"]
    return "crossing_target_on_port", ["Rule 15", "Rule 17"]


class QARetriever:
    def __init__(self, path):
        self.items = []
        if path:
            try:
                with open(path) as f:
                    self.items = json.load(f)
            except FileNotFoundError:
                print(f"(QA dataset '{path}' not found — no retrieval grounding)", file=sys.stderr)

    def retrieve(self, rule_refs, k=4):
        if not self.items:
            return []
        hits = [it for it in self.items
                if any(r in it.get("rule_ref", "") for r in rule_refs)]
        hits.sort(key=lambda x: 0 if x.get("question_type") in ("scenario", "application") else 1)
        return hits[:k]


SYSTEM_PROMPT = """You are the navigation decision system for an autonomous \
surface vessel (own-ship), reasoning with no time pressure — the simulated \
world is frozen while you think, so take as long as you need to reach a \
COLREG-compliant decision. Reply with ONLY a JSON object, no other text:
{"heading": <0-359.9 float, true/MOOS heading, 0=north clockwise>,
 "speed": <float, m/s>,
 "rule_applied": "<e.g. Rule 15, Rule 17>",
 "reasoning": "<one or two sentences>"}

Rules of thumb:
- If you are the give-way vessel, alter early and substantially, normally \
to starboard, and avoid crossing ahead if avoidable.
- If you are the stand-on vessel, hold course and speed unless it becomes \
apparent the other vessel is not keeping clear.
- In a head-on situation, alter to starboard regardless of role.
- With multiple contacts, choose one course/speed that keeps a safe CPA \
from ALL of them simultaneously.
- If no contact poses meaningful risk, resume heading toward the goal."""


def build_user_prompt(own, contacts, goal, qa_context):
    lines = [f"OWN-SHIP: x={own['x']:.1f} y={own['y']:.1f} heading={own['heading']:.1f} "
             f"speed={own['speed']:.2f} goal=({goal[0]:.1f},{goal[1]:.1f})", "", "CONTACTS:"]
    for name, c in contacts.items():
        lines.append(
            f"  {name}: x={c['x']:.1f} y={c['y']:.1f} heading={c['heading']:.1f} "
            f"speed={c['speed']:.2f} | relative_bearing={c['rel_brg']:.1f} "
            f"range={c['range']:.1f}m CPA={c['cpa']:.1f}m TCPA={c['tcpa']:.1f}s "
            f"encounter_type={c['encounter_type']}"
        )
    if qa_context:
        lines += ["", "RELEVANT COLREG REFERENCE:"]
        for qa in qa_context:
            lines.append(f"  - [{qa.get('rule_ref')}] Q: {qa.get('question')}")
            lines.append(f"    A: {qa.get('answer')}")
    lines += ["", "Return your decision as the specified JSON object now."]
    return "\n".join(lines)


class LLMDecisionMaker:
    def __init__(self, model="claude-sonnet-4-6", api_client=None):
        self.model = model
        self.client = api_client or (anthropic.Anthropic() if anthropic else None)

    def decide(self, own, contacts, goal, qa_context):
        prompt = build_user_prompt(own, contacts, goal, qa_context)
        if self.client is None:
            raise RuntimeError("No LLM client configured")
        resp = self.client.messages.create(
            model=self.model, max_tokens=400, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        return json.loads(text), prompt


# ---------------------------------------------------------------------
# Multi-vehicle pause controller — one pymoos connection per vehicle,
# used ONLY to publish USM_SIM_PAUSED into that vehicle's own MOOSDB.
# ---------------------------------------------------------------------
class VehiclePauser:
    def __init__(self, name, host, port):
        self.name = name
        self.comms = pymoos.comms()
        self.comms.set_on_connect_callback(lambda: True)
        self.comms.run(host, port, f"pauser_{name}")
        # give the connection a moment to establish before first use
        time.sleep(0.3)

    def set_paused(self, paused: bool):
        self.comms.notify("USM_SIM_PAUSED", "true" if paused else "false", pymoos.time())


class MultiVehiclePause:
    def __init__(self, vehicles):
        """vehicles: list of (name, host, port) tuples, own-ship included."""
        self.pausers = [VehiclePauser(name, host, port) for name, host, port in vehicles]

    def pause_all(self):
        for p in self.pausers:
            p.set_paused(True)

    def resume_all(self):
        for p in self.pausers:
            p.set_paused(False)


# ---------------------------------------------------------------------
# Own-ship state/contact reader + decision publisher (same pattern as
# colreg_llm_bridge.py, unchanged)
# ---------------------------------------------------------------------
class OwnShipLink:
    def __init__(self, name, host, port):
        self.name = name
        self.own = {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 0.0}
        self.contacts_raw = {}
        self.comms = pymoos.comms()
        self.comms.set_on_connect_callback(self._on_connect)
        self.comms.set_on_mail_callback(self._on_mail)
        self.comms.run(host, port, f"{name}_llm_bridge")

    def _on_connect(self):
        for v in ("NAV_X", "NAV_Y", "NAV_HEADING", "NAV_SPEED"):
            self.comms.register(v, 0)
        self.comms.register("NODE_REPORT", 0)
        return True

    def _on_mail(self):
        for msg in self.comms.fetch():
            key = msg.key()
            if key == "NAV_X":
                self.own["x"] = msg.double()
            elif key == "NAV_Y":
                self.own["y"] = msg.double()
            elif key == "NAV_HEADING":
                self.own["heading"] = msg.double()
            elif key == "NAV_SPEED":
                self.own["speed"] = msg.double()
            elif key == "NODE_REPORT":
                self._parse_node_report(msg.string())
        return True

    def _parse_node_report(self, s):
        fields = {}
        for kv in s.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                fields[k.strip().upper()] = v.strip()
        name = fields.get("NAME")
        if not name or name == self.name:
            return
        try:
            self.contacts_raw[name] = {
                "x": float(fields.get("X", 0.0)), "y": float(fields.get("Y", 0.0)),
                "heading": float(fields.get("HDG", fields.get("HEADING", 0.0))),
                "speed": float(fields.get("SPD", fields.get("SPEED", 0.0))),
            }
        except ValueError:
            pass

    def snapshot(self):
        """Safe to call once vehicles are paused — state is stable."""
        own = dict(self.own)
        contacts = {}
        rule_refs = set()
        for name, c in dict(self.contacts_raw).items():
            brg, rng = bearing_and_range(own["x"], own["y"], c["x"], c["y"])
            rel_brg = relative_bearing(own["heading"], brg)
            cpa, tcpa = cpa_tcpa(own["x"], own["y"], own["heading"], own["speed"],
                                  c["x"], c["y"], c["heading"], c["speed"])
            enc_type, refs = classify_encounter(rel_brg, own["heading"], c["heading"])
            rule_refs.update(refs)
            contacts[name] = {**c, "range": rng, "rel_brg": rel_brg, "cpa": cpa,
                               "tcpa": tcpa, "encounter_type": enc_type}
        return own, contacts, list(rule_refs)

    def publish_decision(self, heading, speed):
        now = pymoos.time()
        self.comms.notify("DESIRED_HEADING", float(heading), now)
        self.comms.notify("DESIRED_SPEED", float(speed), now)


def parse_vehicle_spec(spec):
    """'name:port' or 'name:host:port' -> (name, host, port)."""
    parts = spec.split(":")
    if len(parts) == 2:
        return parts[0], "localhost", int(parts[1])
    if len(parts) == 3:
        return parts[0], parts[1], int(parts[2])
    raise ValueError(f"Bad vehicle spec: {spec}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--own-ship", required=True, help="name:port or name:host:port")
    ap.add_argument("--other-vehicles", default="",
                     help="Comma-separated name:port entries for every OTHER vehicle "
                          "(target ships) whose uSimMarine must also be paused/resumed "
                          "in lockstep with own-ship, e.g. 'ts1:9002,ts2:9003'")
    ap.add_argument("--goal-x", type=float, default=0.0)
    ap.add_argument("--goal-y", type=float, default=2400.0)
    ap.add_argument("--qa-dataset", default=None)
    ap.add_argument("--step-duration", type=float, default=5.0,
                     help="Wall-clock seconds to let the world run (unpaused) between "
                          "decision cycles, before pausing again")
    ap.add_argument("--risk-horizon", type=float, default=300.0,
                     help="Only call the LLM if some contact's TCPA is below this many "
                          "seconds; otherwise skip the LLM call this cycle to save cost")
    ap.add_argument("--default-speed", type=float, default=2.5)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    own_name, own_host, own_port = parse_vehicle_spec(args.own_ship)
    other_specs = [parse_vehicle_spec(s) for s in args.other_vehicles.split(",") if s]

    print(f"Connecting to own-ship '{own_name}' at {own_host}:{own_port} ...")
    link = OwnShipLink(own_name, own_host, own_port)

    all_vehicles = [(own_name, own_host, own_port)] + other_specs
    print(f"Setting up pause control for: {[v[0] for v in all_vehicles]}")
    pauser = MultiVehiclePause(all_vehicles)

    retriever = QARetriever(args.qa_dataset)
    llm = None if args.dry_run else LLMDecisionMaker(model=args.model)

    print("Warming up (letting first NAV_*/NODE_REPORT mail arrive)...")
    time.sleep(2.0)

    print(f"Starting pause/decide/resume loop. step_duration={args.step_duration}s "
          f"risk_horizon={args.risk_horizon}s. Ctrl-C to stop.\n")

    try:
        while True:
            # 1. Freeze the world
            pauser.pause_all()
            time.sleep(0.2)  # let the pause take effect before snapshotting

            # 2. Snapshot state (stable now)
            own, contacts, rule_refs = link.snapshot()

            # 3. Risk gate: only bother the LLM if something is actually close
            min_tcpa = min((c["tcpa"] for c in contacts.values()), default=float("inf"))
            need_decision = contacts and min_tcpa <= args.risk_horizon

            if need_decision:
                qa_context = retriever.retrieve(rule_refs, k=4)
                t0 = time.time()
                if args.dry_run:
                    decision = {"heading": own["heading"], "speed": args.default_speed,
                                "rule_applied": "n/a", "reasoning": "dry run"}
                else:
                    decision, _ = llm.decide(own, contacts, (args.goal_x, args.goal_y), qa_context)
                elapsed = time.time() - t0
                print(f"[decision] min_tcpa={min_tcpa:.1f}s -> LLM took {elapsed:.1f}s "
                      f"(world was frozen for all of it) -> {decision}")
                link.publish_decision(decision["heading"], decision["speed"])
            else:
                print(f"[skip] min_tcpa={min_tcpa:.1f}s > risk_horizon "
                      f"({args.risk_horizon}s) — no LLM call this cycle")

            # 4. Resume the world
            pauser.resume_all()

            # 5. Let simulated time advance before the next check
            time.sleep(args.step_duration)
    except KeyboardInterrupt:
        print("Stopped. Resuming all vehicles before exit.")
        pauser.resume_all()


if __name__ == "__main__":
    main()
