#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colreg_llm_bridge.py — drive a MOOS-IvP vehicle's steering/speed decisions
from an external LLM, bypassing pHelmIvP / BHV_AvoidCollision entirely.

ARCHITECTURE
    This script is a standalone MOOSDB client (via pymoos). It:
      1. Subscribes to own-ship's NAV_X, NAV_Y, NAV_HEADING, NAV_SPEED and
         to NODE_REPORT for every other vehicle (target ships / contacts).
      2. On a slow decision cadence (default every 5s of MOOS time, NOT
         every control tick — LLM calls are too slow for that), computes
         bearing/range/CPA/TCPA to each contact and builds a structured
         prompt.
      3. Optionally retrieves the most relevant items from your 500-item
         COLREG Q&A dataset (colreg_qa_500.json) to ground the prompt in
         the specific rule text for the detected encounter type — a
         lightweight form of retrieval-augmented generation. This is a
         simple keyword/geometry-based retriever, not embeddings; swap in
         a real vector index if you want something more robust.
      4. Calls the LLM (Anthropic API by default; swap in your fine-tuned
         model's endpoint if you have one) and parses a structured JSON
         decision: {heading, speed, rule_applied, reasoning}.
      5. Publishes DESIRED_HEADING / DESIRED_SPEED directly into own-ship's
         MOOSDB. pMarinePID (still running) turns these into rudder/thrust
         commands. pHelmIvP is NOT running on this vehicle at all — see
         the companion opship_llm.moos for the required ANTLER change.

REQUIRED MOOS-SIDE CHANGE
    In the vehicle's .moos file, remove `Run = pHelmIvP` from the ANTLER
    block (and you no longer need opship.bhv at all). Keep uSimMarine and
    pMarinePID running — this script replaces pHelmIvP's job, nothing else.
    See opship_llm.moos in this same folder for a ready-made example.

INSTALL
    pip install pymoos anthropic --break-system-packages

USAGE
    python3 colreg_llm_bridge.py \\
        --moos-host localhost --moos-port 9001 --moos-community opship \\
        --own-ship-name opship \\
        --qa-dataset colreg_qa_500.json \\
        --decision-period 5.0 \\
        --default-speed 2.5 \\
        --safe-distance 50

CAVEATS
    - pymoos's exact API (comms.run signature, msg accessor names) has
      shifted slightly across releases; check `python3 -c "import pymoos;
      help(pymoos.comms)"` against your installed version if you hit
      AttributeErrors, and adjust the small number of pymoos calls marked
      below.
    - NODE_REPORT string field names (NAME=, X=, Y=, HDG=/HEADING=,
      SPD=/SPEED=) are stable across MOOS-IvP versions in my experience,
      but verify against a live NODE_REPORT string from your install
      (echo it with `uPokeDB` / `MOOSDB` scope tools) before trusting the
      parser below unmodified.
    - This script is a starting scaffold — it has NOT been run against a
      live MOOS-IvP install (not available in this environment). Treat it
      as a verified-by-inspection reference implementation, not a
      guaranteed drop-in.
"""
import argparse
import json
import math
import re
import sys
import threading
import time
from collections import deque

try:
    import pymoos
except ImportError:
    print("This script requires pymoos: pip install pymoos --break-system-packages",
          file=sys.stderr)
    raise

try:
    import anthropic
except ImportError:
    anthropic = None  # only required if you actually call the Anthropic API


# ---------------------------------------------------------------------
# Geometry helpers (same conventions as the mission generator: heading
# 0 = north, clockwise, local x/y meters)
# ---------------------------------------------------------------------
def bearing_and_range(ox, oy, tx, ty):
    dx, dy = tx - ox, ty - oy
    rng = math.hypot(dx, dy)
    brg = math.degrees(math.atan2(dx, dy)) % 360.0
    return brg, rng

def relative_bearing(own_heading, true_bearing):
    """Bearing of the contact relative to own-ship's bow, -180..+180,
    positive = starboard side."""
    rel = (true_bearing - own_heading + 540) % 360 - 180
    return rel

def cpa_tcpa(ox, oy, ohdg, ospd, tx, ty, thdg, tspd):
    """Closest point of approach distance and time, given both vessels'
    current position/heading/speed and assuming both hold course/speed."""
    oh, th = math.radians(ohdg), math.radians(thdg)
    vox, voy = ospd * math.sin(oh), ospd * math.cos(oh)
    vtx, vty = tspd * math.sin(th), tspd * math.cos(th)
    dx, dy = tx - ox, ty - oy
    dvx, dvy = vtx - vox, vty - voy
    rel_speed_sq = dvx ** 2 + dvy ** 2
    if rel_speed_sq < 1e-6:
        return math.hypot(dx, dy), 0.0
    t_cpa = -(dx * dvx + dy * dvy) / rel_speed_sq
    t_cpa = max(0.0, t_cpa)
    cx, cy = dx + dvx * t_cpa, dy + dvy * t_cpa
    return math.hypot(cx, cy), t_cpa

def classify_encounter(rel_brg, own_hdg, tgt_hdg):
    """Very rough COLREG encounter classification for prompt grounding /
    retrieval keyword selection. Not a substitute for the LLM's own
    reasoning — just picks which Q&A items to retrieve."""
    course_diff = (tgt_hdg - own_hdg + 540) % 360 - 180  # -180..180
    if abs(rel_brg) <= 6 and abs(abs(course_diff) - 180) <= 20:
        return "head_on", ["Rule 14"]
    if abs(rel_brg) > 112.5:
        return "overtaking_geometry", ["Rule 13"]
    if rel_brg > 0:
        return "crossing_target_on_starboard", ["Rule 15", "Rule 16"]
    else:
        return "crossing_target_on_port", ["Rule 15", "Rule 17"]


# ---------------------------------------------------------------------
# Lightweight retrieval over the 500-item Q&A dataset
# ---------------------------------------------------------------------
class QARetriever:
    def __init__(self, path):
        self.items = []
        if path:
            try:
                with open(path) as f:
                    self.items = json.load(f)
            except FileNotFoundError:
                print(f"(QA dataset '{path}' not found — proceeding without retrieval grounding)",
                      file=sys.stderr)

    def retrieve(self, rule_refs, k=4):
        """Pull a few Q&A items whose rule_ref matches the detected
        encounter's rule(s). Simple exact/substring match on rule_ref —
        swap in embeddings if you want semantic retrieval instead."""
        if not self.items:
            return []
        hits = []
        for it in self.items:
            for r in rule_refs:
                if r in it.get("rule_ref", ""):
                    hits.append(it)
                    break
        # prefer 'scenario' and 'application' items over bare 'recall' for
        # decision-relevant grounding
        hits.sort(key=lambda x: 0 if x.get("question_type") in ("scenario", "application") else 1)
        return hits[:k]


# ---------------------------------------------------------------------
# LLM decision call
# ---------------------------------------------------------------------
SYSTEM_PROMPT = """You are the navigation decision system for an autonomous \
surface vessel (own-ship). You must choose a course and speed that complies \
with the International Regulations for Preventing Collisions at Sea \
(COLREGs). You are given own-ship's current state, the state of every \
detected contact (target ship) including computed bearing, range, CPA \
(closest point of approach) and TCPA (time to CPA) assuming no one \
manoeuvres, and reference COLREG rule text relevant to the detected \
encounter type(s).

Reply with ONLY a JSON object, no other text, in exactly this shape:
{"heading": <0-359.9 float, true/MOOS heading, 0=north clockwise>,
 "speed": <float, m/s>,
 "rule_applied": "<e.g. Rule 15, Rule 17>",
 "reasoning": "<one or two sentences>"}

Rules of thumb:
- If you are the give-way vessel, alter early and substantially (course \
change generally preferred over speed change alone if sea room allows), \
normally to starboard, and avoid crossing ahead if avoidable.
- If you are the stand-on vessel, hold course and speed unless it becomes \
apparent the other vessel is not keeping clear, in which case take \
whatever action best avoids collision — and avoid altering to port toward \
a vessel on your own port side if you must act.
- In a head-on situation, alter to starboard regardless of role.
- When multiple contacts are present, choose one course/speed that keeps \
a safe CPA from ALL of them simultaneously; do not solve encounters \
independently if the manoeuvres conflict.
- If no contact poses a meaningful collision risk (CPA already safe), \
resume your original heading toward the mission goal."""


def build_user_prompt(own, contacts, goal, qa_context):
    lines = []
    lines.append(f"OWN-SHIP: x={own['x']:.1f} y={own['y']:.1f} "
                 f"heading={own['heading']:.1f} speed={own['speed']:.2f} "
                 f"goal=({goal[0]:.1f},{goal[1]:.1f})")
    lines.append("")
    lines.append("CONTACTS:")
    for name, c in contacts.items():
        lines.append(
            f"  {name}: x={c['x']:.1f} y={c['y']:.1f} heading={c['heading']:.1f} "
            f"speed={c['speed']:.2f} | relative_bearing={c['rel_brg']:.1f} "
            f"(negative=port, positive=starboard) range={c['range']:.1f}m "
            f"CPA={c['cpa']:.1f}m TCPA={c['tcpa']:.1f}s "
            f"encounter_type={c['encounter_type']}"
        )
    if qa_context:
        lines.append("")
        lines.append("RELEVANT COLREG REFERENCE (from training Q&A set):")
        for qa in qa_context:
            lines.append(f"  - [{qa.get('rule_ref')}] Q: {qa.get('question')}")
            lines.append(f"    A: {qa.get('answer')}")
    lines.append("")
    lines.append("Return your decision as the specified JSON object now.")
    return "\n".join(lines)


class LLMDecisionMaker:
    def __init__(self, model="claude-sonnet-4-6", api_client=None):
        self.model = model
        self.client = api_client or (anthropic.Anthropic() if anthropic else None)

    def decide(self, own, contacts, goal, qa_context):
        prompt = build_user_prompt(own, contacts, goal, qa_context)
        if self.client is None:
            raise RuntimeError("No LLM client configured (anthropic not installed / no client passed)")
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        text = text.strip()
        # be tolerant of stray markdown fences
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        decision = json.loads(text)
        return decision, prompt


# ---------------------------------------------------------------------
# MOOS bridge
# ---------------------------------------------------------------------
class MoosLLMBridge:
    def __init__(self, host, port, community, own_ship_name, goal,
                 qa_path, decision_period, default_speed, model, dry_run=False):
        self.own_ship_name = own_ship_name
        self.goal = goal
        self.decision_period = decision_period
        self.default_speed = default_speed
        self.dry_run = dry_run

        self.own = {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": default_speed}
        self.contacts_raw = {}  # name -> {x,y,heading,speed}
        self.lock = threading.Lock()

        self.retriever = QARetriever(qa_path)
        self.llm = LLMDecisionMaker(model=model) if not dry_run else None

        self.comms = pymoos.comms()
        self.comms.set_on_connect_callback(self._on_connect)
        self.comms.set_on_mail_callback(self._on_mail)
        # NOTE: exact positional/keyword args for comms.run() vary by
        # pymoos version — check yours if this line errors.
        self.comms.run(host, port, community)

    def _on_connect(self):
        for v in ("NAV_X", "NAV_Y", "NAV_HEADING", "NAV_SPEED"):
            self.comms.register(v, 0)
        self.comms.register("NODE_REPORT", 0)
        return True

    def _on_mail(self):
        for msg in self.comms.fetch():
            key = msg.key()
            with self.lock:
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
        # Typical NODE_REPORT string: "NAME=ts1,X=123.4,Y=567.8,SPD=2.10,
        # HDG=180.0,TYPE=SHIP,..."  Field set/order can vary — adjust the
        # key names below if your version differs (check with a live scope).
        fields = {}
        for kv in s.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                fields[k.strip().upper()] = v.strip()
        name = fields.get("NAME")
        if not name or name == self.own_ship_name:
            return
        try:
            self.contacts_raw[name] = {
                "x": float(fields.get("X", 0.0)),
                "y": float(fields.get("Y", 0.0)),
                "heading": float(fields.get("HDG", fields.get("HEADING", 0.0))),
                "speed": float(fields.get("SPD", fields.get("SPEED", 0.0))),
            }
        except ValueError:
            pass  # malformed field, skip this update

    def _build_contact_context(self):
        with self.lock:
            own = dict(self.own)
            contacts_raw = dict(self.contacts_raw)
        contacts = {}
        all_rule_refs = set()
        for name, c in contacts_raw.items():
            brg, rng = bearing_and_range(own["x"], own["y"], c["x"], c["y"])
            rel_brg = relative_bearing(own["heading"], brg)
            cpa, tcpa = cpa_tcpa(own["x"], own["y"], own["heading"], own["speed"],
                                  c["x"], c["y"], c["heading"], c["speed"])
            enc_type, rule_refs = classify_encounter(rel_brg, own["heading"], c["heading"])
            all_rule_refs.update(rule_refs)
            contacts[name] = {**c, "range": rng, "rel_brg": rel_brg,
                               "cpa": cpa, "tcpa": tcpa, "encounter_type": enc_type}
        return own, contacts, list(all_rule_refs)

    def _publish_decision(self, decision):
        if self.dry_run:
            print(f"[DRY RUN] would publish DESIRED_HEADING={decision['heading']} "
                  f"DESIRED_SPEED={decision['speed']}")
            return
        now = pymoos.time()
        self.comms.notify("DESIRED_HEADING", float(decision["heading"]), now)
        self.comms.notify("DESIRED_SPEED", float(decision["speed"]), now)

    def decision_loop(self):
        while True:
            own, contacts, rule_refs = self._build_contact_context()
            qa_context = self.retriever.retrieve(rule_refs, k=4) if rule_refs else []
            try:
                if self.dry_run:
                    decision = {"heading": own["heading"], "speed": self.default_speed,
                                "rule_applied": "n/a", "reasoning": "dry run, no LLM call"}
                else:
                    decision, prompt = self.llm.decide(own, contacts, self.goal, qa_context)
                print(f"t={pymoos.time():.1f}  decision={decision}")
                self._publish_decision(decision)
            except Exception as e:
                print(f"[WARN] decision cycle failed ({e}); holding last command", file=sys.stderr)
            time.sleep(self.decision_period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moos-host", default="localhost")
    ap.add_argument("--moos-port", type=int, required=True)
    ap.add_argument("--moos-community", required=True,
                     help="Community name this bridge connects AS (e.g. 'opship_llm_bridge')")
    ap.add_argument("--own-ship-name", required=True,
                     help="This vehicle's own NAME as it appears in its own NODE_REPORT/registrations")
    ap.add_argument("--goal-x", type=float, default=0.0)
    ap.add_argument("--goal-y", type=float, default=2400.0)
    ap.add_argument("--qa-dataset", default=None, help="Path to colreg_qa_500.json")
    ap.add_argument("--decision-period", type=float, default=5.0,
                     help="Seconds of MOOS/wallclock time between LLM decision calls")
    ap.add_argument("--default-speed", type=float, default=2.5)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--dry-run", action="store_true",
                     help="Skip LLM calls; just verify the MOOS pub/sub wiring")
    args = ap.parse_args()

    bridge = MoosLLMBridge(
        host=args.moos_host, port=args.moos_port, community=args.moos_community,
        own_ship_name=args.own_ship_name, goal=(args.goal_x, args.goal_y),
        qa_path=args.qa_dataset, decision_period=args.decision_period,
        default_speed=args.default_speed, model=args.model, dry_run=args.dry_run,
    )
    print(f"Bridge connected. Own-ship='{args.own_ship_name}' on port {args.moos_port}. "
          f"Deciding every {args.decision_period}s. Ctrl-C to stop.")
    try:
        bridge.decision_loop()
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
