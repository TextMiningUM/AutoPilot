#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates a targeted drill set fixing two specific, confirmed confusions in
the agent's basic navigation maths:
  1. TCPA misread (TCPA=0 read as "collision now" instead of "closest point
     is now/already passed").
  2. Relative bearing sign confusion (port/starboard on +/-).

SIGN CONVENTION USED (must match the project's own established convention,
from narrate.py / colreg_llm_bridge_paused.py / evaluate_run.py earlier in
this project -- consistency with what the agent already sees in its own
state data matters more than which of two equally valid conventions is
"more standard"):

    relative_bearing_signed_deg = (true_bearing_to_target - own_heading + 540) % 360 - 180
    range: (-180, +180]
    POSITIVE = target is on the STARBOARD side (clockwise from the bow)
    NEGATIVE = target is on the PORT side (counter-clockwise from the bow)
    0 deg   = dead ahead
    +90 deg = starboard beam
    -90 deg = port beam
    +/-180 deg = dead astern

This is mathematically identical to tdgil.com's 0-360 clockwise convention
("045R" = 45 deg clockwise of heading), just expressed as a signed value
instead of a raw 0-360 value: values 0-180 in the 0-360 form correspond to
0 to +180 here (starboard); values 180-360 in the 0-360 form correspond to
-180 to 0 here (port). Both representations are drilled below, since the
agent may encounter either in different parts of its own pipeline, and
converting between them is itself a place sign errors creep in.

Every numeric answer in this file is computed by the functions below, never
hand-typed, specifically to avoid reintroducing a sign error into the data
meant to fix a sign error.
"""
import json
import math
import random

random.seed(7)
items = []
_id_counter = 1

def add(category, topic, qtype, question, answer, explanation, difficulty="medium"):
    global _id_counter
    items.append({
        "id": f"navmath_{_id_counter:04d}",
        "category": category,
        "topic": topic,
        "question_type": qtype,
        "question": question.strip(),
        "answer": answer.strip(),
        "explanation": explanation.strip(),
        "difficulty": difficulty,
    })
    _id_counter += 1


# =====================================================================
# Shared geometry helpers (exact, reused from the project's own convention)
# =====================================================================
def relative_bearing_signed(own_heading, true_bearing):
    return (true_bearing - own_heading + 540) % 360 - 180

def to_0_360_form(signed_bearing):
    return signed_bearing % 360

def side_of(signed_bearing):
    if abs(signed_bearing) < 1e-9:
        return "dead ahead (neither port nor starboard)"
    if abs(abs(signed_bearing) - 180) < 1e-9:
        return "dead astern (neither port nor starboard)"
    return "starboard" if signed_bearing > 0 else "port"

def fmt_deg(x):
    return f"{x:.0f}"


# =====================================================================
# PART 1 -- TCPA interpretation drills
# =====================================================================
add("Basic Navigation Maths", "TCPA definition", "definition",
    "In your own words: what does TCPA (Time to Closest Point of Approach) measure?",
    "TCPA is the time remaining, assuming both vessels hold their current course and speed, "
    "until the range between them reaches its minimum value (the CPA). It is a countdown to "
    "a moment in time, not a statement about whether a collision is happening.",
    "Grounded in the tdgil.com CPA lesson: TCPA is computed as the distance from the current "
    "position to the CPA point divided by the relative speed -- it answers 'when', not 'is there danger'.")

add("Basic Navigation Maths", "TCPA definition", "definition",
    "Does TCPA by itself tell you whether a collision risk exists?",
    "No. TCPA only tells you WHEN the closest point of approach occurs. Whether that closest "
    "approach is dangerous depends entirely on CPA (how close), not on TCPA (how soon). A very "
    "small TCPA with a large CPA means 'passing safely very soon'; a very large TCPA with a "
    "tiny CPA means 'on a dangerous course, but not yet close'. You always need both numbers "
    "together.",
    "This is the single most important corrective fact for the confirmed TCPA-misread error: "
    "TCPA answers 'when', CPA answers 'how close'. Neither alone answers 'is this dangerous'.")

# --- TCPA = 0 exactly ---
tcpa_zero_dangerous_cpas = [5, 10, 15, 20]
tcpa_zero_safe_cpas = [300, 450, 600, 800]

for cpa in tcpa_zero_dangerous_cpas:
    add("Basic Navigation Maths", "TCPA = 0", "application",
        f"A contact shows TCPA = 0 s and CPA = {cpa} m. What is happening right now, precisely?",
        f"Right now, at this instant, the two vessels are at their closest point of approach to "
        f"each other, and that closest distance is {cpa} m. TCPA=0 means 'the closest point is "
        f"occurring now', not 'a collision is occurring now' -- but because CPA is only {cpa} m "
        f"here, this specific case genuinely is a high-risk, very-close-quarters moment.",
        "TCPA=0 always means 'closest approach is now'; whether 'now' is dangerous depends "
        "entirely on the paired CPA value, which here happens to be small.")

for cpa in tcpa_zero_safe_cpas:
    add("Basic Navigation Maths", "TCPA = 0", "application",
        f"A contact shows TCPA = 0 s and CPA = {cpa} m. What is happening right now, precisely?",
        f"Right now, at this instant, the two vessels are at their closest point of approach to "
        f"each other, and that closest distance is {cpa} m. This is NOT a collision and not even "
        f"a close-quarters situation -- {cpa} m is a comfortable safety margin. TCPA=0 only means "
        f"the moment of minimum range has arrived; it does not by itself mean danger.",
        "This is the direct corrective example for the confirmed confusion: TCPA=0 with a large "
        "CPA is a normal, safe passing event, not a collision.")

add("Basic Navigation Maths", "TCPA = 0", "application",
    "True or false: 'TCPA = 0' always means a collision is happening right now.",
    "False. TCPA = 0 means the closest point of approach is occurring right now -- it says "
    "nothing about the distance at that closest point. A collision requires CPA to also be "
    "essentially zero (the vessels' hulls actually coincide). TCPA=0 with CPA=500m is simply "
    "the moment two vessels pass each other with 500 m of clearance.",
    "Direct true/false correction of the confirmed misread pattern.")

# --- TCPA small positive (imminent) ---
for tcpa in [3, 8, 15, 25]:
    for cpa, label in [(1200, "far/safe"), (25, "dangerously close")]:
        risk_word = "low" if cpa > 100 else "high"
        add("Basic Navigation Maths", "TCPA small positive", "application",
            f"TCPA = {tcpa} s, CPA = {cpa} m. Closest approach has not happened yet -- it will "
            f"happen very soon. Is this a collision right now? What should the agent expect in "
            f"the next {tcpa} seconds?",
            f"No, this is not a collision right now. The closest point of approach will occur "
            f"in {tcpa} seconds, and at that moment the minimum range will be about {cpa} m. "
            f"{'Given the large CPA, this is a routine, low-risk passing event soon.' if cpa > 100 else 'Given the very small CPA, this is a genuinely dangerous developing situation that needs immediate action before those ' + str(tcpa) + ' seconds elapse.'}",
            f"TCPA={tcpa}s means the closest-approach moment is {tcpa} seconds in the future, not "
            f"now. Risk assessment ({risk_word}) comes from the CPA value ({cpa} m), not from how "
            f"small TCPA is.")

# --- TCPA large positive (far future) ---
for tcpa in [400, 600, 900]:
    add("Basic Navigation Maths", "TCPA large positive", "application",
        f"TCPA = {tcpa} s ({tcpa/60:.1f} min), CPA = 150 m. How urgent is this right now?",
        f"Not urgent right now. The closest approach is still {tcpa/60:.1f} minutes away. There is "
        f"ample time to observe how the situation develops and to take early action per Rule 16 "
        f"well before it becomes a close-quarters situation -- but the projected CPA of 150 m "
        f"means this is worth continuing to monitor rather than dismissing, since a projection "
        f"this far out is also more likely to change as both vessels' actual courses evolve.",
        f"A large TCPA means plenty of time remains; it does not mean the situation is "
        f"unimportant if the projected CPA is tight, only that there's no need to react in the "
        f"next few seconds.")

# --- TCPA negative (already passed) ---
for tcpa in [-5, -15, -45]:
    for cpa, label in [(20, "was dangerously close"), (600, "was always safe")]:
        add("Basic Navigation Maths", "TCPA negative", "application",
            f"TCPA = {tcpa} s, CPA = {cpa} m. What does a NEGATIVE TCPA mean here, and what is "
            f"the current situation?",
            f"A negative TCPA means the closest point of approach already happened, "
            f"{abs(tcpa)} seconds ago, assuming both vessels held course and speed since then. "
            f"The minimum range reached at that moment was about {cpa} m. The vessels are now "
            f"moving apart (diverging) rather than closing. "
            f"{'This was a genuinely close call that has now passed.' if cpa < 100 else 'This was never a close call -- it passed at a comfortable distance.'} "
            f"No avoiding action is needed now on the basis of this projection alone.",
            "Negative TCPA is the mirror image of the main confusion: it does not mean 'collision "
            "already happened', it means the projected closest-approach moment is in the past, "
            "and (assuming the projection was accurate) the vessels are now opening range.")

add("Basic Navigation Maths", "TCPA negative", "application",
    "A contact's TCPA changes from +8 s to -3 s between two consecutive updates. What just happened?",
    "The closest point of approach moment has just passed, roughly between those two updates -- "
    "the vessels were closing, reached minimum range, and are now diverging. This is an expected, "
    "normal transition as any encounter resolves, not itself an alarming event; what matters is "
    "what the CPA value was at the moment TCPA crossed zero.",
    "Demonstrates TCPA crossing from positive to negative as a normal event marking the passing "
    "moment, not a fault or a new danger.")

add("Basic Navigation Maths", "TCPA negative", "application",
    "Should an agent ever treat a very large negative TCPA (e.g. -900 s) combined with a stale "
    "contact age as meaningful for current decision-making?",
    "Generally no. A large negative TCPA means the projected closest approach was a long time "
    "ago under the course/speed assumptions of that projection. If the contact data is also "
    "stale (large age_s), the projection itself is unreliable -- the target has likely changed "
    "course or speed since, and a fresh bearing/range reading should be used instead of trusting "
    "an old negative-TCPA projection.",
    "Connects TCPA sign interpretation to the broader point (raised earlier in this project, re: "
    "stale contact age_s) that old projections should not be treated as current fact.")

# =====================================================================
# PART 2 -- Relative bearing sign convention drills
# =====================================================================
add("Basic Navigation Maths", "Relative bearing convention", "definition",
    "State the sign convention this system uses for relative bearing.",
    "relative_bearing = (true_bearing_to_target - own_heading + 540) % 360 - 180, giving a "
    "value in the range (-180, +180]. POSITIVE means the contact is on the STARBOARD side "
    "(clockwise from the bow). NEGATIVE means the contact is on the PORT side (counter-clockwise "
    "from the bow). 0 deg is dead ahead; +90 deg is the starboard beam; -90 deg is the port "
    "beam; +/-180 deg is dead astern.",
    "This is the project's own established convention (used throughout narrate.py, "
    "colreg_llm_bridge_paused.py, evaluate_run.py) -- consistency with this exact convention "
    "matters more than which of several equally valid conventions is used, since the agent's "
    "own state data is generated with this one.")

add("Basic Navigation Maths", "Relative bearing convention", "definition",
    "How does this system's signed convention relate to the 0-360 clockwise 'XXXR' notation "
    "used in classic radar plotting (e.g. tdgil.com's Bearings lesson)?",
    "They are the same underlying angle, expressed two different ways. The 0-360 clockwise form "
    "(e.g. '045R' = 45 deg clockwise of the bow) maps directly onto this system's signed form: "
    "0-360 values from 0 to 180 correspond to +0 to +180 here (starboard); 0-360 values from "
    "180 to 360 correspond to -180 to -0 here (port). Converting: signed = (value_0_360 + 180) % 360 - 180.",
    "Explicit bridge between the two representations, since converting between them is itself "
    "a place sign errors can be introduced.")

# --- Given heading + true bearing, compute relative bearing and side ---
heading_bearing_pairs = [
    (0, 30), (0, 330), (45, 90), (45, 0), (90, 180), (90, 45),
    (180, 90), (180, 270), (270, 350), (270, 190), (135, 135), (315, 45),
    (60, 240), (200, 20), (10, 200), (350, 100),
]
for own_hdg, true_brg in heading_bearing_pairs:
    rel = relative_bearing_signed(own_hdg, true_brg)
    zero360 = to_0_360_form(rel)
    side = side_of(rel)
    add("Basic Navigation Maths", "Relative bearing calculation", "application",
        f"Own-ship heading is {own_hdg:03d} deg. A contact's true bearing is {true_brg:03d} deg. "
        f"What is the relative bearing (signed), and which side is the contact on?",
        f"Relative bearing = ({true_brg} - {own_hdg} + 540) mod 360 - 180 = {fmt_deg(rel)} deg. "
        f"Since this value is {'positive' if rel > 0 else 'negative' if rel < 0 else 'zero'}, the "
        f"contact is on the {side}. (Equivalent 0-360 clockwise notation: {fmt_deg(zero360)}R.)",
        f"Direct application of the sign convention: {fmt_deg(rel)} deg {'> 0 -> starboard' if rel>0 else '< 0 -> port' if rel<0 else '= 0 -> dead ahead'}.")

# --- Given a signed relative bearing, state the side (pure sign drill, many values) ---
pure_sign_values = [5, 15, 30, 45, 60, 75, 89, -5, -15, -30, -45, -60, -75, -89,
                    91, 120, 150, 175, -91, -120, -150, -175]
for rel in pure_sign_values:
    side = side_of(rel)
    add("Basic Navigation Maths", "Relative bearing sign drill", "recall",
        f"A contact has relative bearing {'+' if rel>0 else ''}{rel} deg in this system's signed "
        f"convention. Which side is it on: port or starboard?",
        f"{side.capitalize()}.",
        f"Sign rule: positive = starboard, negative = port. {rel} is "
        f"{'positive, so starboard' if rel > 0 else 'negative, so port'}.",
        difficulty="easy")

# --- Boundary cases ---
for rel, desc in [(0, "dead ahead"), (180, "dead astern"), (-180, "dead astern"),
                   (90, "exactly on the starboard beam"), (-90, "exactly on the port beam")]:
    add("Basic Navigation Maths", "Relative bearing boundary cases", "recall",
        f"A contact has relative bearing exactly {rel} deg. Describe its position and state "
        f"whether it counts as 'port' or 'starboard' in the strict sense.",
        f"This is {desc}. At exactly {rel} deg, the contact is neither purely port nor purely "
        f"starboard -- it sits exactly on the boundary (dead ahead or dead astern), or exactly "
        f"abeam if +/-90.",
        "Boundary values (0, +/-90, +/-180) are edge cases where a strict port/starboard label "
        "doesn't cleanly apply the way it does for any other value; an agent should recognise "
        "these rather than force an arbitrary port-or-starboard answer.",
        difficulty="easy")

# --- 0-360 clockwise form -> signed form conversion drills ---
for zero360 in [10, 45, 90, 135, 179, 181, 200, 225, 270, 315, 350]:
    signed = zero360 if zero360 <= 180 else zero360 - 360
    side = side_of(signed)
    add("Basic Navigation Maths", "Bearing notation conversion", "application",
        f"A radar-style bearing is given as {zero360:03d}R (0-360 clockwise notation, per the "
        f"tdgil.com Bearings convention). Convert to this system's signed convention and state "
        f"the side.",
        f"{zero360:03d}R converts to {'+' if signed>0 else ''}{signed} deg in signed form "
        f"(values 0-180 stay the same and are starboard; values 180-360 become "
        f"value-360 and are port). The contact is on the {side}.",
        f"Conversion rule: signed = value_0_360 if value_0_360 <= 180 else value_0_360 - 360.",
        difficulty="easy")

# --- R+S=T drills, grounded directly in the tdgil.com Bearings page ---
for own_hdg, rel_signed in [(90, 20), (170, 60), (320, -110), (45, -170), (0, 175)]:
    zero360 = to_0_360_form(rel_signed)
    true_brg = (own_hdg + zero360) % 360
    add("Basic Navigation Maths", "R+S=T formula", "application",
        f"Own-ship heading (S) is {own_hdg:03d} deg. Relative bearing (R, this system's signed "
        f"form) to a contact is {'+' if rel_signed>0 else ''}{rel_signed} deg. Using the R+S=T "
        f"rule (Relative bearing + Ship's heading = True bearing), find the true bearing (T) to "
        f"the contact.",
        f"Convert R to 0-360 clockwise form first: {'+' if rel_signed>0 else ''}{rel_signed} deg "
        f"-> {fmt_deg(zero360)}R. Then T = (S + R) mod 360 = ({own_hdg} + {fmt_deg(zero360)}) mod "
        f"360 = {fmt_deg(true_brg)} deg.",
        "Grounded directly in the tdgil.com Bearings lesson's R+S=T formula; note R must be in "
        "the 0-360 clockwise form for this formula, not the signed form -- another place a "
        "silent conversion error can creep in if not made explicit.")

print(f"Generated {len(items)} items")
from collections import Counter
cnt = Counter(it["topic"] for it in items)
for k, v in sorted(cnt.items()):
    print(f"  {k}: {v}")

with open("/home/claude/nav_drills/nav_maths_drills.json", "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2, ensure_ascii=False)
print("Wrote nav_maths_drills.json")
