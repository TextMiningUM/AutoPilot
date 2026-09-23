#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates simple_colreg.json — a plain-language COLREG reference written in
this project's own situation_report register (range, closing speed, CPA/
TCPA, risk assessment language), not legal-treaty phrasing. Covers Parts
A-E and Annexes I-IV.

EXAMPLES ARE DELIBERATELY DISTINCT FROM oow_colreg_scenarios.json
That file is the evaluation set (325 items, 13 categories: head_on,
crossing_give_way_fine/broad/abeam, crossing_stand_on_fine/broad,
overtaking_give_way/stand_on, risk_cleared_resume, double_crossing_squeeze,
headon_plus_crossing, overtake_plus_crossing, converging_cluster). Every
example below uses different concrete bearings/speeds/vessel types than
those categories' parametrization (their pattern: own_speed clustered
around 10.0, target cruise speed 17.0, bearings snapped to fine/broad/abeam
buckets on power-driven-only encounters). This file's examples vary speed,
bearing, and vessel type more freely, and cover many rules the eval set
doesn't touch at all (lights, sound signals, narrow channels, vessel
hierarchy, restricted visibility) specifically to avoid doubling as a
disguised copy of the eval categories.
"""
import json

items = []

# Official rule text, transcribed from COLREG-Consolidated-2018.pdf (the
# uploaded document) -- kept as a SEPARATE field from chunk_text (which
# stays plain-language) so embedding the plain explanation isn't diluted
# by legal phrasing. Grouped rule_refs (e.g. "Rule 20-22") include each
# constituent rule's text concatenated with a label, since those chunks
# cover more than one rule number.
OFFICIAL_TEXT = {
"Rule 1": (
"(a) These Rules shall apply to all vessels upon the high seas and in all waters "
"connected therewith navigable by seagoing vessels.\n"
"(b) Nothing in these Rules shall interfere with the operation of special rules made "
"by an appropriate authority for roadsteads, harbours, rivers, lakes or inland "
"waterways connected with the high seas and navigable by seagoing vessels. Such "
"special rules shall conform as closely as possible to these Rules."
),
"Rule 2": (
"(a) Nothing in these Rules shall exonerate any vessel, or the owner, master or crew "
"thereof, from the consequences of any neglect to comply with these Rules or of the "
"neglect of any precaution which may be required by the ordinary practice of "
"seamen, or by the special circumstances of the case.\n"
"(b) In construing and complying with these Rules due regard shall be had to all "
"dangers of navigation and collision and to any special circumstances, including "
"the limitations of the vessels involved, which may make a departure from these "
"Rules necessary to avoid immediate danger."
),
"Rule 3": (
"For the purpose of these Rules, except where the context otherwise requires: "
"(b) The term 'power-driven vessel' means any vessel propelled by machinery. "
"(c) The term 'sailing vessel' means any vessel under sail provided that propelling "
"machinery, if fitted, is not being used. "
"(i) The word 'underway' means that a vessel is not at anchor, or made fast to the "
"shore, or aground. "
"(k) Vessels shall be deemed to be in sight of one another only when one can be "
"observed visually from the other. "
"(l) The term 'restricted visibility' means any condition in which visibility is "
"restricted by fog, mist, falling snow, heavy rainstorms, sandstorms or any other "
"similar causes."
),
"Rule 5": (
"Every vessel shall at all times maintain a proper look-out by sight and hearing as "
"well as by all available means appropriate in the prevailing circumstances and "
"conditions so as to make a full appraisal of the situation and of the risk of "
"collision."
),
"Rule 6": (
"Every vessel shall at all times proceed at a safe speed so that she can take proper "
"and effective action to avoid collision and be stopped within a distance "
"appropriate to the prevailing circumstances and conditions. In determining a safe "
"speed the following factors shall be among those taken into account: "
"(a) By all vessels: (i) the state of visibility; (ii) the traffic density including "
"concentrations of fishing vessels or any other vessels; (iii) the manoeuvrability "
"of the vessel with special reference to stopping distance and turning ability in "
"the prevailing conditions; (iv) at night the presence of background light such as "
"from shore lights or from back scatter of her own lights; (v) the state of wind, "
"sea and current, and the proximity of navigational hazards; (vi) the draught in "
"relation to the available depth of water. "
"(b) Additionally, by vessels with operational radar: (i) the characteristics, "
"efficiency and limitations of the radar equipment; (ii) any constraints imposed by "
"the radar range scale in use; (iii) the effect on radar detection of the sea "
"state, weather and other sources of interference; (iv) the possibility that small "
"vessels, ice and other floating objects may not be detected by radar at an "
"adequate range; (v) the number, location and movement of vessels detected by "
"radar; (vi) the more exact assessment of the visibility that may be possible when "
"radar is used to determine the range of vessels or other objects in the vicinity."
),
"Rule 7": (
"(a) Every vessel shall use all available means appropriate to the prevailing "
"circumstances and conditions to determine if risk of collision exists. If there is "
"any doubt such risk shall be deemed to exist. "
"(b) Proper use shall be made of radar equipment if fitted and operational, "
"including long-range scanning to obtain early warning of risk of collision and "
"radar plotting or equivalent systematic observation of detected objects. "
"(c) Assumptions shall not be made on the basis of scanty information, especially "
"scanty radar information. "
"(d) In determining if risk of collision exists the following considerations shall "
"be among those taken into account: (i) such risk shall be deemed to exist if the "
"compass bearing of an approaching vessel does not appreciably change; (ii) such "
"risk may sometimes exist even when an appreciable bearing change is evident, "
"particularly when approaching a very large vessel or a tow or when approaching a "
"vessel at close range."
),
"Rule 8": (
"(a) Any action to avoid collision shall be taken in accordance with the Rules of "
"this Part and shall, if the circumstances of the case admit, be positive, made in "
"ample time and with due regard to the observance of good seamanship. "
"(b) Any alteration of course and/or speed to avoid collision shall, if the "
"circumstances of the case admit, be large enough to be readily apparent to another "
"vessel observing visually or by radar; a succession of small alterations of "
"course and/or speed should be avoided. "
"(c) If there is sufficient sea room, alteration of course alone may be the most "
"effective action to avoid a close-quarters situation provided that it is made in "
"good time, is substantial and does not result in another close-quarters "
"situation. "
"(d) Action taken to avoid collision with another vessel shall be such as to result "
"in passing at a safe distance. The effectiveness of the action shall be carefully "
"checked until the other vessel is finally past and clear. "
"(e) If necessary to avoid collision or allow more time to assess the situation, a "
"vessel shall slacken her speed or take all way off by stopping or reversing her "
"means of propulsion."
),
"Rule 9": (
"(a) A vessel proceeding along the course of a narrow channel or fairway shall keep "
"as near to the outer limit of the channel or fairway which lies on her starboard "
"side as is safe and practicable. "
"(b) A vessel of less than 20 metres in length or a sailing vessel shall not impede "
"the passage of a vessel which can safely navigate only within a narrow channel or "
"fairway. "
"(c) A vessel engaged in fishing shall not impede the passage of any other vessel "
"navigating within a narrow channel or fairway. "
"(d) A vessel shall not cross a narrow channel or fairway if such crossing impedes "
"the passage of a vessel which can safely navigate only within such channel or "
"fairway. "
"(e)(i) In a narrow channel or fairway when overtaking can take place only if the "
"vessel to be overtaken has to take action to permit safe passing, the vessel "
"intending to overtake shall indicate her intention by sounding the appropriate "
"signal prescribed in Rule 34(c)(i). "
"(g) Any vessel shall, if the circumstances of the case admit, avoid anchoring in a "
"narrow channel."
),
"Rule 10": (
"(b) A vessel using a traffic separation scheme shall: (i) proceed in the "
"appropriate traffic lane in the general direction of traffic flow for that lane; "
"(ii) so far as practicable keep clear of a traffic separation line or separation "
"zone; (iii) normally join or leave a traffic lane at the termination of the lane, "
"but when joining or leaving from either side shall do so at as small an angle to "
"the general direction of traffic flow as practicable. "
"(c) A vessel shall so far as practicable avoid crossing traffic lanes, but if "
"obliged to do so shall cross on a heading as nearly as practicable at right angles "
"to the general direction of traffic flow."
),
"Rule 12": (
"(a) When two sailing vessels are approaching one another, so as to involve risk of "
"collision, one of them shall keep out of the way of the other as follows: "
"(i) when each has the wind on a different side, the vessel which has the wind on "
"the port side shall keep out of the way of the other; "
"(ii) when both have the wind on the same side, the vessel which is to windward "
"shall keep out of the way of the vessel which is to leeward; "
"(iii) if a vessel with the wind on the port side sees a vessel to windward and "
"cannot determine with certainty whether the other vessel has the wind on the port "
"or on the starboard side, she shall keep out of the way of the other."
),
"Rule 13": (
"(a) Notwithstanding anything contained in the Rules of Part B, Sections I and II "
"any vessel overtaking any other shall keep out of the way of the vessel being "
"overtaken. "
"(b) A vessel shall be deemed to be overtaking when coming up with another vessel "
"from a direction more than 22.5 degrees abaft her beam, that is, in such a "
"position with reference to the vessel she is overtaking, that at night she would "
"be able to see only the sternlight of that vessel but neither of her sidelights. "
"(c) When a vessel is in any doubt as to whether she is overtaking another, she "
"shall assume that this is the case and act accordingly. "
"(d) Any subsequent alteration of the bearing between the two vessels shall not "
"make the overtaking vessel a crossing vessel within the meaning of these Rules or "
"relieve her of the duty of keeping clear of the overtaken vessel until she is "
"finally past and clear."
),
"Rule 14": (
"(a) When two power-driven vessels are meeting on reciprocal or nearly reciprocal "
"courses so as to involve risk of collision each shall alter her course to "
"starboard so that each shall pass on the port side of the other. "
"(b) Such a situation shall be deemed to exist when a vessel sees the other ahead "
"or nearly ahead and by night she could see the masthead lights of the other in a "
"line or nearly in a line and/or both sidelights and by day she observes the "
"corresponding aspect of the other vessel. "
"(c) When a vessel is in any doubt as to whether such a situation exists she shall "
"assume that it does exist and act accordingly."
),
"Rule 15": (
"When two power-driven vessels are crossing so as to involve risk of collision, "
"the vessel which has the other on her own starboard side shall keep out of the "
"way and shall, if the circumstances of the case admit, avoid crossing ahead of "
"the other vessel."
),
"Rule 16": (
"Every vessel which is directed to keep out of the way of another vessel shall, so "
"far as possible, take early and substantial action to keep well clear."
),
"Rule 17": (
"(a)(i) Where one of two vessels is to keep out of the way the other shall keep her "
"course and speed. "
"(a)(ii) The latter vessel may however take action to avoid collision by her "
"manoeuvre alone, as soon as it becomes apparent to her that the vessel required "
"to keep out of the way is not taking appropriate action in compliance with these "
"Rules. "
"(b) When, from any cause, the vessel required to keep her course and speed finds "
"herself so close that collision cannot be avoided by the action of the give-way "
"vessel alone, she shall take such action as will best aid to avoid collision. "
"(c) A power-driven vessel which takes action in a crossing situation in "
"accordance with sub-paragraph (a)(ii) of this Rule to avoid collision with "
"another power-driven vessel shall, if the circumstances of the case admit, not "
"alter course to port for a vessel on her own port side. "
"(d) This Rule does not relieve the give-way vessel of her obligation to keep out "
"of the way."
),
"Rule 18": (
"Except where Rules 9, 10 and 13 otherwise require: "
"(a) A power-driven vessel underway shall keep out of the way of: (i) a vessel not "
"under command; (ii) a vessel restricted in her ability to manoeuvre; (iii) a "
"vessel engaged in fishing; (iv) a sailing vessel. "
"(b) A sailing vessel underway shall keep out of the way of: (i) a vessel not "
"under command; (ii) a vessel restricted in her ability to manoeuvre; (iii) a "
"vessel engaged in fishing. "
"(c) A vessel engaged in fishing when underway shall, so far as possible, keep out "
"of the way of: (i) a vessel not under command; (ii) a vessel restricted in her "
"ability to manoeuvre. "
"(d)(i) Any vessel other than a vessel not under command or a vessel restricted in "
"her ability to manoeuvre shall, if the circumstances of the case admit, avoid "
"impeding the safe passage of a vessel constrained by her draught, exhibiting the "
"signals in Rule 28. (ii) A vessel constrained by her draught shall navigate with "
"particular caution having full regard to her special condition."
),
"Rule 19": (
"(a) This Rule applies to vessels not in sight of one another when navigating in "
"or near an area of restricted visibility. "
"(b) Every vessel shall proceed at a safe speed adapted to the prevailing "
"circumstances and conditions of restricted visibility. A power-driven vessel "
"shall have engines ready for immediate manoeuvre. "
"(d) A vessel which detects by radar alone the presence of another vessel shall "
"determine if a close-quarters situation is developing and/or risk of collision "
"exists. If so, she shall take avoiding action in ample time, provided that when "
"such action consists of an alteration of course, so far as possible the "
"following shall be avoided: (i) an alteration of course to port for a vessel "
"forward of the beam, other than for a vessel being overtaken; (ii) an alteration "
"of course towards a vessel abeam or abaft the beam. "
"(e) Except where it has been determined that a risk of collision does not exist, "
"every vessel which hears apparently forward of her beam the fog signal of "
"another vessel, or which cannot avoid a close-quarters situation with another "
"vessel forward of her beam, shall reduce her speed to the minimum at which she "
"can be kept on her course. She shall if necessary take all her way off and in "
"any event navigate with extreme caution until danger of collision is over."
),
"Rule 20-22": (
"Rule 20(b): The Rules concerning lights shall be complied with from sunset to "
"sunrise. Rule 20(d): The Rules concerning shapes shall be complied with by day. "
"Rule 22: The lights prescribed in these Rules shall have an intensity... so as to "
"be visible at minimum ranges of: in vessels of 50 metres or more, a masthead "
"light 6 miles, a sidelight 3 miles; in vessels of 12 to 50 metres, a masthead "
"light 5 miles (3 miles if under 20m), a sidelight 2 miles; in vessels of less "
"than 12 metres, a masthead light 2 miles, a sidelight 1 mile."
),
"Rule 23": (
"(a) A power-driven vessel underway shall exhibit: (i) a masthead light forward; "
"(ii) a second masthead light abaft of and higher than the forward one; except "
"that a vessel of less than 50 metres in length shall not be obliged to exhibit "
"such light but may do so; (iii) sidelights; (iv) a sternlight. "
"(d)(i) A power-driven vessel of less than 12 metres in length may in lieu of the "
"lights prescribed in paragraph (a) exhibit an all-round white light and "
"sidelights."
),
"Rule 24": (
"(a) A power-driven vessel when towing shall exhibit: (i) instead of the light "
"prescribed in Rule 23(a)(i) or (a)(ii), two masthead lights in a vertical line. "
"When the length of the tow exceeds 200 metres, three such lights in a vertical "
"line; (ii) sidelights; (iii) a sternlight; (iv) a towing light in a vertical line "
"above the sternlight; (v) when the length of the tow exceeds 200 metres, a "
"diamond shape where it can best be seen."
),
"Rule 25": (
"(a) A sailing vessel underway shall exhibit: (i) sidelights; (ii) a sternlight. "
"(b) In a sailing vessel of less than 20 metres in length the lights prescribed in "
"paragraph (a) may be combined in one lantern carried at or near the top of the "
"mast where it can best be seen. "
"(e) A vessel proceeding under sail when also being propelled by machinery shall "
"exhibit forward where it can best be seen a conical shape, apex downwards."
),
"Rule 26": (
"(b) A vessel when engaged in trawling... shall exhibit: (i) two all-round lights "
"in a vertical line, the upper being green and the lower white, or a shape "
"consisting of two cones with their apexes together in a vertical line one above "
"the other. "
"(c) A vessel engaged in fishing, other than trawling, shall exhibit: (i) two "
"all-round lights in a vertical line, the upper being red and the lower white, or "
"a shape consisting of two cones with apexes together in a vertical line one "
"above the other; (ii) when there is outlying gear extending more than 150 metres "
"horizontally from the vessel, an all-round white light or a cone apex upwards in "
"the direction of the gear."
),
"Rule 27": (
"(a) A vessel not under command shall exhibit: (i) two all-round red lights in a "
"vertical line where they can best be seen; (ii) two balls or similar shapes in a "
"vertical line where they can best be seen. "
"(b) A vessel restricted in her ability to manoeuvre, except a vessel engaged in "
"mineclearance operations, shall exhibit: (i) three all-round lights in a "
"vertical line where they can best be seen. The highest and lowest of these "
"lights shall be red and the middle light shall be white; (ii) three shapes in a "
"vertical line where they can best be seen. The highest and lowest of these "
"shapes shall be balls and the middle one a diamond."
),
"Rule 28-29": (
"Rule 28: A vessel constrained by her draught may, in addition to the lights "
"prescribed for power-driven vessels in Rule 23, exhibit where they can best be "
"seen three all-round red lights in a vertical line, or a cylinder. "
"Rule 29(a): A vessel engaged on pilotage duty shall exhibit: (i) at or near the "
"masthead, two all-round lights in a vertical line, the upper being white and the "
"lower red; (ii) when underway, in addition, sidelights and a sternlight."
),
"Rule 30": (
"(a) A vessel at anchor shall exhibit where it can best be seen: (i) in the fore "
"part, an all-round white light or one ball; (ii) at or near the stern and at a "
"lower level than the light prescribed in sub-paragraph (i), an all-round white "
"light. "
"(d) A vessel aground shall exhibit the lights prescribed in paragraph (a) or (b) "
"of this Rule and in addition, where they can best be seen: (i) two all-round red "
"lights in a vertical line; (ii) three balls in a vertical line."
),
"Rule 32-34": (
"Rule 32(b): The term 'short blast' means a blast of about one second's duration. "
"Rule 32(c): The term 'prolonged blast' means a blast of from four to six "
"seconds' duration. "
"Rule 34(a): When vessels are in sight of one another, a power-driven vessel "
"underway, when manoeuvring as authorized or required by these Rules, shall "
"indicate that manoeuvre by the following signals on her whistle: one short "
"blast to mean 'I am altering my course to starboard'; two short blasts to mean "
"'I am altering my course to port'; three short blasts to mean 'I am operating "
"astern propulsion'. "
"Rule 34(d): When vessels in sight of one another are approaching each other and "
"from any cause either vessel fails to understand the intentions or actions of "
"the other, or is in doubt whether sufficient action is being taken by the other "
"to avoid collision, the vessel in doubt shall immediately indicate such doubt by "
"giving at least five short and rapid blasts on the whistle."
),
"Rule 35": (
"(a) A power-driven vessel making way through the water shall sound at intervals "
"of not more than 2 minutes one prolonged blast. "
"(b) A power-driven vessel underway but stopped and making no way through the "
"water shall sound at intervals of not more than 2 minutes two prolonged blasts "
"in succession with an interval of about 2 seconds between them. "
"(c) A vessel not under command, a vessel restricted in her ability to "
"manoeuvre, a vessel constrained by her draught, a sailing vessel, a vessel "
"engaged in fishing and a vessel engaged in towing or pushing another vessel "
"shall, instead of the signals prescribed in paragraphs (a) or (b) of this Rule "
"sound at intervals of not more than 2 minutes three blasts in succession, "
"namely one prolonged followed by two short blasts."
),
"Rule 36-37": (
"Rule 36: If necessary to attract the attention of another vessel any vessel may "
"make light or sound signals that cannot be mistaken for any signal authorized "
"elsewhere in these Rules, or may direct the beam of her searchlight in the "
"direction of the danger, in such a way as not to embarrass any vessel. "
"Rule 37: When a vessel is in distress and requires assistance she shall use or "
"exhibit the signals described in Annex IV to these Regulations."
),
"Rule 38": (
"Any vessel (or class of vessels)... the keel of which is laid or which is at a "
"corresponding stage of construction before the entry into force of these "
"Regulations may be exempted from compliance therewith as follows: (a) The "
"installation of lights with ranges prescribed in Rule 22, until four years "
"after the date of entry into force of these Regulations... (c) The "
"repositioning of lights as a result of conversion from Imperial to metric units "
"and rounding off measurement figures, permanent exemption."
),
"Annex I": (
"Section 2(a): On a power-driven vessel of 20 metres or more in length the "
"masthead lights shall be placed... at a height above the hull of not less than "
"6 metres... Section 6(a): Shapes shall be black and of the following sizes: (i) "
"a ball shall have a diameter of not less than 0.6 metre; (ii) a cone shall have "
"a base diameter of not less than 0.6 metre and a height equal to its diameter. "
"Section 7: The chromaticity of all navigation lights shall conform to... "
"standards... specified... by the International Commission on Illumination (CIE)."
),
"Annex II": (
"Section 2(a): Vessels of 20 m or more in length when engaged in trawling... "
"shall exhibit: (i) when shooting their nets: two white lights in a vertical "
"line; (ii) when hauling their nets: one white light over one red light in a "
"vertical line; (iii) when the net has come fast upon an obstruction: two red "
"lights in a vertical line. "
"Section 3: Vessels engaged in fishing with purse seine gear may exhibit two "
"yellow lights in a vertical line. These lights shall flash alternately every "
"second... These lights may be exhibited only when the vessel is hampered by its "
"fishing gear."
),
"Annex III": (
"Section 1(a): The fundamental frequency of the signal shall lie within the "
"range 70-700Hz. "
"Section 1(b): the fundamental frequency of a whistle shall be between the "
"following limits: (i) 70-200 Hz, for a vessel 200 metres or more in length; "
"(ii) 130-350 Hz, for a vessel 75 but less than 200 metres in length; (iii) "
"250-700 Hz, for a vessel less than 75 metres in length. "
"Section 2(a): A bell or gong... shall produce a sound pressure level of not "
"less than 110 dB at a distance of 1 metre from it."
),
"Annex IV": (
"1. The following signals, used or exhibited either together or separately, "
"indicate distress and need of assistance: (a) a gun or other explosive signal "
"fired at intervals of about a minute; (b) a continuous sounding with any "
"fog-signalling apparatus; (c) rockets or shells, throwing red stars fired one "
"at a time at short intervals; (e) a signal sent by radiotelephony consisting of "
"the spoken word 'MAYDAY'; (k) slowly and repeatedly raising and lowering arms "
"outstretched to each side. "
"2. The use or exhibition of any of the foregoing signals except for the "
"purpose of indicating distress and need of assistance... is prohibited."
),
}

def add(part, rule_ref, title, plain, examples):
    # stable, human+machine-readable chunk id: part + rule number
    part_slug = part.lower().replace(" - ", "_").replace(" ", "_")
    rule_slug = rule_ref.lower().replace(" ", "_")
    chunk_id = f"{part_slug}__{rule_slug}"

    # chunk_text: the single, self-contained string meant to be embedded
    # as ONE chunk. Every field a retriever would need to make sense of
    # this rule on its own -- without having to pull in neighbouring
    # array items -- lives inside this one string, clearly labelled.
    # Deliberately plain-language ONLY -- official_text is kept as a
    # separate field so legal phrasing doesn't dilute this embedding.
    examples_block = "\n".join(f"  {i+1}. {ex}" for i, ex in enumerate(examples))
    chunk_text = (
        f"[{part} | {rule_ref}: {title}]\n"
        f"{plain.strip()}\n"
        f"Examples:\n"
        f"{examples_block}"
    )

    official_text = OFFICIAL_TEXT.get(rule_ref, "").strip()

    items.append({
        "chunk_id": chunk_id,
        "part": part,
        "rule_ref": rule_ref,
        "title": title,
        "plain_explanation": plain.strip(),
        "examples": examples,
        "official_text": official_text,
        "official_text_source": "COLREG-Consolidated-2018.pdf (IMO, 1972 as amended)" if official_text else None,
        "chunk_text": chunk_text,
    })

# =====================================================================
# PART A — GENERAL
# =====================================================================
add("Part A - General", "Rule 1", "Application",
    "These rules apply anywhere you're operating on open water, plus any connected water a "
    "seagoing vessel could reach. A local authority can add stricter local rules for a harbour, "
    "river, or lake, but those can't contradict these rules — only tighten them.",
    ["Own-ship transits from open sea into a harbour approach: the same COLREG rules keep "
     "applying, plus whatever local traffic rules the port authority has layered on top."])

add("Part A - General", "Rule 2", "Responsibility / good seamanship",
    "Following the letter of these rules doesn't excuse bad judgement. If ordinary prudent "
    "seamanship — or the specific circumstances — called for something extra, you're still on "
    "the hook for not doing it. And if departing from the rules is the only way to avoid an "
    "immediate danger, you're allowed to.",
    ["Own-ship technically has right of way, but a nearby vessel is clearly not maneuvering "
     "predictably (erratic heading changes, no response to hails) — Rule 2 expects own-ship to "
     "increase caution beyond the bare minimum the geometry alone would require."])

add("Part A - General", "Rule 3", "Key definitions",
    "A few terms recur constantly in situation reports and need precise meaning: 'power-driven "
    "vessel' = under engine; 'sailing vessel' = under sail with engine off; 'underway' = not "
    "anchored, moored, or aground; 'vessels in sight of one another' = each can actually be "
    "seen, not just detected on radar/AIS; 'restricted visibility' = fog, heavy rain, or "
    "anything similarly reducing sight.",
    ["A contact is detected at 8000 m by AIS but visibility is 400 m due to fog — the two "
     "vessels are NOT 'in sight of one another' yet, even though own-ship has full positional "
     "data on the contact."])

# =====================================================================
# SECTION I — ANY CONDITION OF VISIBILITY (Rules 4-10)
# =====================================================================
add("Section I - Any Visibility", "Rule 5", "Look-out",
    "Keep watching and listening at all times, using every tool available (radar, AIS, "
    "binoculars, hydrophone) — not just one sensor. The goal is a full, current picture of the "
    "situation and whether risk of collision exists.",
    ["Own-ship's AIS shows no nearby contacts, but a small unlit vessel without AIS could still "
     "be present — Rule 5 means own-ship can't treat 'AIS is empty' as equivalent to 'area is "
     "clear'."])

add("Section I - Any Visibility", "Rule 6", "Safe speed",
    "Your speed has to let you stop or maneuver in time given the conditions — visibility, "
    "traffic density, your own stopping distance and turning ability, sea state, and (if you "
    "have radar) its range-scale limitations and how well it's actually detecting small or "
    "distant contacts right now.",
    ["Own-ship is doing 15 kt in an area with six other AIS contacts inside a 3 km radius and "
     "visibility dropping to 1500 m — even with no single contact yet showing high risk, Rule 6 "
     "argues for reducing speed given the traffic density and shrinking visibility together."])

add("Section I - Any Visibility", "Rule 7", "Determining risk of collision",
    "Use everything available to work out whether risk of collision exists, and if there's any "
    "doubt, treat it as if the risk exists. A steady compass bearing on an approaching contact "
    "(the bearing barely changing) is the classic sign — even if the range is still opening "
    "slowly. Don't draw conclusions from a thin radar picture; a couple of scattered returns "
    "aren't enough to decide there's no risk.",
    ["Contact bearing holds at 047 +/- 1 deg over three successive readings while range closes "
     "from 4000 m to 3200 m — bearing isn't changing, so risk of collision should be treated as "
     "real even though 3200 m still feels comfortable."])

add("Section I - Any Visibility", "Rule 8", "Action to avoid collision",
    "Whatever you do, do it early, do it decisively, and make it big enough that another vessel "
    "watching visually or on radar can actually tell you've done something — a string of tiny "
    "course nudges is worse than one clear turn. If there's sea room, a course change alone is "
    "often better than a speed change alone. Check that your action actually opens a safe gap, "
    "not just a technically-different number. And if you need thinking time, it's fine to slow "
    "down or stop to buy it.",
    ["Own-ship nudges course by 3 deg, waits, nudges another 4 deg, waits again — this pattern "
     "is exactly what Rule 8(b) warns against; a single clear 25-30 deg alteration would be far "
     "more readable to the other vessel and far more effective."])

add("Section I - Any Visibility", "Rule 9", "Narrow channels",
    "Inside a narrow channel or fairway, keep to the outer edge on your own starboard side if "
    "it's safe to do so — don't hug the centre or the wrong side. Small craft and sailing "
    "vessels must not get in the way of a large vessel that can only safely navigate within the "
    "channel. Don't cross a channel if it would force a channel-bound vessel to react to you. "
    "Overtaking inside a channel needs a whistle-signal exchange first. And avoid anchoring in "
    "the channel altogether if you can help it.",
    ["A 12 m sailing yacht wants to cross a dredged channel while a 200 m bulk carrier is "
     "transiting it and physically cannot leave the channel to avoid the yacht — the yacht must "
     "wait, regardless of who'd normally have right of way in open water."])

add("Section I - Any Visibility", "Rule 10", "Traffic separation schemes",
    "Inside a traffic separation scheme, go with the flow of that lane, stay clear of the "
    "separation zone/line as much as possible, and join or leave at a shallow angle near the "
    "ends of the lane rather than cutting across. If you must cross the whole scheme, cross as "
    "close to a right angle as you can, as quickly as possible — don't loiter in it.",
    ["Own-ship needs to cross a TSS to reach a berth on the far side — instead of angling "
     "diagonally across (spending more time inside the lanes than necessary), own-ship should "
     "turn onto a heading close to 90 deg from the lane's general traffic direction for the "
     "crossing itself."])

# =====================================================================
# SECTION II — VESSELS IN SIGHT OF ONE ANOTHER (Rules 11-18)
# =====================================================================
add("Section II - In Sight", "Rule 12", "Sailing vessels meeting",
    "Between two sailing vessels: if the wind is on different sides for each, the one with wind "
    "on the port side keeps clear. If the wind is on the same side for both, whichever is "
    "upwind keeps clear of the one downwind. If you can't tell which side the wind is on for "
    "the other vessel, assume you're the one who has to keep clear.",
    ["Two sailboats converging, own-ship on a port tack (wind over the port side), the other on "
     "a starboard tack — own-ship is the one required to keep clear, regardless of which vessel "
     "reaches the crossing point first."])

add("Section II - In Sight", "Rule 13", "Overtaking",
    "This one overrides every other steering rule in this section: whoever is overtaking must "
    "keep clear of the vessel being overtaken, full stop, no matter what category either vessel "
    "normally falls into. You're overtaking if you're coming up on someone from more than 22.5 "
    "deg abaft their beam — close enough that at night you'd only see their sternlight, not "
    "either sidelight. If it's genuinely unclear whether you're overtaking or crossing, assume "
    "overtaking. And once you're the overtaking vessel, you stay the overtaking vessel — a "
    "later bearing shift doesn't hand the obligation back.",
    ["Own-ship, doing 9 kt, is closing on a slower fishing vessel doing 4 kt dead ahead, from a "
     "position well behind its beam — even though fishing vessels normally rank above "
     "power-driven vessels in the general hierarchy, own-ship as the overtaking vessel still "
     "must keep clear here, not the other way around."])

add("Section II - In Sight", "Rule 14", "Head-on situation",
    "Two power-driven vessels meeting more or less bow-to-bow on reciprocal courses both alter "
    "to starboard, so each passes on the other's port side. Nobody is 'stand-on' here — it's a "
    "shared, mutual obligation. If there's any doubt whether it's truly head-on, treat it as "
    "head-on.",
    ["Own-ship on heading 090, contact dead ahead on heading 273 (nearly reciprocal) — both "
     "vessels are expected to turn starboard; own-ship shouldn't wait to see what the other "
     "vessel does first, since the duty applies to both simultaneously."])

add("Section II - In Sight", "Rule 15", "Crossing situation",
    "Two power-driven vessels crossing with risk of collision: whichever one has the other on "
    "its own starboard side is the give-way vessel, and should avoid cutting across the other's "
    "bow if at all practical.",
    ["Own-ship on heading 000, contact bearing 050 relative (on own-ship's starboard side), "
     "closing steadily — own-ship is give-way here and should turn to pass behind the contact, "
     "not try to beat it across."])

add("Section II - In Sight", "Rule 16", "Give-way vessel's duty",
    "If you're the one required to keep clear, do it early and substantially — this isn't the "
    "moment for a token gesture.",
    ["Own-ship is give-way in a crossing situation with 6 minutes of TCPA remaining — waiting "
     "until 90 seconds out to react would violate the spirit of 'early,' even if there's "
     "technically still time to avoid a collision at that point."])

add("Section II - In Sight", "Rule 17", "Stand-on vessel's duty",
    "If you're the stand-on vessel, hold your course and speed — don't start second-guessing "
    "and maneuvering just because a vessel is approaching. You're allowed to act on your own if "
    "it becomes clear the other vessel isn't actually keeping clear, and you're required to act "
    "if collision can no longer be avoided by the give-way vessel alone. When you do act, avoid "
    "turning to port toward a vessel that's on your own port side, if you have room to do "
    "something else. None of this excuses the give-way vessel from its own obligation.",
    ["Own-ship is stand-on in a crossing situation; the give-way contact's bearing has stopped "
     "changing and range keeps closing with 90 seconds of TCPA left — this is exactly the point "
     "where Rule 17(b) requires own-ship to act, not continue waiting."])

add("Section II - In Sight", "Rule 18", "Responsibilities between vessel types",
    "Outside of the narrow-channel, TSS, and overtaking exceptions above, there's a general "
    "pecking order: power-driven vessels keep clear of NUC (not under command), RAM (restricted "
    "in ability to manoeuvre), fishing vessels, and sailing vessels, in roughly that priority. "
    "Sailing vessels keep clear of NUC, RAM, and fishing vessels. Fishing vessels keep clear of "
    "NUC and RAM where possible. A vessel constrained by her draught gets extra consideration "
    "from everyone except NUC/RAM vessels, but isn't automatically top of the hierarchy the way "
    "NUC/RAM are.",
    ["Own-ship (power-driven) is approaching a vessel showing a diamond-over-two-balls day shape "
     "(restricted in ability to manoeuvre) — own-ship must keep clear here even if, by pure "
     "crossing-angle geometry alone, own-ship would otherwise have been the stand-on vessel."])

# =====================================================================
# SECTION III — RESTRICTED VISIBILITY (Rule 19)
# =====================================================================
add("Section III - Restricted Visibility", "Rule 19", "Conduct in restricted visibility",
    "This applies when vessels aren't in sight of each other and visibility is reduced. "
    "Everyone must run at a speed suited to the conditions, with engines ready for immediate "
    "manoeuvre. If you pick up a contact on radar alone and it looks like risk of collision, act "
    "early — but avoid turning to port for a contact that's forward of your beam (unless you're "
    "overtaking it), and avoid turning toward a contact that's abeam or behind your beam. If you "
    "hear a fog signal forward of your beam, or can't avoid a close-quarters situation with "
    "something forward of your beam, slow to the minimum speed that still lets you hold your "
    "course, and stop completely if you have to. There's no 'stand-on' role here — both vessels "
    "share the duty to act cautiously.",
    ["Own-ship, in fog, picks up a radar contact 25 deg off the port bow with a closing CPA — "
     "turning further to port toward that contact would violate Rule 19(d); own-ship should "
     "turn starboard or reduce speed instead."])

# =====================================================================
# PART C — LIGHTS AND SHAPES (Rules 20-31)
# =====================================================================
add("Part C - Lights and Shapes", "Rule 20-22", "General light rules and visibility ranges",
    "Lights run from sunset to sunrise (and in restricted visibility during the day too); "
    "shapes are a daytime-only equivalent. Bigger vessels need lights visible from further "
    "away — a 50 m+ vessel needs a masthead light seen from 6 nm, a sub-12 m vessel only needs "
    "2 nm.",
    ["A 20 m fishing vessel's masthead light is only visible from 2.5 nm in clear conditions — "
     "below the 5 nm minimum required for her size class, worth flagging as a possible "
     "equipment fault rather than assuming it's within spec."])

add("Part C - Lights and Shapes", "Rule 23", "Power-driven vessels underway",
    "A power-driven vessel underway shows a forward masthead light, a second higher masthead "
    "light further aft if she's 50 m or more, sidelights (green to starboard, red to port), and "
    "a sternlight. Small vessels under 12 m can simplify down to a single all-round white light "
    "plus sidelights.",
    ["A contact shows only sidelights and a sternlight at night, no masthead light visible — "
     "combined with no other markings, this points toward a sailing vessel rather than a "
     "power-driven one (Rule 25), since a power-driven vessel underway is required to show a "
     "masthead light."])

add("Part C - Lights and Shapes", "Rule 24", "Towing and pushing",
    "A vessel towing shows two (or three, for a long tow) masthead lights stacked vertically "
    "instead of the usual one or two, plus a yellow towing light above the sternlight. The "
    "towed vessel or object shows sidelights and a sternlight of its own, and a diamond shape if "
    "the tow is long.",
    ["A vessel shows three masthead lights in a vertical line instead of the usual configuration "
     "— that specifically signals a tow exceeding 200 m in length, not just 'a tug with "
     "something behind it.'"])

add("Part C - Lights and Shapes", "Rule 25", "Sailing vessels and vessels under oars",
    "A sailing vessel underway shows sidelights and a sternlight, no masthead light. Under 20 m, "
    "she can combine all three into a single tricolour lantern at the masthead instead. A "
    "vessel under sail that's also running her engine counts legally as power-driven and must "
    "show power-driven lights plus a cone shape, apex down, forward.",
    ["A yacht is sailing but her engine is also engaged to help make way — even though the sails "
     "are up, she's legally a power-driven vessel for COLREG purposes and must be treated (and "
     "must show lights) accordingly, not as a sailing vessel."])

add("Part C - Lights and Shapes", "Rule 26", "Fishing vessels",
    "Trawling shows green-over-white all-round lights (or two cones point-to-point by day). "
    "Fishing with other gear shows red-over-white instead, plus an extra white light in the "
    "direction of any outlying gear extending beyond 150 m.",
    ["A vessel shows red over white (not green over white) — that's fishing with static or "
     "line gear, not trawling; the practical implication is different, since her gear may extend "
     "well beyond her own hull in a fixed direction rather than trailing directly astern."])

add("Part C - Lights and Shapes", "Rule 27", "Vessels not under command or restricted in ability to manoeuvre",
    "NUC shows two all-round red lights stacked vertically (two balls by day). RAM shows three "
    "lights stacked red-white-red (ball-diamond-ball by day) — the middle light/shape is the "
    "giveaway that distinguishes RAM from NUC.",
    ["A contact shows red-white-red all-round lights rather than just two reds — that middle "
     "white light means this is a RAM vessel, not NUC; worth noting since the practical "
     "implication (she may still be manoeuvring, just in a constrained way) differs from a "
     "vessel that's fully unable to manoeuvre."])

add("Part C - Lights and Shapes", "Rule 28-29", "Constrained by draught and pilot vessels",
    "A vessel constrained by her draught may show three red lights stacked vertically (a "
    "cylinder shape by day), in addition to normal power-driven lights. A vessel on pilotage "
    "duty shows white-over-red at the masthead.",
    ["A large tanker in a shallow approach channel shows three vertical red lights above her "
     "normal navigation lights — that's a draught-constrained signal, meaning she has very "
     "limited ability to deviate from her track regardless of what COLREG category would "
     "otherwise apply."])

add("Part C - Lights and Shapes", "Rule 30", "Anchored vessels and vessels aground",
    "An anchored vessel under 50 m shows a single all-round white light (a ball by day); 50 m+ "
    "shows one forward and one lower one aft. A vessel aground shows the anchor lights plus two "
    "red lights vertically (three balls by day).",
    ["A contact shows a single all-round white light and isn't moving at all on successive "
     "position updates — that's consistent with an anchored vessel under 50 m, not a "
     "power-driven vessel with a light fault, given zero speed over multiple reports."])

# =====================================================================
# PART D — SOUND AND LIGHT SIGNALS (Rules 32-37)
# =====================================================================
add("Part D - Sound and Light Signals", "Rule 32-34", "Manoeuvring and warning signals",
    "One short blast means 'I'm altering to starboard,' two short blasts means 'I'm altering to "
    "port,' three short blasts means 'I'm going astern.' If you're unsure what another vessel is "
    "doing, or doubt they're doing enough to avoid collision, sound at least five short rapid "
    "blasts — the danger signal — immediately.",
    ["A give-way vessel starts a starboard turn but the stand-on vessel's bearing/range trend "
     "doesn't change and TCPA keeps shrinking — this is exactly when the stand-on vessel should "
     "sound the five-short-blast danger signal, not just keep watching."])

add("Part D - Sound and Light Signals", "Rule 35", "Sound signals in restricted visibility",
    "A power-driven vessel making way sounds one prolonged blast every 2 minutes; stopped and "
    "not making way, it's two prolonged blasts. NUC, RAM, constrained-by-draught, sailing, "
    "fishing, and towing/pushing vessels all use one prolonged plus two short blasts instead, "
    "regardless of whether they're making way.",
    ["A sound signal comes in as one prolonged blast followed by two short — that pattern alone "
     "identifies the source as one of the special-category vessels (NUC/RAM/CBD/sailing/fishing/"
     "towing), not an ordinary power-driven vessel, before any visual contact is even made."])

add("Part D - Sound and Light Signals", "Rule 36-37", "Attracting attention and distress",
    "You can use any light or sound signal to get another vessel's attention as long as it "
    "can't be confused with a signal already defined elsewhere in these rules. Actual distress "
    "signals are a separate, specific list (see Annex IV) — don't improvise those.",
    ["Own-ship wants to warn a distracted-looking nearby vessel of risk without implying an "
     "actual emergency — a searchlight beam aimed carefully (not blinding anyone) is appropriate "
     "under Rule 36; sounding what could be mistaken for a distress signal would not be."])

# =====================================================================
# PART E — EXEMPTIONS
# =====================================================================
add("Part E - Exemptions", "Rule 38", "Exemptions for older vessels",
    "Vessels built before certain rule changes took effect get narrow, specific transitional "
    "exemptions on things like exact light positioning — this only applies to genuinely older "
    "construction, and never exempts anyone from the actual steering and sailing conduct rules.",
    ["An older vessel's masthead light sits slightly outside the modern positioning spec due to "
     "a pre-amendment construction date — that specific technical detail may be exempted, but "
     "she's still fully bound by Rules 4-19 like any other vessel."])

# =====================================================================
# ANNEX I — Technical details of lights and shapes
# =====================================================================
add("Annex I", "Annex I", "Technical positioning of lights and shapes",
    "This is the engineering detail behind Part C: exact heights, spacing, colour "
    "specifications, and intensity formulas for lights, and minimum sizes/colours for shapes. "
    "Mostly relevant for equipment design and inspection, not moment-to-moment navigation "
    "decisions.",
    ["A recognition question about whether a contact's two masthead lights are correctly spaced "
     "to be distinguishable at 1000 m is an Annex I technical-compliance question, not something "
     "that changes which COLREG steering rule applies to the encounter."])

# =====================================================================
# ANNEX II — Additional fishing-fleet signals
# =====================================================================
add("Annex II", "Annex II", "Extra signals for fishing vessels working close together",
    "On top of the normal Rule 26 fishing lights, vessels fishing near each other as a fleet "
    "use extra white-light patterns (trawlers) or a flashing yellow pair (purse seiners) to "
    "signal specific states like shooting nets, hauling, or being hampered by gear.",
    ["A trawler shows an extra pair of white lights beyond her normal green-over-white — that "
     "additional signal is telling nearby fishing vessels something operational (e.g. hauling "
     "nets), not communicating anything to a passing merchant vessel outside the fleet."])

# =====================================================================
# ANNEX III — Technical details of sound signal appliances
# =====================================================================
add("Annex III", "Annex III", "Technical specs for whistles, bells, and gongs",
    "Defines the frequency range, minimum loudness, and audibility range required for a "
    "vessel's whistle based on her length, plus construction standards for bells and gongs. "
    "Equipment-design detail, not a navigation-decision rule.",
    ["A vessel's whistle operates at a frequency outside the required 70-700 Hz range for her "
     "size class — that's an Annex III equipment-compliance issue, unrelated to whether her "
     "actual manoeuvring decisions were COLREG-correct."])

# =====================================================================
# ANNEX IV — Distress signals
# =====================================================================
add("Annex IV", "Annex IV", "Recognised distress signals",
    "A specific, closed list of signals count as distress: things like continuous fog-horn "
    "sounding, red flares, the spoken word 'Mayday,' slowly raising and lowering both arms, and "
    "various radio/EPIRB signals. Using any of these for anything other than genuine distress is "
    "against the rules — and improvising a different signal to mean distress doesn't count "
    "either.",
    ["A vessel fires a single flare that throws white stars, not red — because Annex IV "
     "specifically requires red stars, this particular signal doesn't meet the defined distress "
     "signal, even though it's clearly meant to draw attention."])

# ---------------------------------------------------------------------
print(f"Generated {len(items)} rule entries")
total_examples = sum(len(it["examples"]) for it in items)
print(f"Total examples: {total_examples}")

with open("/home/claude/simple_colreg/simple_colreg.json", "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2, ensure_ascii=False)
print("Wrote simple_colreg.json")
