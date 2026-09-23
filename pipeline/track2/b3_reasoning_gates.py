"""Fase B3 (RAG-rebuild-v2 plan, 2026-09-22): deterministic acceptance gates every
LLM-generated reasoning row must pass before being accepted into a training/review set.
Shared by BOTH Track-2 generators (build_oow_scenarios.py / build_oow_scenarios_leo.py)
so a row produced by either can never bypass a check the other one enforces.

A row that fails ANY gate is retried (the gate's failure messages are fed back to the
teacher as feedback, see run_gated_generation()) up to a caller-supplied max attempts,
then DROPPED and logged -- NEVER forced through. Per-gate rejection counts are the
residual-error measurement the RAG-rebuild-v2 plan asks for.
"""
from __future__ import annotations
import json
import re

from pipeline.oow_agent_spec import validate_action_json

GATE_NAMES = ("schema", "decision_match", "rule_match", "contact_consistency",
             "number_consistency", "threshold_wording", "risk_consistency")

_NUMBER_WITH_UNIT_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:m\b|metres?|meters?|knots?|kn\b|degrees?|deg\b|"
    r"minutes?|min\b|seconds?|secs?|s\b)",
    re.IGNORECASE,
)
_ANY_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_ABOVE_WORDS = ("above", "exceeds", "exceeding", "greater than", "more than", "over",
               "in excess of", "outside")
_BELOW_WORDS = ("below", "under", "beneath", "less than", "within", "inside", "shy of")
# Standing rules (2/5/6/7/11) always apply and are legitimately mentioned as background
# without being "the" encounter/conduct rule -- never counted as an unexpected citation.
_STANDING_RULES = {"2", "5", "6", "7", "11"}
# Fixed COLREG-defined numeric constants the reasoning is entitled to cite regardless of
# whether they happen to appear in a given situation text -- e.g. Rule 13's own
# definition of "overtaking" (more than 22.5 deg abaft the beam) is a REGULATORY
# threshold, not an invented situational fact.
_KNOWN_REGULATORY_CONSTANTS = {22.5}


def extract_numbers_with_units(text: str) -> set[float]:
    """Numbers immediately followed by a distance/speed/angle/time unit word -- excludes
    bare small integers (rule numbers, ordinals) that would otherwise false-positive.
    Used for the REASONING side only -- see extract_any_numbers() for the situation
    side, which has no such units to require (e.g. "heading 173.8," has no unit word)."""
    return {round(float(m), 1) for m in _NUMBER_WITH_UNIT_RE.findall(text)}


def extract_any_numbers(text: str) -> set[float]:
    """Every standalone number in `text`, no unit required -- for the SITUATION side,
    which is fully trusted/deterministic (heading/bearing values there have no trailing
    unit word), so there is no false-positive risk from being permissive here."""
    return {round(float(m), 1) for m in _ANY_NUMBER_RE.findall(text)}


def gate_schema(obj: dict | None) -> str | None:
    if obj is None:
        return "no parseable JSON object in the response"
    errors = validate_action_json(obj)
    return "; ".join(errors) if errors else None


def gate_decision_match(obj: dict, expected_action: str, expected_degrees: float | None) -> str | None:
    """The action/degrees were ALREADY decided (answer-side facts, not the teacher's to
    choose) -- the teacher must restate them verbatim, never invent a different one."""
    if obj.get("action") != expected_action:
        return f"action {obj.get('action')!r} != the already-decided {expected_action!r}"
    if expected_degrees is None:
        if obj.get("degrees") is not None:
            return f"degrees must be null for {expected_action!r}, got {obj.get('degrees')!r}"
    elif obj.get("degrees") != expected_degrees:
        return f"degrees {obj.get('degrees')!r} != the already-decided {expected_degrees!r}"
    return None


def gate_rule_match(obj: dict, expected_encounter_rule: str, expected_conduct_rule: str,
                    reasoning: str) -> str | None:
    """The JSON's encounter_rule/conduct_rule must equal the ground-truth pair (see
    pipeline.oow_agent_spec.classify_rules()), AND the reasoning prose must cite the
    same rule numbers and no others (standing rules 2/5/6/7/11 excepted)."""
    if obj.get("encounter_rule") != expected_encounter_rule:
        return (f"encounter_rule {obj.get('encounter_rule')!r} != ground truth "
               f"{expected_encounter_rule!r}")
    if obj.get("conduct_rule") != expected_conduct_rule:
        return f"conduct_rule {obj.get('conduct_rule')!r} != ground truth {expected_conduct_rule!r}"
    mentioned = set(re.findall(r"Rule\s+(\d+)", reasoning)) - _STANDING_RULES
    expected = {r.split(" ", 1)[1] for r in (expected_encounter_rule, expected_conduct_rule) if r != "none"}
    extra = mentioned - expected
    if extra:
        return f"reasoning cites Rule(s) {sorted(extra)} not in the expected {sorted(expected)}"
    if expected and not (mentioned & expected):
        return f"reasoning never cites any of the expected Rule(s) {sorted(expected)}"
    return None


def gate_contact_consistency(reasoning: str, decisive_contact_name: str | None) -> str | None:
    """The reasoning must name the contact that actually drove the decision -- catches
    the model reasoning about a DIFFERENT (non-decisive) contact in a multi-contact
    situation (the leo00003/leo00009 failure mode)."""
    if decisive_contact_name is None:
        return None  # no real risk -- no decisive contact to check
    if decisive_contact_name not in reasoning:
        return f"reasoning never mentions the decisive contact {decisive_contact_name!r} by name"
    return None


def gate_number_consistency(reasoning: str, situation: str,
                            extra_allowed: frozenset[float] = frozenset()) -> str | None:
    """Every CPA/distance/speed/angle/time number in the reasoning must also appear
    (within rounding) in the situation text, the given decision's own degrees, or a
    known fixed COLREG regulatory constant (_KNOWN_REGULATORY_CONSTANTS) -- catches
    invented numbers. Reasoning-side numbers are extracted strictly (unit required, so
    rule-number mentions like "Rule 14" never count); situation-side numbers are
    extracted permissively (no unit required -- heading/bearing values in the situation
    text have none) since that side is fully trusted/deterministic. Signed situation
    values (rel.bearing is signed, negative=port) are matched against BOTH their signed
    and absolute form -- naturally-phrased reasoning almost always drops the sign in
    favour of "X degrees to port/starboard" wording, which is correct prose, not an
    invented number."""
    reasoning_nums = extract_numbers_with_units(reasoning)
    situation_raw = extract_any_numbers(situation)
    situation_nums = (situation_raw | {abs(v) for v in situation_raw}
                     | {round(v, 1) for v in extra_allowed} | _KNOWN_REGULATORY_CONSTANTS)
    invented = sorted(n for n in reasoning_nums if not any(abs(n - s) <= 1.0 for s in situation_nums))
    if invented:
        return f"reasoning states number(s) {invented} not found in the situation text"
    return None


def gate_threshold_wording(reasoning: str, cpa_m: float | None, safe_distance_m: float,
                          tcpa_s: float | None = None, risk_horizon_s: float | None = None) -> str | None:
    """If the reasoning explicitly compares a CPA to the safe passing distance using an
    above/below-type word, that comparison must be arithmetically correct (the
    leo00022 failure mode: '272 m well above 500 m', when 272 < 500). Quality-review
    STAP 5 (2026-09-23): the SAME check now also applies to TCPA vs the row's OWN
    risk_horizon_s (per-row now, never a hardcoded 300s) -- both `safe_distance_m` and
    `risk_horizon_s` MUST be the row's actual sampled values (see
    build_teacher_payload()), never the historical fixed defaults, so this gate cannot
    silently mis-judge a row that sampled a different safe_distance_m/risk_horizon_s."""
    if cpa_m is not None:
        safe_str = f"{safe_distance_m:.0f}"
        if safe_str in reasoning:
            low = reasoning.lower()
            idx = low.find(safe_str.lower())
            window = low[max(0, idx - 100):idx + 100]
            said_above = any(w in window for w in _ABOVE_WORDS)
            said_below = any(w in window for w in _BELOW_WORDS)
            actual_below = cpa_m < safe_distance_m
            if said_above and not said_below and actual_below:
                return (f"reasoning claims CPA is above/exceeds the {safe_distance_m:.0f}m safe "
                       f"distance, but CPA={cpa_m:.0f}m is actually below it")
            if said_below and not said_above and not actual_below:
                return (f"reasoning claims CPA is below/under the {safe_distance_m:.0f}m safe "
                       f"distance, but CPA={cpa_m:.0f}m is actually at or above it")
    if tcpa_s is not None and risk_horizon_s is not None:
        horizon_str = f"{risk_horizon_s:.0f}"
        if horizon_str in reasoning:
            low = reasoning.lower()
            idx = low.find(horizon_str.lower())
            window = low[max(0, idx - 100):idx + 100]
            said_above = any(w in window for w in _ABOVE_WORDS)
            said_below = any(w in window for w in _BELOW_WORDS)
            actual_below = 0 <= tcpa_s < risk_horizon_s
            if said_above and not said_below and actual_below:
                return (f"reasoning claims TCPA is above/exceeds the {risk_horizon_s:.0f}s risk "
                       f"horizon, but TCPA={tcpa_s:.0f}s is actually within it")
            if said_below and not said_above and not actual_below:
                return (f"reasoning claims TCPA is below/under the {risk_horizon_s:.0f}s risk "
                       f"horizon, but TCPA={tcpa_s:.0f}s is actually at or beyond it (or negative)")
    return None


def gate_risk_consistency(reasoning: str, real_risk: bool) -> str | None:
    """The reasoning's own risk conclusion must match the deterministic ground truth."""
    low = reasoning.lower()
    no_risk_phrases = ("no real risk", "no collision risk", "no give-way", "no stand-on",
                       "not a collision risk", "no rule applies", "no risk of collision")
    if real_risk and any(p in low for p in no_risk_phrases):
        return "reasoning claims no real risk exists, but ground truth says real risk is present"
    return None


def run_gates(obj: dict | None, reasoning: str | None, *, situation: str, expected_action: str,
             expected_degrees: float | None, expected_encounter_rule: str, expected_conduct_rule: str,
             real_risk: bool, cpa_m: float | None, safe_distance_m: float,
             decisive_contact_name: str | None, tcpa_s: float | None = None,
             risk_horizon_s: float | None = None) -> dict[str, str]:
    """Runs every gate; returns {gate_name: failure_message} for only the gates that
    failed (empty dict == accept)."""
    if obj is None or not isinstance(reasoning, str) or not reasoning.strip():
        return {"schema": "no parseable response from the teacher"}
    checks = {
        "schema": gate_schema(obj),
        "decision_match": gate_decision_match(obj, expected_action, expected_degrees),
        "rule_match": gate_rule_match(obj, expected_encounter_rule, expected_conduct_rule, reasoning),
        "contact_consistency": gate_contact_consistency(reasoning, decisive_contact_name),
        "number_consistency": gate_number_consistency(
            reasoning, situation, frozenset({expected_degrees} if expected_degrees is not None else set())),
        "threshold_wording": gate_threshold_wording(reasoning, cpa_m, safe_distance_m, tcpa_s, risk_horizon_s),
        "risk_consistency": gate_risk_consistency(reasoning, real_risk),
    }
    return {name: msg for name, msg in checks.items() if msg}


def extract_first_json_object(text: str) -> dict | None:
    """Best-effort: find the first {...} block and parse it. Tolerant of a teacher
    wrapping the object in prose or markdown fences."""
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    return obj
                start = None
    return None


def generate_gated_row(client, model: str, system_prompt: str, payload: dict, *,
                       situation: str, expected_action: str, expected_degrees: float | None,
                       expected_encounter_rule: str, expected_conduct_rule: str,
                       real_risk: bool, cpa_m: float | None, safe_distance_m: float,
                       decisive_contact_name: str | None, tcpa_s: float | None = None,
                       risk_horizon_s: float | None = None, max_attempts: int = 3,
                       max_tokens: int = 1024) -> tuple[dict | None, list[dict]]:
    """One row, one decision, gated + retried (feeding the gate failures back to the
    teacher as feedback) up to `max_attempts` times. Returns (accepted_obj_or_None,
    attempt_log) -- attempt_log has one entry per attempt with its gate failures, so a
    caller can tally per-gate rejection counts even for a row that eventually succeeds.
    NEVER forces a failing row through -- a row that exhausts max_attempts is dropped
    (returns None), the caller decides what to log/report."""
    user_content = json.dumps(payload, ensure_ascii=False, indent=1)
    attempt_log: list[dict] = []
    for attempt in range(1, max_attempts + 1):
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        text = resp.content[0].text
        obj = extract_first_json_object(text)
        reasoning = obj.get("reasoning") if obj else None
        failures = run_gates(
            obj, reasoning, situation=situation, expected_action=expected_action,
            expected_degrees=expected_degrees, expected_encounter_rule=expected_encounter_rule,
            expected_conduct_rule=expected_conduct_rule, real_risk=real_risk, cpa_m=cpa_m,
            safe_distance_m=safe_distance_m, decisive_contact_name=decisive_contact_name,
            tcpa_s=tcpa_s, risk_horizon_s=risk_horizon_s,
        )
        attempt_log.append({"attempt": attempt, "failures": failures, "raw": text[:800]})
        if not failures:
            return obj, attempt_log
        user_content = (
            json.dumps(payload, ensure_ascii=False, indent=1)
            + "\n\nYour previous answer was REJECTED for these reasons:\n- "
            + "\n- ".join(failures.values())
            + "\n\nProduce a corrected response, following the original instructions exactly. "
             "Reply with ONLY the JSON object, no other text."
        )
    return None, attempt_log
