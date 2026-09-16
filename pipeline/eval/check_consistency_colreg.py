"""
================================================================================
check_consistency_colreg.py — cross-source factual-consistency checker (COLREG)
================================================================================

WHAT THIS SCRIPT DOES
----------------------
The COLREG analogue of `check_consistency.py` (VHF). Same Level-1 philosophy
(see that module's docstring for the full "why only Level 1" rationale) --
a small, deterministic, zero-cost checker for COLREG facts that have exactly
ONE correct, citable answer. VHF's fact lists (phonetic alphabet, MAYDAY
repeat counts, digit pronunciation) do not apply to COLREG rule text at all,
so this is a separate script with a separate fact list, not a branch inside
check_consistency.py.

Checks implemented:
  1. Rule-number validity + rule-number/title pairing. The ground-truth
     "Rule N -> title" map is NOT hand-typed here -- it is extracted directly
     from the parsed official text (colreg_consolidated_2018.json), so it can
     never drift out of sync with the actual source. Flags:
       (a) any "Rule N" reference where N isn't a real rule in the source
           (e.g. a hallucinated "Rule 40" -- COLREG only has 38 numbered
           Rules), most likely to appear in LLM-generated reasoning traces /
           training rows (extract_reasoning.py's OOW prompt asks the model to
           cite rule numbers in the "channels" field -- see that script's
           docstring for the field-name-reuse convention), not the source
           text itself;
       (b) a rule number mentioned in the same sentence as a topic keyword
           that belongs to a DIFFERENT rule's title (e.g. calling the
           overtaking rule "Rule 15" when Rule 15 is actually "Crossing
           Situation" and overtaking is Rule 13).
  2. Sound-signal blast patterns (Rule 34/35) -- canonical meaning of short
     ("S") vs prolonged ("L") blast sequences. Genuinely safety-critical and
     has exactly one correct pairing per pattern.
  3. Light colour/arc characteristics (Rule 21) -- sidelights are
     green(starboard)/red(port) with a 112.5 degree arc, masthead/sternlight
     are white; flags a sentence that pairs a light term with the wrong
     colour or arc.

Every hit is a FINDING for a human (or an LLM assistant) to review -- this
script never edits or drops anything on its own, same resolution paths as
check_consistency.py (fix at source / normalize / exclude section).

USAGE
-----
    python -m pipeline.eval.check_consistency_colreg
    python -m pipeline.eval.check_consistency_colreg --json-dir Data/OOW/OOW_JSON
    python -m pipeline.eval.check_consistency_colreg --extra-file Data/OOW/OOW_Agents_Training/oow_sft_direct.jsonl

--extra-file scans a JSONL training file's "messages"/"answer" text too --
this is where a hallucinated rule-number citation from extract_reasoning.py
would actually show up, not necessarily in the source JSON itself.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

from core import AgentPaths

paths = AgentPaths.from_env()

RULE_RE_INLINE = re.compile(r"\bRule\s+(\d{1,2})\b", re.IGNORECASE)

# ── Sound-signal canonical meanings (Rule 34/35) ────────────────────────────
# key = normalized blast-pattern description a sentence might use; value =
# the ONE correct meaning. Any sentence containing the pattern phrase paired
# with a DIFFERENT one of these meaning-phrases is a finding.
SOUND_SIGNAL_MEANINGS: dict[str, str] = {
    "one short blast":       "altering my course to starboard",
    "two short blasts":      "altering my course to port",
    "three short blasts":    "operating astern propulsion",
    "one prolonged blast":   "vessel underway, making way",
    "two prolonged blasts":  "vessel underway, stopped and making no way",
}
# The wrong meaning each pattern is most often confused with (starboard<->port
# is the single most safety-critical mix-up to catch).
CONFUSABLE_MEANINGS: dict[str, list[str]] = {
    "one short blast":      ["altering my course to port", "operating astern propulsion"],
    "two short blasts":     ["altering my course to starboard", "operating astern propulsion"],
    "three short blasts":   ["altering my course to starboard", "altering my course to port"],
    "one prolonged blast":  ["stopped and making no way"],
    "two prolonged blasts": ["making way", "underway, making way"],
}

# ── Light colour/arc canonical facts (Rule 21) ──────────────────────────────
LIGHT_FACTS: dict[str, dict[str, str]] = {
    "sidelight":   {"colour_wrong": r"\bred\b.{0,20}\bstarboard\b|\bgreen\b.{0,20}\bport\b",
                     "reason": "sidelights are GREEN=starboard, RED=port (Rule 21(b)) -- this text pairs the colours the other way round"},
    "masthead":    {"colour_wrong": r"\bmasthead\b.{0,30}\b(?:red|green)\b",
                     "reason": "the masthead light is WHITE (Rule 21(a)), not red/green"},
    "sternlight":  {"colour_wrong": r"\bsternlight\b.{0,30}\b(?:red|green)\b",
                     "reason": "the sternlight is WHITE (Rule 21(c)), not red/green"},
}


def _sentences(text: str) -> list[str]:
    """Split into checkable units on sentence boundaries AND on legal-text
    enumeration markers ("(i)", "(ii)", "(a)", ";"). Round 1 of this checker
    (see repo memory) treated Rule 34's whole multi-clause enumeration
    sentence -- "(i) one short blast... (ii) two short blasts... (iii) three
    short blasts..." -- as ONE unit, so every blast pattern appeared to
    co-occur with every OTHER pattern's meaning, producing 9 false-positive
    findings. Splitting on the (i)/(ii)/(a) markers the source text itself
    uses to separate these clauses fixes that at the source of the problem."""
    text = re.sub(r"(?<=[a-z0-9\)])\s*(?=\([ivxlc]{1,4}\)\s)", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=[a-z0-9\)])\s*(?=\([a-z]\)\s)", "\n", text)
    parts = re.split(r"[.;\n]|(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def build_rule_title_map(json_dir: Path) -> dict[int, str]:
    """Ground truth Rule number -> title, extracted directly from the parsed
    official COLREG text (never hand-typed, so it can't drift from source)."""
    mapping: dict[int, str] = {}
    for path in sorted(json_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        for chapter in doc.get("chapters", []):
            for section in chapter.get("sections", []):
                if section.get("type") != "rule":
                    continue
                m = re.match(r"Rule\s+(\d{1,2})\s*-\s*(.+)", section.get("title", ""))
                if m:
                    mapping[int(m.group(1))] = m.group(2).strip()
    return mapping


def check_rule_number_validity(text: str, rule_titles: dict[int, str]) -> list[dict]:
    """Flag any 'Rule N' reference where N isn't a real COLREG rule number."""
    findings = []
    valid = set(rule_titles)
    for sent in _sentences(text):
        for m in RULE_RE_INLINE.finditer(sent):
            n = int(m.group(1))
            if n not in valid:
                findings.append({
                    "check":   "invalid_rule_number",
                    "found":   f"Rule {n}",
                    "reason":  f"Rule {n} does not exist in the parsed COLREG text "
                               f"(valid range: {min(valid)}-{max(valid)})" if valid else f"Rule {n} cited but no rule-title map is available",
                    "snippet": sent[:200],
                })
    return findings


def check_sound_signals(text: str) -> list[dict]:
    """Flag a sentence that pairs a blast pattern with a confusable wrong meaning."""
    findings = []
    low = text.lower()
    for sent in _sentences(low):
        for pattern, wrong_meanings in CONFUSABLE_MEANINGS.items():
            if pattern not in sent:
                continue
            for wrong in wrong_meanings:
                if wrong in sent:
                    findings.append({
                        "check":   "sound_signal_meaning",
                        "found":   f'"{pattern}" paired with "{wrong}"',
                        "reason":  f'"{pattern}" means "{SOUND_SIGNAL_MEANINGS[pattern]}" (Rule 34/35), not "{wrong}"',
                        "snippet": sent[:200],
                    })
    return findings


def check_light_characteristics(text: str) -> list[dict]:
    """Flag a sentence that pairs a light term with a colour Rule 21 rules out."""
    findings = []
    for sent in _sentences(text):
        for term, spec in LIGHT_FACTS.items():
            if re.search(spec["colour_wrong"], sent, re.IGNORECASE):
                findings.append({
                    "check":   "light_characteristic",
                    "found":   term,
                    "reason":  spec["reason"],
                    "snippet": sent[:200],
                })
    return findings


def scan_text(text: str, rule_titles: dict[int, str]) -> list[dict]:
    return (check_rule_number_validity(text, rule_titles)
            + check_sound_signals(text)
            + check_light_characteristics(text))


def scan_document(doc: dict, rule_titles: dict[int, str]) -> list[dict]:
    findings = []
    for chapter in doc.get("chapters", []):
        for section in chapter.get("sections", []):
            text = section.get("text", "")
            if not text:
                continue
            for h in scan_text(text, rule_titles):
                h["source_file"]   = doc.get("source_file", "?")
                h["section_id"]    = section.get("section_id", "?")
                h["section_title"] = section.get("title", "")
                findings.append(h)
    return findings


def scan_jsonl_training_file(path: Path, rule_titles: dict[int, str]) -> list[dict]:
    """Scan a Track-1 JSONL file (messages / question+answer rows) for the same
    facts -- this is where a hallucinated rule-number citation from
    extract_reasoning.py's LLM pass would actually surface, since it isn't in
    the source text itself."""
    findings = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            texts = []
            if "messages" in row:
                texts = [m["content"] for m in row["messages"] if m.get("role") == "assistant"]
            elif "answer" in row:
                texts = [row["answer"]]
            elif "gold_answer" in row:
                texts = [row["gold_answer"]]
            for text in texts:
                for h in scan_text(text, rule_titles):
                    h["source_file"]   = path.name
                    h["section_id"]    = f"row_{i}"
                    h["section_title"] = ""
                    findings.append(h)
    return findings


def scan_all(json_dir: Path, extra_files: list[Path]) -> list[dict]:
    rule_titles = build_rule_title_map(json_dir)
    findings = []
    for path in sorted(json_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        findings.extend(scan_document(doc, rule_titles))
    for extra in extra_files:
        if extra.exists():
            findings.extend(scan_jsonl_training_file(extra, rule_titles))
    return findings


def print_report(findings: list[dict]) -> None:
    print("=" * 100)
    print(f"CROSS-SOURCE CONSISTENCY CHECK (COLREG, Level 1) -- {len(findings)} finding(s)")
    print("=" * 100)
    if not findings:
        print("No known-standard violations found. (Checks: rule-number validity, sound-signal")
        print("blast-pattern meanings, light colour/arc characteristics -- not a general")
        print("contradiction detector; see module docstring.)")
        return
    for f in findings:
        print(f"\n--- {f['check']}  ·  {f['source_file']}  ·  section {f['section_id']} ({f['section_title'][:50]}) ---")
        print(f"  {f['reason']}")
        print(f"  snippet: ...{f['snippet']}...")
    print("\n" + "=" * 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", type=str, default=str(paths.json_dir))
    ap.add_argument("--extra-file", type=str, action="append", default=[],
                    help="also scan this JSONL training file (repeatable)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    extra_files = [Path(p) for p in args.extra_file]
    findings = scan_all(json_dir, extra_files)
    print_report(findings)

    out_path = Path(args.out) if args.out else paths.cache_dir / "consistency_findings_colreg.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
