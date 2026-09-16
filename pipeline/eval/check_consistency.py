"""
================================================================================
check_consistency.py — cross-source factual-consistency checker (Level 1)
================================================================================

WHAT THIS SCRIPT DOES
----------------------
A document can be 99% excellent and still contain one wrong or non-standard
claim (a mis-spelled phonetic-alphabet code word, a distress call repeated
the wrong number of times, ...). Left uncaught, that one claim gets chunked
(§ 9), turned into a reasoning trace (§ 11), and then multiplied into several
Q&A/DPO/reflection training rows (§ 12) — so a single bad sentence in a
source document can quietly out-vote the correct version that is already
present, elsewhere, in the rest of the corpus.

This script does NOT try to detect arbitrary contradictions between any two
pieces of text (that requires semantic judgement — see the module docstring
section "WHY ONLY LEVEL 1" below). Instead it is a small, deterministic,
zero-cost checker for a short list of facts that have exactly ONE correct,
citable, standardized answer:

  1. NATO/ITU phonetic-alphabet code words (A=ALFA, not "Alpha"; B=BRAVO,
     not "Beta"; ...) — see PHONETIC_ALPHABET below.
  2. Distress/urgency/safety call-sign repetition counts (MAYDAY x3,
     PAN PAN x3, SECURITE x3) — see REPEAT_PROWORDS below.
  3. Standardized VHF digit pronunciation inside quoted digit-by-digit
     readouts (positions, MMSI, POB counts) — e.g. "FIFE" not "FIVE",
     "NINER" not "NINE" — see NUMBER_WORDS below. Only checked inside a
     quote that already contains one correctly-pronounced digit, so this
     never fires on ordinary prose numbers or whole-word channel names.

Every hit is a FINDING for a human (or an LLM assistant) to review — this
script never edits or drops anything on its own. See "HOW FINDINGS GET
RESOLVED" below for what happens after a finding is confirmed real.

WHY ONLY LEVEL 1 (not general contradiction detection)
--------------------------------------------------------
Not all "inconsistency" is bad. Two source documents can legitimately give
different working channels for the same purpose in different regions (e.g.
"the coastguard working channel is 67 in the UK" vs "... 72 in NL") — that
is correct regional variation, not a data-quality bug, and training on both
(properly scoped) is exactly the kind of diversity that helps a model
generalize. Building a general-purpose "do these two excerpts agree"
detector would need semantic judgement (an LLM-judge pass over clustered
chunks) and would risk flagging that legitimate variation as a false
"contradiction". This script deliberately stays inside the much smaller,
much safer set of facts that have exactly one universally-correct answer
regardless of region or context, so every finding it raises is a genuine
candidate for a fix — high precision over recall.

HOW FINDINGS GET RESOLVED
--------------------------
This script only ever reports; it never edits a source file or a JSON on its
own. For each finding the reviewer picks one of three paths:

  (a) FIX AT SOURCE (preferred whenever the bad claim is a clean, isolated
      wrong word/number and the source is a plain .txt/.md) — edit the file
      directly, delete the stale JSON under Data/VHF/VHF_JSON/, and rebuild
      §9-§12. This is a full, residue-free fix.
  (b) NORMALIZE — for a section where the source can't be hand-edited (a
      PDF) or a source-text edit isn't worth it, add a scoped
      {section_id, find, replace} entry to consistency_normalizations.json.
      § 8.7's cleanup_all_jsons() reads this file and rewrites only that
      exact section's text before chunking, so the fix survives every
      re-parse without touching the original file and with zero data loss.
      This must stay scoped per section_id, never a blind find/replace over
      the whole corpus — the same word can be a genuine misspelling in one
      section and a legitimate designator name (e.g. "Search Pattern Alpha")
      in another; only a human reviewing each finding can tell the two apart.
  (c) EXCLUDE THE SECTION — for a section where the wrong claim can't be
      cleanly separated from otherwise-useful text even with a normalization
      rule, add its section_id to consistency_exclusions.json instead, which
      the same cleanup pass reads and drops entirely before chunking.

USAGE
-----
    python -m pipeline.eval.check_consistency
    python -m pipeline.eval.check_consistency --json-dir Data/VHF/VHF_JSON --out Data/VHF/VHF_Agents_Training/consistency_findings.json

This is meant to be run as many times as useful (once now, then again after
adding any new source document) — same iterate-until-clean philosophy as
analyze_gaps.py (§ 13.6).
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

from core import AgentPaths

paths = AgentPaths.from_env()

# ── Level-1 canonical reference data (VHF domain) ──────────────────────────
# ITU-R M.1172 / IMO SMCP standardized phonetic alphabet. Key = correct code
# word; value = commonly-seen incorrect/non-standard variants worth flagging.
PHONETIC_ALPHABET: dict[str, list[str]] = {
    "ALFA":     ["Alpha"],
    "BRAVO":    ["Beta"],
    "JULIETT":  ["Juliet", "Juliette"],
    "NOVEMBER": [],
    "WHISKEY":  ["Whisky"],
    "XRAY":     [],
}

# Distress/urgency/safety prowords with a standardized triple-repeat opening.
REPEAT_PROWORDS: dict[str, int] = {
    "MAYDAY":   3,
    "PAN PAN":  3,
    "SECURITE": 3,
    "SECURITY": 3,
}

# Standardized VHF digit pronunciation (ITU/NATO). Only digits that actually
# change from ordinary English are listed -- SIX, SEVEN and ZERO are already
# spoken the same way. Key = correct word; value = the ordinary-English word
# it must not be confused with inside a digit-by-digit readout.
NUMBER_WORDS: dict[str, str] = {
    "WUN":   "ONE",
    "TOO":   "TWO",
    "TREE":  "THREE",
    "FOWER": "FOUR",
    "FIFE":  "FIVE",
    "AIT":   "EIGHT",
    "NINER": "NINE",
}
# A quote only counts as a digit-by-digit readout (and is worth checking for
# the ordinary-English variants above) if it already contains at least one
# digit spoken in the standardized form -- this is what keeps the check from
# firing on ordinary prose that happens to contain the word "one" or "four".
_PHONETIC_NUMBER_SIGNAL_RE = re.compile(
    r"\bZERO\b|\b" + r"\b|\b".join(NUMBER_WORDS) + r"\b"
)


def _variant_pattern(variant: str) -> re.Pattern:
    return re.compile(rf"(?<![A-Za-z]){re.escape(variant)}(?![A-Za-z])")


# Shared quote-matching pattern for the two checks below. Must be generous
# enough for a full worked transmission example (some run past 400 chars) --
# a too-tight cap here silently drops the whole quote from being checked at
# all, rather than just truncating what's shown, which is a much worse bug.
_QUOTE_RE = re.compile(r'"([^"]{0,2000})"')


def check_phonetic_variants(text: str) -> list[dict]:
    """Find whole-word occurrences of a non-standard phonetic-alphabet spelling."""
    findings = []
    for canonical, variants in PHONETIC_ALPHABET.items():
        for variant in variants:
            for m in _variant_pattern(variant).finditer(text):
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 60)
                findings.append({
                    "check":     "phonetic_variant",
                    "canonical": canonical,
                    "found":     variant,
                    "reason":    f'ITU/NATO standard spelling is "{canonical}", not "{variant}"',
                    "snippet":   text[start:end].replace("\n", " "),
                })
    return findings


def check_repeat_counts(text: str) -> list[dict]:
    """Find a distress/urgency/safety proword repeated a non-standard number of
    consecutive times (only inside quoted transmission-like text, to avoid
    flagging ordinary narrative mentions of the word)."""
    findings = []
    for quote in _QUOTE_RE.findall(text):
        for proword, expected in REPEAT_PROWORDS.items():
            pattern = re.compile(rf"\b{re.escape(proword)}\b(?:[\s,]+{re.escape(proword)}\b)+", re.IGNORECASE)
            for m in pattern.finditer(quote):
                actual = len(re.findall(rf"\b{re.escape(proword)}\b", m.group(0), re.IGNORECASE))
                if actual != expected and actual >= 2:
                    findings.append({
                        "check":     "repeat_count",
                        "proword":   proword,
                        "expected":  expected,
                        "found":     actual,
                        "reason":    f'"{proword}" is repeated {actual}x here; the standard opening repeats it {expected}x',
                        "snippet":   quote[:160],
                    })
    return findings


def check_number_pronunciation(text: str) -> list[dict]:
    """Find ordinary-English number words inside a quoted digit-by-digit
    readout (position, MMSI, POB count, ...) that should use the standardized
    VHF pronunciation instead. Only checks quotes that already contain at
    least one correctly-pronounced digit (ZERO or one of NUMBER_WORDS), so
    ordinary prose numbers and whole-word channel names (e.g. "Channel
    Sixteen") are never flagged."""
    findings = []
    for quote in _QUOTE_RE.findall(text):
        if not _PHONETIC_NUMBER_SIGNAL_RE.search(quote):
            continue
        for canonical, variant in NUMBER_WORDS.items():
            for m in _variant_pattern(variant).finditer(quote):
                start = max(0, m.start() - 60)
                end = min(len(quote), m.end() + 60)
                findings.append({
                    "check":     "number_pronunciation",
                    "canonical": canonical,
                    "found":     variant,
                    "reason":    f'Standardized VHF digit pronunciation is "{canonical}", not "{variant}", in a digit-by-digit readout',
                    "snippet":   quote[start:end],
                })
    return findings


def scan_document(doc: dict) -> list[dict]:
    """Run all Level-1 checks over every section of one parsed JSON document."""
    findings = []
    for chapter in doc.get("chapters", []):
        for section in chapter.get("sections", []):
            text = section.get("text", "")
            if not text:
                continue
            hits = (check_phonetic_variants(text) + check_repeat_counts(text)
                    + check_number_pronunciation(text))
            for h in hits:
                h["source_file"] = doc.get("source_file", "?")
                h["section_id"]  = section.get("section_id", "?")
                h["section_title"] = section.get("title", "")
                findings.append(h)
    return findings


def scan_all(json_dir: Path) -> list[dict]:
    """Run scan_document() over every JSON in json_dir (skips files starting with '_')."""
    findings = []
    for path in sorted(json_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        findings.extend(scan_document(doc))
    return findings


def print_report(findings: list[dict]) -> None:
    print("=" * 100)
    print(f"CROSS-SOURCE CONSISTENCY CHECK (Level 1)  —  {len(findings)} finding(s)")
    print("=" * 100)
    if not findings:
        print("No known-standard violations found. (This only checks the fixed list of facts")
        print("in PHONETIC_ALPHABET / REPEAT_PROWORDS above — it is not a general contradiction")
        print("detector; see the module docstring for why.)")
        return
    for f in findings:
        print(f"\n--- {f['check']}  ·  {f['source_file']}  ·  section {f['section_id']} ({f['section_title'][:50]}) ---")
        print(f"  {f['reason']}")
        print(f"  snippet: ...{f['snippet']}...")
    print("\n" + "=" * 100)
    print("NEXT STEP: for each finding, either —")
    print("  (a) fix at source — edit the .txt/.md, delete the stale JSON, re-run § 8-§ 12; or")
    print("  (b) normalize — add a scoped {section_id, find, replace} entry to")
    print("      consistency_normalizations.json (for PDFs, or when not worth a source edit); or")
    print("  (c) exclude the section — add its section_id to consistency_exclusions.json so")
    print("      § 8.7's cleanup pass drops it before chunking ever sees it.")
    print("Re-run this script after adding any new source document — as many times as useful.")
    print("=" * 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", type=str, default=str(paths.json_dir))
    ap.add_argument("--out", type=str, default=None,
                    help="output JSON path (default: <cache_dir>/consistency_findings.json)")
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    findings = scan_all(json_dir)
    print_report(findings)

    out_path = Path(args.out) if args.out else paths.cache_dir / "consistency_findings.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
