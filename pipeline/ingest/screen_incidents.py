"""Screen Data/<domain>/<domain>_Incidents PDFs for COLREG (collision-avoidance)
relevance before spending any Track-1 extraction effort on them.

Most marine accident-investigation reports are groundings, fires, flooding,
machinery failures or man-overboard cases that never touch Rule-of-the-Road
decision-making. With 800+ PDFs in the corpus, hand-opening each one isn't
practical, so this scores every report on COLREG-specific vocabulary density
(rule numbers, give-way/stand-on, risk-of-collision, crossing/overtaking/
head-on situations, ...) minus vocabulary that signals an unrelated accident
type, and also locates the page where an "Analysis"/"Conclusions"/"Causal
factors" section starts (where a genuinely relevant report concentrates its
Rule-of-the-Road reasoning, vs. the vessel-particulars/weather boilerplate
that fills most of the rest of the document).

Output is a single ranked JSON (Data/<domain>/<domain>_Agents_Training/
incident_screening.json) so a human only needs to skim a short shortlist and
a handful of pages per report -- not the whole 800-PDF corpus.

Run with: python -m pipeline.ingest.screen_incidents
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pdfplumber

from core import AgentPaths

paths = AgentPaths.from_env()
INCIDENTS_DIR = paths.incidents_dir
CACHE_DIR = paths.cache_dir / "incidents_text_cache"
OUT_FILE = paths.cache_dir / "incident_screening.json"

# Positive signal: vocabulary that only shows up when the report actually
# discusses Rule-of-the-Road / collision-avoidance decision-making.
COLREG_KEYWORDS: dict[str, int] = {
    "colreg": 3, "colregs": 3, "collision regulations": 3,
    "rule of the road": 3, "give-way": 2, "give way vessel": 2,
    "stand-on": 2, "stand on vessel": 2, "risk of collision": 3,
    "crossing situation": 2, "head-on situation": 2, "overtaking vessel": 2,
    "close-quarters": 2, "close quarters situation": 2,
    "action to avoid collision": 2, "proper look-out": 2, "look-out": 1,
    "safe speed": 1, "sound signal": 1, "collision avoidance": 2,
    "narrow channel": 1, "traffic separation scheme": 1,
    "not under command": 1, "restricted in her ability to manoeuvre": 1,
    "vessel traffic service": 1, "cpa": 1, "closest point of approach": 2,
    "arpa": 1, "collision between": 3,
}
for _n in range(1, 20):
    COLREG_KEYWORDS[f"rule {_n} "] = 2

# Negative signal: suggests the accident is unrelated to Rule-of-the-Road
# decision-making (grounding, fire, flooding, personnel injury, machinery).
NON_COLREG_KEYWORDS: dict[str, int] = {
    "grounding": 1, "ran aground": 1, "engine room fire": 2, "flooding": 1,
    "man overboard": 2, "machinery failure": 1, "capsize": 1, "capsized": 1,
    "loss of stability": 1, "cargo shift": 1, "person overboard": 2,
    "fatality": 1, "electrocution": 1, "mooring line": 1,
}

ANALYSIS_HEADING_RE = re.compile(
    r"^(ANALYSIS|SECTION\s*\d+\s*[-:]?\s*ANALYSIS|CONCLUSIONS?|"
    r"CAUSAL FACTORS|FINDINGS(\s+OF\s+FACT)?)\s*$",
    re.IGNORECASE,
)


def find_pdfs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.pdf"))


def extract_pages_cached(pdf_path: Path) -> list[str]:
    """Return per-page text, cached as JSON so re-runs don't re-parse the PDF."""
    rel = pdf_path.relative_to(INCIDENTS_DIR)
    cache_file = CACHE_DIR / rel.with_suffix(".json")
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as e:  # corrupt/scanned/encrypted PDF
        pages = [f"__EXTRACT_ERROR__: {e}"]
    cache_file.write_text(json.dumps(pages), encoding="utf-8")
    return pages


def score_pages(pages: list[str]) -> dict:
    full = "\n".join(pages).lower()
    pos, hits = 0, {}
    for kw, weight in COLREG_KEYWORDS.items():
        c = full.count(kw)
        if c:
            pos += c * weight
            hits[kw] = c
    neg, neg_hits = 0, {}
    for kw, weight in NON_COLREG_KEYWORDS.items():
        c = full.count(kw)
        if c:
            neg += c * weight
            neg_hits[kw] = c

    analysis_page = None
    for i, ptext in enumerate(pages):
        for line in (ptext or "").split("\n"):
            line = line.strip()
            if line and ANALYSIS_HEADING_RE.match(line):
                analysis_page = i
                break
        if analysis_page is not None:
            break

    return {
        "colreg_score": pos,
        "non_colreg_score": neg,
        "net_score": pos - neg,
        "colreg_hits": hits,
        "non_colreg_hits": neg_hits,
        "analysis_page": analysis_page,
        "n_pages": len(pages),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Only scan the first N PDFs (smoke-testing).")
    args = ap.parse_args()

    pdfs = find_pdfs(INCIDENTS_DIR)
    if args.limit:
        pdfs = pdfs[: args.limit]
    print(f"Found {len(pdfs)} PDFs under {INCIDENTS_DIR}")
    results = []
    for i, pdf_path in enumerate(pdfs, 1):
        pages = extract_pages_cached(pdf_path)
        info = score_pages(pages)
        info["path"] = str(pdf_path.relative_to(INCIDENTS_DIR)).replace("\\", "/")
        results.append(info)
        if i % 50 == 0:
            print(f"  scored {i}/{len(pdfs)}")

    results.sort(key=lambda r: r["net_score"], reverse=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_relevant = sum(1 for r in results if r["net_score"] >= 5)
    print(f"\nWrote {OUT_FILE}")
    print(f"{n_relevant}/{len(results)} reports score net_score >= 5 (likely COLREG-relevant)")
    print("\nTop 20 by net_score:")
    for r in results[:20]:
        print(f"  {r['net_score']:>4}  {r['path']}  (analysis section @ page {r['analysis_page']}, {r['n_pages']} pages)")


if __name__ == "__main__":
    main()
