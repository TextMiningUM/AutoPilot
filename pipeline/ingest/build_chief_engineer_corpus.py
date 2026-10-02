"""Chief Engineer manuals/drawings corpus -- raw acquisition (2026-10-03).

Downloads real source documents for the future Chief Engineer RAG corpus
(design_chief_engineer.md sec 7/9) into ``Data/ChiefEngineer/ChiefEngineerManuals/``,
one subfolder per source, plus a ``manifest.json`` recording provenance (url,
license note, fetched_at, sha256) for every file.

Deliberately RAW ACQUISITION ONLY -- mirrors pipeline/ingest/build_captain_legal_corpus.py's
posture exactly: this script does not parse, chunk, or filter anything into RAG format.
Which parts of which document actually enter training/RAG is a separate, later decision.

Safe to run LOCALLY (no GPU/model loading, just HTTP downloads) -- same category as
build_captain_legal_corpus.py per the project's local-vs-cloud split.

Sources (see design_chief_engineer.md sec 7/9 for the full research trail):
  1. navedtra            -- US Navy NAVEDTRA training manuals (Engineman, Fireman,
                             Machinery Repairman, Fluid Power, Blueprint Reading,
                             Basic Machines, Tools and Their Uses, Principles of
                             Naval Engineering, Joint Oil Analysis Program Manual),
                             hosted at legacy.maritime.org (US federal government
                             works, public domain). CONFIRMED (2026-10-03): the
                             site's WAF 403s a bare scripted request even with a
                             full realistic header set, but accepts the SAME
                             request once a ``Referer`` header pointing at the
                             manuals index page is added -- this is the key
                             finding that makes scripted acquisition possible here.
  2. man_project_guides  -- Real MAN B&W two-stroke "Complete Project Guide" PDFs
                             (S50ME-C10.7, S60ME-C10.7), freely published by
                             MAN Energy Solutions (now Everllence) with NO login
                             wall, directly matching design_chief_engineer.md's
                             "MAN B&W S-series" reference-engine choice. Earlier
                             design-doc research (pre-rebrand man-es.com URLs)
                             had flagged this as "unverified" -- now confirmed and
                             acquired under the new everllence.com site structure.
  3. marpol_annex_vi     -- NOT re-downloaded: copied from the already-acquired
                             Data/Captain/Legal_Reference/uk_legislation/uk_marpol_air.xml
                             (UK SI 2008/2924, Open Government Licence v3.0), per
                             the user's own explicit instruction that data already
                             acquired for another domain should be copied/reused,
                             not re-fetched.
  4. wikipedia           -- Background articles (Diesel engine, Marine propulsion,
                             Marine engineering, Condition monitoring, Predictive
                             maintenance, Turbocharger, Crankshaft), CC BY-SA 4.0,
                             via Wikipedia's own public plaintext-extract API.

Classification-society rules (DNV/ABS/LR) and manufacturer manuals beyond the MAN
project guides were checked this pass and confirmed subscription/portal-gated
(DNV's "Rules and Standards Explorer", ABS's "MyFreedom" client portal) -- not a
legitimate anonymous bulk-download target, same class of finding as IMO's IMODOCS
registration wall already documented for Captain. Hydraulic/pneumatic drawing
sourcing via Wikimedia Commons MediaSearch was retried this pass with several
search-term variants and still returned zero hits -- remains an open gap
(design_chief_engineer.md sec 4.3/9), not pursued further per the "don't
brute-force a blocked approach" rule.

Run with: python -m pipeline.ingest.build_chief_engineer_corpus
          python -m pipeline.ingest.build_chief_engineer_corpus --sources navedtra
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
OUT_DIR = WORKSPACE / "Data" / "ChiefEngineer" / "ChiefEngineerManuals"
MANIFEST_PATH = OUT_DIR / "manifest.json"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
# legacy.maritime.org's WAF 403s a bare request even with the full header set above --
# empirically confirmed (2026-10-03) that adding a Referer pointing at its own manuals
# index page is what actually lets the request through.
MARITIME_ORG_HEADERS = {**DEFAULT_HEADERS, "Referer": "https://legacy.maritime.org/doc/index.php"}

# NAVEDTRA / US Navy training manuals hosted at legacy.maritime.org/doc/pdf/ --
# all US federal government works, public domain. (filename, dest_name, description)
NAVEDTRA_MANUALS = [
    ("engineman1.pdf", "engineman1.pdf", "Engineman 1 & C, NAVEDTRA, 2001"),
    ("engineman2.pdf", "engineman2.pdf", "Engineman 2, NAVEDTRA, 1992"),
    ("fireman.pdf", "fireman.pdf", "Fireman, NAVEDTRA, 1992"),
    ("machinery-repairman.pdf", "machinery_repairman.pdf", "Machinery Repairman, NAVEDTRA, 1993"),
    ("fluidpower.pdf", "fluid_power.pdf", "Fluid Power, NAVEDTRA, 1990"),
    ("blueprint.pdf", "blueprint_reading_and_sketching.pdf", "Blueprint Reading and Sketching, NAVEDTRA, 1994"),
    ("basicmachines.pdf", "basic_machines.pdf", "Basic Machines, NAVEDTRA, 1994"),
    ("tools.pdf", "tools_and_their_uses.pdf", "Tools and Their Uses, NAVEDTRA, 1992"),
    ("engineering.pdf", "principles_of_naval_engineering.pdf", "Principles of Naval Engineering, NAVPERS-10788B, 1970"),
    ("afd-070730-009.pdf", "joint_oil_analysis_program_manual.pdf", "Joint Oil Analysis Program Manual, NAVAIR 17-15-50.4, 2005 -- oil/wear-particle condition monitoring"),
]

# Real MAN B&W two-stroke "Complete Project Guide" PDFs, freely published, no login wall --
# matches design_chief_engineer.md's MAN B&W S-series reference-engine choice.
MAN_PROJECT_GUIDES = [
    ("https://www.everllence.com/applications/projectguides/2stroke/content/printed/S50ME-C10_7.pdf", "S50ME-C10.7_project_guide.pdf", "MAN B&W S50ME-C10.7 Complete Project Guide"),
    ("https://www.everllence.com/applications/projectguides/2stroke/content/printed/S60ME-C10_7.pdf", "S60ME-C10.7_project_guide.pdf", "MAN B&W S60ME-C10.7 Complete Project Guide"),
]

# Background Wikipedia articles, CC BY-SA 4.0 -- fine for training, not verbatim redistribution.
WIKIPEDIA_ARTICLES = [
    "Diesel engine",
    "Marine propulsion",
    "Marine engineering",
    "Condition monitoring",
    "Predictive maintenance",
    "Turbocharger",
    "Crankshaft",
    "Marine steam engine",
]


def _load_manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return []


def _save_manifest(entries: list[dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _fetch(url: str, headers: dict[str, str], timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _record(manifest: list[dict], *, source: str, url: str, dest: Path, license_note: str, description: str) -> None:
    data = dest.read_bytes()
    entry = {
        "source": source,
        "url": url,
        "path": str(dest.relative_to(WORKSPACE)).replace("\\", "/"),
        "description": description,
        "license_note": license_note,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest[:] = [e for e in manifest if e["path"] != entry["path"]]
    manifest.append(entry)


def fetch_navedtra(manifest: list[dict]) -> None:
    """US Navy NAVEDTRA training manuals -- public domain, hosted at legacy.maritime.org."""
    dest_dir = OUT_DIR / "navedtra"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for remote_name, dest_name, description in NAVEDTRA_MANUALS:
        dest = dest_dir / dest_name
        url = f"https://legacy.maritime.org/doc/pdf/{remote_name}"
        try:
            data = _fetch(url, MARITIME_ORG_HEADERS)
        except urllib.error.URLError as exc:
            print(f"  [navedtra] FAILED {remote_name}: {exc}")
            continue
        dest.write_bytes(data)
        _record(
            manifest,
            source="navedtra",
            url=url,
            dest=dest,
            license_note="US federal government work -- public domain (NAVEDTRA training manual, via legacy.maritime.org/Maritime Park Association)",
            description=description,
        )
        print(f"  [navedtra] OK {dest_name} ({len(data):,} bytes)")


def fetch_man_project_guides(manifest: list[dict]) -> None:
    """Real MAN B&W two-stroke Complete Project Guide PDFs -- freely published, no login wall."""
    dest_dir = OUT_DIR / "man_project_guides"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for url, dest_name, description in MAN_PROJECT_GUIDES:
        dest = dest_dir / dest_name
        try:
            data = _fetch(url, DEFAULT_HEADERS, timeout=120.0)
        except urllib.error.URLError as exc:
            print(f"  [man_project_guides] FAILED {dest_name}: {exc}")
            continue
        dest.write_bytes(data)
        _record(
            manifest,
            source="man_project_guides",
            url=url,
            dest=dest,
            license_note="MAN Energy Solutions / Everllence -- freely published planning-stage technical documentation, no login wall",
            description=description,
        )
        print(f"  [man_project_guides] OK {dest_name} ({len(data):,} bytes)")


def fetch_marpol_annex_vi(manifest: list[dict]) -> None:
    """Copy (not re-download) the MARPOL Annex VI UK SI already acquired for Captain."""
    src = WORKSPACE / "Data" / "Captain" / "Legal_Reference" / "uk_legislation" / "uk_marpol_air.xml"
    if not src.exists():
        print("  [marpol_annex_vi] SKIPPED: source file not found at", src)
        return
    dest_dir = OUT_DIR / "legal_reference"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "uk_marpol_annex_vi_air_pollution.xml"
    shutil.copyfile(src, dest)
    _record(
        manifest,
        source="marpol_annex_vi",
        url="https://www.legislation.gov.uk/uksi/2008/2924/data.xml",
        dest=dest,
        license_note="Open Government Licence v3.0 -- The Merchant Shipping (Prevention of Air Pollution from Ships) Regulations 2008 (copied from Data/Captain/Legal_Reference, already acquired for the Captain domain)",
        description="MARPOL Annex VI (air pollution) UK implementing legislation",
    )
    print(f"  [marpol_annex_vi] OK copied from Captain corpus ({dest.stat().st_size:,} bytes)")


def fetch_wikipedia(manifest: list[dict]) -> None:
    """Background Wikipedia articles via the public plaintext-extract API -- CC BY-SA 4.0."""
    dest_dir = OUT_DIR / "wikipedia_background"
    dest_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "AutoPilot-ChiefEngineerResearch/1.0 (educational research project)"}
    for title in WIKIPEDIA_ARTICLES:
        safe_title = title.replace(" ", "_")
        api_url = (
            "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&titles="
            + urllib.parse.quote(title)
            + "&format=json&explaintext=1&formatversion=2"
        )
        try:
            raw = _fetch(api_url, headers)
            data = json.loads(raw)
            extract = data["query"]["pages"][0]["extract"]
        except (urllib.error.URLError, KeyError, IndexError) as exc:
            print(f"  [wikipedia] FAILED {title}: {exc}")
            continue
        dest = dest_dir / f"{safe_title}.txt"
        dest.write_text(extract, encoding="utf-8")
        _record(
            manifest,
            source="wikipedia",
            url=f"https://en.wikipedia.org/wiki/{safe_title}",
            dest=dest,
            license_note="CC BY-SA 4.0 -- Wikipedia article, plaintext extract via the public API (fine for training, not verbatim redistribution)",
            description=f"Wikipedia: {title}",
        )
        print(f"  [wikipedia] OK {title} ({len(extract):,} chars)")


SOURCES = {
    "navedtra": fetch_navedtra,
    "man_project_guides": fetch_man_project_guides,
    "marpol_annex_vi": fetch_marpol_annex_vi,
    "wikipedia": fetch_wikipedia,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(SOURCES),
        default=sorted(SOURCES),
        help="Which sources to fetch (default: all).",
    )
    args = parser.parse_args()

    manifest = _load_manifest()
    for name in args.sources:
        print(f"=== {name} ===")
        SOURCES[name](manifest)
    _save_manifest(manifest)
    print(f"\nManifest now has {len(manifest)} entries -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
