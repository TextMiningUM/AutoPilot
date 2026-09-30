"""Captain legal/regulatory reference corpus -- raw acquisition (2026-09-30).

Downloads real source documents for the future Captain RAG corpus (ISM/SOLAS/
STCW/MARPOL-equivalent text + BMP5) into ``Data/Captain/Legal_Reference/``, one
subfolder per source, plus a ``manifest.json`` recording provenance (url,
license note, fetched_at, sha256) for every file.

Deliberately RAW ACQUISITION ONLY -- this script does not parse, chunk, or
filter anything into RAG format. Which parts of which document actually enter
training/RAG is a separate, later decision (design_captain_missions.md
sections 12 and 14.1), not made here.

Sources, smallest/simplest first (see design_captain_missions.md sec 12 for
the full research trail):
  1. bmp5           -- BMP5 (Best Management Practices to Deter Piracy), one
                       PDF, industry-published (BIMCO/ICS/IGP&I/INTERTANKO/
                       OCIMF), freely distributed. The official
                       maritimeglobalsecurity.org host 403s scripted requests;
                       the Liberian ship registry's (LISCR) own online library
                       mirrors the identical PDF and does not.
  2. uk_legislation  -- UK Merchant Shipping statutory instruments that
                       implement STCW/the ISM Code, via legislation.gov.uk's
                       real public content-negotiation API (Open Government
                       Licence v3.0, anonymous, no auth). Exact SI references
                       were looked up via that site's own title-search
                       redirect (``/id?title=...``), not guessed.
  3. ecfr            -- US Code of Federal Regulations sections that
                       implement the same substance (Coast Guard shipping
                       regs), DISCOVERED via eCFR's own real search API
                       (query -> hierarchy) rather than guessing title/part
                       numbers, then fetched via the versioner endpoint.

IMO's own resolution texts (the ISM Code's actual parent, Resolution
A.741(18)) are deliberately NOT scraped here: imo.org confirmed (2026-09-30)
that full resolution text from 2000-present requires IMODOCS registration,
and pre-2000 resolutions are "available in print" only -- neither is a
legitimate anonymous bulk-download target.

Run with: python -m pipeline.ingest.build_captain_legal_corpus
          python -m pipeline.ingest.build_captain_legal_corpus --sources bmp5
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
OUT_DIR = WORKSPACE / "Data" / "Captain" / "Legal_Reference"
MANIFEST_PATH = OUT_DIR / "manifest.json"
USER_AGENT = "Mozilla/5.0 (AutoPilot-CaptainDesign-research/1.0)"


def _load_manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return []


def _save_manifest(entries: list[dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _fetch(url: str, timeout: float = 30.0, accept_encoding_gzip: bool = False) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if accept_encoding_gzip:
        # eCFR's versioner "full" endpoint documents this as REQUIRED (not optional) --
        # confirmed empirically: omitting it returns HTTP 406, not a normal response.
        headers["Accept-Encoding"] = "gzip"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw


def _record(manifest: list[dict], *, source: str, url: str, dest: Path, license_note: str) -> None:
    data = dest.read_bytes()
    entry = {
        "source": source,
        "url": url,
        "path": str(dest.relative_to(WORKSPACE)).replace("\\", "/"),
        "license_note": license_note,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest[:] = [e for e in manifest if e["path"] != entry["path"]]
    manifest.append(entry)


def fetch_bmp5(manifest: list[dict]) -> None:
    """BMP5 -- one PDF. Try the official host first, fall back to a verified mirror."""
    dest = OUT_DIR / "bmp5" / "BMP5.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    mirrors = [
        "https://www.maritimeglobalsecurity.org/media/1038/bmp5-high_res.pdf",
        "https://www.liscr.com/sites/default/files/online_library/BMP5.pdf",
    ]
    for url in mirrors:
        try:
            data = _fetch(url)
        except urllib.error.HTTPError as exc:
            print(f"  [bmp5] {url} -> HTTP {exc.code}, trying next mirror")
            continue
        if not data.startswith(b"%PDF"):
            print(f"  [bmp5] {url} -> not a PDF, trying next mirror")
            continue
        dest.write_bytes(data)
        _record(manifest, source="bmp5", url=url, dest=dest,
                 license_note="BIMCO/ICS/IGP&I Clubs/INTERTANKO/OCIMF industry guidance, freely distributed")
        print(f"  [bmp5] saved {dest.relative_to(WORKSPACE)} ({len(data)} bytes) from {url}")
        return
    print("  [bmp5] FAILED -- no mirror succeeded")


# UK Merchant Shipping statutory instruments -- exact references looked up via
# legislation.gov.uk's own title-search redirect (https://www.legislation.gov.uk/id?title=...)
# or free-text search (https://www.legislation.gov.uk/all?text=...), never guessed. Each is
# the CURRENT (in-force) base instrument for its topic (amendment-only SIs skipped).
UK_INSTRUMENTS = [
    ("uk_stcw", "uksi/2022/1342", "The Merchant Shipping (Standards of Training, Certification and Watchkeeping) Regulations 2022"),
    ("uk_ism", "uksi/2026/194", "The Merchant Shipping (International Safety Management (ISM) Code) Regulations 2026"),
    # MARPOL Annex I (oil), II (noxious liquid substances in bulk), IV (sewage), VI (air pollution).
    ("uk_marpol_oil", "uksi/2019/42", "The Merchant Shipping (Prevention of Oil Pollution) Regulations 2019"),
    ("uk_marpol_nls", "uksi/2018/68", "The Merchant Shipping (Prevention of Pollution from Noxious Liquid Substances in Bulk) Regulations 2018"),
    ("uk_marpol_sewage", "uksi/2020/620", "The Merchant Shipping (Prevention of Pollution by Sewage from Ships) Regulations 2020"),
    ("uk_marpol_air", "uksi/2008/2924", "The Merchant Shipping (Prevention of Air Pollution from Ships) Regulations 2008"),
]


def fetch_uk_legislation(manifest: list[dict]) -> None:
    """UK Merchant Shipping SIs implementing STCW/the ISM Code, via legislation.gov.uk's data.xml API."""
    dest_dir = OUT_DIR / "uk_legislation"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for slug, ref, title in UK_INSTRUMENTS:
        url = f"https://www.legislation.gov.uk/{ref}/data.xml"
        dest = dest_dir / f"{slug}.xml"
        try:
            data = _fetch(url)
        except urllib.error.HTTPError as exc:
            print(f"  [uk_legislation] {title} -> HTTP {exc.code}")
            continue
        dest.write_bytes(data)
        _record(manifest, source="uk_legislation", url=url, dest=dest,
                 license_note="Open Government Licence v3.0 -- " + title)
        print(f"  [uk_legislation] saved {dest.relative_to(WORKSPACE)} ({len(data)} bytes) -- {title}")


# eCFR search queries -> which Captain-relevant concept they stand in for.
# Restricted to title 46 (Shipping) so generic phrases (e.g. "safety management
# system") don't pull in unrelated titles (aviation/rail/etc. also use the term).
ECFR_QUERIES = [
    ("ecfr_ism", "safety management system", "46"),
    ("ecfr_stcw", "training certification watchkeeping", "46"),
]

# Known, directly-confirmed eCFR parts -- not discovered via search, but via a real
# CROSS-REFERENCE found inside an already-fetched document (46 CFR 138.225 itself names
# "33 CFR part 96" as the actual US implementation of the ISM Code), so no guessing.
ECFR_KNOWN_PARTS = [
    ("ecfr_ism_implementation", "33", "96", "ISM Code US implementation, cross-referenced from 46 CFR 138.225"),
]


def _ecfr_fetch_part(dest_dir: Path, manifest: list[dict], title_as_of: dict[str, str],
                      *, slug: str, title: str, part: str, license_note: str) -> None:
    if title not in title_as_of:
        titles_meta = json.loads(_fetch("https://www.ecfr.gov/api/versioner/v1/titles.json"))
        match = next((t for t in titles_meta.get("titles", []) if str(t.get("number")) == title), None)
        if match is None:
            print(f"  [ecfr] could not resolve a valid as_of date for title {title}, skipping fetch")
            return
        title_as_of[title] = match["up_to_date_as_of"]
    as_of = title_as_of[title]
    xml_url = f"https://www.ecfr.gov/api/versioner/v1/full/{as_of}/title-{title}.xml?part={part}"
    dest = dest_dir / f"{slug}_title{title}_part{part}.xml"
    try:
        data = _fetch(xml_url, timeout=60.0, accept_encoding_gzip=True)
    except urllib.error.HTTPError as exc:
        print(f"  [ecfr] fetch failed for title {title} part {part}: HTTP {exc.code}")
        return
    dest.write_bytes(data)
    _record(manifest, source="ecfr", url=xml_url, dest=dest, license_note=license_note)
    print(f"  [ecfr] saved {dest.relative_to(WORKSPACE)} ({len(data)} bytes) -- title {title} part {part}")


def fetch_ecfr(manifest: list[dict]) -> None:
    """Discover + fetch US CFR sections implementing the same substance, via eCFR's real search API
    (never a guessed title/part number), plus a short list of already-confirmed known parts."""
    dest_dir = OUT_DIR / "ecfr"
    dest_dir.mkdir(parents=True, exist_ok=True)
    title_as_of: dict[str, str] = {}

    for slug, title, part, note in ECFR_KNOWN_PARTS:
        _ecfr_fetch_part(dest_dir, manifest, title_as_of, slug=slug, title=title, part=part,
                          license_note=f"US Government Work (public domain) -- {note}")

    for slug, query, title in ECFR_QUERIES:
        search_url = (
            "https://www.ecfr.gov/api/search/v1/results"
            f"?query={urllib.parse.quote(query)}&hierarchy[title]={title}&per_page=5"
        )
        try:
            hits = json.loads(_fetch(search_url))
        except (urllib.error.HTTPError, json.JSONDecodeError) as exc:
            print(f"  [ecfr] search failed for {query!r}: {exc}")
            continue
        results = hits.get("results", [])
        (dest_dir / f"{slug}_search_hits.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        if not results:
            print(f"  [ecfr] no hits for {query!r} in title {title}")
            continue
        top = results[0]
        part = top["hierarchy"].get("part")
        if not part:
            print(f"  [ecfr] top hit for {query!r} has no part number, skipping fetch")
            continue
        _ecfr_fetch_part(dest_dir, manifest, title_as_of, slug=slug, title=title, part=part,
                          license_note=f"US Government Work (public domain) -- discovered via search query {query!r}")


SOURCES = {
    "bmp5": fetch_bmp5,
    "uk_legislation": fetch_uk_legislation,
    "ecfr": fetch_ecfr,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES),
                         help="Which sources to fetch (default: all, smallest first).")
    args = parser.parse_args()

    manifest = _load_manifest()
    for name in args.sources:
        print(f"== {name} ==")
        SOURCES[name](manifest)
    _save_manifest(manifest)
    print(f"\nManifest: {MANIFEST_PATH.relative_to(WORKSPACE)} ({len(manifest)} files total)")


if __name__ == "__main__":
    sys.exit(main())
