import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\jcsch\Documents\Python\Auto Pilot")
from pipeline.ingest.screen_incidents import score_pages

JSON_DIR = Path(r"C:\Users\jcsch\Documents\Python\Auto Pilot\Data\OOW\OOW_JSON")

rows = []
for p in JSON_DIR.glob("chirp_mfb_*.json"):
    doc = json.loads(p.read_text(encoding="utf-8"))
    for ch in doc.get("chapters", []):
        for s in ch.get("sections", []):
            if s.get("concepts"):
                continue
            score = score_pages([s["text"]])["net_score"]
            rows.append((score, p.name, ch["title"], s["type"], s["text"][:150]))

rows.sort(reverse=True)
print(f"{len(rows)} empty-concept CHIRP sections total")
print("\nTop 15 by net_score among the empty-concept ones (should ideally be near 0 if tagging is fine):")
for score, fname, title, stype, snippet in rows[:15]:
    print(f"score={score:4d}  {fname}  [{stype}] {title!r}")
    print(f"    {snippet!r}")
