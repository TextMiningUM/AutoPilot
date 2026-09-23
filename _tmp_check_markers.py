import json
import re
from pathlib import Path

d = Path(r"C:\Users\jcsch\Documents\Python\Auto Pilot\Data\OOW\OOW_Agents_Training\chirp_text_cache")
RE = re.compile(r"what the reporter told us", re.IGNORECASE)
RE_ENDS = re.compile(r"report\s+ends", re.IGNORECASE)
RE_OUTLINE = re.compile(r"OUTLINE\s*:", re.IGNORECASE)

rows = []
for f in sorted(d.glob("*.json")):
    pages = json.loads(f.read_text(encoding="utf-8"))
    full = "\n".join(pages)
    rows.append((f.name, len(RE.findall(full)), len(RE_ENDS.findall(full)), len(RE_OUTLINE.findall(full))))

for name, wtru, ends, outline in rows:
    print(f"{name:60s} what_the_reporter={wtru:3d} report_ends={ends:3d} outline={outline:3d}")
