import json
from collections import Counter
from pathlib import Path

d = Path(r"C:\Users\jcsch\Documents\Python\Auto Pilot\Data\OOW\OOW_Agents_Training\chirp_text_cache")
ctrl = Counter()
n_files = 0
for f in d.glob("*.json"):
    n_files += 1
    pages = json.loads(f.read_text(encoding="utf-8"))
    for p in pages:
        for ch in p:
            if ord(ch) < 0x20 and ch != "\n":
                ctrl[ch] += 1
print("n_files", n_files)
for ch, n in sorted(ctrl.items(), key=lambda kv: -kv[1]):
    print(f"{hex(ord(ch))!s:6s} count={n}")

# also check Report Ends marker presence in raw cache
import re
RE = re.compile(r"report\s+ends", re.IGNORECASE)
total_matches = 0
per_file_hits = {}
for f in d.glob("*.json"):
    pages = json.loads(f.read_text(encoding="utf-8"))
    full = "\n".join(pages)
    n = len(RE.findall(full))
    if n:
        per_file_hits[f.name] = n
    total_matches += n
print("\ntotal 'report ends' matches across cache:", total_matches)
print("files with matches:", per_file_hits)
