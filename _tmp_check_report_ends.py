import json
import re
from pathlib import Path

d = Path(r"C:\Users\jcsch\Documents\Python\Auto Pilot\Data\OOW\OOW_Agents_Training\chirp_text_cache")
RE = re.compile(r"report\s+ends", re.IGNORECASE)
RE_LOOSE = re.compile(r"report.{0,5}ends", re.IGNORECASE | re.DOTALL)

for name in ["MFB-56-September-2019-72dpi-RGB-for-online.json",
             "MFB-57-November-2019-72dpi-RGB-for-online.json",
             "MFB-60-August-2020.json",
             "MFB-65-November-2021-72dpi-RGB-for-online.json",
             "MFB-67-English-edition-72dpi-RGB-for-online.json"]:
    p = d / name
    if not p.exists():
        print(name, "MISSING")
        continue
    pages = json.loads(p.read_text(encoding="utf-8"))
    full = "\n".join(pages)
    strict = len(RE.findall(full))
    loose = RE_LOOSE.findall(full)
    print(name, "strict_matches=", strict, "loose_matches=", loose[:5])
