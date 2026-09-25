import json
from pathlib import Path
from collections import Counter

JSON_DIR = Path(r"C:\Users\jcsch\Documents\Python\Auto Pilot\Data\OOW\OOW_JSON")

for pattern, label in [("chirp_mfb_*.json", "chirp"), ("leo_moos_cases.json", "moos"),
                       ("incident_marginal_*.json", "incident_marginal")]:
    n_sections = 0
    n_empty = 0
    concept_counter = Counter()
    for p in JSON_DIR.glob(pattern):
        doc = json.loads(p.read_text(encoding="utf-8"))
        for ch in doc.get("chapters", []):
            for s in ch.get("sections", []):
                n_sections += 1
                concepts = s.get("concepts") or []
                if not concepts:
                    n_empty += 1
                concept_counter.update(concepts)
    print(f"{label:20s} n_sections={n_sections:5d} n_empty_concepts={n_empty:5d} "
          f"({100*n_empty/n_sections:.1f}%)" if n_sections else f"{label}: no sections found")
    print("  top concepts:", concept_counter.most_common(15))
    print()
