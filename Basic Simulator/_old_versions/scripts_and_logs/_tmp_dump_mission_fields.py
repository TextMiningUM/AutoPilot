import json
from pathlib import Path

d = Path(__file__).parent / "Data" / "missions"
for mid in [f"Imazu{i:02d}" for i in range(1, 23)] + [f"UM{i:02d}" for i in range(1, 14)]:
    p = d / f"{mid}.json"
    if not p.exists():
        continue
    j = json.loads(p.read_text(encoding="utf-8"))
    n_targets = len(j.get("targets", []))
    print(f"{mid:8s} n_targets={n_targets} role={j.get('own_ship_role')!r:50s} rules={j.get('rule_refs')}")
