import json
from pathlib import Path
from collections import Counter

p = Path("Data/missions/_llm_runs/Imazu01__v11_super_colreg_rag_cot__W0_base__v10v11_smoke.json")
run = json.loads(p.read_text(encoding="utf-8"))
print("outcome:", run.get("outcome"))
ev = run.get("evaluation") or {}
print("verdict:", ev.get("verdict"), "composite:", ev.get("composite_score"), "safety:", ev.get("safety"))
print()

cited_ctr = Counter()
chunk_ctr = Counter()
for cp in run["checkpoints"]:
    dec = cp.get("decision") or {}
    if dec.get("encounter_rule") and dec["encounter_rule"] != "none":
        cited_ctr[dec["encounter_rule"]] += 1
    for cid in (cp.get("debug") or {}).get("retrieved_chunk_ids") or []:
        chunk_ctr[cid] += 1

print("encounter_rule citations:", dict(cited_ctr))
print("retrieved chunk_id frequency:", dict(chunk_ctr))
print("n checkpoints:", len(run["checkpoints"]))
print()

for cp in run["checkpoints"]:
    dec = cp.get("decision") or {}
    print(f"step={cp.get('step'):>3} action={dec.get('action'):<12} deg={str(dec.get('degrees')):<6} "
          f"enc={dec.get('encounter_rule'):<10} cond={dec.get('conduct_rule'):<10}")
    print(f"    reasoning: {(dec.get('reasoning') or '')[:400]}")
