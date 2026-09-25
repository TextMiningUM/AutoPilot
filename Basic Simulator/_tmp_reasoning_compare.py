import json, re

d = json.load(open("Data/missions/_llm_runs/Imazu01__v11_super_colreg_rag_cot__oow_qwen_sft_lora_v2+oow_qwen_dpo_lora_v2__compare_v2.json", encoding="utf-8"))
cps = d.get("checkpoints", [])
lens = [len(cp.get("reasoning_raw") or "") for cp in cps]
waits = [len(re.findall(r"\bwait\b", cp.get("reasoning_raw") or "", re.I)) for cp in cps]
print("SFT+DPO v11: n=", len(cps), "avg_len=", sum(lens) / max(1, len(lens)), "avg_wait=", sum(waits) / max(1, len(waits)))
print("=== sample reasoning_raw (checkpoint 0) ===")
print(cps[0].get("reasoning_raw"))
