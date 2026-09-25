import json

d = json.load(open("Data/missions/_llm_runs/Imazu01__v11_super_colreg_rag_cot__oow_qwen_sft_lora_v2+oow_qwen_dpo_lora_v2__compare_v2.json", encoding="utf-8"))
ev = d.get("evaluation", {})
print("verdict:", ev.get("verdict"), "composite:", ev.get("composite_score"))
for axis in ("safety", "temporal", "spatial", "manoeuvre", "smoothness", "compliance"):
    print(axis, ev.get(axis))
