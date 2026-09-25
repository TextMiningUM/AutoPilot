import json
d = json.load(open("Data/missions/_llm_runs/Imazu01__v0_base__oow_qwen_sft_lora_v2+oow_qwen_dpo_lora_v2+oow_qwen_reflect_lora_v2fix__reflect_v2fix_test.json", encoding="utf-8"))
for cp in d["checkpoints"][-6:]:
    dec = cp.get("decision", {})
    print("step", cp.get("step"), "action=", dec.get("action"), "parse_error=", dec.get("_parse_error"))
    raw = cp.get("reasoning_raw") or cp.get("debug", {}).get("raw_output")
    print("  raw[:400]=", repr((raw or "")[:400]))
