import json
from pathlib import Path

log = json.loads(Path("Data/missions/_llm_runs/s01_head_on__v2_cot.json").read_text(encoding="utf-8"))
for cp in log["checkpoints"]:
    raw = cp["debug"].get("raw_response", "")
    print(f"step={cp['step']} raw_len_chars={len(raw)} has_close_think={'</think>' in raw} "
         f"has_brace={'{' in raw and '}' in raw} parse_error={cp['decision'].get('_parse_error')}")
