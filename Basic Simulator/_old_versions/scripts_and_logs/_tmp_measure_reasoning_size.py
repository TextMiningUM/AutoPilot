"""One-off: how much of each run log's size is already the raw <think>/reasoning text
(debug.raw_response) vs the rest, to answer 'how many KB would storing the reasoning
process add' -- it's already stored today for CoT/PG configs; this measures exactly how
much."""
import json
import statistics
from pathlib import Path

d = Path(__file__).parent / "Data" / "missions" / "Missions data v2 20260922"
files = sorted(d.glob("*__*__units_v1.json"))
print("n_files", len(files))
sizes = [f.stat().st_size for f in files]
print("total_kb", round(sum(sizes) / 1024, 1), "avg_kb_per_file", round(statistics.mean(sizes) / 1024, 1))

raw_bytes_total = 0
raw_count = 0
cp_count = 0
per_config = {}
for f in files:
    doc = json.loads(f.read_text(encoding="utf-8"))
    config = doc.get("config")
    per_config.setdefault(config, {"n": 0, "raw_bytes": 0, "cp": 0, "file_bytes": 0})
    per_config[config]["file_bytes"] += f.stat().st_size
    for cp in doc.get("checkpoints", []):
        cp_count += 1
        per_config[config]["cp"] += 1
        rr = cp.get("debug", {}).get("raw_response")
        if rr:
            raw_count += 1
            b = len(rr.encode("utf-8"))
            raw_bytes_total += b
            per_config[config]["raw_bytes"] += b
    per_config[config]["n"] += 1

print("total checkpoints", cp_count, "checkpoints with raw_response", raw_count)
print("raw_response total KB", round(raw_bytes_total / 1024, 1))
if raw_count:
    print("raw_response mean bytes/checkpoint", round(raw_bytes_total / raw_count, 1))
print()
print(f"{'config':16s} {'files':>5s} {'checkpoints':>11s} {'file_KB':>9s} {'raw_KB':>8s} {'raw_%_of_file':>13s} {'mean_B/cp':>10s}")
for cfg, v in sorted(per_config.items()):
    mean_b = v["raw_bytes"] / v["cp"] if v["cp"] else 0
    pct = 100 * v["raw_bytes"] / v["file_bytes"] if v["file_bytes"] else 0
    print(f"{cfg:16s} {v['n']:5d} {v['cp']:11d} {v['file_bytes']/1024:9.1f} {v['raw_bytes']/1024:8.1f} {pct:12.1f}% {mean_b:10.1f}")
