"""Print + save a manifest of all training data derived from the 30 JSONs.

Reads: _cache/*.jsonl + vhf_gold_answers.json
Writes: _cache/training_data_manifest.json + prints a sample row per file.
"""
from __future__ import annotations
import json
from pathlib import Path

W = Path(__file__).resolve().parent
CACHE = W / "Data" / "VHF" / "VHF_Agents_Training"

FILES = [
    ("SFT direct",   CACHE / "vhf_sft_direct.jsonl"),
    ("SFT CoT",      CACHE / "vhf_sft_cot.jsonl"),
    ("SFT RAG",      CACHE / "vhf_sft_rag.jsonl"),
    ("Multi-hop",    CACHE / "vhf_multihop.jsonl"),
    ("DPO pairs",    CACHE / "vhf_dpo_pairs.jsonl"),
    ("Reflection",   CACHE / "vhf_reflection.jsonl"),
]
GOLD = W / "Data" / "VHF" / "VHF_Eval" / "vhf_gold_answers.json"


def count_lines(p: Path) -> int:
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for _ in f: n += 1
    return n


def first_record(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.loads(f.readline())


def total_tokens_approx(p: Path) -> int:
    total = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            total += len(line) // 4  # rough char->token
    return total


def main():
    rows = []
    for label, path in FILES:
        if not path.exists():
            print(f"MISSING: {path.name}"); continue
        n = count_lines(path)
        sz = path.stat().st_size
        tok = total_tokens_approx(path)
        rows.append({"name": label, "file": path.name, "rows": n, "bytes": sz, "approx_tokens": tok})

    gold_n = len(json.loads(GOLD.read_text(encoding="utf-8")))

    print("=" * 90)
    print(f"{'Dataset':<15} {'File':<26} {'Rows':>6} {'MB':>7} {'~tokens':>10}")
    print("=" * 90)
    for r in rows:
        print(f"{r['name']:<15} {r['file']:<26} {r['rows']:>6} {r['bytes']/1024/1024:>7.2f} {r['approx_tokens']:>10,}")
    print("-" * 90)
    print(f"{'TOTAL TRAIN':<15} {'':<26} {sum(r['rows'] for r in rows):>6} "
          f"{sum(r['bytes'] for r in rows)/1024/1024:>7.2f} "
          f"{sum(r['approx_tokens'] for r in rows):>10,}")
    print(f"{'Gold eval':<15} {'vhf_gold_answers.json':<26} {gold_n:>6}   held-out (not in train)")

    manifest = {
        "gold_eval_rows": gold_n,
        "datasets": rows,
        "total_train_rows": sum(r["rows"] for r in rows),
        "total_train_bytes": sum(r["bytes"] for r in rows),
    }
    (CACHE / "training_data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest saved to _cache/training_data_manifest.json")

    print("\n" + "=" * 90)
    print("Sample row per dataset")
    print("=" * 90)
    for label, path in FILES:
        if not path.exists(): continue
        r = first_record(path)
        print(f"\n--- {label} ({path.name}) ---")
        if "messages" in r:
            for m in r["messages"]:
                content = m["content"]
                shown = content[:220].replace("\n", " ")
                print(f"  [{m['role']}] {shown}{' ...' if len(content) > 220 else ''}")
        elif "prompt" in r and "chosen" in r:
            print(f"  [prompt user] {r['prompt'][-1]['content'][:200]}")
            print(f"  [chosen]      {r['chosen'][0]['content'][:200]} ...")
            print(f"  [rejected]    {r['rejected'][0]['content'][:200]} ...")
            print(f"  perturbation: {r.get('perturbation')}")


if __name__ == "__main__":
    main()
