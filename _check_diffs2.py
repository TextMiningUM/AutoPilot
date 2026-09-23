import subprocess

files = [
    "cloud/run_all_oow.sh",
    "pipeline/compress/compress_distill.py",
    "pipeline/eval/eval_finetuned.py",
    "pipeline/eval/eval_oow_scenarios.py",
    "pipeline/eval/run_ablation.py",
]

for f in files:
    print("=" * 30, f)
    local_diff = subprocess.run(
        ["git", "diff", "HEAD", "--", f],
        capture_output=True, text=True, check=True
    ).stdout
    print("--- cloud's uncommitted change (vs its own HEAD) ---")
    print(local_diff[:4000] if local_diff else "(no change)")
    print()
