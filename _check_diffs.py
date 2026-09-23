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
    diff = subprocess.run(
        ["git", "diff", "--stat", "32cbbb1", "origin/main", "--", f],
        capture_output=True, text=True, check=True
    ).stdout
    print("--- origin/main changed this file vs cloud's base (32cbbb1) ---")
    print(diff or "(no change)")
    local_diff = subprocess.run(
        ["git", "diff", "--stat", "--", f],
        capture_output=True, text=True, check=True
    ).stdout
    print("--- cloud's uncommitted local diff ---")
    print(local_diff or "(no change)")
    print()
