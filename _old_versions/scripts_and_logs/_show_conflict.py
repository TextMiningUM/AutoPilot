import subprocess

for ref, label in [(":2", "ours (cloud HEAD)"), (":3", "theirs (origin/main)")]:
    print("=" * 20, label)
    result = subprocess.run(
        ["git", "show", f"{ref}:cloud/run_baseline_ablation_n50.sh"],
        capture_output=True, text=True
    )
    print(result.stdout if result.returncode == 0 else f"(missing: {result.stderr.strip()})")
    print()
