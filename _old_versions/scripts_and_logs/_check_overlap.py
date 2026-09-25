import subprocess

origin_files = subprocess.run(
    ["git", "diff", "--name-only", "32cbbb1", "origin/main"],
    capture_output=True, text=True, check=True
).stdout.splitlines()

status_lines = subprocess.run(
    ["git", "status", "--porcelain"],
    capture_output=True, text=True, check=True
).stdout.splitlines()
cloud_files = [line[3:].strip().strip('"') for line in status_lines]

overlap = set(origin_files) & set(cloud_files)
print(len(overlap), "overlapping files")
for f in sorted(overlap):
    print(f)
