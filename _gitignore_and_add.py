import subprocess

with open(".gitignore", "a", encoding="utf-8") as f:
    f.write("\n# Local/cloud-only safeguard snapshots (models_v1: model weights, way too\n")
    f.write("# large for git; Eval_v1: duplicates data already tracked under Data/OOW,Data/VHF)\n")
    f.write("models_v1/\n")
    f.write("Eval_v1/\n")

subprocess.run(["git", "add", "-A"], check=True)
status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
print("files staged:", len(status.splitlines()))
print(status[:2000])
