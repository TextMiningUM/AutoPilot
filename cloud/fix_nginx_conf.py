"""One-shot fix for a nginx.conf whose 'http {' block got a mangled insertion.
Run on the cloud instance as root: sudo python3 fix_nginx_conf.py
"""
from pathlib import Path

P = Path("/etc/nginx/nginx.conf")
lines = P.read_text().splitlines(keepends=True)

idx = next(i for i, l in enumerate(lines) if l.strip() == "http {")
# The mangled map block sits right after "http {" -- replace up to the blank line.
end = idx + 1
while end < len(lines) and lines[end].strip() != "":
    end += 1

good_block = [
    "    map $http_upgrade $connection_upgrade {\n",
    "        default upgrade;\n",
    '        "" close;\n',
    "    }\n",
]
lines[idx + 1:end] = good_block
P.write_text("".join(lines))
print("nginx.conf patched")

