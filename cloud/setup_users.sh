#!/bin/bash
# Creates the shared "autopilot" group + team Linux accounts, and fixes
# permissions on the AutoPilot repo so JupyterHub users can read/write it.
set -euo pipefail

REPO_DIR="/home/ubuntu/AutoPilot"
USERS=(leo sim michiel jan alex)

sudo groupadd -f autopilot
sudo usermod -aG autopilot ubuntu

PASS_FILE="/root/tljh_initial_passwords.txt"
sudo touch "$PASS_FILE"
sudo chmod 600 "$PASS_FILE"

for u in "${USERS[@]}"; do
    if ! id -u "$u" >/dev/null 2>&1; then
        sudo useradd -m -G autopilot -s /bin/bash "$u"
        pw="$(openssl rand -base64 12)"
        echo "${u}:${pw}" | sudo chpasswd
        echo "${u}:${pw}" | sudo tee -a "$PASS_FILE" > /dev/null
        echo "created user: $u"
    else
        echo "user already exists: $u"
    fi
done

# Shared read/write access to the repo for the whole team; setgid so new
# files/dirs inherit the "autopilot" group automatically.
sudo chgrp -R autopilot "$REPO_DIR"
sudo chmod -R g+rwX "$REPO_DIR"
sudo find "$REPO_DIR" -type d -exec chmod g+s {} \;

echo "---"
getent group autopilot
echo "---"
for u in "${USERS[@]}"; do id "$u"; done
echo "---"
echo "Initial passwords written to $PASS_FILE (root-only). Retrieve via:"
echo "  ssh ... 'sudo cat $PASS_FILE'"
