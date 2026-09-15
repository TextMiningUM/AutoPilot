#!/bin/bash
# Gives each teammate their OWN independent git clone (own git history, own
# pull/push) instead of a shared working directory. The shared .venv stays
# shared via the global "autopilot-venv" Jupyter kernel (independent of
# where the notebook file itself lives), so nobody re-installs torch/CUDA.
set -euo pipefail

REPO_URL="https://github.com/TextMiningUM/AutoPilot.git"
USERS=(leo sim michiel jan alex)

for u in "${USERS[@]}"; do
    dest="/home/${u}/AutoPilot"
    if [ -d "$dest/.git" ]; then
        echo "already cloned: $u"
        continue
    fi
    sudo -u "$u" git clone --quiet "$REPO_URL" "$dest"
    # Share the existing OpenAI key (per team decision) — copied, not
    # symlinked, so each clone is fully independent.
    if [ -f /home/ubuntu/AutoPilot/.env ]; then
        sudo cp /home/ubuntu/AutoPilot/.env "$dest/.env"
        sudo chown "${u}:${u}" "$dest/.env"
        sudo chmod 600 "$dest/.env"
    fi
    echo "cloned: $u -> $dest"
done
