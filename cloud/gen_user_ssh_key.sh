#!/bin/bash
# Generates a fresh SSH keypair for a teammate, installs the PUBLIC key into
# their own authorized_keys, and leaves the PRIVATE key in /tmp so it can be
# scp'd off the server immediately (never printed to any log/chat).
set -euo pipefail
u="$1"
ssh_dir="/home/${u}/.ssh"
key_path="/tmp/${u}_id_ed25519"

sudo mkdir -p "$ssh_dir"
sudo chmod 700 "$ssh_dir"

rm -f "$key_path" "${key_path}.pub"
ssh-keygen -t ed25519 -f "$key_path" -N "" -C "${u}@autopilot-server" -q

sudo tee -a "${ssh_dir}/authorized_keys" < "${key_path}.pub" > /dev/null
sudo chmod 600 "${ssh_dir}/authorized_keys"
sudo chown -R "${u}:${u}" "$ssh_dir"

echo "public key installed for ${u}"
echo "private key staged at: ${key_path} (root-readable only until fetched)"
sudo chmod 600 "$key_path"
