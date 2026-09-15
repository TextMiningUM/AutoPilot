#!/bin/bash
# Shared storage for trained/exported models, accessible rwx by the whole
# "autopilot" group (ubuntu + all teammates). Default ACLs ensure every new
# file/dir automatically inherits group rwx, regardless of who creates it.
set -euo pipefail

SHARE_DIR="/srv/shared-models"

sudo mkdir -p "$SHARE_DIR"
sudo chgrp autopilot "$SHARE_DIR"
sudo chmod 2775 "$SHARE_DIR"   # setgid + rwxrwxr-x

if ! command -v setfacl >/dev/null 2>&1; then
    sudo apt-get install -y acl
fi

# Default ACL: new files/dirs created by ANY member automatically get
# group rwx, not just the creator's own umask-limited permissions.
sudo setfacl -d -m g:autopilot:rwx "$SHARE_DIR"
sudo setfacl -m g:autopilot:rwx "$SHARE_DIR"

echo "--- ls -la ---"
ls -la "$SHARE_DIR" -d
echo "--- getfacl ---"
getfacl "$SHARE_DIR"
