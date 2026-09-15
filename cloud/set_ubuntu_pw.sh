#!/bin/bash
set -euo pipefail
pw="$(openssl rand -base64 12)"
echo "ubuntu:${pw}" | sudo chpasswd
echo "ubuntu:${pw}" | sudo tee -a /root/tljh_initial_passwords.txt > /dev/null
echo "ubuntu password set"
