#!/bin/bash
# Runs once, on first boot.
set -euo pipefail
apt-get update -y
apt-get install -y docker.io docker-compose-v2
systemctl enable --now docker
touch /var/run/bootstrap-done
