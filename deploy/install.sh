#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo
echo "=== OptionEdge 2.0 / research engine 20.0 ==="

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker ontbreekt; automatisch installeren..."
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl docker.io docker-compose-v2
  systemctl enable --now docker
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose plugin ontbreekt; automatisch installeren..."
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y docker-compose-v2
fi

# 2 GB emergency swap on the Hetzner server.
if ! swapon --show | grep -q "/swapfile"; then
  echo "2 GB nood-swap aanmaken..."
  if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
  fi
  swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  sysctl vm.swappiness=10 || true
fi

echo "Eventuele oude OptionEdge-containers stoppen (volumes blijven behouden)..."
docker compose down --remove-orphans 2>/dev/null || true

echo "Containers bouwen..."
docker compose build

echo "OptionEdge starten..."
docker compose up -d

echo "Wachten tot website reageert..."
OK=0
for i in $(seq 1 60); do
  if curl -fsS --max-time 3 http://127.0.0.1/_stcore/health >/dev/null 2>&1; then
    OK=1
    break
  fi
  sleep 2
done

echo
docker compose ps
echo

if [ "$OK" -eq 1 ]; then
  echo "OPTIONEDGE_20_OK"
else
  echo "OPTIONEDGE_20_START_FAILED"
  echo
  echo "Laatste logs:"
  docker compose logs --tail=160 web api postgres worker_market worker_trades worker_features worker_news worker_model worker_inventory
  exit 1
fi
