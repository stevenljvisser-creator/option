#!/usr/bin/env bash
set -euo pipefail

echo "LET OP: legacy upgrade_11.sh. Gebruik voor OptionEdge 2.0 deploy/remote_upgrade_v20.sh."

APP_DIR="/opt/marketscope/MarketScope_10_2_WEB_SETUP"
cd "$APP_DIR"

echo
echo "============================================================"
echo " MarketScope 11.0 - volledige serverupgrade"
echo "============================================================"
echo

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker ontbreekt; installeren..."
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl docker.io docker-compose-v2
  systemctl enable --now docker
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose ontbreekt; installeren..."
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y docker-compose-v2
fi

# Keep SSH and the MarketScope website reachable if UFW is enabled.
if command -v ufw >/dev/null 2>&1; then
  if ufw status | grep -qi "Status: active"; then
    ufw allow 22/tcp >/dev/null || true
    ufw allow 80/tcp >/dev/null || true
  fi
fi

# Emergency swap for the 4 GB CX23. It lives only on the Hetzner server.
if ! swapon --show | grep -q "/swapfile"; then
  echo "2 GB nood-swap configureren..."
  if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
  fi
  swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  sysctl vm.swappiness=10 >/dev/null 2>&1 || true
fi

echo "Nieuwe MarketScope images bouwen terwijl de huidige website nog draait..."
docker compose build

echo
echo "Overschakelen naar MarketScope 11.0..."
docker compose down --remove-orphans
docker compose up -d

echo
echo "Wachten op de website..."
WEB_OK=0
for i in $(seq 1 60); do
  if curl -fsS --max-time 4 http://127.0.0.1/_stcore/health >/dev/null 2>&1; then
    WEB_OK=1
    break
  fi
  sleep 2
done

if [ "$WEB_OK" -ne 1 ]; then
  echo
  echo "FOUT: website antwoordt intern niet."
  echo
  docker compose ps
  echo
  docker compose logs --tail=180 web api worker_market worker_news postgres
  exit 1
fi

echo "Website reageert."

echo "Wachten op workers..."
sleep 12

STATUS_JSON="$(curl -fsS --max-time 8 http://127.0.0.1:8501/_stcore/health || true)"

echo
echo "Containerstatus:"
docker compose ps

echo
echo "Geheugen:"
free -h

echo
echo "MARKETSCOPE_11_OK"
