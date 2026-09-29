#!/usr/bin/env bash
set -Eeuo pipefail

PAYLOAD="${1:-/tmp/optionedge20_server.tar.gz}"
LOG="/root/optionedge20-upgrade.log"
STAGE="/opt/marketscope/staging_optionedge20"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; echo "FOUT regel=$LINENO commando=$BASH_COMMAND exit=$rc"; exit $rc' ERR

echo "OptionEdge 2.0.2 / engine 20.0 — veilige upgrade"
docker compose version >/dev/null
test -f "$PAYLOAD"

TARGET=""
CID="$(docker ps --filter publish=80 --format '{{.ID}}' | head -n 1 || true)"
if [ -n "$CID" ]; then
  TARGET="$(docker inspect -f '{{ index .Config.Labels "com.docker.compose.project.working_dir" }}' "$CID" 2>/dev/null || true)"
fi
if [ -z "$TARGET" ] || [ ! -f "$TARGET/docker-compose.yml" ]; then
  while IFS= read -r path; do
    candidate="$(dirname "$path")"
    if [ -f "$candidate/docker-compose.yml" ]; then TARGET="$candidate"; break; fi
  done < <(find /opt/marketscope /opt/optionedge -maxdepth 4 -name docker-compose.yml -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2-)
fi

NEW_INSTALL=0
if [ -z "$TARGET" ] || [ ! -f "$TARGET/docker-compose.yml" ]; then
  TARGET="/opt/optionedge"
  mkdir -p "$TARGET"
  NEW_INSTALL=1
fi
if [ -L "$TARGET" ]; then
  echo "Veiligheidsstop: doelmap mag geen symlink zijn."; exit 21
fi
TARGET="$(readlink -f "$TARGET")"
case "$TARGET" in
  /opt/marketscope/*|/opt/optionedge|/opt/optionedge/*) ;;
  *) echo "Veiligheidsstop: onverwachte doelmap $TARGET"; exit 20;;
esac
echo "Doelmap: $TARGET"

rm -rf "$STAGE"
mkdir -p "$STAGE"
tar -xzf "$PAYLOAD" -C "$STAGE" --strip-components=1
test -f "$STAGE/docker-compose.yml"
cd "$STAGE"
docker compose config >/dev/null
docker compose build api

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="/opt/marketscope/code-backups/pre_optionedge20_${STAMP}.tar.gz"
if [ "$NEW_INSTALL" -eq 0 ]; then
  mkdir -p /opt/marketscope/code-backups
  cd "$TARGET"
  tar -czf "$BACKUP" app api core worker deploy tests .streamlit docker-compose.yml requirements.txt README.md DATA_VEREISTEN.md INSTALLATIE_EN_UPGRADE.md PROJECTSTATUS.md 2>/dev/null || true
fi

rollback() {
  if [ "$NEW_INSTALL" -eq 0 ] && [ -f "$BACKUP" ]; then
    echo "Automatische coderollback; volumes en Object Storage zijn niet gewijzigd."
    cd "$TARGET"
    rm -rf app api core worker deploy tests .streamlit
    rm -f docker-compose.yml requirements.txt README.md DATA_VEREISTEN.md INSTALLATIE_EN_UPGRADE.md PROJECTSTATUS.md
    tar -xzf "$BACKUP" -C "$TARGET"
    docker compose up -d --build --remove-orphans || true
  fi
}

cd "$TARGET"
rm -rf app api core worker deploy tests .streamlit
rm -f docker-compose.yml requirements.txt README.md DATA_VEREISTEN.md INSTALLATIE_EN_UPGRADE.md PROJECTSTATUS.md
cp -a "$STAGE"/. "$TARGET"/

if ! docker compose up -d --remove-orphans --force-recreate; then rollback; exit 30; fi

# Streamlit health alone is not enough to prove API/database compatibility.
WEB_OK=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1/_stcore/health >/dev/null 2>&1; then WEB_OK=1; break; fi
  sleep 2
done
if [ "$WEB_OK" -ne 1 ]; then
  echo "KRITIEKE CHECK MISLUKT: Streamlit web health niet bereikbaar."
  docker compose logs --tail=240 web api postgres || true
  rollback; exit 31
fi

# Verify the additive database migration directly against the ORM metadata.
if ! docker compose exec -T api python - <<'PY'
from sqlalchemy import inspect
from core.db import engine, Base
from core import models  # registers metadata

insp=inspect(engine)
errors=[]
existing=set(insp.get_table_names())
for table in Base.metadata.sorted_tables:
    if table.name not in existing:
        errors.append(f"missing table: {table.name}")
        continue
    actual={c['name'] for c in insp.get_columns(table.name)}
    expected={c.name for c in table.columns}
    missing=sorted(expected-actual)
    if missing:
        errors.append(f"{table.name}: missing columns {missing}")
if errors:
    print("SCHEMA_MISMATCH")
    for e in errors: print(" -",e)
    raise SystemExit(1)
print("SCHEMA_OK")
PY
then
  echo "KRITIEKE CHECK MISLUKT: databaseschema is niet compatibel met de nieuwe code."
  docker compose logs --tail=240 api postgres || true
  rollback; exit 32
fi

# Critical endpoints must work. Data-dependent status pages are also checked,
# but a temporary storage/data issue is a warning and must not roll back healthy code.
if ! docker compose exec -T web python - <<'PY'
import requests,sys,time
base="http://api:8000"

last=""
for _ in range(90):
    try:
        r=requests.get(base+"/health",timeout=4)
        last=f"HTTP {r.status_code}: {r.text[:500]}"
        if r.ok: break
    except Exception as exc:
        last=f"{type(exc).__name__}: {exc}"
    time.sleep(2)
else:
    print("CRITICAL health FAILED",last)
    raise SystemExit(1)

critical=[
    ("health","/health"),
    ("setup","/setup/status"),
    ("earnings","/earnings/status"),
    ("scorecard","/model/scorecard"),
    ("server","/server/status"),
]
optional=[
    ("readiness","/data/readiness"),
    ("design","/research/design"),
    ("inventory","/inventory/latest"),
    ("project","/project/status"),
    ("vectors","/agents/vectors/latest"),
    ("deep","/deep-relationships/latest"),
    ("fusion","/fusion-model/latest"),
    ("alpha","/alpha-research/latest"),
    ("safety","/trading/safety-status"),
    ("paper","/paper-trading/status"),
    ("monitoring","/monitoring/latest"),
    ("gpu","/gpu-news/status"),
    ("schema","/data/schema"),
]

def check(name,path,required):
    prefix="CRITICAL" if required else "OPTIONAL"
    try:
        r=requests.get(base+path,timeout=12)
        version_ok=(name!="health" or (r.ok and r.json().get("version")=="20.0"))
        ok=bool(r.ok and version_ok)
        body=r.text.replace("\n"," ")[:700]
        print(f"{prefix} {name:12s} {'OK' if ok else 'FAILED'} HTTP={r.status_code} {body}")
        return ok
    except Exception as exc:
        print(f"{prefix} {name:12s} FAILED {type(exc).__name__}: {exc}")
        return False

critical_ok=True
for name,path in critical:
    critical_ok=check(name,path,True) and critical_ok
optional_failures=[]
for name,path in optional:
    if not check(name,path,False): optional_failures.append(name)
if optional_failures:
    print("WAARSCHUWING: niet-kritieke statuschecks faalden:",", ".join(optional_failures))
if not critical_ok:
    raise SystemExit(1)
print("CRITICAL_API_CHECKS_OK")
PY
then
  echo "KRITIEKE CHECK MISLUKT: hierboven staat exact welke API-route faalde en met welke fouttekst."
  docker compose logs --tail=240 web api worker_features worker_model postgres || true
  rollback; exit 33
fi

docker compose exec -T web python - <<'PY' || true
import requests
try:
    r=requests.post("http://api:8000/inventory/scan",json={},timeout=10)
    print("inventory scan:",r.status_code,r.text[:300])
except Exception as exc:
    print("inventory scan warning:",type(exc).__name__,exc)
PY

echo "OPTIONEDGE_20_2_OK"
