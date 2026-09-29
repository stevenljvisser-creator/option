# Veilige installatie en upgrade

## Bestaande OptionEdge-server

Gebruik bij voorkeur het meegeleverde upgradepakket; dit zoekt de actieve
Docker Compose-map via de container op poort 80. De upgrade:

1. bouwt de nieuwe image eerst in staging;
2. maakt een codebackup;
3. vervangt alleen applicatiecode;
4. gebruikt nooit `docker compose down -v`;
5. behoudt PostgreSQL named volumes, secrets en alle Object Storage-data;
6. controleert site, API, workers en v20-routes;
7. rolt bij een mislukte healthcheck terug naar de codebackup.

Handmatig in dezelfde bestaande Compose-map:

```bash
docker compose build api
docker compose up -d --remove-orphans --force-recreate
docker compose ps
curl -fsS http://127.0.0.1/_stcore/health
```

Verwijder geen volumes en wis de bestaande Object Storage-prefixen v2/v3/v4
niet. De nieuwe fusie-uitvoer gebruikt uitsluitend v5.

## Nieuwe server

```bash
chmod +x deploy/install.sh
sudo ./deploy/install.sh
```

Open vervolgens poort 80 en voltooi de setup in de browser. Start daarna een
Object Storage-scan op **Data & gereedheid**.

Voor de nieuwe kwartaalcijferslaag zijn daarnaast nodig:

- een SEC User-Agent met naam en contact-e-mailadres;
- een FMP API-key.

Beide kunnen bij eerste configuratie of later op **Systeem** worden ingesteld.
Gebruik daarna **Test verbindingen**. SEC/FMP-fouten blokkeren nooit het lezen
van al eerder opgeslagen earningssnapshots.

## GPU-machine van Floris

Kopieer `GPU_AGENT_FLORIS_20GB` naar de GPU-host. Zie de README in die map.
Gebruik een afzonderlijke restricted S3-key. De agent hoeft niet in hetzelfde
Hetzner-account of netwerk te staan als de webserver.

## Aanbevolen volgorde na upgrade

1. Object Storage opnieuw scannen.
2. Alle blokkers op **Data & gereedheid** oplossen.
3. SEC/FMP op NVDA importeren en de kwartaalmapping controleren; daarna dezelfde
   earningsagent voor alle bestaande tickers starten.
4. Nieuws-LLM-backfill uitvoeren.
5. De versnelde Clean Feature Store, events en Pair Store bouwen.
6. 384D dagvectoren bouwen.
7. Alpha Research starten; de permanente scorecard en `data_usage` controleren.
8. Pas na volledige historische quote-/dieptedata een niet-voorlopig
   kostenonderzoek uitvoeren.
9. Overtuigende kandidaten via exact dezelfde risklaag paper traden.
10. Live blijft uit totdat alle promotie-, monitoring- en infrastructuurgates
    aantoonbaar slagen.
