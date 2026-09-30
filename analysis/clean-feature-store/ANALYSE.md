# OptionEdge 20.0 — Clean Feature Store analyse

Datum: 2026-09-30. Baseline HEAD: `205bffc`. Geen applicatiecode gewijzigd, geen containers benaderd, geen productieverbinding geopend, geen marktdata gegenereerd of gedownload. Alleen dit rapport en een offline diagnostisch script toegevoegd. Geen functionele wijziging, dus nog geen nieuw functioneel checkpoint nodig.

## Conclusie en bewijsgrens

De code bevat aantoonbaar zes database-instellingenlezingen per HEAD/GET/PUT en per eerste LIST-pagina; vervolgpaginering heeft één extra instellingenlezing per pagina. Iedere instellingenlezing opent een Session, doet een primary-key SELECT en sluit die weer. De boto3-clientcache voorkomt dit NIET: instellingen worden vóór de cachelookup opgehaald. Bij bestaande settings gaat het dus om zes SELECTs plus transactieafhandeling per normale storage-operatie. Bij ontbrekende instellingen/exceptions kan dit aantal afwijken.

Andere concrete oorzaken van vermijdbaar werk: legacy gedeelde optie-/tradebestanden worden per ticker geheel gelezen; earnings doet voor iedere unieke timestamp vier filters/sorts over earningshistorie; brede frames worden meermaals gekopieerd/gesorteerd; automatische featurejobs gebruiken geen missing-only importplan en run_features schrijft geen featurecursor; na iedere job volgt een volledige inventoryscan. Vier parallelle dagen delen volgens compose 650 MB. Welke oorzaak de werkelijke weken/maanden-runtime domineert, is zonder datasets en telemetry nog niet vast te stellen. Een gegarandeerde minuten/uren-runtime zou nu ongefundeerd zijn.

De kopie bevat circa 2,8 MB code/git/cache, geen gevonden marktpartities of inventarissnapshot. `/tmp` bevat installatiebundels, geen gevonden benchmarkdata. Python3 is aanwezig maar pandas ontbreekt; daardoor zijn applicatieprofielen niet uitvoerbaar. Er is om bestaande lokale benchmarkdata en een ontwikkelverbinding gevraagd. Geen afhankelijkheden geïnstalleerd en geen werkende productie-runtime hergebruikt.

## Call-flow

```text
Streamlit render_standard_agent('features') [app/streamlit_app.py:304]
  POST /agents/features/preview
    build_plan → load_snapshot → get_json(v3/inventory/latest.json)
  POST /agents/features/start [api/main.py:1135]
    setup_complete → AppSetting SELECTs
    active-job SELECT → build_plan → INSERT AgentJob(payload.import_plan)
worker.runner main → next_job (queued SELECT) → run_agent_job [worker/tasks.py:168]
  ensure_company_rows → initial_counts → weekday/default/importplan
  dispatch features → run_features [worker/tasks.py:862]
    planned_dates_for_ticker → batches van 1..8, standaard 4
    ThreadPoolExecutor → build_one → save_feature_day
      build_feature_day [core/feature_engine.py:286]
        HEAD stock → GET/gzip/CSV → stock_features
        _load_option_day → LIST per ticker, fallback LIST dag → GET/gzip/CSV
          filter underlying → concat → volledige-row drop_duplicates
        normalize_timestamp → numeric conversion → sort → OCC parsing
        backward stock merge_asof, tolerance 2 minuten
        dte/moneyness/intrinsic/extrinsic → contract pct_change(1,5,30)
        _load_trade_summary → LIST/GET → groupby → ticker/minute left merge
        _oi_snapshot → HEAD/GET → dedup ticker → left merge
        _earnings_records → cached latest processed SEC/FMP snapshot
          attach_earnings_features → prepare eenmaal → per unieke ts lookup → merge
        _load_sentiment → HEAD/GET professionele v4-nieuwslaag
          _news_asof_features → rolling nieuws → 3 backward asof joins
        eventflags → ensure_quote_columns
        _attach_future_option_targets → contract-aware forward asof +5min tolerantie
        targets → geselecteerde featurekolommen
      put_df → volledige CSV-string → UTF8 bytes → gzip level 1 → PUT featuredag
    main thread: write_receipt → PUT JSON; update_company SELECT/UPDATE/commit
      job update/commit → batch gc.collect
    finalize_company_rows → finish_by_progress → finish
      queue inventoryjob → worker_inventory scan_inventory
        LIST heel market-data/ → GET historische manifests → PUT snapshots/latest
```

Celery bestaat alleen als leeg/commentaarbestand `worker/celery_app.py`; geen Celery-tasks, broker of concurrencyinstellingen actief. `worker.runner` voert één job synchroon uit. Features delen de queue met open_interest/events/macro/pairs/vectors; lange eerdere jobs vertragen de start. Auto-scheduler draait in worker_market en kan ook featurejobs aanmaken.

## Opslag, granulariteit en persistentie

PostgreSQL database `marketscope`: `app_settings` voor configuratie/secrets; `agent_jobs` voor queue/status/importplan; `agent_company_progress` per job/ticker; `agent_configs` en `agent_cursors` voor auto. Feature-rijen gaan NIET naar PostgreSQL. `core/db.py` gebruikt SQLAlchemy QueuePool met defaults (normaal pool_size 5/max_overflow 10), pre_ping; compose PostgreSQL max_connections 28, work_mem 4 MB, RAM 300 MB. Werkelijke pools/queries/indexen niet live geïnspecteerd. Job/status hebben enkelvoudige indexen; progress heeft unique(job_id,ticker), dus die lookup is al geïndexeerd. Geen bewijs dat een ontbrekende index primaire oorzaak is.

Object keys (T=ticker, D=dag):

| Rol | Key/prefix |
|---|---|
| Stocks | `market-data/v2/stocks/minute/T/YYYY/MM/D.csv.gz` |
| Opties | `market-data/v2/options/minute/YYYY/MM/D/by-ticker/T/*.csv.gz`, fallback dagprefix zonder by-ticker |
| Tradefeatures | `market-data/v3/option-trade-minute/YYYY/MM/D/by-ticker/T/*.csv.gz`, fallback gedeelde part-files |
| OI | `market-data/v3/open-interest/T/D.csv.gz` |
| Earnings | `market-data/v5/earnings/processed/T/` laatste basename-observationstamp, eenmaal gecachet per proces/ticker |
| Nieuws | `market-data/v4/professional-news/T/YYYY/MM/D.csv.gz` |
| Features | `market-data/v4/features/T/YYYY/MM/D.csv.gz` |
| Receipts | `market-data/v3/inventory/receipts/features/YYYY/MM/D/T.json` |
| Inventory | `market-data/v3/inventory/latest.json` en `snapshots/<stamp>.json` |

De outputeenheid is ticker × dag; rijen zijn optiecontract × timestamp. Expiry/strike worden uit OCC-symbolen afgeleid. Geen losse PUT/commit per record/minuut. Stockbasis, joinresultaten en featuresubstappen zijn alleen in RAM; trade-minute-summaries en processed earnings bestaan wel persistent. Features en receipts checkpointen op dagniveau; geen checkpoint binnen een dag.

Missing-only bestaat al voor handmatige starts op basis van inventarissnapshot. De snapshot kan verouderd zijn, en fysieke aanwezigheid/receipt bewijst nog geen correcte horizon/sourceversie. Automatische jobs hebben geen importplan; planned_dates_for_ticker valt terug op alle weekdagen. run_features roept mark_cursor niet aan. Een force-run overschrijft dezelfde featurekey, ongeacht horizon. Kernbestanden naast bovenstaande: core/config.py, core/storage.py, core/runtime_settings.py, core/import_planner.py, core/storage_inventory.py, core/earnings.py, core/quotes.py, core/models.py, core/db.py, docker-compose.yml. Theoretische Black-Scholes-berekeningen zitten in downstream pairfeatures en zijn geen onderdeel van dit buildpad.

## Uitgevoerde offline microbenchmark

Reproduceerbaar: `python3 analysis/clean-feature-store/benchmark_call_counts.py`. Het script extraheert de bestaande storagefuncties via AST en voert ze uit met tellende mocks. Geen netwerk/database en geen applicatie-imports (worker.tasks importeert init_db met schemawrites!). JSON bevat bronhash, CPU-tijd en alle 12 metingen. Dit meet uitsluitend wrapperoverhead en request-/settingsmultiplicatie, NIET marktthroughput.

| Operatie | 1.000 calls: seconden | 10.000 | 100.000 | Settingslezingen bij 100.000 |
|---|---:|---:|---:|---:|
| HEAD | .003664 | .032382 | .388692 | 600.000 |
| GET | .005102 | .031541 | .358907 | 600.000 |
| PUT | .003452 | .032728 | .347584 | 600.000 |
| LIST, één pagina | .003516 | .041229 | .456385 | 600.000 |

Peak RSS script 17.024 KiB; CPU bij 100.000 calls respectievelijk .316/.352/.337/.407 sec. Werkelijke DBqueries en S3requests: nul (mocks tellen de bedoelde aanroepen). Marketrecords/outputbytes/outputhash: niet van toepassing. Bronhash staat in JSON. De tijden zijn geen voorspelling voor I/O.

Hostsnapshot: RAM totaal 3,73 GiB, beschikbaar 1,76 GiB; swap totaal 2 GiB, ongeveer 847 MiB bezet. Dit is hoststatus, GEEN bewijs dat de featureworker momenteel swapt of OOMt. Docker-/workergebruik is bewust niet gemeten. Dag/ticker/1.000–100.000-records profielen, downloadtijd, CSV, joins, writes en outputhash zijn nog niet meetbaar zonder data/dependencies.

Analytische minimumdag bij 1 stockfile, 1 tickeroptiefile, geen trades/OI/nieuws, warme earningscache: 3 HEAD + 2 GET + 3 LIST (tradefallback) + 2 PUT = 10 requests, 60 settings-SELECTs. Aanwezige trades/OI/nieuws en cold earnings verhogen dit; iedere extra partfile voegt GET en zes SELECTs toe. Retries niet inbegrepen. LIST over >1.000 keys voegt paginering toe. Legacy-layout versterkt GET/CSV-werk met ongeveer aantal tickers dat dezelfde bestanden nodig heeft; metadata overhead blijft afzonderlijk.

## Top 10, in implementatievolgorde na meting

Rangschikking is voorlopig: verwachte winst versus risico, niet een gemeten runtime-ranking. Tijden zijn ontwikkelschattingen inclusief gerichte verificatie.

| P | Bottleneck / bestand-functie | Oorzaak en bewijs | Voorstel | Verwachte winst | Risico / validatie | Tijd |
|---|---|---|---|---|---|---|
| 1 | Settings N+1 — storage.client/bucket_name/runtime_settings.get_setting | Offline exact 6 lezingen per operatie; Session.get per lezing | Run-scoped immutable storagecontext, expliciete refresh nieuwe job | Tot ~6Q−6 settingsqueries minder bij Q requests; totaal afhankelijk DB-aandeel | Laag-middel: credentialrotatie/exceptiongedrag; gelijke requests en outputhash, wisselende settings testen | 0,5–1 dag |
| 2 | Herbouw auto — tasks.check_auto_agents/run_features | Geen importplan, geen featurecursor | Missing-only plan ook auto; resume met bronversies/receipts | Bij 99% al aanwezige geldige output tot ~100× minder buildwerk, geen pure full-rebuildwinst | Middel: gewijzigde bronnen/horizon niet overslaan; crash/resume/calendar-tests | 1–2 dagen |
| 3 | Legacy herlezingen — feature_engine._load_option_day/_load_trade_summary | Alle gedeelde partfiles per ticker, filter pas na CSV | Lees gedeelde dag eenmaal en routeer per ticker naar tijdelijke cleanpartities | Inputdeel potentieel tot T× voor T tickers | Middel: behouden authoritative by-ticker, dedup/filtervolgorde, rowmultiset | 2–4 dagen |
| 4 | Geheugendruk — run_features/build_feature_day/compose | 4 brede dagen in 650 MB, meerdere kopieën | Meet 1/2/4; geheugenbudget per taak en bounded scheduling | Onbekend; mogelijk grootste winst bij swap/OOM, anders geen winst | Laag bij scheduling; controle RSS/oom/retries en output | 0,5–1,5 dag |
| 5 | Earnings per minuut — attach_earnings_features/earnings_features_at | Iedere unieke ts 4 historische filters/sorts; voorbereiden al gecachet per dag | Op PIT-change boundaries selecties vooraf bepalen, age vectoriseren | Onderdeel van O(U×E log E) richting O(E log E+U); totaal onbekend | Hoog: revisies/ties/fail-closed; oracle per timestamp en beschikbaarheidsgrenzen | 2–4 dagen |
| 6 | CSV/gzip — storage.get_df/put_df | Full bytes, decompress, dtype inference, CSV-string/UTF8/gzip gelijktijdig | Eerst beperkte reads als semantisch veilig; later additionele Parquetpartities | Onbekend tot codec benchmark; compressie is al level 1 | Middel: CSV-roundtrip dtypes/NaN/dedup; geen float32 zonder bewijs | 2–4 dagen |
| 7 | Brede targetsort/copy — _attach_future_option_targets | Kopie hele frame, meerdere brede sorts/joins | Smalle sleutel-/waardetabel + stabiel row-id; kolommen daarna terugkoppelen | Alleen gemeten sort/join/copydeel, mogelijk lagere RSS | Middel: tie-order bij gelijke keys, duplicates, forward tolerance | 1–2 dagen |
| 8 | Nieuws redundantie — _news_asof_features | 3 sorts/asof over contractrijen met identieke timestamps | Bereken mapping per unieke timestamp, map terug met row-id | Join-input N→U, theoretisch N/U reductie inputgrootte | Middel: huidige positionele toewijzing kan bij ongesorteerde ts afwijken; niet stil corrigeren | 1–2 dagen |
| 9 | Volledige inventory — finish/scan_inventory | Global LIST en historische manifest GET na job | Delta receipts/manifestmetadata + periodieke volledige audit | Scandeel O(alle objecten)→O(gewijzigd), geen directe buildwinst | Middel: externally gewijzigde objecten en snapshotstaleness | 2–3 dagen |
| 10 | Batchbarrières/queue — run_features/worker.runner | Nieuwe threadpool/batch; wachten op traagste dag, één gedeelde jobworker | Eén bounded pool met continue aanvulling; later aparte featuresqueue | Onbekend, significant alleen bij scheve dagkosten | Middel: RAMbudget, main-thread DB, jobclaim/concurrent writers | 1–2 dagen |

## Expliciete checklist overige bottlenecks

Geen iterrows/apply of nested per-record loops in feature_engine. Wel per-object loops en earnings per unieke minuut; iterrows/apply in earnings-import/combine vallen buiten de featurebuild met processed snapshots. Optiemomentum en targets zijn al gegroepeerd/vectorized, OCC-parsing reeds vectorized (maar regex nog per rij, unieke contractmapping is later onderzoek). Geen historische stockfullscan; stock/day. Legacy opties/trades wel volledige dag voor alle tickers. Geen caching van die frames; earnings en boto-client wel caching, cold concurrent cachemisses kunnen dezelfde earnings laden. DataFrame object/string en float64-geheugen niet gemeten. Geen aanwijzing voor SQL SELECT * op featuredata: SQLAlchemy select(Model) leest hele controlrows, niet marktframes. Geen commits/writes per marktrecord. Trades groupby en OI dedup beogen many-to-one, maar merge-cardinaliteit wordt niet expliciet gevalideerd. CSV/gzip decode/encode is volledig in geheugen. Python/Pandas/CSV/earnings hebben deels GIL-beperking; threads overlappen I/O maar maken CPU geen garantie parallel. Multiprocessing kan RAM en serialisatie juist vergroten. Stopcheck alleen vóór batch. Exceptions bij option/trade GET en earnings-load kunnen stil data weglaten; exists maskeert alle errors als afwezig, retries/backoff tot max_attempts 8/readtimeout 120s kunnen lang duren. Dit vereist error/requesttelemetry vóór agressieve retries of cachingwijzigingen. Geen extra inner-loop validatie behalve PITfilters; deze mogen niet verdwijnen.

## Architectuur en tools

Bestaande ticker/jaar/maand/dag-featurepartities zijn een geschikte basis, dus geen herbouw van het project. Voeg eventueel een gestandaardiseerde cleanbasislaag naast bestaande rawdata toe, met immutable bronmanifest (ETag én waar nodig inhoudshash), code/schema/horizonversie en complete status. Alleen nieuwe/gewijzigde dependency-partities verwerken. Compacte Parquetfeaturepartities additioneel aanbieden via compatibele readers, CSV behouden tot gelijkheid/downstreamcompatibiliteit bewezen. Metadata/checkpoints verwijzen exact naar objectversies; model-ready manifest selecteert expliciete features en aparte labels.

PyArrow: kandidaat voor typed Parquet en geselecteerde kolommen, bewezen pas via codec/RSS-tests. DuckDB: kandidaat om legacy CSV once te scannen/partitioneren zonder brede pandasframes; CSV gzip kan alsnog volledige scan vereisen. Polars: alleen als profiel sort/join/groupby dominant toont; asof-ties/nulls/rolling moeten identiek bewezen. Vectorized NumPy: kansrijk voor earnings-boundarymapping en numerieke kolommen, behoud float64. concurrent.futures is al actief; benut bestaand bounded threading eerst. Multiprocessing pas bij bewezen CPU-bound werk en toereikend RAM, zonder grote DataFrame-pickles. Geen dependencywijziging nu. Tools niet online onderzocht; voorstellen zijn onderzoekskandidaten, geen actuele API/performanceclaims.

## Reproduceerbaar benchmark- en integriteitsplan

1. Gebruik bestaande complete partitions: één ticker/dag met alle bronnen, daarnaast lege/missing optionele bronnen, dense en sparse dagen, legacy en by-ticker, earnings na marktclose/revisies. Geen willekeurige head(N) die targets/rollinghistory kapotmaakt.
2. Fixturemanifest bevat objectkeys, contenthashes, sizes, ticker/date, sourceformat, horizon, settings zonder secrets, git SHA, exacte Python/packageversies, hardware/threadinstellingen. Remote reads uitsluitend read-only ontwikkelcredentials; outputs naar /tmp geïsoleerde directory. Nooit api.main/worker.tasks rechtstreeks importeren zonder isolatie vanwege init_db.
3. Meet afzonderlijk remote download en lokaal replay; koude/warme cache apart. 1 warm-up plus 3 herhalingen; median/p95 en profiler-overhead apart. Stop bij RAMbudget, geen grotere runs na OOM/swap.
4. Meet perf_counter/process_time per loader, gzip decode, read_csv, stockfeatures, normalization/OCC, iedere join, earnings/news, targets, selectie, to_csv/UTF8/gzip, upload, receipts/control-DB/planning. Profiler cProfile; wrappercounters voor functies; SQLAlchemy before/after_cursor_execute op geïsoleerde engine; botocore events tellen LIST/HEAD/GET/PUT en retries. GET Body.read apart tellen om netwerkstreamtijd niet te missen. CPU/RSS iedere 50 ms via psutil, procesboom voor multiprocessing, swap deltas en cgroupstats uit ontwikkelworker. Hostswap is geen procesattributie.
5. Reeksen: 1 ticker × dag → ticker × maand → 5 tickers × maand. Records 1k/10k/100k alleen op complete behouden contracttrajecten met correcte context; rapporteer daadwerkelijke input/outputcounts, unieke ts/contracts en tijdsbereik. Eerst kleinere complete dag als 100k te zwaar is.
6. Resultaatschema: scope, sourcehash/codeSHA, input/outputrecords, seconds/records_per_second, stage times, cpu_seconds/cpu_percent (conventie vastleggen), peak RSS, swap delta, DB SELECT/UPDATE/commits, S3 requests/retries/bytes, outputbytes, canonical/outputfile SHA256. Lezen en schrijven inclusief meting; output niet naar productie.
7. Vergelijk rowmultiset inclusief duplicaten, ticker/UTC ts/expiry/strike/type, kolommen en volgorde, nullmask/infinities en cardinaliteit. Canonical sort met tie-id; SHA voor exacte vergelijking, numeriek aanvankelijk rtol=1e-12/atol=1e-12 (per feature beoordelen), structurele waarden exact. CSV-roundtrip apart van in-memory output vergelijken; gzip-bytehash alleen met vaste metadata of inhoud decomprimeren.
8. PIT: controleer availability <= ts, backward stock tolerance 2min, future target uitsluitend zelfde contract vanaf ts+h binnen 5min; targets uitgesloten modelinputs. Controleer after-close/BMO/AMC, revisies, timezone/DST, ontbrekende minuten, markturen, expiraties, nulprijzen/nulls, dagsgrenzen, duplicate timestamps/contracten en horizon 15/30/60. Optiemomentum is momenteel rijlag, ondanks m-naam; optimalisatie moet dat behouden. Geen overnight context toevoegen zonder aparte functionele toestemming. Bestaande nieuwsrolling blijft na laatste artikel staan tot volgende artikel; eventuele correctnessfix separaat, niet mengen met performance.
9. Geen verandering theoretische opties/labels/targets. Downstream pair/modeltests en existing earnings/leakage-tests draaien in geïsoleerde devomgeving zodra dependencies beschikbaar zijn. Nog niet uitgevoerd.

## Runtime en drie scenario's

Niet bekend: totale records R, ticker-dagen D, echte throughput v en aandeel per stadium. Geen gemeten marktthroughput, dus huidige volledige runtime en speedups kunnen NIET numeriek berekend worden. Gebruik daadwerkelijke end-to-end elapsed (niet individuele parallelle regels/s optellen): T0=R/v, of sum(t_dag)/effectieve concurrency plus planning/inventory. Stratificeer legacy/dense/sparse, p95, cold cache en retrykosten.

A: quick wins settingscontext, goede missing-only planning, memorybudget. SA=1/((1-fA)+fA/sA), TA=T0/SA. Als DBsettings bijvoorbeeld 50% tijd zijn en 10× sneller worden, SA=1,82 (illustratie, geen verwachting). Bij partial updates remaining/full-ratio apart rapporteren; dit is geen full-rebuildspeedup.

B: A plus legacy once-per-day cleanbasis, smalle joins en PITearnings. TB=T0×som(f_i/s_i), SB=T0/TB. Alleen inputdeel van shared-layout kan T× verminderen; dit niet vermenigvuldigen met alle andere globale factoren.

C: B plus gemeten Parquet/enginekeuze en RAMgebudgetteerde CPUparallelisatie. TC bestaat uit werkelijke compute/IO/serialization/metadata; storagebandwidth/CPU/RAM begrenzen parallelisme. Geen garantie voor minuten of uren.

Doeltest: voor 1 uur moet v>=R/3600, voor 4 uur v>=R/14400, voor 10 minuten v>=R/600. Vergelijk ook minimaal te lezen bytes / gemeten storagebandwidth. Zodra manifest en kleine echte profielen beschikbaar zijn, tabel vullen met T0/TA/TB/TC en onzekerheidsinterval. Nu is het verwachte totaalresultaat minder redundant werk en lager geheugen, maar niet kwantificeerbaar.

## Eerste kleine wijziging en rollback

Eerste stap na toestemming en echte baselineprofiel: storage-settingscontext voor één featurejob zodat dezelfde endpoint/region/bucket/access/secret-set niet per object zesmaal wordt gelezen. Behoud algemene live settingssemantiek voor andere callers, boto-cache en requestinhoud; secrets nooit loggen. Ontwerp exact refresh/invalidation/jobgrenzen voordat implementatie start. Beoogde bestanden core/storage.py, core/runtime_settings.py en worker/tasks.py, plus gerichte isolatietest. Als scope voor veilige context te groot blijkt, eerst alleen diagnostische timings in development toevoegen na checkpoint.

Vóór elke functionele stap: schoon statusreview, afzonderlijke checkpointcommit `checkpoint: before <measure>` op werkbranch, noteer SHA in benchmarkmanifest; huidige startreferentie 205bffc. Analysis-artifacts afzonderlijk committen indien gewenst; geen commit nu gemaakt. Iedere maatregel eigen commit en echte before/aftermanifest. Rollback via git revert van de maatregelcommit; baseline niet resetten en geen bestaande userwijzigingen verliezen. Engine/Parquet/metadatawijzigingen elk aparte commits, readers achter compatibiliteitskeuze; oude keys behouden, nieuwe outputs nieuwe versioned prefix, geen overwrites. Revert maakt oude readers actief; geen bucketcleanup nodig. Future changes mogen productie niet deployen zonder afzonderlijke opdracht. In-place overschreven objecten zijn niet met git terug te halen: daarom in benchmark/optimalisatiefase uitsluitend aparte outputs.

Stopstatus: analyse van codepad afgerond; representatieve performanceprofiling en exacte runtimeoorzaak wachten op bestaande data/devruntime. Geen applicatiecode veranderen vóór toestemming.
