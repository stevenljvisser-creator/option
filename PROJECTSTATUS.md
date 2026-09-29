# OptionEdge 2.0 — gecontroleerde projectstatus

Deze status maakt onderscheid tussen **gebouwde software** en **data/resultaten
die pas op de echte Hetzner-omgeving kunnen ontstaan**. Een aanwezig scherm of
object is nooit automatisch bewijs dat een dataset inhoudelijk compleet is.

## Volledig uitgevoerd in de code

- De bestaande PostgreSQL-volumes, secrets en Object Storage-prefixen v2/v3/v4
  blijven behouden; nieuwe fusie-uitvoer gebruikt `market-data/v5`.
- Volledig klikbare Python/Streamlit-interface in de donkere Expectra-stijl.
- Eén versievaste 384D vector per ticker-handelsdag:
  192 nieuws/events, 96 opties, 48 aandeel/markt, 32 verwachting en 16
  kwaliteit/masks.
- Ruwe BGE-M3-chunkrepresentaties van 1024D en Qwen-eventvelden blijven apart
  opgeslagen en worden samen target-vrij in het nieuwsblok geprojecteerd.
- Theoretische én praktisch waargenomen verwachtingswaarde, tijdwaarde,
  theory/market-gap en put-call-parity staan expliciet in de bronfeatures.
- Dagelijkse as-of op de marktsluiting; te oude contractprints worden
  uitgesloten. Toekomstwaarden staan alleen in fysiek gescheiden labels.
- Effectieve nieuws-beschikbaarheid (`available_utc`) voorkomt dat een gewone
  webbackfill die pas later is gezien ongemerkt in een historische vector komt.
- GPU-agent voor Floris' 20-GB-GPU met Qwen 14B-AWQ/8B-AWQ-fallback, BGE-M3,
  revisies, deduplicatie, rechtenbeleid, broncoverage en Object Storage-koppeling.
- Gesorteerde Object Storage-inventaris en harde poorten voor vectorbouw en
  training; v4-compatibiliteitsnieuws telt niet als v5-bronbewijs.
- Diepe dagelijkse verbandscan met discovery/confirmation, horizonafhankelijke
  purge/embargo, effectgrootte, Benjamini–Hochberg FDR en within-ticker-controle.
- Dimensievergelijking 128/256/384/512/768 op dezelfde chronologische perioden;
  selectie op validatie-Brier, gevolgd door calibratie en één finale hold-out.
- Nieuws-ablation, call/put-kansmodellen, rendementsmodellen, verwachte
  extrinsieke waarde en bruto/paper-edge in de website.
- Modelcoach kan na een expliciet verzoek vectoren, v20-verbandanalyse of
  Fusie 2.0-training inplannen; hij omzeilt de datapoort niet.
- Veilige upgrade met codebackup, healthchecks en rollback; geen
  `docker compose down -v` en geen verwijdering van Object Storage-data.
- Append-only SEC/FMP-earningslaag met officiële SEC-actuals, FMP-consensus,
  earningsdatum/-tijd, fiscale kwartalen, surprises en point-in-time
  beschikbaarheidsvelden. Jaar-EPS wordt nooit foutief tot Q4 afgetrokken.
- Tijdelijke SEC/FMP-uitval houdt oude snapshots en verwerkte historie intact;
  importstatus, fouten, laatste succesvolle bronimports en status per ticker
  zijn zichtbaar.
- Alpha Research Engine met simpele baseline, Ridge, purged walk-forward,
  embargo, één onaangeraakte hold-out, realistische maar zo nodig voorlopige
  kosten, NO_TRADE en exacte feature-ablaties.
- De primaire alpha-run gebruikt alle geschikte geïmporteerde ticker-dagen in
  de gekozen periode zonder verborgen row cap. Eén vooraf gedefinieerd near-ATM
  contract per ticker-dag voorkomt pseudo-replicatie.
- Economische scorecard met EV_net, expected value per trade, netto return,
  Sharpe, Sortino, profit factor, drawdown, win rate, payoff, turnover,
  exposure en trades. Oude resultaten blijven zichtbaar bij onvolledige nieuwe
  imports en hebben een databasefallback naast het volledige S3-manifest.
- Afzonderlijke leakage-, risk-, paper-trading- en monitoringlagen met
  fail-closed kill switch. Live orders zijn hard uitgeschakeld.
- Clean Feature Store versneld door vectorized OCC-parsing, één gegroepeerde
  contractbewuste targetjoin en earningscache per ticker; contractisolatie is
  met een regressietest vastgelegd.

## Nog uit te voeren op de echte omgeving

Deze onderdelen zijn bewust niet als “gereed” gemarkeerd voordat de website de
echte opslag heeft gescand:

1. Upgrade op de bestaande server uitvoeren en de eerste read-only
   Object Storage-inventaris afronden.
2. GPU-agent op Floris' machine installeren, modellen downloaden en een echte
   CUDA/VRAM-run uitvoeren.
3. SEC User-Agent en FMP API-key instellen; eerst de echte NVDA-import uitvoeren
   en de gerapporteerde kwartalen handmatig tegen bronfilings controleren.
   Daarna kan dezelfde generieke worker alle ingestelde OptionEdge-tickers
   zonder codewijziging verwerken.
4. Een gelicentieerd, versie-vast historisch nieuwsarchief toevoegen als brede
   historische nieuwsdekking gewenst is. Google News/RSS alleen bewijst geen
   volledigheid en “alle internetnieuws” is niet aantoonbaar.
5. Alle door **Data & gereedheid** gemelde bronhiaten importeren, in het
   bijzonder contractreferentie/corporate actions, risicovrije curve,
   earnings-tijden en volledige v5-nieuwslagen/coverage-receipts.
6. 384D-vectoren en Alpha Research op de echte complete data draaien. Alleen
   vooraf geselecteerde kandidaten mogen de finale hold-out openen.
7. Historische entry/exit NBBO, quote-size, spread, slippage, fees, minimum tick
   en fillregels toevoegen; tot die tijd blijft het kostenmodel voorlopig.
8. Voldoende live papertrades verzamelen en backtest/predicted/paperresultaten
   vergelijken; brokeradapter, operationele driftbaseline en kill-switchtests
   afronden. Pas daarna kan beperkte live inzet überhaupt worden beoordeeld.

## Bewust zichtbaar als volgende onderzoeksfase

De interface noemt deze methoden expliciet als nog niet volledig uitgevoerd:

- historische embedding-neighbours;
- formele eventstudy met abnormal returns en sectorbenchmark;
- vooraf vastgelegde lag-analyse;
- uitgebreidere regime-/interactiepatronen met voldoende support (basisregimes
  worden al point-in-time gerapporteerd);
- extra calibratie slope/intercept en betrouwbaarheidsintervallen.

Ze horen pas te worden toegevoegd nadat de verplichte data compleet is; eerder
uitvoeren zou vooral meer schijnprecisie opleveren.

## Besluit over 384 dimensies

384D blijft de primaire, interpreteerbare opslagvariant. Meer dimensies zijn
niet automatisch beter: 512D of 768D wordt alleen gekozen wanneer die variant
op identieke, latere validatieperioden een lagere Brier-score laat zien en de
finale test stabiel blijft. De ruwe bronnen blijven behouden, dus deze keuze
vereist geen nieuwe nieuws- of marktimport.

## Technische verificatie vóór verpakking

- 38 server-/onderzoekstests geslaagd.
- 6 GPU-agentlogica-tests geslaagd.
- Alle Pythonmodules compileren.
- FastAPI OpenAPI bevat alle earnings-, alpha-, scorecard-, risk-, paper- en
  monitoringroutes; worker- en Streamlitmodules compileren.
- Docker Compose-structuur en alle shellscripts syntactisch gevalideerd.
- Een echte modeldownload, echte Object Storage-verbinding en productie-training
  kunnen alleen met de bijbehorende credentials, data en GPU worden bevestigd.

## Update 23 september 2026 — earnings, volledige databenutting en snelheid

- SEC EDGAR Company Facts/XBRL en FMP earnings blijven de point-in-time earningslaag; ruwe snapshots en processed snapshots zijn append-only in `market-data/v5/earnings/`.
- Modeltraining gebruikt standaard alle beschikbare geschikte Feature Store-bestanden en rijen; representatieve sampling is niet langer de productiestandaard.
- Modelruns registreren `data_policy` en `data_usage`, zodat achteraf controleerbaar is hoeveel data werkelijk is gebruikt.
- De Clean Feature Store hergebruikt de S3 connection pool, bereidt earnings eenmaal per featuredag voor en verwerkt featuredagen begrensd parallel (standaard vier workers).
- De earnings-agent gebruikt standaard alleen bedrijfstickers; SPY en QQQ blijven onderdeel van markt-, optie-, regime- en featuredata, maar niet van corporate earnings-import.
