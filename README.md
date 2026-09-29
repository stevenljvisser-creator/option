# OptionEdge 2.0 / research engine 20.0

OptionEdge onderzoekt wanneer de theoretische en marktgeobserveerde
verwachtingswaarde in opties waarschijnlijk afwijkt van wat daarna gebeurt. De
nieuwe fusielaag combineert per ticker-handelsdag nieuws, opties, het aandeel en
verwachtingswaarde in één versievaste vector.

Het primaire doel is nu alpha-onderzoek voor een uiteindelijk robuust
handelsmodel. Historische accuracy of een fraaie backtest is geen einddoel. Een
kandidaat moet buiten de trainingsperiode, na kosten en met aanvaardbaar risico
standhouden. De software garandeert geen winst en echte orderuitvoering staat
hard uit totdat alle promotiepoorten zijn bewezen.

## Onderzoekscontract

De primaire vector heeft 384 dimensies:

| Coördinaten | Blok | Dimensies |
|---|---|---:|
| 0–191 | nieuws en gebeurtenissen | 192 |
| 192–287 | optie-oppervlak en activiteit | 96 |
| 288–335 | aandeel en marktcontext | 48 |
| 336–367 | theoretische én praktische verwachtingswaarde | 32 |
| 368–383 | dekking, kwaliteit en masks | 16 |

Ruwe nieuwschunks blijven als 1024D BGE-M3-embeddings opgeslagen. De 192D
nieuwscomponent combineert die semantische representatie met een vaste,
target-vrije samenvatting van de Qwen-eventvelden (eventtype, richting, impact,
horizon, surprise, relevantie, importance, uncertainty en novelty). De
fusietraining vergelijkt 128/256/384/512/768D op dezelfde chronologische folds.
384D is dus de vooraf gekozen hoofdvariant, niet een vooraf uitgeroepen winnaar.

De verwachtingswaardeblok bevat expliciet:

- Black–Scholes/volatiliteitsbenchmark en verdisconteerde theoretische waarde;
- praktisch waargenomen call-/putprijs en extrinsieke tijdwaarde;
- theorie-markt-gap en put-call-parity-residual;
- risicovrije rente, looptijd, strike en volatiliteitscontext.

Gerealiseerde toekomstige payoff, toekomstige extrinsieke waarde en toekomstig
aandeelrendement zijn uitsluitend targets. Ze komen nooit in dezelfde
dagvector terecht.

## Nieuws-LLM

De losse map `GPU_AGENT_FLORIS_20GB` bevat de GPU-worker:

- Qwen3-14B-AWQ met automatische 8B-fallback;
- BAAI/BGE-M3, 1024D per gededupliceerde chunk;
- schema-gevalideerde eventextractie per ticker;
- publicatie-, `first_seen`- en effectieve `available_utc`-tijd, revisies, rechten, bronkwaliteit,
  fetchstatus en duplicaatcluster;
- koppeling via beperkte Hetzner Object Storage-credentials, ook als GPU en
  webserver in verschillende accounts staan;
- broncoverage en mislukkingen per ticker-maand.

Letterlijk alle internetnieuws is niet bewijsbaar. De site toont daarom exact
welke bronuniversums zijn doorzocht, waar data ontbreekt en welke fetches door
robots, paywall, licentie, timeout of parsing mislukten. Voor volwaardige
historische dekking is een gelicentieerde feed nodig.

Een gewone historische webcrawl geldt niet automatisch als point-in-time
archief. Voor niet-verifieerbare backfills wordt de lokale `first_seen_utc` als
beschikbaarheidstijd gebruikt; alleen een expliciet geconfigureerd versiearchief
of een officiële SEC-acceptatietijd mag de historische publicatietijd gebruiken.

## Relatie- en modelonderzoek

- één onafhankelijke observatie per ticker-handelsdag;
- chronologische discovery (vroeg) en confirmation (later), gesplitst op hele
  datums;
- maximaal 1.200 kandidaatsignalen;
- Spearman, mutual information en quartile-effecten;
- Benjamini–Hochberg-correctie in discovery én confirmation;
- een verband telt pas als zowel globaal als na within-ticker demeaning de
  richting en FDR-drempel standhouden;
- dimensieselectie op validatie-Brier;
- aparte calibratieperiode en finale onaangeraakte hold-out;
- ablation zonder nieuws om de echte bijdrage van het taalmodel te meten.

Dit levert voorspellende verbanden en hypotheses, geen causaliteitsbewijs.

## Earnings en kwartaalcijfers

De earningslaag gebruikt twee gescheiden rollen:

- SEC EDGAR Company Facts/XBRL en Submissions voor officiële werkelijke
  kwartaalcijfers, filingdatum, acceptatietijd, boekjaar, kwartaal en formulier;
- Financial Modeling Prep voor earningsdatum/-tijd en de marktverwachtingen
  voor EPS en omzet.

Revenue, net income, EPS, assets, liabilities en operationele cashflow worden
per fiscal quarter genormaliseerd. Cumulatieve SEC-flowwaarden worden alleen
gequarteriseerd wanneer de voorgaande kwartalen aantoonbaar beschikbaar zijn;
EPS wordt nooit afgetrokken omdat EPS niet additief is. SEC en FMP worden per
ticker/kwartaal gekoppeld en leveren absolute en procentuele EPS- en
omzetverrassingen.

Iedere SEC- en FMP-respons wordt content-addressed en append-only bewaard. Een
FMP-schatting telt alleen als historische feature wanneer OptionEdge die
daadwerkelijk vóór de earnings-cutoff heeft waargenomen. Een latere backfill
blijft bewaard als `backfill_unverified`, maar wordt niet terug de geschiedenis
in gelekt. Tijdelijke bronuitval verwijdert of overschrijft nooit eerder
opgeslagen data.

## Alpha Research Engine

De primaire onderzoeksmotor:

- catalogiseert alle geïmporteerde, geschikte ticker-handelsdagen in de gekozen
  periode; er geldt in dit primaire pad geen verborgen rij- of daglimiet;
- gebruikt één vooraf vastgelegde near-ATM contractobservatie per ticker-dag als
  onafhankelijke handelseenheid, nadat alle aanwezige contracten zijn bekeken;
- begint met een historische-meanbaseline en Ridge, en reserveert één volledig
  onaangeraakte eindperiode;
- gebruikt purged expanding walk-forward-validatie met horizonafhankelijke purge
  en embargo;
- test exacte ablaties zonder nieuws, embeddings, opties, earnings, technische
  features en theoretische optiewaarde;
- rapporteert EV netto, netto return, Sharpe, Sortino, profit factor, maximale
  drawdown, win rate, payoff, turnover, exposure en aantal trades;
- maakt kosten voorlopig zolang entry/exit bid-ask en diepte ontbreken;
- kan expliciet `NO_TRADE` kiezen wanneer verwachte edge niet boven kosten,
  veiligheidsmarge en onzekerheid uitkomt.

Iedere run, voorspelling, kostenset, featurelijst, fold en metric wordt
append-only opgeslagen. De website toont daarnaast een permanente scorecard.
Een onvolledige nieuwe import wist oude resultaten niet; een compacte kopie van
de hold-outmetrics blijft ook bij de bestaande modelrun staan.

## Risk, paper trading en monitoring

Signal generation, risk en execution zijn gescheiden. De onafhankelijke
risklaag controleert datakwaliteit/leakage, edge na kosten, liquiditeit, spread,
onzekerheid, exposure, concentratie, correlatie, volatiliteit, drawdown en het
dagverlies. Paper-signalen en afwikkelingen zijn append-only en gebruiken
dezelfde beslisroute als een toekomstige execution-adapter.

Monitoring meet onder andere verwacht versus werkelijk resultaat, Brier en
calibratie, feature-/prediction drift, drawdown, spreads, liquiditeit en
uitvoeringskosten. Kill-switchredenen zijn expliciet. Live trading blijft in
deze versie hard uit en er is nog geen brokeradapter.

## Snellere Clean Feature Store

Deze update verwerkt featuredagen begrensd parallel (standaard 4 workers) en
hergebruikt één boto3/S3 connection pool per workerproces. De earnings-history
wordt per featuredag één keer naar point-in-time timestamps genormaliseerd in
plaats van opnieuw per minuut. Dit verandert geen features of leakage-regels;
het verlaagt alleen onnodige S3- en pandas-overhead.

## Volledige databenutting bij training

Nieuwe modelruns gebruiken standaard **alle beschikbare geschikte rijen** uit
alle Feature Store-bestanden in de gekozen periode. De eerdere representatieve
file/row-sampling blijft alleen als expliciete backwards-compatible debugmodus
bestaan. Modelmetrics slaan `data_policy` en `data_usage` op zodat achteraf
controleerbaar is hoeveel bestanden en rijen werkelijk zijn gebruikt.

De bestaande opslagcontracten zijn behouden. De zware OCC-contractparser is
gevectoriseerd, toekomstige optiontargets worden met één contractbewuste
`merge_asof` gekoppeld in plaats van een Python-loop per contract, en de
verwerkte earningshistorie wordt per ticker in de worker gecachet. De worker
toont regels per seconde. Targets blijven strikt per optioncontract geïsoleerd.

## Nieuwe Object Storage-laag

Bestaande v2/v3/v4-data blijft intact. Nieuwe v5-uitvoer staat onder:

```text
market-data/v5/news-articles/
market-data/v5/news-chunks/
market-data/v5/news-embeddings/
market-data/v5/news-events/
market-data/v5/daily-vectors/
market-data/v5/daily-labels/
market-data/v5/deep-relationships/
market-data/v5/fusion-models/
market-data/v5/gpu-news-agent/
market-data/v5/earnings/raw/sec/
market-data/v5/earnings/raw/fmp/
market-data/v5/earnings/processed/
market-data/v5/earnings/status/
market-data/v5/alpha-research/
market-data/v5/paper-trading/
market-data/v5/trading-control/
market-data/v5/monitoring/
```

Elke 384D vector krijgt daarnaast een raw `.npz`, metadata-JSON, schemahash,
bronpaden en expliciete maskvector.

## Interface

De Streamlit-interface gebruikt de donkere marine/cyaan Expectra-stijl in de
bestaande OptionEdge-omgeving. Belangrijkste pagina's:

- Overzicht en projectpoorten;
- Onderzoeksontwerp;
- Data & gereedheid;
- Dagvectoren;
- Nieuws-LLM;
- Diepe verbanden;
- Fusie- en dimensietraining;
- permanente modelscorecard en Alpha Research;
- Kans & edge;
- Verbeteragent;
- lopende taken, agents en roadmap.

De data-pagina toont geïmporteerde datasets gesorteerd per laag en toont apart
alles wat nog moet worden geïmporteerd. Modeltraining is zichtbaar geblokkeerd
zolang de verplichte data-poorten volgens de laatste Object Storage-scan niet
compleet zijn.

## Starten

```bash
docker compose build
docker compose up -d
```

Open daarna poort 80. De eerste configuratie vraagt de bestaande Hetzner
Object Storage- en Massive-credentials. Stel voor de earningslaag ook een SEC
User-Agent met contactadres en een FMP API-key in. Zie
`INSTALLATIE_EN_UPGRADE.md` voor de
veilige upgradeprocedure en `DATA_VEREISTEN.md` voor de volledige datasetlijst.
`PROJECTSTATUS.md` scheidt exact wat in de software gereed is van wat pas na een
echte opslagscan, GPU-run en training als voltooid mag worden beschouwd.

## Belangrijkste operationele grens

Zonder historische entry- én exit-bid/ask, diepte, spread, slippage, fees en
fillregels zijn kostengecorrigeerde uitkomsten expliciet **voorlopig**. Er mag
geen live-winstclaim worden gedaan. Daarna zijn nog overtuigende paper trading,
stabiele monitoring, brokerintegratie en alle promotiepoorten nodig.
