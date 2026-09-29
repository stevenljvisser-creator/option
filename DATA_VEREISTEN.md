# Data-audit vóór modeltraining

De API-route `/data/readiness` en de pagina **Data & gereedheid** voeren deze
controle rechtstreeks op de laatste Object Storage-inventaris uit. Een object
bestaat is niet genoeg: begin/einddatum en tickers moeten eveneens aantoonbaar
gedekt zijn.

## Verplichte brondata

| Data | Minimaal nodig | Waarom |
|---|---|---|
| Aandeel minute bars | alle tickers, vanaf 2022-09-16 | rendement, volume, volatiliteit, point-in-time context |
| SPY en QQQ minute bars | dezelfde periode | algemeen marktregime en abnormaal rendement |
| Option minute aggregates | calls en puts, alle perioden | prijs, volume, moneyness, expiratie en surface |
| Option trades | alle tickers en dagen | feitelijke activiteit en uitvoerbare tradeproxy |
| Contractreferentie | multiplier, strike, expiry, type | joins en aangepaste contracten correct behandelen |
| Corporate actions | splits, symbol changes, mergers | historische prijzen/contracten vergelijkbaar houden |
| Risicovrije curve | point-in-time en looptijdpassend | theoretische waardering |
| Earnings | publicatietijd plus surprise | sterke tijdsgebonden confounder |

## SEC/FMP earningscontract

| Bron | Rol | Point-in-time-eis |
|---|---|---|
| SEC EDGAR Company Facts/XBRL + Submissions | officiële omzet, net income, EPS, assets, liabilities, cashflow, filing en fiscal period | pas beschikbaar vanaf SEC-acceptatietijd; bij ontbrekende acceptance conservatief na filingdatum |
| FMP earnings | earningsdatum/-tijd, EPS- en omzetconsensus en eventuele FMP-actual | iedere waargenomen respons is een immutable snapshot; alleen vóór de eventcutoff waargenomen estimates zijn trainingsveilig |

Nieuwe objecten staan onder `market-data/v5/earnings/raw/sec/`,
`raw/fmp/`, `processed/` en `status/`. Een post-event historische backfill wordt
wel bewaard, maar krijgt `backfill_unverified` en mag geen pre-event feature of
surprise vormen. Bij bronuitval wordt niets verwijderd of overschreven.

## Verplichte nieuwslaag

Voor context begint nieuws bij voorkeur een jaar eerder, op 2021-09-16.

| Laag | Opgeslagen controle |
|---|---|
| Bronartikel/revisie | canonical URL, bron, published, first_seen, effective available-at, hash, revision, rights, fetchstatus |
| Duplicatie | exact content hash plus syndication/title-cluster |
| Chunk | article/revision-id, volgorde, hash, lengte en rechtenstatus |
| Embedding | BGE-M3 1024D, modelversie, chunk-id en attention-signalen |
| LLM-event | type, richting, impact, horizon, relevance, importance, uncertainty, novelty, facts, causal chain |
| Coverage | query/feed, ticker, periode, aantallen, duplicaten, robots/paywall/fetch/parse-fouten |

Een Google News/RSS-resultaat is een discoverybron, niet vanzelf een volledige
historische nieuwsset. Voor onderzoek met een sterke coverageclaim zijn
gelicentieerde historische feeds nodig, met heldere full-text- en
embeddingrechten. Een feed mag alleen `point_in_time_archive=true` krijgen als
ook historische versies en beschikbaarheidstijden contractueel/technisch
betrouwbaar zijn; anders wordt de lokale first-seen-tijd gebruikt en kan een
late backfill niet in een oudere vector terechtkomen.

## Sterk aanbevolen

- point-in-time open interest, IV en Greeks;
- dividendhistorie en ex-datums;
- borrow fee, availability en short-interestcontext;
- sector-/factorbenchmarks naast SPY/QQQ;
- exchange calendar met early closes en holidays;
- betrouwbare tijdzones voor earnings, filings en persberichten;
- bronlatency: `published_utc`, `first_seen_utc` en `processed_utc` apart.

## Nodig vóór een nettowinstclaim

- historische NBBO/bid/ask-quotes;
- bid/ask-size en relatieve spread;
- slippage- en fillregel die vooraf is vastgelegd;
- commissions/fees;
- liquiditeits- en minimumvolumecriteria;
- contractselectieregel die niet achteraf wordt geoptimaliseerd.

Zonder deze laag blijft `positive_option_outcome` een bruto/paper-target.

## Afgeleide lagen en go/no-go

1. Clean point-in-time Feature Store.
2. Call/put Pair Store voor dezelfde strike en expiratie.
3. Eén 384D vector per ticker-handelsdag met raw sidecar en schemahash.
4. Targets per horizon, fysiek gescheiden van inputs.
5. Discovery/confirmation-rapport met FDR en effectgrootte.
6. Dimensiebenchmark op identieke purged folds.
7. Calibratie plus finale hold-out.
8. Drift- en coverage-monitoring voor paper-live gebruik.

## Benutting voor Alpha Research

De primaire alpha-run inventariseert alle geschikte Pair Store ticker-dagen
binnen de gekozen periode. Er wordt niet stil gesampled of op een maximumaantal
rijen afgekapt. Alle aanwezige contracten worden bij de vooraf vastgelegde
selectieregel betrokken; vervolgens wordt precies één near-ATM contract per
ticker-dag als onafhankelijke trade-observatie gebruikt. Dit voorkomt dat
duizenden sterk afhankelijke contractminuten als duizenden onafhankelijke
bewijzen tellen.

Een dag is alleen geschikt wanneer de benodigde target, dagvector en provenance
aanwezig zijn. Uitgesloten dagen en aantallen vóór/na de vectorjoin worden in
`data_usage` gerapporteerd. “Alle data gebruiken” betekent dus alle geschikte
point-in-time observaties gebruiken, niet ontbrekende of leakage-onveilige data
met verzonnen waarden aanvullen.

Modelresultaten zijn onafhankelijk van de actuele importknoppen zichtbaar via
de permanente scorecard. Volledige experimenten en voorspellingen blijven
append-only in Object Storage; compacte hold-outmetrics staan tevens bij de
bestaande modelrun.

De website mag pas `training_ready=true` tonen als alle verplichte training- en
vectorpoorten aantoonbaar gereed zijn.
