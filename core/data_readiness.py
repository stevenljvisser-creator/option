"""Turn an Object Storage inventory into an explicit research readiness gate."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Iterable

from core.research_design import NEWS_CONTEXT_START, RESEARCH_START


DEFAULT_TICKERS = [
    "NVDA", "AMD", "AVGO", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
    "JPM", "BAC", "GS", "LLY", "UNH", "XOM", "CVX", "CAT", "BA", "WMT", "COST",
]
BENCHMARK_TICKERS = ["SPY", "QQQ"]


REQUIREMENTS = [
    {"id":"stock_minute","order":10,"group":"Marktbron","label":"Aandeel minute bars","datasets":["stocks"],"needed":"verplicht","gate":"training","start":RESEARCH_START,"tickers":"all","why":"Koers, volume, volatiliteit en tijdsynchrone marktcontext."},
    {"id":"benchmarks","order":20,"group":"Marktbron","label":"SPY en QQQ minute bars","datasets":["stocks"],"needed":"verplicht","gate":"training","start":RESEARCH_START,"tickers":"benchmarks","why":"Abnormaal rendement en marktregime; voorkomt dat algemeen beursnieuws als tickereffect telt."},
    {"id":"option_minute","order":30,"group":"Optiebron","label":"Optie minute aggregates","datasets":["options"],"needed":"verplicht","gate":"training","start":RESEARCH_START,"tickers":"companies","why":"Historische call/put-prijs, volume, moneyness- en expiratiesurface."},
    {"id":"option_trades","order":40,"group":"Optiebron","label":"Optietransacties","datasets":["option_trades"],"needed":"verplicht","gate":"training","start":RESEARCH_START,"tickers":"companies","why":"Werkelijke transacties, activiteit en uitvoerbare tradeproxy."},
    {"id":"option_quotes","order":50,"group":"Optiebron","label":"Historische bid/ask quotes","datasets":["quotes"],"needed":"later verplicht","gate":"netto-uitkomst","start":RESEARCH_START,"tickers":"companies","why":"Mid, spread, slippage en netto-winstkans. Zonder quotes blijft de uitkomst bruto/paper."},
    {"id":"contract_reference","order":60,"group":"Optiebron","label":"Contractreferentie en corporate actions","datasets":["option_reference","corporate_actions"],"needed":"verplicht","gate":"training","start":RESEARCH_START,"tickers":"companies","why":"Correcte multiplier, expiratie, splits, symbol changes en aangepaste contracten."},
    {"id":"open_interest","order":70,"group":"Optiebron","label":"Point-in-time open interest","datasets":["open_interest"],"needed":"sterk aanbevolen","gate":"volwaardig model","start":RESEARCH_START,"tickers":"companies","why":"Positionering en liquiditeitscontext; snapshots mogen niet achteraf worden teruggezet."},
    {"id":"risk_free","order":80,"group":"Referentie","label":"Risicovrije rentecurve","datasets":["macro"],"needed":"verplicht","gate":"training","start":RESEARCH_START,"tickers":"none","why":"Theoretische optieprijs; gebruik een looptijdpassende curve in plaats van één vaste rente."},
    {"id":"dividends_borrow","order":90,"group":"Referentie","label":"Dividend en borrow/short context","datasets":["dividends","borrow"],"needed":"sterk aanbevolen","gate":"volwaardig model","start":RESEARCH_START,"tickers":"companies","why":"Optietheorie, early exercise en put-call parity worden anders systematisch vertekend."},
    {"id":"earnings","order":100,"group":"Events","label":"SEC-kwartaalcijfers + point-in-time FMP-consensus","datasets":["earnings_processed"],"needed":"verplicht","gate":"training","start":RESEARCH_START,"tickers":"companies","why":"Officiële SEC-actuals, versievaste FMP-verwachtingen, earningsdatum en surprises zonder stille historische vervanging."},
    {"id":"news_raw","order":110,"group":"Nieuws","label":"Bronnieuws en revisies","datasets":["news_articles"],"needed":"verplicht","gate":"vectorbouw","start":NEWS_CONTEXT_START,"tickers":"companies","why":"Versievaste v5-artikelen uit officiële bronnen, professionele media en newswires met published/first-seen tijd. De v4-compatibiliteitslaag telt niet als bewijs."},
    {"id":"news_chunks","order":120,"group":"Nieuws","label":"Gededupeerde nieuwschunks","datasets":["news_chunks"],"needed":"verplicht","gate":"vectorbouw","start":NEWS_CONTEXT_START,"tickers":"companies","why":"Volledige tekst waar toegestaan; overlap, taal, entiteiten, rechten en syndicatiecluster vastgelegd."},
    {"id":"news_embeddings","order":130,"group":"Nieuws","label":"1024D chunk embeddings","datasets":["news_embeddings"],"needed":"verplicht","gate":"vectorbouw","start":NEWS_CONTEXT_START,"tickers":"companies","why":"Ruwe semantische informatie blijft beschikbaar vóór dagelijkse attention-pooling."},
    {"id":"llm_events","order":140,"group":"Nieuws","label":"LLM eventextractie","datasets":["news_events"],"needed":"verplicht","gate":"vectorbouw","start":NEWS_CONTEXT_START,"tickers":"companies","why":"Eventtype, richting, impactketen, horizon, onzekerheid, novelty en tickerrelevantie."},
    {"id":"news_coverage","order":145,"group":"Nieuws","label":"Meetbare broncoverage en foutreceipts","datasets":["news_coverage"],"needed":"verplicht","gate":"vectorbouw","start":NEWS_CONTEXT_START,"tickers":"companies","why":"Bewijst welke ticker-maanden en bronuniversums zijn onderzocht, inclusief duplicaten, robots/paywall- en fetchfouten."},
    {"id":"clean_features","order":150,"group":"Afgeleid","label":"Point-in-time Feature Store","datasets":["clean_features"],"needed":"verplicht","gate":"vectorbouw","start":RESEARCH_START,"tickers":"companies","why":"Gesynchroniseerde markt-, optie-, nieuws- en eventkenmerken met as-of audit trail."},
    {"id":"pairs","order":160,"group":"Afgeleid","label":"Call/put Pair Store","datasets":["clean_pairs"],"needed":"verplicht","gate":"vectorbouw","start":RESEARCH_START,"tickers":"companies","why":"Zelfde strike/expiratie, extrinsieke waarde, theoretische benchmark en parity residual."},
    {"id":"daily_vectors","order":170,"group":"Modelinput","label":"Dagelijkse fusievectoren","datasets":["daily_vectors"],"needed":"verplicht","gate":"training","start":RESEARCH_START,"tickers":"companies","why":"Eén versievaste 384D vector per ticker-handelsdag plus masks en provenance."},
    {"id":"labels","order":180,"group":"Modelinput","label":"Toekomstlabels per horizon","datasets":["daily_labels","clean_pairs"],"needed":"verplicht","gate":"training","start":RESEARCH_START,"tickers":"companies","why":"Uit de Pair Store afleidbaar en na bouw apart opgeslagen; nooit gemengd met de inputvector."},
    {"id":"quotes_layer","order":190,"group":"Gereserveerd","label":"Quotes C2-vergelijking","datasets":["quotes"],"needed":"gereserveerd","gate":"C2","start":RESEARCH_START,"tickers":"companies","why":"C1 zonder quotes versus C2 met quotes, zonder eerdere imports opnieuw te doen."},
    {"id":"exchange_calendar","order":200,"group":"Referentie","label":"Exchange calendar, holidays en early closes","datasets":["exchange_calendar"],"needed":"sterk aanbevolen","gate":"volwaardig model","start":RESEARCH_START,"tickers":"none","why":"Voorkomt foutieve dagelijkse toewijzing op feestdagen en verkorte handelsdagen."},
    {"id":"sector_factors","order":210,"group":"Referentie","label":"Sector- en factorbenchmarks","datasets":["sector_factors"],"needed":"sterk aanbevolen","gate":"volwaardig model","start":RESEARCH_START,"tickers":"companies","why":"Scheidt tickerspecifieke reactie van sector-, stijl- en marktfactoren."},
]


def _as_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def _wanted_tickers(mode: str, requested: Iterable[str]) -> list[str]:
    companies = [str(x).upper() for x in requested if str(x).strip() and str(x).upper() not in BENCHMARK_TICKERS]
    if mode == "benchmarks":
        return BENCHMARK_TICKERS.copy()
    if mode == "companies":
        return companies
    if mode == "all":
        return companies + [x for x in BENCHMARK_TICKERS if x not in companies]
    return []


def _dataset(snapshot: dict[str, Any], names: list[str]) -> tuple[str | None, dict[str, Any]]:
    datasets = (snapshot or {}).get("datasets") or {}
    for name in names:
        row = datasets.get(name) or {}
        if int(row.get("objects", 0) or 0) > 0:
            return name, row
    return None, {}


def readiness(snapshot: dict[str, Any] | None, tickers: Iterable[str] | None = None,
              end: date | None = None) -> dict[str, Any]:
    snapshot = snapshot or {}
    requested = [str(x).upper() for x in (tickers or DEFAULT_TICKERS)]
    end = end or (date.today() - timedelta(days=1))
    rows: list[dict[str, Any]] = []

    for req in REQUIREMENTS:
        code, ds = _dataset(snapshot, req["datasets"])
        objects = int(ds.get("objects", 0) or 0)
        first = _as_date(ds.get("first_date"))
        last = _as_date(ds.get("last_date"))
        wanted = _wanted_tickers(req["tickers"], requested)
        present_tickers = {str(x).upper() for x in (ds.get("tickers") or [])}
        missing_tickers = [x for x in wanted if x not in present_tickers]
        start = _as_date(req["start"])

        date_ok = bool(first and start and first <= start + timedelta(days=14) and last and last >= end - timedelta(days=10))
        ticker_ok = not wanted or not missing_tickers
        if objects <= 0:
            state = "uitgesteld" if req["needed"] in {"gereserveerd", "later verplicht"} else "ontbreekt"
        elif date_ok and ticker_ok:
            state = "gereed"
        else:
            state = "gedeeltelijk"

        rows.append({
            **req,
            "state": state,
            "dataset_found": code,
            "objects": objects,
            "bytes": int(ds.get("bytes", 0) or 0),
            "date_count": int(ds.get("date_count", 0) or 0),
            "ticker_count": int(ds.get("ticker_count", 0) or 0),
            "first_date": ds.get("first_date"),
            "last_date": ds.get("last_date"),
            "missing_tickers": missing_tickers,
            "coverage_note": (
                "Volledige periode en tickerlijst aantoonbaar aanwezig."
                if state == "gereed" else
                "Dataset aanwezig, maar periode of tickerdekking is nog niet compleet."
                if state == "gedeeltelijk" else
                "Nog niet aangetroffen in de laatste opslagscan."
            ),
        })

    training_gate = [x for x in rows if x["gate"] == "training" and x["needed"] == "verplicht"]
    vector_gate = [x for x in rows if x["gate"] == "vectorbouw" and x["needed"] == "verplicht"]
    training_blockers = [x for x in training_gate if x["state"] != "gereed"]
    vector_blockers = [x for x in vector_gate if x["state"] != "gereed"]
    blockers = training_blockers + vector_blockers
    required = [x for x in rows if x["needed"] == "verplicht"]
    score = sum(x["state"] == "gereed" for x in required) / max(1, len(required))

    return {
        "generated_from": snapshot.get("generated_at"),
        "research_start": RESEARCH_START,
        "news_context_start": NEWS_CONTEXT_START,
        "requested_tickers": requested,
        "readiness_score": score,
        "vector_ready": not vector_blockers,
        "training_ready": not blockers,
        "blocker_count": len(blockers),
        "blockers": [{"id":x["id"], "label":x["label"], "state":x["state"], "gate":x["gate"]} for x in blockers],
        "vector_blockers": [{"id":x["id"], "label":x["label"], "state":x["state"]} for x in vector_blockers],
        "training_blockers": [{"id":x["id"], "label":x["label"], "state":x["state"]} for x in training_blockers],
        "rows": sorted(rows, key=lambda x: (x["order"], x["label"])),
        "important_limit": (
            "Geen openbare nieuwsverzamelaar kan letterlijk alle internetberichten garanderen. "
            "OptionEdge rapporteert daarom per bron en ticker meetbare dekking, tijdigheid, duplicaten, "
            "licentiebeperkingen en mislukte fetches."
        ),
    }
