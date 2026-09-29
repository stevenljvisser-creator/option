from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date,datetime,timedelta,timezone
import re
import math

from core.config import (
    RAW_ROOT,DATA_V3_ROOT,TRADES_ROOT,TRADE_FEATURE_ROOT,OI_ROOT,EARNINGS_ROOT,
    SENTIMENT_ROOT,FEATURE_ROOT,QUOTE_ROOT,MODEL_ROOT,MACRO_ROOT,PAIR_ROOT,
    RELATIONSHIP_ROOT,PAIR_MODEL_ROOT,GF_NEWS_ROOT,GF_SNAPSHOT_ROOT,RICH_NEWS_ROOT,
    INVENTORY_ROOT,INVENTORY_RECEIPT_ROOT,INVENTORY_SNAPSHOT_ROOT,
    EARNINGS_SEC_RAW_ROOT,EARNINGS_FMP_RAW_ROOT,EARNINGS_PROCESSED_ROOT,EARNINGS_STATUS_ROOT,
    ALPHA_RESEARCH_ROOT,PAPER_TRADING_ROOT,TRADING_CONTROL_ROOT,MONITORING_ROOT,
)
from core.storage import list_objects_meta,get_json,put_json

DATASET_LABELS={
    "stocks":"Aandelen minute bars",
    "options":"Optie minute bars",
    "news":"Nieuws basis (Massive)",
    "option_trades":"Option trades",
    "option_trade_features":"Option trade minute-features",
    "open_interest":"Open interest / IV / Greeks",
    "earnings":"Legacy earnings (Massive/Benzinga)",
    "earnings_sec_raw":"SEC EDGAR Company Facts (append-only)",
    "earnings_fmp_raw":"FMP earningsverwachtingen (append-only)",
    "earnings_processed":"SEC + FMP gecombineerde kwartaalcijfers",
    "earnings_status":"SEC/FMP importstatus en fouten",
    "sentiment":"Sentiment + eventclassificatie",
    "gpu_sentiment":"GPU FinBERT sentiment",
    "features":"Technische + option features",
    "quotes":"Quotes",
    "google_finance_news":"Google Finance nieuws",
    "google_finance_snapshots":"Google Finance snapshots",
    "news_rich":"Verrijkt historisch nieuws",
    "macro":"Risicovrije rente",
    "pairs":"Call/Put Pair Feature Store",
    "relationships":"Relationship Discovery",
    "pair_models":"FINAL multi-horizon modellen",
    "legacy_models":"Legacy modellen",
    "professional_news":"OptionEdge professioneel nieuws",
    "clean_features":"OptionEdge clean Feature Store",
    "clean_pairs":"OptionEdge clean Call/Put Pair Store",
    "clean_relationships":"OptionEdge relationships",
    "clean_pair_models":"OptionEdge FINAL modellen",
    "events":"OptionEdge markt- en nieuwsevents",
    "option_reference":"Optiecontract-referentie",
    "corporate_actions":"Corporate actions",
    "dividends":"Dividendhistorie",
    "borrow":"Borrow / short-context",
    "news_articles":"Bronnieuws + revisies",
    "news_chunks":"Gededupeerde nieuwschunks",
    "news_embeddings":"1024D nieuwsembeddings",
    "news_events":"LLM eventextractie",
    "news_coverage":"Nieuwsbroncoverage en foutreceipts",
    "daily_vectors":"384D dagelijkse fusievectoren",
    "daily_labels":"Toekomstlabels per horizon",
    "deep_relationships":"Diepe verbandanalyse",
    "fusion_models":"Fusie- en dimensiemodellen",
    "alpha_research":"Alpha-experimenten, voorspellingen en scorecards",
    "paper_trading":"Append-only paper-signalen en afwikkelingen",
    "trading_control":"Risk-, promotie- en kill-switchstatus",
    "monitoring":"Model-, feature-, voorspelling- en execution-drift",
    "exchange_calendar":"Exchange calendar",
    "sector_factors":"Sector- en factorbenchmarks",
}

AGENT_DATASETS={
    "stocks":["stocks"],
    "options":["options"],
    "news":["news"],
    "option_trades":["option_trades","option_trade_features"],
    "open_interest":["open_interest"],
    "earnings":["earnings_processed","earnings_sec_raw","earnings_fmp_raw"],
    "google_finance":["google_finance_news","google_finance_snapshots"],
    "rich_news":["news_rich"],
    "sentiment":["sentiment"],
    "features":["clean_features"],
    "macro":["macro"],
    "pairs":["clean_pairs"],
    "events":["events"],
    "relationships":["relationships"],
    "pair_model":["pair_models"],
    "vectors":["daily_vectors"],
    "deep_relationships":["deep_relationships"],
    "alpha_research":["alpha_research"],
}

DATE_RE=re.compile(r"(\d{4}-\d{2}-\d{2})")
STAMP_RE=re.compile(r"(\d{8}T\d{6}Z)")

PATTERNS=[
    ("stocks",re.compile(r"^market-data/v2/stocks/minute/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2})\.csv\.gz$")),
    ("options",re.compile(r"^market-data/v2/options/minute/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2})/(?:part-\d+\.csv\.gz|by-ticker/[^/]+/part-\d+\.csv\.gz|manifest\.json)$")),
    ("news",re.compile(r"^market-data/v2/news/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2})\.csv\.gz$")),
    ("option_trades",re.compile(r"^market-data/v3/option-trades/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2})/(?:part-\d+\.csv\.gz|by-ticker/[^/]+/part-\d+\.csv\.gz|manifest\.json)$")),
    ("option_trade_features",re.compile(r"^market-data/v3/option-trade-minute/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2})/(?:part-\d+\.csv\.gz|by-ticker/[^/]+/part-\d+\.csv\.gz|manifest\.json)$")),
    ("open_interest",re.compile(r"^market-data/v3/open-interest/([^/]+)/(\d{4}-\d{2}-\d{2})\.csv\.gz$")),
    ("earnings",re.compile(r"^market-data/v3/earnings/([^/]+)/(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv\.gz$")),
    ("earnings_sec_raw",re.compile(r"^market-data/v5/earnings/raw/sec/([^/]+)/(\d{4})/(\d{2})/(\d{2})/.+\.json\.gz$")),
    ("earnings_fmp_raw",re.compile(r"^market-data/v5/earnings/raw/fmp/([^/]+)/(\d{4})/(\d{2})/(\d{2})/.+\.json\.gz$")),
    ("earnings_processed",re.compile(r"^market-data/v5/earnings/processed/([^/]+)/(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})/.+\.csv\.gz$")),
    ("earnings_status",re.compile(r"^market-data/v5/earnings/status/runs/\d{4}/\d{2}/(.+)\.json$")),
    ("sentiment",re.compile(r"^market-data/v3/sentiment/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2})\.csv\.gz$")),
    ("gpu_sentiment",re.compile(r"^market-data/v3/gpu-sentiment/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2})\.csv\.gz$")),
    ("features",re.compile(r"^market-data/v3/features/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2})\.csv\.gz$")),
    ("quotes",re.compile(r"^market-data/v3/quotes/(?:([^/]+)/)?(?:\d{4}/\d{2}/)?(\d{4}-\d{2}-\d{2})")),
    ("google_finance_news",re.compile(r"^market-data/v3/google-finance/news/([^/]+)/(\d{4}-\d{2}-\d{2})\.csv\.gz$")),
    ("google_finance_snapshots",re.compile(r"^market-data/v3/google-finance/snapshots/([^/]+)/(\d{4}-\d{2}-\d{2})\.json$")),
    ("news_rich",re.compile(r"^market-data/v3/news-rich/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2})\.csv\.gz$")),
    ("macro",re.compile(r"^market-data/v3/macro/(.+)$")),
    ("pairs",re.compile(r"^market-data/v3/option-pairs/h(\d+)/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2})\.csv\.gz$")),
    ("relationships",re.compile(r"^market-data/v3/relationships/(.+)$")),
    ("pair_models",re.compile(r"^market-data/v3/pair-models/(.+)$")),

    ("professional_news",re.compile(r"^market-data/v4/professional-news/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2})\.csv\.gz$")),
    ("events",re.compile(r"^market-data/v4/events/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2})\.csv\.gz$")),
    ("clean_features",re.compile(r"^market-data/v4/features/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2})\.csv\.gz$")),
    ("clean_pairs",re.compile(r"^market-data/v4/option-pairs/h(\d+)/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2})\.csv\.gz$")),
    ("clean_relationships",re.compile(r"^market-data/v4/relationships/(.+)$")),
    ("clean_pair_models",re.compile(r"^market-data/v4/pair-models/(.+)$")),
    ("legacy_models",re.compile(r"^market-data/v3/models/(.+)$")),

    ("option_reference",re.compile(r"^market-data/v5/option-reference/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2}).*$")),
    ("corporate_actions",re.compile(r"^market-data/v5/corporate-actions/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2}).*$")),
    ("dividends",re.compile(r"^market-data/v5/dividends/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2}).*$")),
    ("borrow",re.compile(r"^market-data/v5/borrow/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2}).*$")),
    ("news_articles",re.compile(r"^market-data/v5/news-articles/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2}).*$")),
    ("news_chunks",re.compile(r"^market-data/v5/news-chunks/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2}).*$")),
    ("news_embeddings",re.compile(r"^market-data/v5/news-embeddings/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2}).*$")),
    ("news_events",re.compile(r"^market-data/v5/news-events/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2}).*$")),
    ("news_coverage",re.compile(r"^market-data/v5/gpu-news-agent/receipts/([^/]+)/\d{4}/(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.json$")),
    ("daily_vectors",re.compile(r"^market-data/v5/daily-vectors/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2}).*$")),
    ("daily_labels",re.compile(r"^market-data/v5/daily-labels/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2}).*$")),
    ("exchange_calendar",re.compile(r"^market-data/v5/exchange-calendar/(.+)$")),
    ("sector_factors",re.compile(r"^market-data/v5/sector-factors/([^/]+)/\d{4}/\d{2}/(\d{4}-\d{2}-\d{2}).*$")),
    ("deep_relationships",re.compile(r"^market-data/v5/deep-relationships/(.+)$")),
    ("fusion_models",re.compile(r"^market-data/v5/fusion-models/(.+)$")),
    ("alpha_research",re.compile(r"^market-data/v5/alpha-research/(.+)$")),
    ("paper_trading",re.compile(r"^market-data/v5/paper-trading/(.+)$")),
    ("trading_control",re.compile(r"^market-data/v5/trading-control/(.+)$")),
    ("monitoring",re.compile(r"^market-data/v5/monitoring/(.+)$")),
]

def utcnow():
    return datetime.now(timezone.utc)

def _iso(x):
    if x is None:
        return None
    if hasattr(x,"isoformat"):
        return x.isoformat()
    return str(x)

def _parse_date(s):
    try:
        return datetime.strptime(str(s),"%Y-%m-%d").date()
    except Exception:
        return None

def _new_summary():
    return {
        "objects":0,
        "bytes":0,
        "dates":set(),
        "tickers":set(),
        "ticker_dates":defaultdict(set),
        "horizons":set(),
        "first_modified":None,
        "last_modified":None,
        "manifest_count":0,
        "rows_by_ticker":defaultdict(int),
        "data_dates":set(),
        "manifest_dates":set(),
        "legacy_ticker_detail_unknown":False,
    }

def _update_modified(ds,last_modified):
    if last_modified is None:
        return
    old_first=ds["first_modified"]
    old_last=ds["last_modified"]
    if old_first is None or last_modified<old_first:
        ds["first_modified"]=last_modified
    if old_last is None or last_modified>old_last:
        ds["last_modified"]=last_modified

def classify_object(key):
    if key.startswith(INVENTORY_ROOT+"/"):
        return None,None
    for dataset,rx in PATTERNS:
        m=rx.match(key)
        if m:
            return dataset,m
    return None,None

def _apply_basic(ds_name,m,key,meta,summaries,option_manifest_keys,trade_manifest_keys):
    ds=summaries[ds_name]
    ds["objects"]+=1
    ds["bytes"]+=int(meta.get("size",0) or 0)
    _update_modified(ds,meta.get("last_modified"))

    g=m.groups() if m is not None else ()

    if ds_name in ["stocks","news","sentiment","gpu_sentiment","features","open_interest",
                   "google_finance_news","google_finance_snapshots","news_rich",
                   "professional_news","clean_features","events",
                   "option_reference","corporate_actions","dividends","borrow",
                   "news_articles","news_chunks","news_embeddings","news_events",
                   "daily_vectors","daily_labels","sector_factors"]:
        ticker=g[0]
        d=_parse_date(g[1])
        ds["tickers"].add(ticker)
        if d:
            ds["dates"].add(d)
            ds["ticker_dates"][ticker].add(d)

    elif ds_name=="news_coverage":
        ticker,start_text,end_text=g
        ds["tickers"].add(ticker)
        for value in [start_text,end_text]:
            d=_parse_date(value)
            if d:
                ds["dates"].add(d)
                ds["ticker_dates"][ticker].add(d)

    elif ds_name=="earnings":
        ticker,start,end=g
        ds["tickers"].add(ticker)
        d1=_parse_date(start);d2=_parse_date(end)
        if d1:ds["dates"].add(d1)
        if d2:ds["dates"].add(d2)

    elif ds_name in ["earnings_sec_raw","earnings_fmp_raw"]:
        ticker,year,month,day=g
        ds["tickers"].add(ticker)
        d=_parse_date(f"{year}-{month}-{day}")
        if d:
            ds["dates"].add(d)
            ds["ticker_dates"][ticker].add(d)

    elif ds_name=="earnings_processed":
        ticker,start,end=g
        ds["tickers"].add(ticker)
        for value in [start,end]:
            d=_parse_date(value)
            if d:
                ds["dates"].add(d)
                ds["ticker_dates"][ticker].add(d)

    elif ds_name=="options":
        d=_parse_date(g[0])
        if d:ds["dates"].add(d)
        if key.endswith("/manifest.json"):
            ds["manifest_count"]+=1
            option_manifest_keys.append((d,key))

    elif ds_name in ["option_trades","option_trade_features"]:
        d=_parse_date(g[0])
        if d:ds["dates"].add(d)
        if key.endswith("/manifest.json"):
            ds["manifest_count"]+=1
            trade_manifest_keys.append((ds_name,d,key))
        elif ds_name=="option_trades":
            # Legacy completeness is determined after all dates/manifests are counted.
            pass

    elif ds_name in ["pairs","clean_pairs"]:
        old_h,ticker,ds_date=g
        d=_parse_date(ds_date)
        ds["horizons"].add(f"h{old_h}")
        ds["tickers"].add(ticker)
        if d:
            ds["dates"].add(d)
            ds["ticker_dates"][ticker].add(d)

    elif ds_name=="quotes":
        ticker=g[0]
        d=_parse_date(g[1])
        if ticker:
            ds["tickers"].add(ticker)
            if d:ds["ticker_dates"][ticker].add(d)
        if d:ds["dates"].add(d)

    else:
        # Generic timestamp/date extraction for macro/models/relationships.
        dm=DATE_RE.search(key)
        if dm:
            d=_parse_date(dm.group(1))
            if d:ds["dates"].add(d)
        sm=STAMP_RE.search(key)
        if sm:
            try:
                d=datetime.strptime(sm.group(1),"%Y%m%dT%H%M%SZ").date()
                ds["dates"].add(d)
            except Exception:
                pass

def _read_option_manifest(item):
    d,key=item
    try:
        x=get_json(key)
        return d,key,x,None
    except Exception as e:
        return d,key,None,f"{type(e).__name__}: {e}"

def _read_trade_manifest(item):
    dataset,d,key=item
    try:
        x=get_json(key)
        return dataset,d,key,x,None
    except Exception as e:
        return dataset,d,key,None,f"{type(e).__name__}: {e}"

def _receipt_info(key):
    # market-data/v3/inventory/receipts/{agent}/{YYYY}/{MM}/{date}/{ticker}.json
    prefix=INVENTORY_RECEIPT_ROOT+"/"
    if not key.startswith(prefix):
        return None
    rest=key[len(prefix):]
    parts=rest.split("/")
    if len(parts)<5:
        return None
    agent=parts[0]
    ds=parts[3]
    ticker=parts[4].rsplit(".",1)[0]
    d=_parse_date(ds)
    if not d:
        return None
    return agent,d,ticker

def scan_inventory(callback=None):
    """
    Exact physical inventory of objects currently present in Hetzner Object Storage.
    It does not decide whether an import should be skipped.
    """
    summaries=defaultdict(_new_summary)
    option_manifest_keys=[]
    trade_manifest_keys=[]
    receipt_dates=defaultdict(set)
    receipt_ticker_dates=defaultdict(lambda:defaultdict(set))
    unknown_object_count=0
    unknown_object_examples=[]
    total_objects=0
    total_bytes=0

    def list_cb(scanned,pages,truncated):
        if callback:
            # Object listing is the largest fixed stage. Progress is deliberately
            # capped because S3 does not expose total object count in advance.
            p=min(.58,.05+.04*math.log1p(max(1,pages)))
            callback(p,f"Hetzner scannen: {scanned:,} objecten gelezen ({pages} pagina's)…")

    for meta in list_objects_meta("market-data/",callback=list_cb):
        key=meta["key"]

        rec=_receipt_info(key)
        if rec:
            agent,d,ticker=rec
            receipt_dates[agent].add(d)
            receipt_ticker_dates[agent][ticker].add(d)
            continue

        # Inventory/GPU control metadata are not market observations.
        if (key.startswith(INVENTORY_ROOT+"/")
                or key.startswith("market-data/v4/gpu-agent/")
                or key.startswith("market-data/v5/gpu-news-agent/")):
            continue

        total_objects+=1
        total_bytes+=int(meta.get("size",0) or 0)

        ds_name,m=classify_object(key)
        if ds_name is None:
            if not key.startswith(INVENTORY_ROOT+"/"):
                unknown_object_count+=1
                if len(unknown_object_examples)<30:
                    unknown_object_examples.append(key)
            continue
        _apply_basic(
            ds_name,m,key,meta,summaries,
            option_manifest_keys,trade_manifest_keys
        )

    if callback:
        callback(.62,f"{total_objects:,} objecten gevonden. Optie-manifests verifiëren…")

    # Options already had manifests in earlier versions. Read these to recover exact
    # per-ticker coverage and row totals without opening the large option part files.
    if option_manifest_keys:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures=[ex.submit(_read_option_manifest,x) for x in option_manifest_keys]
            for i,f in enumerate(as_completed(futures),1):
                d,key,x,err=f.result()
                if x:
                    ds=summaries["options"]
                    requested=x.get("tickers") or []
                    rows=x.get("rows_by_ticker") or {}
                    for ticker in requested:
                        ds["tickers"].add(str(ticker))
                        if d:ds["ticker_dates"][str(ticker)].add(d)
                    for ticker,n in rows.items():
                        ds["rows_by_ticker"][str(ticker)]+=int(n or 0)
                if callback and (i%50==0 or i==len(futures)):
                    callback(
                        .62+.14*i/max(1,len(futures)),
                        f"Optie-manifests: {i:,}/{len(futures):,}"
                    )

    # Newer trade imports also write manifests. Legacy files remain physically
    # inventoried, but their old per-ticker composition cannot be proven from keys.
    if trade_manifest_keys:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures=[ex.submit(_read_trade_manifest,x) for x in trade_manifest_keys]
            for i,f in enumerate(as_completed(futures),1):
                dataset,d,key,x,err=f.result()
                if x:
                    ds=summaries[dataset]
                    requested=x.get("tickers") or []
                    rows=x.get("rows_by_ticker") or {}
                    for ticker in requested:
                        ds["tickers"].add(str(ticker))
                        if d:ds["ticker_dates"][str(ticker)].add(d)
                    for ticker,n in rows.items():
                        ds["rows_by_ticker"][str(ticker)]+=int(n or 0)

    if callback:
        callback(.84,"Inventaris samenvatten en dateranges berekenen…")

    final={}
    for name in DATASET_LABELS:
        ds=summaries.get(name,_new_summary())
        if name in ["option_trades","option_trade_features"]:
            ds["legacy_ticker_detail_unknown"]=bool(
                set(ds.get("data_dates",set()))-set(ds.get("manifest_dates",set()))
            )
        dates=sorted(ds["dates"])
        ticker_dates={
            t:[d.isoformat() for d in sorted(vals)]
            for t,vals in sorted(ds["ticker_dates"].items())
        }
        legacy_unknown=bool(ds["legacy_ticker_detail_unknown"])
        if name in ["option_trades","option_trade_features"]:
            legacy_unknown = int(ds["manifest_count"]) < len(dates)

        final[name]={
            "dataset":name,
            "label":DATASET_LABELS[name],
            "objects":int(ds["objects"]),
            "bytes":int(ds["bytes"]),
            "size_gb":round(int(ds["bytes"])/(1024**3),3),
            "date_count":len(dates),
            "dates":[d.isoformat() for d in dates],
            "first_date":dates[0].isoformat() if dates else None,
            "last_date":dates[-1].isoformat() if dates else None,
            "tickers":sorted(ds["tickers"]),
            "ticker_count":len(ds["tickers"]),
            "ticker_dates":ticker_dates,
            "horizons":sorted(ds["horizons"]),
            "manifest_count":int(ds["manifest_count"]),
            "rows_by_ticker":dict(sorted(ds["rows_by_ticker"].items())),
            "first_modified":_iso(ds["first_modified"]),
            "last_modified":_iso(ds["last_modified"]),
            "legacy_ticker_detail_unknown":legacy_unknown,
        }

    receipts={}
    for agent,dates in receipt_dates.items():
        receipts[agent]={
            "dates":[d.isoformat() for d in sorted(dates)],
            "ticker_dates":{
                t:[d.isoformat() for d in sorted(vals)]
                for t,vals in sorted(receipt_ticker_dates[agent].items())
            }
        }

    snapshot={
        "version":"20.0",
        "generated_at":utcnow().isoformat(),
        "source":"direct Hetzner S3 ListObjectsV2 + manifests",
        "total_market_data_objects":total_objects,
        "total_market_data_bytes":total_bytes,
        "total_market_data_gb":round(total_bytes/(1024**3),3),
        "datasets":final,
        "receipts":receipts,
        "unknown_object_count":unknown_object_count,
        "unknown_object_examples":unknown_object_examples,
        "notes":[
            "Object counts, bytes, keys, dates and modifications are read directly from Hetzner.",
            "Options use their historical manifests for exact ticker/date coverage.",
            "Legacy option-trade files before manifests have exact date/object/byte coverage but may lack provable per-ticker detail.",
            "A missing legacy news file can mean either zero articles or no stored object; new verification receipts remove this ambiguity going forward.",
        ],
    }

    stamp=utcnow().strftime("%Y%m%dT%H%M%SZ")
    key=f"{INVENTORY_SNAPSHOT_ROOT}/{stamp}.json"
    put_json(key,snapshot)
    put_json(f"{INVENTORY_ROOT}/latest.json",snapshot)

    if callback:
        callback(.99,"Hetzner-inventaris opgeslagen.")
    return snapshot,key

def weekday_dates(start,end):
    d=start
    out=[]
    while d<=end:
        if d.weekday()<5:
            out.append(d)
        d+=timedelta(days=1)
    return out

def coverage_for_agent(snapshot,agent,tickers,start,end):
    """
    Coverage is based on direct S3 presence and verification receipts.
    Denominator follows the application's weekday import grid and is therefore
    labelled 'requested weekday units', not claimed as an exchange-calendar proof.
    """
    datasets=AGENT_DATASETS.get(agent,[])
    if not datasets:
        return None

    start=_parse_date(start) if not isinstance(start,date) else start
    end=_parse_date(end) if not isinstance(end,date) else end
    if not start or not end:
        return None

    days=[end] if agent=="earnings" else weekday_dates(start,end)
    wanted=[str(t).upper() for t in tickers if str(t).strip()]
    receipt_agent="earnings_v5" if agent=="earnings" else agent
    receipts=(snapshot or {}).get("receipts",{}).get(receipt_agent,{})
    receipt_td={
        t:set(_parse_date(x) for x in vals if _parse_date(x))
        for t,vals in (receipts.get("ticker_dates") or {}).items()
    }
    receipt_days=set(_parse_date(x) for x in receipts.get("dates",[]) if _parse_date(x))

    # Pick the primary dataset for progress.
    primary=datasets[0]
    ds=((snapshot or {}).get("datasets") or {}).get(primary,{})
    ticker_dates={
        t:set(_parse_date(x) for x in vals if _parse_date(x))
        for t,vals in (ds.get("ticker_dates") or {}).items()
    }

    ticker_based=bool(ticker_dates) or agent in [
        "stocks","news","open_interest","earnings","google_finance",
        "rich_news","sentiment","features","pairs"
    ]

    if ticker_based and wanted:
        total=len(days)*len(wanted)
        physical=0
        verified=0
        details=[]
        for ticker in wanted:
            present={d for d in ticker_dates.get(ticker,set()) if start<=d<=end}
            receipted={d for d in receipt_td.get(ticker,set()) if start<=d<=end}
            combined=present|receipted
            physical+=len(present)
            verified+=len(combined)
            details.append({
                "ticker":ticker,
                "physical_days":len(present),
                "verified_days":len(combined),
                "requested_weekdays":len(days),
                "first_present":min(present).isoformat() if present else None,
                "last_present":max(present).isoformat() if present else None,
            })
    else:
        total=len(days)
        present={
            _parse_date(x) for x in (ds.get("dates") or [])
            if _parse_date(x) and start<=_parse_date(x)<=end
        }
        receipted={d for d in receipt_days if start<=d<=end}
        physical=len(present)
        verified=len(present|receipted)
        details=[]

    pct=(verified/total) if total else 0.0
    return {
        "agent":agent,
        "dataset":primary,
        "dataset_label":ds.get("label",primary),
        "snapshot_generated_at":(snapshot or {}).get("generated_at"),
        "objects":ds.get("objects",0),
        "bytes":ds.get("bytes",0),
        "size_gb":ds.get("size_gb",0),
        "first_date":ds.get("first_date"),
        "last_date":ds.get("last_date"),
        "date_count":ds.get("date_count",0),
        "ticker_count":ds.get("ticker_count",0),
        "physical_units":physical,
        "verified_units":verified,
        "requested_weekday_units":total,
        "verified_ratio":pct,
        "ticker_details":details,
        "legacy_ticker_detail_unknown":ds.get("legacy_ticker_detail_unknown",False),
        "basis":"direct S3 objects + verification receipts",
    }

def receipt_key(agent,day,ticker="__day__"):
    safe=str(ticker or "__day__").replace("/","_")
    return f"{INVENTORY_RECEIPT_ROOT}/{agent}/{day:%Y}/{day:%m}/{day}/{safe}.json"

def write_receipt(agent,day,ticker="__day__",status="verified",rows=None,keys=None,message=""):
    key=receipt_key(agent,day,ticker)
    put_json(key,{
        "agent":agent,
        "date":str(day),
        "ticker":ticker,
        "status":status,
        "rows":rows,
        "keys":keys or [],
        "message":message,
        "verified_at":utcnow().isoformat(),
    })
    return key
