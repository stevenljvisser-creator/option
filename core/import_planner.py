from __future__ import annotations
from datetime import date,timedelta,datetime
from core.storage import get_json
from core.storage_inventory import AGENT_DATASETS

DAILY_TICKER_AGENTS={
    "stocks","options","news","rich_news","sentiment","features","pairs","option_trades","events"
}
CURRENT_TICKER_AGENTS={"open_interest","google_finance"}
RANGE_TICKER_AGENTS={"earnings"}
SINGLE_AGENTS={"macro"}

def _parse_date(x):
    if isinstance(x,date):
        return x
    try:
        return datetime.strptime(str(x),"%Y-%m-%d").date()
    except Exception:
        return None

def _weekdays(start,end):
    out=[]
    d=start
    while d<=end:
        if d.weekday()<5:
            out.append(d)
        d+=timedelta(days=1)
    return out

def load_snapshot():
    try:
        return get_json("market-data/v3/inventory/latest.json")
    except Exception:
        return None

def _market_calendar(snapshot,start,end):
    """
    Prefer trading dates already proven by any market dataset in Hetzner.
    This prevents repeated 'missing' work on weekends/US holidays.
    If too little historical evidence is available, fall back to weekdays.
    """
    if not snapshot:
        return _weekdays(start,end)
    datasets=snapshot.get("datasets") or {}
    union=set()
    for name in ["stocks","options","option_trades"]:
        ds=datasets.get(name) or {}
        for x in ds.get("dates") or []:
            d=_parse_date(x)
            if d and start<=d<=end:
                union.add(d)
    weekdays=_weekdays(start,end)
    if len(union)>=max(3,int(len(weekdays)*.55)):
        return sorted(union)
    return weekdays

def _receipt_dates(snapshot,agent,ticker=None):
    if agent=="earnings":
        agent="earnings_v5"
    rec=((snapshot or {}).get("receipts") or {}).get(agent,{})
    if ticker is None:
        return {
            _parse_date(x) for x in rec.get("dates",[])
            if _parse_date(x)
        }
    td=(rec.get("ticker_dates") or {}).get(ticker,[])
    return {_parse_date(x) for x in td if _parse_date(x)}

def _dataset_ticker_dates(snapshot,dataset,ticker):
    ds=((snapshot or {}).get("datasets") or {}).get(dataset,{})
    vals=(ds.get("ticker_dates") or {}).get(ticker,[])
    return {_parse_date(x) for x in vals if _parse_date(x)}

def _dataset_dates(snapshot,dataset):
    ds=((snapshot or {}).get("datasets") or {}).get(dataset,{})
    return {_parse_date(x) for x in ds.get("dates",[]) if _parse_date(x)}

def _agent_primary_dataset(agent):
    vals=AGENT_DATASETS.get(agent) or []
    return vals[0] if vals else None

def build_plan(agent,tickers,start,end,force=False,snapshot=None):
    snapshot=snapshot if snapshot is not None else load_snapshot()
    start=_parse_date(start);end=_parse_date(end)
    tickers=[str(t).strip().upper() for t in tickers if str(t).strip()]
    if not start or not end or start>end:
        raise ValueError("Ongeldige start- of einddatum.")

    dataset=_agent_primary_dataset(agent)
    plan={
        "agent":agent,
        "mode":"force" if force else "missing_only",
        "snapshot_at":(snapshot or {}).get("generated_at"),
        "dataset":dataset,
        "start_date":str(start),
        "end_date":str(end),
        "tickers":tickers,
        "expected_dates":[],
        "missing_by_ticker":{},
        "already_by_ticker":{},
        "legacy_assumed_present_by_ticker":{},
        "expected_units":0,
        "already_present_units":0,
        "remaining_units":0,
        "notes":[],
    }

    if agent in DAILY_TICKER_AGENTS:
        expected=_market_calendar(snapshot,start,end)
        plan["expected_dates"]=[str(d) for d in expected]
        ds=((snapshot or {}).get("datasets") or {}).get(dataset,{})
        dataset_dates=_dataset_dates(snapshot,dataset)
        legacy_unknown=bool(ds.get("legacy_ticker_detail_unknown"))

        for ticker in tickers:
            physical=_dataset_ticker_dates(snapshot,dataset,ticker)
            receipts=_receipt_dates(snapshot,agent,ticker)
            already=[]
            legacy=[]
            missing=[]
            for d in expected:
                if force:
                    missing.append(d);continue
                if d in physical or d in receipts:
                    already.append(d);continue
                # Old trade part-files did not always record per-ticker composition.
                # Avoid repeatedly re-downloading those days; flag as legacy assumed
                # rather than pretending exact ticker coverage.
                if agent=="option_trades" and legacy_unknown and d in dataset_dates:
                    already.append(d);legacy.append(d);continue
                missing.append(d)

            plan["already_by_ticker"][ticker]=[str(d) for d in already]
            plan["missing_by_ticker"][ticker]=[str(d) for d in missing]
            plan["legacy_assumed_present_by_ticker"][ticker]=[str(d) for d in legacy]

        plan["expected_units"]=len(expected)*len(tickers)
        plan["already_present_units"]=sum(len(v) for v in plan["already_by_ticker"].values())
        plan["remaining_units"]=sum(len(v) for v in plan["missing_by_ticker"].values())

        if agent=="option_trades" and any(plan["legacy_assumed_present_by_ticker"].values()):
            plan["notes"].append(
                "Oude trade-dagen zonder per-ticker manifest worden als bestaand behandeld "
                "om eindeloos opnieuw downloaden te voorkomen; dit wordt in de UI als legacy gemarkeerd."
            )

    elif agent in CURRENT_TICKER_AGENTS:
        # Current/snapshot agents are by definition "today"; the UI date range
        # must not cause the same live snapshot to be stored under an old date.
        ds=((snapshot or {}).get("datasets") or {}).get(dataset,{})
        physical_td=ds.get("ticker_dates") or {}
        target=date.today()
        plan["expected_dates"]=[str(target)]
        for ticker in tickers:
            physical={_parse_date(x) for x in physical_td.get(ticker,[]) if _parse_date(x)}
            receipts=_receipt_dates(snapshot,agent,ticker)
            present=(target in physical or target in receipts)
            plan["already_by_ticker"][ticker]=[str(target)] if present and not force else []
            plan["missing_by_ticker"][ticker]=[] if present and not force else [str(target)]
            plan["legacy_assumed_present_by_ticker"][ticker]=[]
        plan["expected_units"]=len(tickers)
        plan["already_present_units"]=sum(len(v) for v in plan["already_by_ticker"].values())
        plan["remaining_units"]=sum(len(v) for v in plan["missing_by_ticker"].values())

    elif agent in RANGE_TICKER_AGENTS:
        # Exact old range files are difficult to prove from a compact inventory,
        # so receipts are authoritative for no-repeat behavior going forward.
        target=end
        plan["expected_dates"]=[str(target)]
        for ticker in tickers:
            receipts=_receipt_dates(snapshot,agent,ticker)
            physical=_dataset_ticker_dates(snapshot,dataset,ticker)
            present=(target in receipts or target in physical)
            plan["already_by_ticker"][ticker]=[str(target)] if present and not force else []
            plan["missing_by_ticker"][ticker]=[] if present and not force else [str(target)]
            plan["legacy_assumed_present_by_ticker"][ticker]=[]
        plan["expected_units"]=len(tickers)
        plan["already_present_units"]=sum(len(v) for v in plan["already_by_ticker"].values())
        plan["remaining_units"]=sum(len(v) for v in plan["missing_by_ticker"].values())

    elif agent in SINGLE_AGENTS:
        receipts=_receipt_dates(snapshot,agent,None)
        target=end
        present=target in receipts
        plan["expected_dates"]=[str(target)]
        plan["expected_units"]=1
        plan["already_present_units"]=1 if present and not force else 0
        plan["remaining_units"]=0 if present and not force else 1
        plan["missing_by_ticker"]={"__day__":[] if present and not force else [str(target)]}
        plan["already_by_ticker"]={"__day__":[str(target)] if present and not force else []}

    else:
        # Agents that are computations rather than imports run when explicitly asked.
        plan["expected_units"]=1
        plan["already_present_units"]=0
        plan["remaining_units"]=1
        plan["notes"].append("Deze agent is een berekening en wordt alleen op expliciet verzoek uitgevoerd.")

    ds=((snapshot or {}).get("datasets") or {}).get(dataset or "",{})
    stored_bytes=int(ds.get("bytes",0) or 0)
    ticker_dates=ds.get("ticker_dates") or {}
    proven_units=sum(len(v or []) for v in ticker_dates.values())
    if proven_units<=0:
        proven_units=int(ds.get("date_count",0) or 0)
    avg_bytes=(stored_bytes/proven_units) if proven_units>0 else None
    plan["stored_dataset_bytes"]=stored_bytes
    plan["average_bytes_per_proven_unit"]=avg_bytes
    plan["estimated_remaining_bytes"]=(
        int(avg_bytes*int(plan.get("remaining_units",0) or 0))
        if avg_bytes is not None else None
    )
    plan["estimated_total_selection_bytes"]=(
        int(avg_bytes*int(plan.get("expected_units",0) or 0))
        if avg_bytes is not None else None
    )

    if not snapshot:
        plan["notes"].append(
            "Er was nog geen Hetzner-inventarissnapshot; OptionEdge kan daarom niet bewijzen wat al aanwezig is."
        )
    elif plan.get("estimated_remaining_bytes") is not None:
        plan["notes"].append(
            "Resterende bytes zijn een schatting op basis van de gemiddelde grootte van reeds opgeslagen eenheden; "
            "de eenhedentelling zelf is leidend."
        )
    return plan

def planned_dates_for_ticker(job,ticker,default_days):
    plan=(job.payload or {}).get("import_plan") or {}
    vals=(plan.get("missing_by_ticker") or {}).get(ticker)
    if vals is None:
        return list(default_days)
    out=[]
    for x in vals:
        d=_parse_date(x)
        if d:out.append(d)
    return out

def planned_tickers_for_day(job,day):
    plan=(job.payload or {}).get("import_plan") or {}
    missing=plan.get("missing_by_ticker")
    if not isinstance(missing,dict):
        return list(job.tickers)
    ds=str(day)
    return [t for t in job.tickers if ds in set(missing.get(t,[]))]

def initial_counts(job,ticker,default_total):
    plan=(job.payload or {}).get("import_plan") or {}
    already=(plan.get("already_by_ticker") or {}).get(ticker)
    missing=(plan.get("missing_by_ticker") or {}).get(ticker)
    if already is None or missing is None:
        return 0,default_total
    return len(already),len(already)+len(missing)
