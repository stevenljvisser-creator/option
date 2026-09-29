from datetime import date,timedelta,datetime,timezone
from dataclasses import asdict
import gc,time,calendar,math
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd
import numpy as np
from sqlalchemy import select,desc

from core.db import SessionLocal,init_db
from core.models import AgentConfig,AgentCursor,AgentJob,AgentCompanyProgress,ModelRun,AppSetting
from core.massive import flat_chunks,option_underlying_series,MassiveREST
from core.storage import put_df,put_json,get_df,get_json,exists,list_keys,put_bytes
from core.config import (
    RAW_ROOT,TRADES_ROOT,TRADE_FEATURE_ROOT,OI_ROOT,EARNINGS_ROOT,
    SENTIMENT_ROOT,FEATURE_ROOT,MODEL_ROOT,GF_NEWS_ROOT,GF_SNAPSHOT_ROOT,RICH_NEWS_ROOT,MACRO_ROOT,
    PAIR_ROOT,RELATIONSHIP_ROOT,PAIR_MODEL_ROOT,SIGNAL_EVENT_ROOT,
    DAILY_VECTOR_ROOT,DEEP_RELATIONSHIP_ROOT,FUSION_MODEL_ROOT,
)
from core.runtime_settings import import_chunk_rows
from core.sentiment import enrich_news
from core.google_finance import GoogleFinanceClient,ArticleEnricher
from core.feature_engine import save_feature_day,clear_earnings_cache
from core.pair_features import save_pair_day,load_rate_series
from core.macro import fetch_risk_free
from core.relationships import discover_relationships
from core.pair_modeling import train_multi_horizon_final,serialize_bundle as serialize_pair_bundle
from core.multi_horizon import build_multi_horizon_rows
from core.horizons import DEFAULT_HORIZONS,horizon_label
from core.modeling import load_training_rows,train_all,serialize_bundle
from core.storage_inventory import scan_inventory,write_receipt
from core.import_planner import planned_dates_for_ticker,planned_tickers_for_day,initial_counts
from core.vector_fusion import save_daily_vector,vector_key
from core.deep_relationships import relationship_scan
from core.fusion_modeling import (
    train_fusion_model,daily_relationship_frame,attach_vectors,CONTRACT_FEATURES,
    serialize_bundle as serialize_fusion_bundle,
)
from core.earnings import (
    FmpEarningsClient,SecEdgarClient,import_earnings_ticker,
    latest_earnings_run,save_earnings_run,
)
from core.alpha_research import (
    CostModel,WalkForwardConfig,assign_point_in_time_regimes,
    run_research_suite,save_research_suite,
)

init_db()

def now(): return datetime.now(timezone.utc)

def weekdays(start,end):
    d=start
    while d<=end:
        if d.weekday()<5: yield d
        d+=timedelta(days=1)

def month_ranges(start,end):
    cur=date(start.year,start.month,1)
    while cur<=end:
        nxt=date(cur.year+1,1,1) if cur.month==12 else date(cur.year,cur.month+1,1)
        yield max(start,cur),min(end,nxt-timedelta(days=1))
        cur=nxt

def ensure_company_rows(db,job,units):
    existing={x.ticker:x for x in db.scalars(
        select(AgentCompanyProgress).where(AgentCompanyProgress.job_id==job.id)
    ).all()}
    for ticker in job.tickers:
        x=existing.get(ticker)
        if x is None:
            x=AgentCompanyProgress(job_id=job.id,ticker=ticker);db.add(x)
        initial,total=initial_counts(job,ticker,units)
        x.status="queued"
        x.total_units=max(1,total)
        x.completed_units=initial
        x.current_fraction=0.0
        x.current_item=""
        x.message=(
            f"{initial:,} al aanwezig in Hetzner • {max(0,total-initial):,} nog te verwerken"
            if initial else f"{max(0,total):,} te verwerken"
        )
    db.commit()

def company(db,job_id,ticker):
    """Return one ticker-progress row for a job, or None.

    This helper was referenced by the v14 worker but accidentally omitted,
    causing `NameError: name 'company' is not defined` during live agent updates.
    """
    return db.scalar(
        select(AgentCompanyProgress).where(
            AgentCompanyProgress.job_id==job_id,
            AgentCompanyProgress.ticker==ticker
        )
    )

def update_company(db,job_id,ticker,done,total,current,msg,status="running",fraction=0.0):
    x=company(db,job_id,ticker)
    if not x:return
    x.completed_units=int(done);x.total_units=int(total)
    x.current_fraction=float(max(0,min(1,fraction)))
    x.current_item=current;x.message=msg;x.status=status
    db.commit()

def finalize_company_rows(db,job,total,end_date):
    rows=db.scalars(select(AgentCompanyProgress).where(
        AgentCompanyProgress.job_id==job.id
    )).all()
    for x in rows:
        x.current_fraction=0.0
        remaining=max(0,int(x.total_units or total)-int(x.completed_units or 0))
        if remaining==0:
            x.status="done"
            x.message="Klaar • ✓ alle geplande eenheden geverifieerd in Hetzner"
        else:
            x.status="warning"
            x.message=(
                f"{int(x.completed_units or 0):,}/{int(x.total_units or total):,} gereed • "
                f"{remaining:,} nog niet voltooid"
            )
    db.commit()

def finish_by_progress(db,job,success_message,partial_message):
    if int(job.completed_units or 0)>=int(job.total_units or 0):
        finish(db,job,"done",success_message)
        return True
    finish(
        db,job,"warning",
        partial_message.format(
            done=int(job.completed_units or 0),
            total=int(job.total_units or 0),
            remaining=max(0,int(job.total_units or 0)-int(job.completed_units or 0))
        )
    )
    return False

def check_stop(db,job):
    db.refresh(job);return bool(job.stop_requested)

def finish(db,job,status,message):
    job.status=status;job.message=message;job.finished_at=now()
    job.current_fraction=0.0;db.commit()

    # After a successful writer job, queue one read-only Hetzner inventory scan.
    # This never changes import behavior and never skips Massive downloads.
    if job.agent!="inventory" and status in ["done","warning"]:
        active=db.scalar(select(AgentJob).where(
            AgentJob.agent=="inventory",
            AgentJob.status.in_(["queued","running"])
        ).order_by(desc(AgentJob.id)))
        if not active:
            inv=AgentJob(
                agent="inventory",mode="auto_after_write",status="queued",
                start_date=None,end_date=None,tickers=[],
                message=f"Automatische Hetzner-verificatie na {job.agent}.",
                payload={"source_job_id":job.id,"source_agent":job.agent}
            )
            db.add(inv);db.commit()

def mark_cursor(db,agent,end_date):
    c=db.get(AgentCursor,agent) or AgentCursor(agent=agent)
    c.last_successful_date=end_date;db.add(c);db.commit()

def speed_text(rows,started):
    rpm=rows/max(0.01,time.monotonic()-started)*60
    return f"{rpm/1_000_000:.1f} mln regels/min" if rpm>=1_000_000 else f"{rpm:,.0f} regels/min"

def run_agent_job(job_id:int):
    db=SessionLocal()
    try:
        job=db.get(AgentJob,job_id)
        if not job:return
        job.status="running";job.started_at=now();job.stop_requested=False;db.commit()
        days=list(weekdays(job.start_date,job.end_date)) if job.start_date and job.end_date else []
        ensure_company_rows(db,job,max(1,len(days)))
        plan=(job.payload or {}).get("import_plan") or {}
        if plan.get("expected_units") is not None:
            job.total_units=max(1,int(plan.get("expected_units") or 1))
            job.completed_units=int(plan.get("already_present_units") or 0)
        else:
            job.total_units=max(1,max(1,len(days))*max(1,len(job.tickers)))
            job.completed_units=0
        job.current_fraction=0.0
        job.message=(
            f"Start: {job.completed_units:,}/{job.total_units:,} eenheden al aanwezig; "
            f"{max(0,job.total_units-job.completed_units):,} resterend."
        )
        db.commit()

        dispatch={
            "stocks":lambda:run_stocks(db,job,days),
            "options":lambda:run_options(db,job,days),
            "news":lambda:run_news(db,job,days),
            "google_finance":lambda:run_google_finance(db,job),
            "rich_news":lambda:run_rich_news(db,job,days),
            "option_trades":lambda:run_option_trades(db,job,days),
            "open_interest":lambda:run_open_interest(db,job),
            "earnings":lambda:run_earnings(db,job),
            "sentiment":lambda:run_sentiment(db,job,days),
            "features":lambda:run_features(db,job,days),
            "macro":lambda:run_macro(db,job),
            "pairs":lambda:run_pairs(db,job,days),
            "vectors":lambda:run_vectors(db,job,days),
            "events":lambda:run_events(db,job,days),
            "relationships":lambda:run_relationships(db,job),
            "deep_relationships":lambda:run_deep_relationships(db,job),
            "pair_model":lambda:run_pair_model_job(db,job),
            "fusion_model":lambda:run_fusion_model_job(db,job),
            "alpha_research":lambda:run_alpha_research_job(db,job),
            "model":lambda:run_model(db,job),
            "inventory":lambda:run_inventory(db,job),
        }
        if job.agent not in dispatch:
            finish(db,job,"error",f"Onbekende agent: {job.agent}")
        else:
            dispatch[job.agent]()
    except Exception as e:
        try:
            job=db.get(AgentJob,job_id)
            if job:finish(db,job,"error",f"{type(e).__name__}: {e}")
        except Exception:pass
    finally:
        db.close();gc.collect()

# ---- existing stock minute importer ----
def run_stocks(db,job,days):
    completed={ticker:initial_counts(job,ticker,len(days))[0] for ticker in job.tickers}
    for day in days:
        if check_stop(db,job):finish(db,job,"stopped","Handmatig gestopt.");return
        active=planned_tickers_for_day(job,day)
        if not active:
            continue
        selected=set(active)
        buffers={t:[] for t in active};counts={t:0 for t in active}
        rows_seen=0;started=time.monotonic()
        try:
            for chunk,fraction,part in flat_chunks("us_stocks_sip/minute_aggs_v1",day):
                rows_seen+=len(chunk)
                sub=chunk[chunk["ticker"].isin(selected)] if "ticker" in chunk else chunk.iloc[0:0]
                for ticker,g in sub.groupby("ticker",sort=False):
                    buffers[ticker].append(g.copy());counts[ticker]+=len(g)
                job.current_fraction=fraction;job.current_item=str(day)
                job.message=f"{day} • {fraction:.1%} • {speed_text(rows_seen,started)}";db.commit()
                del chunk,sub
            for ticker in active:
                stored_keys=[]
                if buffers[ticker]:
                    data_key=f"{RAW_ROOT}/stocks/minute/{ticker}/{day:%Y}/{day:%m}/{day}.csv.gz"
                    put_df(data_key,pd.concat(buffers[ticker],ignore_index=True))
                    stored_keys=[data_key]
                write_receipt(
                    "stocks",day,ticker,
                    status="stored" if counts[ticker] else "verified_empty",
                    rows=counts[ticker],keys=stored_keys
                )
                completed[ticker]+=1;job.completed_units+=1
                update_company(
                    db,job.id,ticker,completed[ticker],initial_counts(job,ticker,len(days))[1],str(day),
                    f"{counts[ticker]:,} regels • ✓ Hetzner geverifieerd","running"
                )
        except Exception as e:
            for ticker in active:
                update_company(
                    db,job.id,ticker,completed[ticker],
                    initial_counts(job,ticker,len(days))[1],str(day),
                    f"NIET voltooid • {type(e).__name__}: {e}"[:220],"warning"
                )
            job.message=f"{day} niet voltooid: {type(e).__name__}: {e}"[:300]
            db.commit()
        gc.collect()
    finalize_company_rows(db,job,len(days),job.end_date)
    if finish_by_progress(
        db,job,
        "Aandelenimport compleet; alleen ontbrekende eenheden zijn verwerkt.",
        "Aandelenimport gedeeltelijk: {done:,}/{total:,} gereed • {remaining:,} resterend."
    ):
        mark_cursor(db,"stocks",job.end_date)

# ---- existing option minute aggregate importer ----
def _merge_day_manifest(key,new_tickers,new_rows,extra=None):
    try:
        old=get_json(key) if exists(key) else {}
    except Exception:
        old={}
    old_rows=dict(old.get("rows_by_ticker") or {})
    old_rows.update({str(k):int(v or 0) for k,v in (new_rows or {}).items()})
    tickers=sorted(set([str(x) for x in (old.get("tickers") or [])]+[str(x) for x in new_tickers]))
    merged=dict(old)
    merged.update({
        "tickers":tickers,
        "rows_by_ticker":old_rows,
        "updated_at":now().isoformat(),
        "storage_layout":"v14_per_ticker_incremental",
    })
    if extra:
        merged.update(extra)
    put_json(key,merged)
    return merged

def run_options(db,job,days):
    completed={ticker:initial_counts(job,ticker,len(days))[0] for ticker in job.tickers}
    for day in days:
        if check_stop(db,job):
            finish(db,job,"stopped","Handmatig gestopt.");return
        active=planned_tickers_for_day(job,day)
        if not active:
            continue

        selected=set(active)
        counts={ticker:0 for ticker in active}
        parts_by_ticker={ticker:0 for ticker in active}
        rows_seen=0
        started=time.monotonic()

        try:
            for chunk,fraction,part in flat_chunks("us_options_opra/minute_aggs_v1",day):
                rows_seen+=len(chunk)
                if "ticker" in chunk:
                    under=option_underlying_series(chunk["ticker"])
                    mask=under.isin(selected)
                    sub=chunk.loc[mask].copy()
                    if not sub.empty:
                        sub.insert(1,"underlying",under.loc[mask].astype(str).values)
                        for ticker,g in sub.groupby("underlying",sort=False):
                            if ticker not in counts:
                                continue
                            # v14: per-ticker keys prevent a partial re-import from
                            # overwriting another ticker's already stored day-parts.
                            key=(
                                f"{RAW_ROOT}/options/minute/{day:%Y}/{day:%m}/{day}/"
                                f"by-ticker/{ticker}/part-{part:05d}.csv.gz"
                            )
                            put_df(key,g.copy())
                            parts_by_ticker[ticker]+=1
                            counts[ticker]+=len(g)

                job.current_fraction=fraction
                job.current_item=str(day)
                job.message=(
                    f"{day} • bronbestand {fraction:.1%} • {speed_text(rows_seen,started)} • "
                    f"{job.completed_units:,}/{job.total_units:,} eenheden gereed"
                )
                for ticker in active:
                    x=company(db,job.id,ticker)
                    if x:
                        x.current_fraction=fraction
                        x.current_item=str(day)
                        x.message=f"{counts[ticker]:,} optie-minute-regels in huidige dag"
                db.commit()
                del chunk

            manifest_key=f"{RAW_ROOT}/options/minute/{day:%Y}/{day:%m}/{day}/manifest.json"
            _merge_day_manifest(
                manifest_key,active,counts,
                extra={"parts_by_ticker":parts_by_ticker,"date":str(day)}
            )

            for ticker in active:
                write_receipt(
                    "options",day,ticker,
                    status="stored" if counts[ticker] else "verified_empty",
                    rows=counts[ticker],keys=[manifest_key]
                )
                completed[ticker]+=1
                job.completed_units+=1
                update_company(
                    db,job.id,ticker,completed[ticker],
                    initial_counts(job,ticker,len(days))[1],str(day),
                    f"{counts[ticker]:,} regels • ✓ Hetzner • alleen ontbrekende ticker/dag","running"
                )
        except Exception as exc:
            for ticker in active:
                # Failed units are NOT counted as completed.
                update_company(
                    db,job.id,ticker,completed[ticker],
                    initial_counts(job,ticker,len(days))[1],str(day),
                    f"NIET voltooid • {type(exc).__name__}: {exc}"[:220],"warning"
                )
            job.message=f"{day} niet voltooid: {type(exc).__name__}: {exc}"[:300]
            db.commit()
        gc.collect()

    finalize_company_rows(db,job,len(days),job.end_date)
    if job.completed_units>=job.total_units:
        mark_cursor(db,"options",job.end_date)
        finish(db,job,"done","Optie-minute-import compleet; bestaande dagen/tickers zijn niet opnieuw geladen.")
    else:
        finish(
            db,job,"warning",
            f"Optie-import gedeeltelijk: {job.completed_units:,}/{job.total_units:,} eenheden geverifieerd."
        )

# ---- news importer ----
def run_news(db,job,days):
    rest=MassiveREST();done={ticker:initial_counts(job,ticker,len(days))[0] for ticker in job.tickers};days_set=set(days)
    for ticker in job.tickers:
        ticker_days=set(planned_dates_for_ticker(job,ticker,days))
        if not ticker_days:
            continue
        for a,b in month_ranges(job.start_date,job.end_date):
            month_days=[d for d in weekdays(a,b) if d in days_set and d in ticker_days]
            if not month_days:continue
            try:
                frame=rest.news(ticker,f"{a}T00:00:00Z",f"{b+timedelta(days=1)}T00:00:00Z")
                groups={}
                if not frame.empty:
                    dt=pd.to_datetime(frame["published_utc"],utc=True,errors="coerce")
                    frame=frame.assign(_date=dt.dt.date)
                    groups={d:g.drop(columns=["_date"]) for d,g in frame.groupby("_date")}
                for d in month_days:
                    daily=groups.get(d);n=0 if daily is None else len(daily)
                    stored_keys=[]
                    if daily is not None and not daily.empty:
                        data_key=f"{RAW_ROOT}/news/{ticker}/{d:%Y}/{d:%m}/{d}.csv.gz"
                        put_df(data_key,daily)
                        stored_keys=[data_key]
                    write_receipt(
                        "news",d,ticker,
                        status="stored" if n else "verified_empty",
                        rows=n,keys=stored_keys
                    )
                    done[ticker]+=1;job.completed_units+=1
                    update_company(
                        db,job.id,ticker,done[ticker],initial_counts(job,ticker,len(days))[1],str(d),
                        f"{n} nieuwsregels • ✓ Hetzner geverifieerd"
                    )
            except Exception as e:
                # The month stays missing; do not fake completion.
                current=month_days[0] if month_days else job.start_date
                update_company(
                    db,job.id,ticker,done[ticker],
                    initial_counts(job,ticker,len(days))[1],str(current),
                    f"NIET voltooid • {type(e).__name__}: {e}"[:220],"warning"
                )
                job.message=f"{ticker} maand niet voltooid: {type(e).__name__}: {e}"[:300]
                db.commit()
    finalize_company_rows(db,job,len(days),job.end_date)
    if finish_by_progress(
        db,job,
        "Nieuwsimport compleet; bestaande ticker/dagen zijn niet opnieuw geladen.",
        "Nieuwsimport gedeeltelijk: {done:,}/{total:,} gereed • {remaining:,} resterend."
    ):
        mark_cursor(db,"news",job.end_date)

# ---- NEW option tick-trades importer ----
def run_option_trades(db,job,days):
    done={ticker:initial_counts(job,ticker,len(days))[0] for ticker in job.tickers}

    for day in days:
        active=planned_tickers_for_day(job,day)
        if not active:
            continue
        selected=set(active)
        counts={ticker:0 for ticker in active}
        raw_parts={ticker:0 for ticker in active}
        feature_parts={ticker:0 for ticker in active}
        rows_seen=0
        started=time.monotonic()

        try:
            for chunk,fraction,part in flat_chunks("us_options_opra/trades_v1",day):
                if check_stop(db,job):
                    finish(db,job,"stopped","Handmatig gestopt.");return
                rows_seen+=len(chunk)
                if "ticker" not in chunk:
                    continue

                under=option_underlying_series(chunk["ticker"])
                mask=under.isin(selected)
                sub=chunk.loc[mask].copy()
                if not sub.empty:
                    sub.insert(1,"underlying",under.loc[mask].astype(str).values)

                    for ticker,g0 in sub.groupby("underlying",sort=False):
                        if ticker not in counts:
                            continue
                        g=g0.copy()
                        raw_key=(
                            f"{TRADES_ROOT}/{day:%Y}/{day:%m}/{day}/"
                            f"by-ticker/{ticker}/part-{part:05d}.csv.gz"
                        )
                        put_df(raw_key,g)
                        raw_parts[ticker]+=1
                        counts[ticker]+=len(g)

                        tscol=next(
                            (c for c in ["sip_timestamp","participant_timestamp","timestamp"] if c in g),
                            None
                        )
                        if tscol and "price" in g and "size" in g:
                            ts=pd.to_numeric(g[tscol],errors="coerce")
                            med=float(ts.dropna().median()) if not ts.dropna().empty else 0
                            unit="ns" if med>1e16 else "ms" if med>1e11 else "s"
                            g["minute"]=pd.to_datetime(ts,unit=unit,utc=True,errors="coerce").dt.floor("min")
                            g["price"]=pd.to_numeric(g["price"],errors="coerce")
                            g["size"]=pd.to_numeric(g["size"],errors="coerce")
                            g["trade_dollar_volume"]=g["price"]*g["size"]
                            g["trade_price_x_size"]=g["price"]*g["size"]
                            g["trade_price_sq_x_size"]=(g["price"]**2)*g["size"]
                            agg=g.groupby(["ticker","underlying","minute"],as_index=False).agg(
                                trade_count=("size","count"),
                                trade_volume=("size","sum"),
                                trade_dollar_volume=("trade_dollar_volume","sum"),
                                max_trade_size=("size","max"),
                                trade_price_x_size=("trade_price_x_size","sum"),
                                trade_price_sq_x_size=("trade_price_sq_x_size","sum")
                            )
                            feature_key=(
                                f"{TRADE_FEATURE_ROOT}/{day:%Y}/{day:%m}/{day}/"
                                f"by-ticker/{ticker}/part-{part:05d}.csv.gz"
                            )
                            put_df(feature_key,agg)
                            feature_parts[ticker]+=1

                job.current_fraction=fraction
                job.current_item=str(day)
                job.message=(
                    f"Option trades • {day} • bronbestand {fraction:.1%} • "
                    f"{speed_text(rows_seen,started)} • {job.completed_units:,}/{job.total_units:,} gereed"
                )
                for ticker in active:
                    x=company(db,job.id,ticker)
                    if x:
                        x.current_fraction=fraction
                        x.current_item=str(day)
                        x.message=f"{counts[ticker]:,} trades in huidige dag"
                db.commit()
                del chunk

            raw_manifest=f"{TRADES_ROOT}/{day:%Y}/{day:%m}/{day}/manifest.json"
            feature_manifest=f"{TRADE_FEATURE_ROOT}/{day:%Y}/{day:%m}/{day}/manifest.json"
            extra={
                "date":str(day),
                "raw_parts_by_ticker":raw_parts,
                "feature_parts_by_ticker":feature_parts,
            }
            _merge_day_manifest(raw_manifest,active,counts,extra=extra)
            _merge_day_manifest(feature_manifest,active,counts,extra=extra)

            for ticker in active:
                write_receipt(
                    "option_trades",day,ticker,
                    status="stored" if counts[ticker] else "verified_empty",
                    rows=counts[ticker],keys=[raw_manifest,feature_manifest]
                )
                done[ticker]+=1
                job.completed_units+=1
                update_company(
                    db,job.id,ticker,done[ticker],
                    initial_counts(job,ticker,len(days))[1],str(day),
                    f"{counts[ticker]:,} trades • ✓ Hetzner • alleen ontbrekende ticker/dag","running"
                )
        except Exception as exc:
            for ticker in active:
                update_company(
                    db,job.id,ticker,done[ticker],
                    initial_counts(job,ticker,len(days))[1],str(day),
                    f"NIET voltooid • {type(exc).__name__}: {exc}"[:220],"warning"
                )
            job.message=f"{day} niet voltooid: {type(exc).__name__}: {exc}"[:300]
            db.commit()
        gc.collect()

    finalize_company_rows(db,job,len(days),job.end_date)
    if job.completed_units>=job.total_units:
        mark_cursor(db,"option_trades",job.end_date)
        finish(db,job,"done","Option trades compleet; bestaande eenheden zijn overgeslagen.")
    else:
        finish(
            db,job,"warning",
            f"Option trades gedeeltelijk: {job.completed_units:,}/{job.total_units:,} eenheden geverifieerd."
        )

# ---- Daily OI/IV/Greeks snapshot. Historical backfill is not fabricated. ----
def run_open_interest(db,job):
    rest=MassiveREST()
    target=date.today()
    done=0
    for ticker in job.tickers:
        planned=planned_dates_for_ticker(job,ticker,[target])
        initial,total=initial_counts(job,ticker,1)
        if not planned:
            done+=initial
            continue
        try:
            d=rest.option_chain(ticker)
            stored_keys=[]
            if not d.empty:
                data_key=f"{OI_ROOT}/{ticker}/{target}.csv.gz"
                put_df(data_key,d);stored_keys=[data_key]
            write_receipt(
                "open_interest",target,ticker,
                status="stored" if len(d) else "verified_empty",
                rows=len(d),keys=stored_keys
            )
            done+=1;job.completed_units+=1
            update_company(db,job.id,ticker,total,total,str(target),
                           f"{len(d):,} contracten • ✓ Hetzner snapshot","done")
        except Exception as exc:
            update_company(db,job.id,ticker,initial,total,str(target),
                           f"NIET voltooid • {type(exc).__name__}: {exc}"[:220],"warning")
        job.current_item=ticker;db.commit()

    finalize_company_rows(db,job,1,target)
    if finish_by_progress(
        db,job,
        "Open-interest/IV/Greeks snapshots compleet.",
        "Open-interest snapshots gedeeltelijk: {done:,}/{total:,} gereed • {remaining:,} resterend."
    ):
        mark_cursor(db,"open_interest",target)

# ---- Official SEC actuals + versioned FMP expectations. ----
def run_earnings(db,job):
    started=now()
    # Reuse HTTP sessions and the SEC ticker/CIK map for the complete batch.
    # Source construction failures remain per-ticker warnings inside the
    # importer, so one missing credential never removes stored history.
    try:shared_sec_client=SecEdgarClient()
    except Exception:shared_sec_client=None
    try:shared_fmp_client=FmpEarningsClient()
    except Exception:shared_fmp_client=None
    previous=latest_earnings_run() or {}
    previous_rows={x.get("ticker"):x for x in (previous.get("ticker_status") or []) if x.get("ticker")}
    ticker_status=dict(previous_rows)
    last_sec=previous.get("last_successful_sec_import")
    last_fmp=previous.get("last_successful_fmp_import")
    run_errors=[]
    for ticker in job.tickers:
        planned=planned_dates_for_ticker(job,ticker,[job.end_date])
        initial,total=initial_counts(job,ticker,1)
        if not planned:
            continue
        if check_stop(db,job):
            finish(db,job,"stopped","Handmatig gestopt.");return
        try:
            result=import_earnings_ticker(
                ticker,job.start_date,job.end_date,
                sec_client=shared_sec_client,fmp_client=shared_fmp_client,
            )
            row=result.as_dict()
            row["updated_at"]=now().isoformat()
            previous_row=previous_rows.get(ticker) or {}
            if result.status!="done" and previous_row:
                row["previous_data_retained"]=True
                row["previous_processed"]=(previous_row.get("processed") or {})
                if not int((row.get("processed") or {}).get("rows",0) or 0):
                    row["processed"]={
                        **(previous_row.get("processed") or {}),
                        "success":False,"not_overwritten":True,
                    }
            ticker_status[ticker]=row
            if result.sec.get("success"):
                last_sec=result.sec.get("fetched_at") or now().isoformat()
            if result.fmp.get("success"):
                last_fmp=result.fmp.get("fetched_at") or now().isoformat()
            run_errors.extend(result.errors)
            events=int(result.processed.get("events",0) or 0)
            rows=int(result.processed.get("rows",0) or 0)
            if result.status=="done":
                write_receipt(
                    "earnings_v5",job.end_date,ticker,
                    status="stored",rows=rows,
                    keys=[x for x in [result.sec.get("key"),result.fmp.get("key"),
                                      result.processed.get("key")] if x],
                    message=(f"SEC + FMP point-in-time earnings {job.start_date} t/m {job.end_date}")
                )
                job.completed_units+=1
                update_company(
                    db,job.id,ticker,total,total,str(job.end_date),
                    f"{events} events / {rows} revisierecords • SEC + FMP ✓ Hetzner","done"
                )
            else:
                update_company(
                    db,job.id,ticker,initial,total,str(job.end_date),
                    (f"{events} events opgeslagen; bronwaarschuwing: "
                     + " | ".join(result.errors))[:300],"warning"
                )
        except Exception as exc:
            message=f"{ticker}: {type(exc).__name__}: {exc}"
            run_errors.append(message)
            retained=previous_rows.get(ticker) or {}
            ticker_status[ticker]={
                "ticker":ticker,"status":"error","updated_at":now().isoformat(),
                "sec":{"success":False},"fmp":{"success":False},
                "processed":{
                    **(retained.get("processed") or {"events":0,"rows":0}),
                    "success":False,"not_overwritten":True,
                },"errors":[message],"previous_data_retained":bool(retained),
            }
            update_company(
                db,job.id,ticker,initial,total,str(job.end_date),
                ("NIET voltooid • SEC/FMP: " f"{type(exc).__name__}: {exc}")[:240],"warning"
            )
        job.current_item=ticker;db.commit()

    all_status=sorted(ticker_status.values(),key=lambda x:str(x.get("ticker","")))
    manifest={
        "run_id":f"earnings-job-{job.id}",
        "job_id":job.id,"started_at":started.isoformat(),"finished_at":now().isoformat(),
        "start_date":str(job.start_date),"end_date":str(job.end_date),"tickers":job.tickers,
        "last_successful_sec_import":last_sec,
        "last_successful_fmp_import":last_fmp,
        "earnings_events":sum(int((x.get("processed") or {}).get("events",0) or 0) for x in all_status),
        "tickers_with_quarterly_facts":sum(
            bool((x.get("sec") or {}).get("rows",0)) for x in all_status
        ),
        "errors":run_errors,"ticker_status":all_status,
        "append_only":True,"legacy_v3_preserved":True,
    }
    try:
        manifest["status_key"]=save_earnings_run(manifest)
    except Exception as exc:
        run_errors.append(f"Statusmanifest: {type(exc).__name__}: {exc}")
    # A long-lived Celery worker may have cached an older processed earnings
    # snapshot.  New feature jobs must see the just-imported immutable version.
    clear_earnings_cache()
    finalize_company_rows(db,job,1,job.end_date)
    complete=finish_by_progress(
        db,job,
        "SEC/FMP earningslaag compleet.",
        "SEC/FMP earnings gedeeltelijk: {done:,}/{total:,} gereed • {remaining:,} resterend."
    )
    if complete:mark_cursor(db,"earnings",job.end_date)

# ---- Google Finance current company/news snapshot ----
def run_google_finance(db,job):
    gf=GoogleFinanceClient();enricher=ArticleEnricher()
    target=date.today()
    for ticker in job.tickers:
        planned=planned_dates_for_ticker(job,ticker,[target])
        initial,total=initial_counts(job,ticker,1)
        if not planned:
            continue
        if check_stop(db,job):
            finish(db,job,"stopped","Handmatig gestopt.");return
        try:
            snapshot,stories=gf.fetch(ticker)
            snapshot_key=f"{GF_SNAPSHOT_ROOT}/{ticker}/{target}.json"
            put_json(snapshot_key,snapshot)
            rich=enricher.enrich_frame(stories,max_articles=40)
            stored_keys=[snapshot_key]
            if not rich.empty:
                rich=enrich_news(rich)
                news_key=f"{GF_NEWS_ROOT}/{ticker}/{target}.csv.gz"
                put_df(news_key,rich);stored_keys.append(news_key)
            write_receipt("google_finance",target,ticker,status="stored",
                          rows=len(rich),keys=stored_keys)
            job.completed_units+=1
            update_company(
                db,job.id,ticker,total,total,str(target),
                f"{len(stories)} headlines • {len(rich)} verrijkt • ✓ Hetzner","done"
            )
        except Exception as exc:
            update_company(
                db,job.id,ticker,initial,total,str(target),
                f"NIET voltooid • {type(exc).__name__}: {exc}"[:240],"warning"
            )
        job.current_item=ticker
        job.message=f"{job.completed_units:,}/{job.total_units:,} bedrijven gereed"
        db.commit()
    finalize_company_rows(db,job,1,target)
    finish_by_progress(
        db,job,
        "Google Finance snapshots en rijk nieuws compleet.",
        "Google Finance gedeeltelijk: {done:,}/{total:,} gereed • {remaining:,} resterend."
    )

# ---- Enrich already imported historical Massive news ----
def run_rich_news(db,job,days):
    enricher=ArticleEnricher()
    done={ticker:initial_counts(job,ticker,len(days))[0] for ticker in job.tickers}
    for ticker in job.tickers:
        ticker_days=planned_dates_for_ticker(job,ticker,days)
        for d in ticker_days:
            if check_stop(db,job):
                finish(db,job,"stopped","Handmatig gestopt.");return
            try:
                raw_key=f"{RAW_ROOT}/news/{ticker}/{d:%Y}/{d:%m}/{d}.csv.gz"
                n=0;enriched=0;stored_keys=[]
                if exists(raw_key):
                    raw=get_df(raw_key)
                    n=len(raw)
                    if "article_url" in raw.columns and not raw.empty:
                        rich=enricher.enrich_frame(raw,max_articles=60)
                        enriched=int((rich.get("enrichment_status","")=="enriched").sum()) if not rich.empty else 0
                        if not rich.empty:
                            rich=enrich_news(rich)
                            rich_key=f"{RICH_NEWS_ROOT}/{ticker}/{d:%Y}/{d:%m}/{d}.csv.gz"
                            put_df(rich_key,rich)
                            stored_keys=[rich_key]
                write_receipt(
                    "rich_news",d,ticker,
                    status="stored" if stored_keys else "verified_empty",
                    rows=enriched,keys=stored_keys
                )
                done[ticker]+=1;job.completed_units+=1
                update_company(
                    db,job.id,ticker,done[ticker],initial_counts(job,ticker,len(days))[1],str(d),
                    f"{n} nieuwsitems • {enriched} verrijkt • ✓ Hetzner"
                )
            except Exception as exc:
                update_company(
                    db,job.id,ticker,done[ticker],initial_counts(job,ticker,len(days))[1],str(d),
                    f"NIET voltooid • {type(exc).__name__}: {exc}"[:220],"warning"
                )
            job.current_item=f"{ticker} {d}"
            job.message=f"{job.completed_units:,}/{job.total_units:,} eenheden gereed"
            db.commit()
    finalize_company_rows(db,job,len(days),job.end_date)
    finish_by_progress(
        db,job,
        "Historische nieuwsverrijking compleet.",
        "Nieuwsverrijking gedeeltelijk: {done:,}/{total:,} gereed • {remaining:,} resterend."
    )

# ---- Derived sentiment agent ----
def run_sentiment(db,job,days):
    done={ticker:initial_counts(job,ticker,len(days))[0] for ticker in job.tickers}
    for ticker in job.tickers:
        ticker_days=planned_dates_for_ticker(job,ticker,days)
        for d in ticker_days:
            try:
                candidates=[
                    f"{RICH_NEWS_ROOT}/{ticker}/{d:%Y}/{d:%m}/{d}.csv.gz",
                    f"{GF_NEWS_ROOT}/{ticker}/{d}.csv.gz",
                    f"{RAW_ROOT}/news/{ticker}/{d:%Y}/{d:%m}/{d}.csv.gz",
                ]
                x=pd.DataFrame()
                for key in candidates:
                    if exists(key):
                        x=enrich_news(get_df(key));break
                n=len(x);stored_keys=[]
                if n:
                    sentiment_key=f"{SENTIMENT_ROOT}/{ticker}/{d:%Y}/{d:%m}/{d}.csv.gz"
                    put_df(sentiment_key,x);stored_keys=[sentiment_key]
                write_receipt(
                    "sentiment",d,ticker,
                    status="stored" if n else "verified_empty",
                    rows=n,keys=stored_keys
                )
                done[ticker]+=1;job.completed_units+=1
                update_company(
                    db,job.id,ticker,done[ticker],initial_counts(job,ticker,len(days))[1],str(d),
                    f"{n} sentimentregels • ✓ Hetzner"
                )
            except Exception as exc:
                update_company(
                    db,job.id,ticker,done[ticker],initial_counts(job,ticker,len(days))[1],str(d),
                    f"NIET voltooid • {type(exc).__name__}: {exc}"[:220],"warning"
                )
            job.current_item=f"{ticker} {d}"
            job.message=f"{job.completed_units:,}/{job.total_units:,} eenheden gereed"
            db.commit()
    finalize_company_rows(db,job,len(days),job.end_date)
    finish_by_progress(
        db,job,
        "Sentimentfeatures compleet.",
        "Sentiment gedeeltelijk: {done:,}/{total:,} gereed • {remaining:,} resterend."
    )

# ---- Derived option-level training features ----
def run_features(db,job,days):
    payload=job.payload or {}
    h=int(payload.get("horizon_minutes",30))
    workers=max(1,min(8,int(payload.get("parallel_workers",4) or 4)))
    counts={ticker:initial_counts(job,ticker,len(days)) for ticker in job.tickers}
    done={ticker:counts[ticker][0] for ticker in job.tickers}
    work=[]
    for ticker in job.tickers:
        for d in planned_dates_for_ticker(job,ticker,days):
            work.append((ticker,d))

    def build_one(ticker,d):
        started=time.monotonic()
        try:
            n=save_feature_day(ticker,d,h)
            elapsed=max(.001,time.monotonic()-started)
            return {
                "ticker":ticker,"day":d,"rows":n,"ok":True,
                "key":f"{FEATURE_ROOT}/{ticker}/{d:%Y}/{d:%m}/{d}.csv.gz",
                "message":f"{n:,} modelregels • {n/elapsed:,.0f} regels/s • ✓ Hetzner",
            }
        except Exception as exc:
            return {
                "ticker":ticker,"day":d,"rows":0,"ok":False,"key":None,
                "message":f"NIET voltooid • {type(exc).__name__}: {exc}"[:220],
            }

    # Process in bounded batches.  This gives the S3/pandas workload useful
    # parallelism without allowing an unbounded queue to consume RAM.
    for offset in range(0,len(work),workers):
        if check_stop(db,job):finish(db,job,"stopped","Handmatig gestopt.");return
        batch=work[offset:offset+workers]
        with ThreadPoolExecutor(max_workers=min(workers,len(batch)),thread_name_prefix="features") as pool:
            futures=[pool.submit(build_one,ticker,d) for ticker,d in batch]
            for future in as_completed(futures):
                result=future.result();ticker=result["ticker"];d=result["day"]
                if result["ok"]:
                    write_receipt(
                        "features",d,ticker,
                        status="stored" if result["rows"] else "verified_empty",
                        rows=result["rows"],keys=[result["key"]] if result["rows"] else []
                    )
                    done[ticker]+=1;job.completed_units+=1
                    status="running"
                else:
                    status="warning"
                msg=result["message"]
                update_company(db,job.id,ticker,done[ticker],counts[ticker][1],str(d),msg,status)
                job.current_item=f"{ticker} {d}"
                job.message=f"{msg} • parallel={workers}"
                db.commit()
        gc.collect()
    finalize_company_rows(db,job,len(days),job.end_date)
    finish_by_progress(
        db,job,
        "Feature Store compleet. Model kan nu handmatig worden getraind.",
        "Feature Store gedeeltelijk: {done:,}/{total:,} gereed • {remaining:,} resterend."
    )

# ---- Dedicated manual model training job ----
def run_model(db,job):
    p=job.payload or {}
    max_rows=int(p.get("max_rows",100000))
    use_all_available=bool(p.get("use_all_available",True))
    target=p.get("target","target_extrinsic_return_30m")
    run_id=int(p.get("model_run_id",0) or 0)
    mr=db.get(ModelRun,run_id) if run_id else None

    def progress(value,message):
        value=float(max(0.0,min(0.98,value)))
        job.current_fraction=value
        job.current_item="Model Lab"
        job.message=message
        if mr:
            mr.status="running"
            mr.progress=value
            mr.message=message
        db.commit()

    try:
        if mr:
            mr.status="running"
            mr.started_at=now()
            mr.progress=0.01
            mr.message="Training starten…"
            db.commit()

        progress(0.01,"Featurecatalogus openen…")

        df=load_training_rows(
            job.tickers,
            job.start_date,
            job.end_date,
            max_rows=max_rows,
            callback=progress,
            use_all_available=use_all_available,
            target=target,
        )
        data_usage=dict(df.attrs.get("data_usage") or {})

        metrics,artifacts=train_all(df,target,callback=progress)
        metrics["data_usage"]=data_usage
        metrics["data_policy"]=(
            "all_available_eligible_rows_no_hidden_cap" if use_all_available
            else "legacy_representative_sampling"
        )

        progress(0.93,"Modelbundle serialiseren…")
        stamp=now().strftime("%Y%m%dT%H%M%SZ")
        key=f"{MODEL_ROOT}/{stamp}/bundle.pkl"

        bundle={
            "created_at":stamp,
            "target":target,
            "tickers":job.tickers,
            "start_date":str(job.start_date),
            "end_date":str(job.end_date),
            "data_usage":data_usage,
            "metrics":metrics,
            "models":artifacts
        }

        put_bytes(key,serialize_bundle(bundle),"application/octet-stream")
        put_json(f"{MODEL_ROOT}/{stamp}/metrics.json",metrics)

        if mr:
            mr.status="done"
            mr.finished_at=now()
            mr.progress=1.0
            mr.message="Training klaar"
            mr.metrics=metrics
            mr.feature_sets={
                k:(v.get("features",[]) if isinstance(v,dict) else [])
                for k,v in metrics.items()
            }
            mr.artifact_key=key
            db.commit()

        job.current_fraction=1.0
        finish(db,job,"done","Handmatige modeltraining klaar.")

    except Exception as exc:
        message=f"{type(exc).__name__}: {exc}"
        if mr:
            mr.status="error"
            mr.finished_at=now()
            mr.message=message
            db.commit()
        job.message=message
        job.current_fraction=0.0
        db.commit()
        raise



def run_inventory(db,job):
    def cb(value,message):
        job.current_fraction=float(max(0,min(.99,value)))
        job.current_item="Hetzner Object Storage"
        job.message=message
        db.commit()

    try:
        job.total_units=1;job.completed_units=0;db.commit()
        cb(.01,"Rechtstreeks verbinden met Hetzner Object Storage…")
        snapshot,key=scan_inventory(callback=cb)

        setting=db.get(AppSetting,"LATEST_INVENTORY_KEY") or AppSetting(key="LATEST_INVENTORY_KEY")
        setting.value=key
        setting.encrypted=False
        db.add(setting)

        generated=db.get(AppSetting,"LATEST_INVENTORY_AT") or AppSetting(key="LATEST_INVENTORY_AT")
        generated.value=snapshot.get("generated_at","")
        generated.encrypted=False
        db.add(generated)

        db.commit()
        job.completed_units=1
        job.current_fraction=1.0
        finish(
            db,job,"done",
            f"Hetzner-inventaris klaar: {snapshot.get('total_market_data_objects',0):,} objecten, "
            f"{snapshot.get('total_market_data_gb',0):.3f} GB."
        )
    except Exception as e:
        finish(db,job,"error",f"{type(e).__name__}: {e}")

def run_macro(db,job):
    try:
        job.message="Historische 3-maands Treasury-rate ophalen…";db.commit()
        d=fetch_risk_free(job.start_date,job.end_date)
        macro_key=f"{MACRO_ROOT}/risk_free_3m.csv.gz"
        put_df(macro_key,d)
        write_receipt(
            "macro",job.end_date,"__day__",
            status="stored",rows=len(d),keys=[macro_key],
            message=f"range {job.start_date} t/m {job.end_date}"
        )
        job.total_units=1;job.completed_units=1;db.commit()
        finish(db,job,"done",f"{len(d)} rentewaarnemingen • ✓ Hetzner.")
    except Exception as e:
        finish(db,job,"error",f"{type(e).__name__}: {e}")


def run_events(db,job,days):
    done={ticker:initial_counts(job,ticker,len(days))[0] for ticker in job.tickers}
    event_cols=[
        "event_stock_breakout_up_30m","event_stock_breakdown_30m",
        "event_high_realized_vol","event_stock_volume_spike",
        "event_option_jump_up","event_option_jump_down","event_earnings_within_5d"
    ]
    for ticker in job.tickers:
        ticker_days=planned_dates_for_ticker(job,ticker,days)
        for d in ticker_days:
            try:
                feature_key=f"{FEATURE_ROOT}/{ticker}/{d:%Y}/{d:%m}/{d}.csv.gz"
                if not exists(feature_key):
                    raise RuntimeError("Clean Feature Store ontbreekt voor deze ticker/dag.")
                x=get_df(feature_key)
                if x.empty:
                    out=pd.DataFrame()
                else:
                    x["ts"]=pd.to_datetime(x["ts"],utc=True,errors="coerce")
                    for c in event_cols+[
                        "news_count_60m","sentiment_60m","event_importance_60m",
                        "bullish_signal_60m","bearish_signal_60m","stock_return_30m",
                        "realized_vol_30m","stock_volume_z30","days_to_earnings"
                    ]:
                        if c in x:x[c]=pd.to_numeric(x[c],errors="coerce").fillna(0)
                    x["event_flag"]=0.0
                    for c in event_cols:
                        if c in x:x["event_flag"]=np.maximum(x["event_flag"],x[c])
                    if "news_count_60m" in x:
                        x["event_flag"]=np.maximum(x["event_flag"],(x["news_count_60m"]>0).astype(float))
                    e=x[x["event_flag"]>0].copy()
                    if e.empty:
                        out=e
                    else:
                        e["minute"]=e["ts"].dt.floor("min")
                        agg={}
                        for c in event_cols:
                            if c in e:agg[c]="max"
                        for c in ["news_count_60m","event_importance_60m","bullish_signal_60m","bearish_signal_60m","realized_vol_30m","stock_volume_z30"]:
                            if c in e:agg[c]="max"
                        for c in ["sentiment_60m","stock_return_30m","days_to_earnings","close_stock"]:
                            if c in e:agg[c]="mean"
                        out=e.groupby("minute",as_index=False).agg(agg)
                        out["ticker"]=ticker
                        labels=[]
                        for _,row in out.iterrows():
                            hits=[]
                            for c in event_cols:
                                if c in out and float(row.get(c,0) or 0)>0:
                                    hits.append(c.replace("event_",""))
                            if float(row.get("news_count_60m",0) or 0)>0:
                                hits.append("professional_news")
                            labels.append("|".join(hits))
                        out["event_labels"]=labels

                key=f"{SIGNAL_EVENT_ROOT}/{ticker}/{d:%Y}/{d:%m}/{d}.csv.gz"
                put_df(key,out)
                write_receipt("events",d,ticker,status="stored",rows=len(out),keys=[key])
                done[ticker]+=1;job.completed_units+=1
                update_company(
                    db,job.id,ticker,done[ticker],initial_counts(job,ticker,len(days))[1],
                    str(d),f"{len(out):,} events • ✓ Hetzner","running"
                )
            except Exception as exc:
                update_company(
                    db,job.id,ticker,done[ticker],initial_counts(job,ticker,len(days))[1],
                    str(d),f"NIET voltooid • {type(exc).__name__}: {exc}"[:220],"warning"
                )
            job.current_item=f"{ticker} {d}"
            job.message=f"Event Engine • {job.completed_units:,}/{job.total_units:,} gereed"
            db.commit()

    finalize_company_rows(db,job,len(days),job.end_date)
    finish_by_progress(
        db,job,
        "Event Engine compleet.",
        "Event Engine gedeeltelijk: {done:,}/{total:,} gereed • {remaining:,} resterend."
    )

def run_pairs(db,job,days):
    horizon=int((job.payload or {}).get("horizon_minutes",30))
    rates=load_rate_series();done={ticker:initial_counts(job,ticker,len(days))[0] for ticker in job.tickers}
    for ticker in job.tickers:
        ticker_days=planned_dates_for_ticker(job,ticker,days)
        for day in ticker_days:
            if check_stop(db,job):finish(db,job,"stopped","Handmatig gestopt.");return
            try:
                n=save_pair_day(ticker,day,horizon,rates)
                pair_key=f"{PAIR_ROOT}/h{horizon}/{ticker}/{day:%Y}/{day:%m}/{day}.csv.gz"
                write_receipt(
                    "pairs",day,ticker,
                    status="stored" if n else "verified_empty",
                    rows=n,keys=[pair_key] if n else []
                )
                msg=f"{n:,} call/put-paren • ✓ Hetzner";status="running"
            except Exception as e:
                msg=f"NIET voltooid • {type(e).__name__}: {e}"[:220];status="warning"
            if status!="warning":
                done[ticker]+=1;job.completed_units+=1
            update_company(db,job.id,ticker,done[ticker],initial_counts(job,ticker,len(days))[1],str(day),msg,status)
            job.current_item=f"{ticker} {day}";job.message=msg;db.commit()
    finalize_company_rows(db,job,len(days),job.end_date)
    finish_by_progress(
        db,job,
        "Call/put Pair Feature Store compleet.",
        "Pair Store gedeeltelijk: {done:,}/{total:,} gereed • {remaining:,} resterend."
    )

def run_vectors(db,job,days):
    """Build exactly one 384D context vector per ticker/trading day."""
    payload=job.payload or {}
    horizon=int(payload.get("horizon_minutes",30))
    force=bool(payload.get("force",False))
    done={ticker:0 for ticker in job.tickers}
    for ticker in job.tickers:
        for day in days:
            if check_stop(db,job):
                finish(db,job,"stopped","Vectorbouw handmatig gestopt.");return
            key=vector_key(ticker,day)
            if exists(key) and not force:
                done[ticker]+=1;job.completed_units+=1
                update_company(db,job.id,ticker,done[ticker],len(days),str(day),"Al aanwezig • overgeslagen")
                continue
            try:
                result=save_daily_vector(ticker,day,horizon_minutes=horizon)
                write_receipt("vectors",day,ticker,status="stored",rows=1,keys=[result["csv_key"],result["npz_key"],result["meta_key"]])
                msg=(f"384D vector • nieuws-chunks {int((result.get('news') or {}).get('chunks',0)):,} "
                     f"• schema {str(result.get('schema_hash',''))[:10]}")
                status="running"
            except FileNotFoundError:
                # Weekend/holiday or a fully absent source day is verified, not a
                # failed vector.  It is deliberately not stored as an all-zero row.
                write_receipt("vectors",day,ticker,status="verified_empty",rows=0,keys=[])
                msg="Geen handels-/nieuwsbrondata • geverifieerd leeg"
                status="running"
            except Exception as exc:
                msg=f"NIET voltooid • {type(exc).__name__}: {exc}"[:220]
                status="warning"
            if status!="warning":
                done[ticker]+=1;job.completed_units+=1
            update_company(db,job.id,ticker,done[ticker],len(days),str(day),msg,status)
            job.current_item=f"{ticker} {day}";job.message=msg;db.commit()
    finalize_company_rows(db,job,len(days),job.end_date)
    finish_by_progress(
        db,job,
        "Dagelijkse 384D fusievectoren compleet.",
        "Vectorbouw gedeeltelijk: {done:,}/{total:,} gereed • {remaining:,} resterend."
    )

def _requested_horizons(payload):
    hs=(payload or {}).get("horizons") or DEFAULT_HORIZONS
    return [h for h in DEFAULT_HORIZONS if h in hs]

def run_deep_relationships(db,job):
    payload=job.payload or {}
    horizons=_requested_horizons(payload)
    targets=[x for x in (payload.get("targets") or []) if x in {
        "future_call_extrinsic_return","future_put_extrinsic_return","future_stock_up"
    }]
    max_rows=int(payload.get("max_rows_per_horizon",30000))
    min_support=int(payload.get("min_support",80))

    def cb(value,message):
        job.current_fraction=float(max(0,min(.97,value)))
        job.current_item="Diepe verbandanalyse"
        job.message=message;db.commit()

    cb(.01,"Point-in-time horizonsets opbouwen…")
    rows_by_h=build_multi_horizon_rows(
        job.tickers,job.start_date,job.end_date,horizons=horizons,
        max_rows_per_horizon=max_rows,
        callback=lambda v,m:cb(.01+.40*v,m),
    )
    bundles={};work=[(h,t) for h in horizons for t in targets if h in rows_by_h and not rows_by_h[h].empty]
    stamp=now().strftime("%Y%m%dT%H%M%SZ")
    for i,(h,target) in enumerate(work,1):
        if check_stop(db,job):finish(db,job,"stopped","Diepe verbandanalyse gestopt.");return
        cb(.43+.52*(i-1)/max(1,len(work)),f"{h} · {target}: discovery en onafhankelijke confirmatie…")
        try:
            daily=daily_relationship_frame(rows_by_h[h],dimensions=384)
            purge={"D1":1,"D2":2,"D3":3,"W1":5}.get(h,1)
            bundle=relationship_scan(
                daily,target,max_features=1200,min_support=min_support,purge_sessions=purge
            )
            bundle["analysis_unit"]="één ticker-handelsdag"
            bundle["fusion_dimensions"]=384
            bundle["ticker_days"]=int(len(daily))
            key=f"{DEEP_RELATIONSHIP_ROOT}/{h}/{target}/{stamp}.json"
            put_json(key,bundle)
            bundles.setdefault(h,{})[target]={"key":key,"summary":{
                "rows":bundle.get("rows"),"tested_features":bundle.get("tested_features"),
                "confirmed_hypotheses":bundle.get("confirmed_hypotheses"),
                "analysis_unit":bundle.get("analysis_unit"),"ticker_days":bundle.get("ticker_days"),
            },"confirmed":bundle.get("confirmed",[])[:50]}
        except Exception as exc:
            bundles.setdefault(h,{})[target]={"error":f"{type(exc).__name__}: {exc}"}
    manifest={
        "version":"20.0","created_at":stamp,"tickers":job.tickers,
        "start_date":str(job.start_date),"end_date":str(job.end_date),
        "horizons":horizons,"targets":targets,"analyses":bundles,
        "causal_claim":False,
        "multiple_testing":"Benjamini-Hochberg FDR + chronologische discovery/confirmation",
    }
    manifest_key=f"{DEEP_RELATIONSHIP_ROOT}/runs/{stamp}.json"
    put_json(manifest_key,manifest)
    setting=db.get(AppSetting,"LATEST_DEEP_RELATIONSHIP_KEY") or AppSetting(key="LATEST_DEEP_RELATIONSHIP_KEY")
    setting.value=manifest_key;setting.encrypted=False;db.add(setting);db.commit()
    job.current_fraction=1.0
    finish(db,job,"done",f"Diepe verbandanalyse klaar: {len(work)} horizon-targetcombinaties.")

def run_relationships(db,job):
    p=job.payload or {}
    horizons=_requested_horizons(p)
    max_rows=int(p.get("max_rows_per_horizon",30000))
    min_train=int(p.get("min_train_support",100))
    min_val=int(p.get("min_validation_support",40))

    def cb(v,m):
        job.current_fraction=float(max(0,min(.97,v)))
        job.current_item="Multi-Horizon Relationship Discovery"
        job.message=m
        db.commit()

    cb(.01,"Multi-horizon call/put targets opbouwen…")
    rows_by_h=build_multi_horizon_rows(
        job.tickers,job.start_date,job.end_date,
        horizons=horizons,
        max_rows_per_horizon=max_rows,
        callback=lambda v,m: cb(v*.55,m)
    )

    bundles={}
    stamp=now().strftime("%Y%m%dT%H%M%SZ")
    usable=[h for h in horizons if h in rows_by_h and not rows_by_h[h].empty]

    for i,h in enumerate(usable,1):
        d=rows_by_h[h]
        cb(.56+.40*(i-1)/max(1,len(usable)),
           f"{horizon_label(h)}: historische verbanden zoeken en valideren…")

        # Respect user support settings, but reduce them for smaller long-horizon
        # samples so day/week models are not impossible to build.
        mt=min(min_train,max(30,int(len(d)*.01)))
        mv=min(min_val,max(12,int(len(d)*.004)))

        bundle=discover_relationships(
            d,
            min_train_support=mt,
            min_validation_support=mv,
            max_order=3
        )
        bundles[h]=bundle
        key=f"{RELATIONSHIP_ROOT}/{h}/{stamp}.json"
        put_json(key,bundle)

        for setting_key in [f"LATEST_RELATIONSHIP_KEY_{h}","LATEST_RELATIONSHIP_KEY"]:
            setting=db.get(AppSetting,setting_key) or AppSetting(key=setting_key)
            setting.value=key
            setting.encrypted=False
            db.add(setting)
        db.commit()

    # One manifest makes it easy for Results to discover all horizons from one run.
    manifest_key=f"{RELATIONSHIP_ROOT}/multi/{stamp}.json"
    put_json(manifest_key,{
        "created_at":stamp,
        "horizons":usable,
        "bundles":bundles,
    })
    setting=db.get(AppSetting,"LATEST_MULTI_RELATIONSHIP_KEY") or AppSetting(key="LATEST_MULTI_RELATIONSHIP_KEY")
    setting.value=manifest_key;setting.encrypted=False;db.add(setting);db.commit()

    job.current_fraction=.99
    finish(
        db,job,"done",
        "Multi-horizon Relationship Discovery klaar: "+
        ", ".join(f"{horizon_label(h)}={len(bundles[h].get('rows',[]))}" for h in usable)
    )

def run_pair_model_job(db,job):
    p=job.payload or {}
    horizons=_requested_horizons(p)
    max_rows=int(p.get("max_rows_per_horizon",30000))
    rid=int(p.get("model_run_id",0) or 0)
    mr=db.get(ModelRun,rid) if rid else None

    def cb(v,m):
        job.current_fraction=float(max(0,min(.98,v)))
        job.current_item="Multi-Horizon FINAL Model"
        job.message=m
        if mr:
            mr.status="running"
            mr.progress=float(max(0,min(.98,v)))
            mr.message=m
        db.commit()

    try:
        if mr:
            mr.status="running";mr.started_at=now();mr.progress=.01;db.commit()

        cb(.01,"Multi-horizon targetdataset opbouwen…")
        rows_by_h=build_multi_horizon_rows(
            job.tickers,job.start_date,job.end_date,
            horizons=horizons,
            max_rows_per_horizon=max_rows,
            callback=lambda v,m: cb(v*.48,m)
        )

        # Re-discover relationships inside this exact training window.
        # This prevents a relationship bundle made with later observations from
        # leaking future information into an earlier model run.
        cb(.49,"Horizon-specifieke historische relaties opnieuw point-in-time valideren…")
        metrics,models,relationship_bundles=train_multi_horizon_final(
            rows_by_h,
            relationship_bundles={},
            callback=lambda v,m: cb(.49+.47*v,m)
        )

        stamp=now().strftime("%Y%m%dT%H%M%SZ")
        key=f"{PAIR_MODEL_ROOT}/multi/{stamp}/bundle.pkl"
        bundle={
            "version":"15.0",
            "created_at":stamp,
            "horizons":metrics.get("overall",{}).get("horizons",horizons),
            "tickers":job.tickers,
            "start_date":str(job.start_date),
            "end_date":str(job.end_date),
            "max_rows_per_horizon":max_rows,
            "metrics":metrics,
            "relationships":relationship_bundles,
            "models":models,
        }
        put_bytes(key,serialize_pair_bundle(bundle),"application/octet-stream")
        put_json(f"{PAIR_MODEL_ROOT}/multi/{stamp}/metrics.json",metrics)

        # Persist relationship bundles that were discovered automatically during training.
        for h,rel in relationship_bundles.items():
            rel_key=f"{RELATIONSHIP_ROOT}/{h}/{stamp}_training.json"
            put_json(rel_key,rel)
            setting=db.get(AppSetting,f"LATEST_RELATIONSHIP_KEY_{h}") or AppSetting(key=f"LATEST_RELATIONSHIP_KEY_{h}")
            setting.value=rel_key;setting.encrypted=False;db.add(setting)

        latest=db.get(AppSetting,"LATEST_MULTI_HORIZON_MODEL_KEY") or AppSetting(key="LATEST_MULTI_HORIZON_MODEL_KEY")
        latest.value=key;latest.encrypted=False;db.add(latest)

        # Compatibility key for older UI/API code.
        compat=db.get(AppSetting,"LATEST_PAIR_MODEL_KEY") or AppSetting(key="LATEST_PAIR_MODEL_KEY")
        compat.value=key;compat.encrypted=False;db.add(compat)
        db.commit()

        if mr:
            mr.status="done";mr.finished_at=now();mr.progress=1.0
            mr.message="Multi-horizon FINAL model klaar"
            mr.metrics=metrics
            mr.artifact_key=key
            db.commit()

        job.current_fraction=1.0
        finish(db,job,"done","Handmatige multi-horizon FINAL modeltraining klaar.")

    except Exception as e:
        msg=f"{type(e).__name__}: {e}"
        if mr:
            mr.status="error";mr.finished_at=now();mr.message=msg;db.commit()
        finish(db,job,"error",msg)

def run_fusion_model_job(db,job):
    payload=job.payload or {}
    horizons=[x for x in (payload.get("horizons") or ["D1","D2","D3","W1"])
              if x in ["D1","D2","D3","W1"]]
    max_rows=int(payload.get("max_rows_per_horizon",30000))
    run_id=int(payload.get("model_run_id",0) or 0)
    mr=db.get(ModelRun,run_id) if run_id else None

    def cb(value,message):
        value=float(max(0,min(.98,value)))
        job.current_fraction=value;job.current_item="384D fusie + dimensiebenchmark";job.message=message
        if mr:
            mr.status="running";mr.progress=value;mr.message=message
            if mr.started_at is None:mr.started_at=now()
        db.commit()

    try:
        cb(.01,"Dagelijkse point-in-time fusietraining starten…")
        metrics,models=train_fusion_model(
            job.tickers,job.start_date,job.end_date,horizons=horizons,
            max_rows_per_horizon=max_rows,callback=cb,
        )
        stamp=now().strftime("%Y%m%dT%H%M%SZ")
        key=f"{FUSION_MODEL_ROOT}/{stamp}/bundle.pkl"
        bundle={
            "version":"20.0","created_at":stamp,"tickers":job.tickers,
            "start_date":str(job.start_date),"end_date":str(job.end_date),
            "horizons":horizons,"metrics":metrics,"models":models,
            "quotes_included":False,"net_profit_claim_allowed":False,
        }
        put_bytes(key,serialize_fusion_bundle(bundle),"application/octet-stream")
        put_json(f"{FUSION_MODEL_ROOT}/{stamp}/metrics.json",metrics)
        setting=db.get(AppSetting,"LATEST_FUSION_MODEL_KEY") or AppSetting(key="LATEST_FUSION_MODEL_KEY")
        setting.value=key;setting.encrypted=False;db.add(setting)
        if mr:
            mr.status="done";mr.finished_at=now();mr.progress=1.0
            mr.message="384D fusie- en dimensiebenchmark klaar"
            mr.metrics=metrics;mr.artifact_key=key
        db.commit()
        job.current_fraction=1.0
        finish(db,job,"done","Fusietraining klaar; final hold-out en nieuwsablaties opgeslagen.")
    except Exception as exc:
        msg=f"{type(exc).__name__}: {exc}"
        if mr:
            mr.status="error";mr.finished_at=now();mr.message=msg;db.commit()
        finish(db,job,"error",msg)


def run_alpha_research_job(db,job):
    """Run auditable alpha hypotheses on every available eligible ticker-day."""
    payload=job.payload or {}
    horizons=[x for x in (payload.get("horizons") or ["D1","D2","D3","W1"])
              if x in ["D1","D2","D3","W1"]]
    run_id=int(payload.get("model_run_id",0) or 0)
    mr=db.get(ModelRun,run_id) if run_id else None
    config=WalkForwardConfig(
        min_train_sessions=int(payload.get("min_train_sessions",80)),
        test_sessions=int(payload.get("test_sessions",20)),
        purge_sessions=5,
        embargo_sessions=int(payload.get("embargo_sessions",1)),
        n_splits=int(payload.get("n_splits",5)),
        holdout_fraction=float(payload.get("holdout_fraction",.15)),
        minimum_holdout_sessions=int(payload.get("minimum_holdout_sessions",20)),
    )
    cost_model=CostModel(
        commission_per_contract_side=float(payload.get("commission_per_contract_side",.65)),
        regulatory_fee_per_contract_side=float(payload.get("regulatory_fee_per_contract_side",.03)),
        slippage_bps_per_side=float(payload.get("slippage_bps_per_side",5.0)),
        fallback_roundtrip_spread_pct=float(payload.get("fallback_roundtrip_spread_pct",.08)),
        safety_margin_return=float(payload.get("safety_margin_return",.005)),
    )

    def cb(value,message):
        value=float(max(0,min(.98,value)))
        job.current_fraction=value;job.current_item="Alpha Research Engine";job.message=message
        if mr:
            mr.status="running";mr.progress=value;mr.message=message
            if mr.started_at is None:mr.started_at=now()
        db.commit()

    try:
        cb(.01,"Alle geïmporteerde ticker-dagen catalogiseren…")
        rows_by_h=build_multi_horizon_rows(
            job.tickers,job.start_date,job.end_date,horizons=horizons,
            max_rows_per_horizon=max(1,int(payload.get("max_rows_per_horizon",30000))),
            callback=lambda v,m:cb(.01+.22*v,m),
            use_all_available=True,independent_daily_contract=True,
        )
        raw_cache={};horizon_results={};all_predictions={};errors=[]
        variants={
            "all_layers":[],"without_news":["news"],"without_embeddings":["embeddings"],
            "without_options":["options"],"without_earnings":["earnings"],
            "without_technical":["technical"],
            "without_theoretical_option_value":["theoretical"],
        }
        for hi,horizon in enumerate(horizons):
            cb(.25+.68*hi/max(1,len(horizons)),f"{horizon}: exacte feature-ablaties opbouwen…")
            base=rows_by_h.get(horizon,pd.DataFrame()).copy()
            usage=dict(getattr(base,"attrs",{}).get("data_usage") or {})
            if base.empty:
                errors.append(f"{horizon}: geen complete targets")
                continue
            base=base.reset_index(drop=True)
            base["research_row_id"]=np.arange(len(base),dtype=int)
            full=attach_vectors(
                base,384,raw_cache,disabled_components=[],include_provenance=True
            )
            full=assign_point_in_time_regimes(full)
            if full.empty:
                errors.append(f"{horizon}: geen gekoppelde dagvectoren")
                continue
            invalid=(
                ~full["vector_metadata_present"].fillna(False).astype(bool)
                | full["vector_future_values_used_as_features"].fillna(True).astype(bool)
                | ~full["earnings_point_in_time_enforced"].fillna(False).astype(bool)
            )
            strong_news=(
                ~full.get("news_present",pd.Series(False,index=full.index)).fillna(False).astype(bool)
                | full["news_availability_basis"].isin(["effective_available_at","max_published_first_seen"])
            )
            invalid |= ~strong_news
            if invalid.any():
                raise ValueError(
                    f"{horizon}: {int(invalid.sum())} vectoren falen de provenance/leakage-audit. "
                    "Bouw deze dagen opnieuw met first-seen timestamps."
                )
            research=full.copy()
            feature_sets={}
            shared=[column for column in CONTRACT_FEATURES if column in research
                    and pd.to_numeric(research[column],errors="coerce").notna().any()]
            for variant,disabled in variants.items():
                if variant=="all_layers":
                    variant_frame=full
                else:
                    variant_frame=attach_vectors(
                        base,384,raw_cache,disabled_components=disabled,
                        include_provenance=False,
                    )
                vector_columns=[f"fusion_{i:03d}" for i in range(384)]
                rename={column:f"{variant}__{column}" for column in vector_columns if column in variant_frame}
                values=variant_frame[["research_row_id",*rename]].rename(columns=rename)
                if variant=="all_layers":
                    research=research.drop(columns=list(rename),errors="ignore").merge(
                        values,on="research_row_id",how="inner"
                    )
                else:
                    research=research.merge(values,on="research_row_id",how="inner")
                variant_shared=list(shared)
                if "options" in disabled:
                    variant_shared=[column for column in variant_shared if not (
                        column.startswith(("call_","put_")) or column in {
                            "strike","dte","benchmark_sigma","risk_free_rate"
                        }
                    )]
                if "theoretical" in disabled:
                    variant_shared=[column for column in variant_shared if not (
                        column.startswith("bs_") or "expectation_gap" in column
                        or "parity" in column or column in {"benchmark_sigma","risk_free_rate"}
                    )]
                feature_sets[variant]=variant_shared+list(rename.values())
            horizon_config=WalkForwardConfig(
                **{**asdict(config),"purge_sessions":{"D1":1,"D2":2,"D3":3,"W1":5}[horizon]}
            )
            suite,predictions=run_research_suite(
                research,feature_sets,
                ["future_call_extrinsic_return","future_put_extrinsic_return"],
                horizon,config=horizon_config,cost_model=cost_model,
            )
            suite["data_usage"]={
                **usage,"rows_after_vector_join":int(len(research)),
                "ticker_days_after_vector_join":int(
                    research[["source_ticker","source_day"]].drop_duplicates().shape[0]
                ),
                "all_imported_eligible_ticker_days_considered":bool(
                    usage.get("use_all_available_ticker_days")
                ),
                "statistical_unit":"one predeclared near-ATM contract per ticker-day",
            }
            horizon_results[horizon]=suite
            for name,frame in predictions.items():
                all_predictions[f"{horizon}|{name}"]=frame

        master={
            "research_version":"alpha-v1","created_at":now().isoformat(),
            "tickers":job.tickers,"start_date":str(job.start_date),"end_date":str(job.end_date),
            "horizons":horizons,"results":horizon_results,"errors":errors,
            "all_imported_eligible_data_policy":True,
            "walk_forward_config":asdict(config),"cost_model":asdict(cost_model),
            "live_trading_authorized":False,
        }
        saved=save_research_suite(master,all_predictions)
        setting=db.get(AppSetting,"LATEST_ALPHA_RESEARCH_KEY") or AppSetting(key="LATEST_ALPHA_RESEARCH_KEY")
        setting.value=saved["manifest_key"];setting.encrypted=False;db.add(setting)
        if mr:
            compact_scorecard=[]
            for horizon,suite in horizon_results.items():
                for experiment in suite.get("experiments") or []:
                    holdout=experiment.get("holdout_metrics") or {}
                    if not holdout:
                        continue
                    compact_scorecard.append({
                        "horizon":horizon,
                        "experiment_id":experiment.get("experiment_id"),
                        "model":experiment.get("model"),
                        "feature_set":experiment.get("feature_set"),
                        "target":experiment.get("target"),
                        "EV_net":holdout.get("EV_net"),
                        "expected_value_per_trade":holdout.get("expected_value_per_trade"),
                        "net_return":holdout.get("net_return"),
                        "sharpe":holdout.get("sharpe"),
                        "sortino":holdout.get("sortino"),
                        "max_drawdown":holdout.get("maximum_drawdown"),
                        "profit_factor":holdout.get("profit_factor"),
                        "trades":holdout.get("number_of_trades"),
                        "costs_provisional":holdout.get("costs_provisional"),
                        "leakage_valid":experiment.get("leakage_valid"),
                    })
            mr.status="done" if horizon_results else "warning";mr.finished_at=now();mr.progress=1.0
            mr.message=("Alpha-onderzoek gereed" if horizon_results else "Geen trainbare alpha-horizon")
            mr.metrics={
                "research_version":"alpha-v1","horizons":list(horizon_results),
                "errors":errors,"all_imported_eligible_data_policy":True,
                "manifest_key":saved["manifest_key"],"scorecard":compact_scorecard,
            }
            mr.artifact_key=saved["manifest_key"]
        if horizon_results:
            job.completed_units=job.total_units
            ticker_units=max(1,sum(1 for _ in weekdays(job.start_date,job.end_date)))
            for ticker in job.tickers:
                update_company(
                    db,job.id,ticker,ticker_units,ticker_units,str(job.end_date),
                    "Alle beschikbare ticker-dagen onderzocht • manifest ✓ Hetzner","done"
                )
        db.commit();job.current_fraction=1.0
        finish(
            db,job,"done" if horizon_results else "warning",
            f"Alpha-onderzoek opgeslagen: {len(horizon_results)}/{len(horizons)} horizons; "
            "live trading blijft uit."
        )
    except Exception as exc:
        msg=f"{type(exc).__name__}: {exc}"
        if mr:
            mr.status="error";mr.finished_at=now();mr.message=msg;db.commit()
        finish(db,job,"error",msg)

def check_auto_agents():
    db=SessionLocal()
    try:
        yesterday=date.today()-timedelta(days=1)
        # model is intentionally NEVER automatically trained.
        for agent in ["stocks","options","news","option_trades","sentiment","features","pairs"]:
            cfg=db.get(AgentConfig,agent)
            if not cfg or not cfg.auto_enabled or not cfg.tickers:continue
            active=db.scalar(select(AgentJob).where(
                AgentJob.agent==agent,AgentJob.status.in_(["queued","running"])
            ).order_by(desc(AgentJob.id)))
            if active:continue
            cur=db.get(AgentCursor,agent)
            start=(cur.last_successful_date+timedelta(days=1)
                   if cur and cur.last_successful_date else cfg.default_start)
            if not start or start>yesterday:continue
            db.add(AgentJob(agent=agent,mode="incremental",status="queued",
                start_date=start,end_date=yesterday,tickers=cfg.tickers,
                message="Automatisch ingepland."))
            db.commit()
    finally:db.close()
