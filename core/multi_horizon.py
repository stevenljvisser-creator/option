from __future__ import annotations
from datetime import datetime
import math,re
from collections import OrderedDict
import numpy as np
import pandas as pd

from core.config import PAIR_ROOT
from core.storage import list_keys,get_df
from core.horizons import HORIZONS,DEFAULT_HORIZONS,horizon_spec

PAIR_RE=re.compile(
    re.escape(PAIR_ROOT)+r"/h(\d+)/([^/]+)/(\d{4})/(\d{2})/(\d{4}-\d{2}-\d{2})\.csv\.gz$"
)
PREFERRED_SOURCE_HORIZONS=[30,15,60,120,240]

def _notify(cb,p,msg):
    if cb:
        cb(float(p),str(msg))

def pair_catalog(tickers,start,end,callback=None):
    """
    Existing pair files contain all current-time features irrespective of the
    old target horizon. Use one preferred copy per ticker/date as the new
    horizon-independent source layer. No market data is downloaded again.
    """
    wanted=set(tickers)
    candidates={}
    seen=0
    _notify(callback,.01,"Bestaande Call/Put Pair Store inventariseren…")
    for key in list_keys(PAIR_ROOT+"/"):
        seen+=1
        m=PAIR_RE.match(key)
        if not m:
            continue
        old_h,ticker,_,_,ds=m.groups()
        if ticker not in wanted:
            continue
        day=pd.Timestamp(ds).date()
        if not (start<=day<=end):
            continue
        old_h=int(old_h)
        rank=PREFERRED_SOURCE_HORIZONS.index(old_h) if old_h in PREFERRED_SOURCE_HORIZONS else 99
        ident=(ticker,day)
        prev=candidates.get(ident)
        if prev is None or rank<prev[0]:
            candidates[ident]=(rank,key)
        if seen%2000==0:
            _notify(callback,.02,f"Pair-catalogus: {seen:,} objecten bekeken…")

    by_ticker={t:{} for t in tickers}
    for (ticker,day),(rank,key) in candidates.items():
        by_ticker.setdefault(ticker,{})[day]=key
    return by_ticker

class DayCache:
    def __init__(self,max_items=18):
        self.max_items=max_items
        self._data=OrderedDict()

    def get(self,key):
        if key in self._data:
            val=self._data.pop(key)
            self._data[key]=val
            return val.copy()
        frame=get_df(key)
        self._data[key]=frame
        while len(self._data)>self.max_items:
            self._data.popitem(last=False)
        return frame.copy()

def _prep(frame):
    d=frame.copy()
    if d.empty:
        return d
    d["minute"]=pd.to_datetime(d["minute"],utc=True,errors="coerce")
    d["expiry"]=pd.to_datetime(d["expiry"],utc=True,errors="coerce")
    for c in [
        "strike","call_extrinsic","put_extrinsic","close_stock",
        "future_call_extrinsic_return","future_put_extrinsic_return",
        "future_stock_return"
    ]:
        if c in d:
            d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["minute","expiry","strike","call_extrinsic","put_extrinsic","close_stock"])
    # Ignore any old horizon target columns. 13.2 computes targets afresh.
    return d

def _sample_even(d,n):
    if len(d)<=n:
        return d.copy()
    idx=np.linspace(0,len(d)-1,n,dtype=int)
    return d.iloc[idx].copy()


def _select_independent_daily_contract(frame):
    """Consider every close contract and keep one predeclared near-ATM unit."""
    if frame is None or frame.empty:
        return frame
    d=frame.copy()
    spot=pd.to_numeric(d.get("close_stock"),errors="coerce")
    strike=pd.to_numeric(d.get("strike"),errors="coerce")
    d["_selection_moneyness"]=(strike/spot.replace(0,np.nan)-1).abs()
    if "dte" in d:
        d["_selection_dte"]=pd.to_numeric(d["dte"],errors="coerce")
    else:
        expiry=pd.to_datetime(d.get("expiry"),utc=True,errors="coerce")
        minute=pd.to_datetime(d.get("minute"),utc=True,errors="coerce")
        d["_selection_dte"]=(expiry-minute).dt.total_seconds()/86400
    d=d.dropna(subset=["_selection_moneyness","_selection_dte"])
    d=d[d["_selection_dte"]>0]
    if d.empty:
        return d
    return d.sort_values(["_selection_moneyness","_selection_dte"]).head(1).drop(
        columns=["_selection_moneyness","_selection_dte"]
    )

def _attach_intraday_target(anchor,full_day,minutes):
    if anchor.empty or full_day.empty:
        return anchor.iloc[0:0]
    out=[]
    for (expiry,strike),g in anchor.groupby(["expiry","strike"],sort=False):
        g=g.sort_values("minute").copy()
        tg=full_day[
            (full_day["expiry"]==expiry)&(full_day["strike"]==strike)
        ].copy()
        if tg.empty:
            continue
        future=tg[["minute","call_extrinsic","put_extrinsic","close_stock"]].rename(columns={
            "minute":"target_minute",
            "call_extrinsic":"target_call_extrinsic",
            "put_extrinsic":"target_put_extrinsic",
            "close_stock":"target_stock_close",
        }).sort_values("target_minute")
        left=g.copy()
        left["wanted_target"]=left["minute"]+pd.Timedelta(minutes=int(minutes))
        matched=pd.merge_asof(
            left.sort_values("wanted_target"),
            future,
            left_on="wanted_target",
            right_on="target_minute",
            direction="forward",
            tolerance=pd.Timedelta(minutes=max(5,int(minutes*.12))),
        )
        out.append(matched)
    return pd.concat(out,ignore_index=True) if out else anchor.iloc[0:0]

def _local_minute_of_day(series):
    local=series.dt.tz_convert("America/New_York")
    return local.dt.hour*60+local.dt.minute


def _daily_close_anchor(frame, day, max_staleness_minutes=30):
    """Carry the last pre-close contract observation to the 16:00 as-of.

    The observed timestamp is retained for audit. Contracts without an update
    in the final window are excluded rather than pretending an old print was a
    current close. Exchange-calendar/early-close support remains a data gate.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=getattr(frame,"columns",[]))
    d=frame.copy()
    close=pd.Timestamp(
        datetime(day.year,day.month,day.day,16,0),tz="America/New_York"
    ).tz_convert("UTC")
    d["minute"]=pd.to_datetime(d["minute"],utc=True,errors="coerce")
    local_day=d["minute"].dt.tz_convert("America/New_York").dt.date
    d=d[(local_day==day)&(d["minute"]<=close)].copy()
    if d.empty:
        return d
    group=["expiry","strike"]
    d=d.sort_values("minute").groupby(group,as_index=False,sort=False).tail(1).copy()
    d["source_observed_minute"]=d["minute"]
    d["source_staleness_minutes"]=(close-d["minute"]).dt.total_seconds()/60
    d=d[d["source_staleness_minutes"].between(0,float(max_staleness_minutes))].copy()
    d["minute"]=close
    return d.reset_index(drop=True)

def _attach_session_target(anchor,target,session_count):
    """
    Match the same strike/expiry at approximately the same New York trading
    minute on the Nth later observed trading session. Holidays/weekends are
    handled by the catalog's actual trading dates, not by clock-day arithmetic.
    """
    if anchor.empty or target.empty:
        result=anchor.copy()
        for c in ["target_minute","target_call_extrinsic","target_put_extrinsic","target_stock_close"]:
            result[c]=np.nan
        return result

    a=anchor.copy()
    t=target.copy()
    a["market_minute"]=_local_minute_of_day(a["minute"])
    t["market_minute"]=_local_minute_of_day(t["minute"])
    t=t.rename(columns={
        "minute":"target_minute",
        "call_extrinsic":"target_call_extrinsic",
        "put_extrinsic":"target_put_extrinsic",
        "close_stock":"target_stock_close",
    })

    pieces=[]
    for (expiry,strike),g in a.groupby(["expiry","strike"],sort=False):
        tg=t[(t["expiry"]==expiry)&(t["strike"]==strike)].copy()
        if tg.empty:
            z=g.copy()
            z["target_minute"]=pd.NaT
            z["target_call_extrinsic"]=np.nan
            z["target_put_extrinsic"]=np.nan
            z["target_stock_close"]=np.nan
            pieces.append(z)
            continue
        tg=tg.sort_values("market_minute")
        left=g.sort_values("market_minute")
        m=pd.merge_asof(
            left,
            tg[["market_minute","target_minute","target_call_extrinsic",
                "target_put_extrinsic","target_stock_close"]],
            on="market_minute",
            direction="nearest",
            tolerance=10,
        )
        pieces.append(m)
    return pd.concat(pieces,ignore_index=True) if pieces else a.iloc[0:0]

def _finish_target(frame,code):
    d=frame.copy()
    d["target_horizon"]=code
    d["horizon_equivalent_market_minutes"]=HORIZONS[code]["equivalent_market_minutes"]
    d["horizon_log_minutes"]=np.log1p(float(HORIZONS[code]["equivalent_market_minutes"]))
    d["future_call_extrinsic_return"]=(
        pd.to_numeric(d["target_call_extrinsic"],errors="coerce")
        /pd.to_numeric(d["call_extrinsic"],errors="coerce").replace(0,np.nan)-1
    )
    d["future_put_extrinsic_return"]=(
        pd.to_numeric(d["target_put_extrinsic"],errors="coerce")
        /pd.to_numeric(d["put_extrinsic"],errors="coerce").replace(0,np.nan)-1
    )
    d["future_stock_return"]=(
        pd.to_numeric(d["target_stock_close"],errors="coerce")
        /pd.to_numeric(d["close_stock"],errors="coerce").replace(0,np.nan)-1
    )
    d["future_stock_up"]=(d["future_stock_return"]>0).astype(float)
    return d

def build_multi_horizon_rows(
    tickers,start,end,horizons=None,max_rows_per_horizon=50000,callback=None,
    use_all_available=True,independent_daily_contract=False
):
    horizons=horizons or DEFAULT_HORIZONS
    for h in horizons:
        horizon_spec(h)

    catalog=pair_catalog(tickers,start,end,callback)
    cache=DayCache(max_items=20)
    # Allocate anchor files approximately evenly across companies and time.
    active=[t for t in tickers if catalog.get(t)]
    if not active:
        raise RuntimeError(
            "Geen bestaande Call/Put Pair Store gevonden. "
            "Draai eerst Agents → Call/Put Pair Feature Store."
        )

    target_anchor_files=max(80,min(500,int(math.ceil(max_rows_per_horizon/220))))
    per_ticker=(None if use_all_available else max(2,target_anchor_files//len(active)))
    anchors=[]

    for ticker in active:
        dates=sorted(catalog[ticker])
        if not dates:
            continue
        if per_ticker is None or len(dates)<=per_ticker:
            picked=dates
        else:
            idx=np.linspace(0,len(dates)-1,per_ticker,dtype=int)
            picked=[dates[i] for i in idx]
        for day in picked:
            anchors.append((ticker,day,catalog[ticker][day]))

    if not anchors:
        raise RuntimeError("Geen anchor-dagen beschikbaar voor multi-horizon training.")

    rows_by_h={h:[] for h in horizons}
    anchor_sample=max(80,int(math.ceil(max_rows_per_horizon/max(1,len(anchors))*1.15)))

    for ai,(ticker,day,key) in enumerate(anchors,1):
        raw=_prep(cache.get(key))
        if raw.empty:
            continue
        # In full-data mode keep every intraday option-pair observation too.
        # Previously ticker-days were complete but intraday rows were still
        # silently sampled here, which made "use_all_available" misleading.
        anchor=raw.copy() if use_all_available else _sample_even(raw,anchor_sample)
        daily_anchor=_daily_close_anchor(raw,day)
        if independent_daily_contract:
            daily_anchor=_select_independent_daily_contract(daily_anchor)
        elif not use_all_available:
            daily_anchor=_sample_even(daily_anchor,anchor_sample)

        dates=sorted(catalog[ticker])
        date_pos={d:i for i,d in enumerate(dates)}
        pos=date_pos[day]

        for code in horizons:
            spec=HORIZONS[code]
            if spec["kind"]=="minutes":
                target=_attach_intraday_target(anchor,raw,spec["value"])
            else:
                if daily_anchor.empty:
                    continue
                future_index=pos+int(spec["value"])
                if future_index>=len(dates):
                    continue
                target_day=dates[future_index]
                # Guard against holes in the Pair Store being mistaken for normal
                # trading sessions. Weekend/holiday gaps are allowed, large data
                # gaps are not.
                max_calendar_gap={1:4,2:5,3:7,5:10}.get(int(spec["value"]),int(spec["value"])+5)
                if (target_day-day).days>max_calendar_gap:
                    continue
                target_raw=_prep(cache.get(catalog[ticker][target_day]))
                target_close=_daily_close_anchor(target_raw,target_day)
                target=_attach_session_target(daily_anchor,target_close,spec["value"])
            target=_finish_target(target,code)
            target["source_ticker"]=ticker
            target["source_day"]=str(day)
            rows_by_h[code].append(target)

        if ai==1 or ai%4==0 or ai==len(anchors):
            _notify(
                callback,
                .03+.42*(ai/len(anchors)),
                f"Multi-horizon targets: {ai}/{len(anchors)} anchor-dagen"
            )

    out={}
    for code,frames in rows_by_h.items():
        if not frames:
            out[code]=pd.DataFrame()
            continue
        d=pd.concat(frames,ignore_index=True)
        d=d.dropna(subset=[
            "future_call_extrinsic_return",
            "future_put_extrinsic_return",
            "future_stock_return",
        ]).sort_values("minute")
        if not use_all_available and len(d)>max_rows_per_horizon:
            idx=np.linspace(0,len(d)-1,max_rows_per_horizon,dtype=int)
            d=d.iloc[idx].copy()
        d.attrs["data_usage"]={
            "catalog_ticker_days":int(sum(len(catalog.get(t,{}) or {}) for t in active)),
            "anchor_ticker_days_considered":int(len(anchors)),
            "eligible_target_rows":int(len(d)),
            "use_all_available_ticker_days":bool(use_all_available),
            "independent_daily_contract":bool(independent_daily_contract),
            "hidden_row_cap_applied":bool(not use_all_available and len(d)>=max_rows_per_horizon),
        }
        out[code]=d

    usable=[h for h,d in out.items() if not d.empty]
    if not usable:
        raise RuntimeError("Geen complete multi-horizon targets konden worden opgebouwd.")
    _notify(
        callback,.47,
        "Multi-horizon dataset gereed: "+
        ", ".join(f"{h}={len(out[h]):,}" for h in usable)
    )
    return out

def stack_horizons(rows_by_h):
    frames=[]
    for code,d in rows_by_h.items():
        if d is None or d.empty:
            continue
        x=d.copy()
        x["target_horizon"]=code
        frames.append(x)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
