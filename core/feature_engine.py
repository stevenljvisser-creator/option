from __future__ import annotations
from datetime import date,timedelta
from functools import lru_cache
import math,re
import numpy as np
import pandas as pd

from core.config import (
    RAW_ROOT,TRADES_ROOT,TRADE_FEATURE_ROOT,OI_ROOT,EARNINGS_ROOT,
    SENTIMENT_ROOT,FEATURE_ROOT,QUOTE_ROOT,GPU_SENTIMENT_ROOT
)
from core.storage import get_df,exists,list_keys,put_df
from core.quotes import ensure_quote_columns
from core.earnings import attach_earnings_features,load_processed_earnings,EARNINGS_VECTOR_FEATURES

OPTION_RE=re.compile(r"^O:([A-Z0-9.\-]+?)(\d{6})([CP])(\d{8})$")

def parse_option_symbol(symbol):
    m=OPTION_RE.match(str(symbol))
    if not m:
        return None
    under,ymd,cp,strike=m.groups()
    expiry=pd.to_datetime(ymd,format="%y%m%d",errors="coerce")
    return {
        "underlying":under,
        "expiry":expiry,
        "option_type":"call" if cp=="C" else "put",
        "strike":int(strike)/1000.0
    }

def _timestamp_col(df):
    for c in ["window_start","timestamp","sip_timestamp","participant_timestamp"]:
        if c in df.columns:
            return c
    return None

def normalize_timestamp(df):
    c=_timestamp_col(df)
    if c is None:
        return df
    s=pd.to_numeric(df[c],errors="coerce")
    if s.dropna().empty:
        df["ts"]=pd.to_datetime(df[c],utc=True,errors="coerce")
        return df
    med=float(s.dropna().median())
    unit="ns" if med>1e16 else "ms" if med>1e11 else "s"
    df["ts"]=pd.to_datetime(s,unit=unit,utc=True,errors="coerce")
    return df

def stock_features(stock):
    if stock.empty:
        return stock
    d=normalize_timestamp(stock.copy()).sort_values("ts")
    for c in ["open","high","low","close","volume","transactions"]:
        if c in d:
            d[c]=pd.to_numeric(d[c],errors="coerce")
    px=d["close"]
    d["stock_return_1m"]=px.pct_change(1)
    d["stock_return_5m"]=px.pct_change(5)
    d["stock_return_30m"]=px.pct_change(30)
    d["stock_return_60m"]=px.pct_change(60)
    r=np.log(px.replace(0,np.nan)).diff()
    d["realized_vol_30m"]=r.rolling(30,min_periods=8).std()*np.sqrt(390*252)
    d["realized_vol_60m"]=r.rolling(60,min_periods=15).std()*np.sqrt(390*252)
    d["sma_10_ratio"]=px/(px.rolling(10,min_periods=3).mean())-1
    d["sma_30_ratio"]=px/(px.rolling(30,min_periods=8).mean())-1
    ema12=px.ewm(span=12,adjust=False).mean()
    ema26=px.ewm(span=26,adjust=False).mean()
    d["macd_ratio"]=(ema12-ema26)/px.replace(0,np.nan)
    delta=px.diff()
    up=delta.clip(lower=0).rolling(14,min_periods=5).mean()
    dn=(-delta.clip(upper=0)).rolling(14,min_periods=5).mean()
    rs=up/dn.replace(0,np.nan)
    d["rsi14"]=100-(100/(1+rs))
    if "volume" in d:
        mu=d["volume"].rolling(30,min_periods=8).mean()
        sd=d["volume"].rolling(30,min_periods=8).std()
        d["stock_volume_z30"]=(d["volume"]-mu)/sd.replace(0,np.nan)
    if all(c in d for c in ["high","low","close"]):
        d["intraday_range"]=(d["high"]-d["low"])/d["close"].replace(0,np.nan)
    keep=[
        "ts","close","stock_return_1m","stock_return_5m","stock_return_30m",
        "stock_return_60m","realized_vol_30m","realized_vol_60m",
        "sma_10_ratio","sma_30_ratio","macd_ratio","rsi14",
        "stock_volume_z30","intraday_range"
    ]
    return d[[c for c in keep if c in d]]

def _load_option_day(ticker,day):
    base=f"{RAW_ROOT}/options/minute/{day:%Y}/{day:%m}/{day}/"
    ticker_prefix=base+f"by-ticker/{ticker}/"
    ticker_keys=[k for k in list_keys(ticker_prefix) if k.endswith(".csv.gz")]

    # v14 per-ticker layout is authoritative for a ticker if present. This
    # prevents double counting against old shared part-files.
    keys=ticker_keys if ticker_keys else [
        k for k in list_keys(base)
        if k.endswith(".csv.gz") and "/by-ticker/" not in k
    ]

    frames=[]
    for key in keys:
        try:
            x=get_df(key)
            if "underlying" in x.columns:
                x=x[x["underlying"].astype(str)==ticker]
            elif "ticker" in x.columns:
                u=x["ticker"].astype(str).str.extract(
                    r"^O:([A-Z0-9.\-]+?)\d{6}[CP]\d{8}$",expand=False
                )
                x=x[u==ticker]
            if not x.empty:
                frames.append(x)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames,ignore_index=True).drop_duplicates()

def _load_trade_summary(ticker,day):
    base=f"{TRADE_FEATURE_ROOT}/{day:%Y}/{day:%m}/{day}/"
    ticker_prefix=base+f"by-ticker/{ticker}/"
    ticker_keys=[k for k in list_keys(ticker_prefix) if k.endswith(".csv.gz")]
    keys=ticker_keys if ticker_keys else [
        k for k in list_keys(base)
        if k.endswith(".csv.gz") and "/by-ticker/" not in k
    ]
    frames=[]
    for key in keys:
        try:
            x=get_df(key)
            if "underlying" in x:
                x=x[x["underlying"].astype(str)==ticker]
            if not x.empty:
                frames.append(x)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    d=pd.concat(frames,ignore_index=True)
    numeric=["trade_count","trade_volume","trade_dollar_volume","max_trade_size",
             "trade_price_x_size","trade_price_sq_x_size"]
    for c in numeric:
        if c in d:d[c]=pd.to_numeric(d[c],errors="coerce").fillna(0)
    keys=[c for c in ["ticker","minute","underlying"] if c in d]
    agg={c:"sum" for c in numeric if c in d}
    d=d.groupby(keys,as_index=False).agg(agg)
    if "trade_volume" in d and "trade_price_x_size" in d:
        d["trade_vwap"]=d["trade_price_x_size"]/d["trade_volume"].replace(0,np.nan)
    return d

def _load_sentiment(ticker,day):
    # OptionEdge 1.0 deliberately excludes all legacy MarketScope news/sentiment.
    # Only the new professional GPU news layer is accepted.
    key=f"{GPU_SENTIMENT_ROOT}/{ticker}/{day:%Y}/{day:%m}/{day}.csv.gz"
    if not exists(key):
        return pd.DataFrame()
    d=get_df(key)
    if "gpu_sentiment_score" in d.columns:
        d["sentiment_score"]=pd.to_numeric(d["gpu_sentiment_score"],errors="coerce")
    elif "sentiment_score" in d.columns:
        d["sentiment_score"]=pd.to_numeric(d["sentiment_score"],errors="coerce")
    else:
        d["sentiment_score"]=0.0
    d["published_utc"]=pd.to_datetime(d["published_utc"],utc=True,errors="coerce")
    return d.sort_values("published_utc")

def _earnings_records(ticker,start,end):
    # New feature builds use only the versioned SEC/FMP layer.  Legacy
    # Massive/Benzinga rows remain stored in v3 for audit, but do not have the
    # observation timestamps required for a defensible point-in-time join.
    return _cached_earnings_records(str(ticker).upper()).copy()


@lru_cache(maxsize=128)
def _cached_earnings_records(ticker):
    """Load immutable processed earnings once per worker process/ticker."""
    try:
        return load_processed_earnings(ticker)
    except Exception:
        return pd.DataFrame()

def clear_earnings_cache():
    """Invalidate worker-local earnings snapshots after a successful import."""
    _cached_earnings_records.cache_clear()

def _oi_snapshot(ticker,day):
    key=f"{OI_ROOT}/{ticker}/{day}.csv.gz"
    if not exists(key):
        return pd.DataFrame()
    return get_df(key)

def _news_asof_features(base,sent):
    out=base.copy()
    category_cols=[
        "earnings_news_60m","guidance_news_60m","analyst_news_60m",
        "product_news_60m","legal_regulatory_news_60m","mna_news_60m",
        "management_news_60m","macro_news_60m"
    ]
    for c in ["sentiment_latest","sentiment_15m","sentiment_60m",
              "news_count_15m","news_count_60m","emotion_60m",
              "surprise_60m","source_quality_60m","event_importance_60m",
              "bullish_signal_60m","bearish_signal_60m",
              "official_source_count_60m","professional_source_count_60m"]+category_cols:
        out[c]=np.nan
    if sent.empty or out.empty:
        return out
    sent=sent.dropna(subset=["published_utc"]).copy()
    for c in [
        "sentiment_score","emotion_intensity","surprise_score","source_quality",
        "event_importance","bullish_signal","bearish_signal",
        "is_official_source","is_professional_source"
    ]:
        if c in sent:
            sent[c]=pd.to_numeric(sent[c],errors="coerce").fillna(0)
        else:
            sent[c]=0.0

    latest=pd.merge_asof(
        out[["ts"]].sort_values("ts"),
        sent[["published_utc","sentiment_score"]].sort_values("published_utc"),
        left_on="ts",right_on="published_utc",direction="backward"
    )
    out.loc[latest.index,"sentiment_latest"]=latest["sentiment_score"].values

    cats=["earnings","guidance","analyst","product","legal_regulatory","mna","management","macro"]
    if "event_category" not in sent:
        sent["event_category"]="other"
    for cat in cats:
        sent[f"cat_{cat}"]=(sent["event_category"].astype(str)==cat).astype(float)

    events=sent.set_index("published_utc").sort_index()
    for win,label in [(15,"15m"),(60,"60m")]:
        roll=events[[
            "sentiment_score","emotion_intensity","surprise_score","source_quality",
            "event_importance","bullish_signal","bearish_signal",
            "is_official_source","is_professional_source"
        ]].rolling(f"{win}min",closed="both").agg(["mean","count","sum"])
        tmp=pd.DataFrame({
            "event_ts":roll.index,
            f"sentiment_{label}":roll[("sentiment_score","mean")].values,
            f"news_count_{label}":roll[("sentiment_score","count")].values,
            f"emotion_{label}":roll[("emotion_intensity","mean")].values,
            f"surprise_{label}":roll[("surprise_score","mean")].values,
        }).sort_values("event_ts")
        if win==60:
            tmp["source_quality_60m"]=roll[("source_quality","mean")].values
            tmp["event_importance_60m"]=roll[("event_importance","mean")].values
            tmp["bullish_signal_60m"]=roll[("bullish_signal","sum")].values
            tmp["bearish_signal_60m"]=roll[("bearish_signal","sum")].values
            tmp["official_source_count_60m"]=roll[("is_official_source","sum")].values
            tmp["professional_source_count_60m"]=roll[("is_professional_source","sum")].values
        if win==60:
            for cat in cats:
                cr=events[f"cat_{cat}"].rolling("60min",closed="both").sum()
                tmp[f"{cat}_news_60m"]=cr.values
        merged=pd.merge_asof(
            out[["ts"]].sort_values("ts"),tmp,
            left_on="ts",right_on="event_ts",direction="backward"
        )
        for c in tmp.columns:
            if c!="event_ts" and c in out.columns:
                out[c]=merged[c].values
    return out


def _attach_future_option_targets(options,horizon_minutes):
    """Vectorized, contract-aware as-of target join used by the clean store."""
    if options is None or options.empty:
        return options.iloc[0:0] if options is not None else pd.DataFrame()
    tolerance=pd.Timedelta("5min")
    left=options.copy()
    left["wanted_ts"]=left["ts"]+pd.Timedelta(minutes=horizon_minutes)
    future=options[["ticker","ts","close_option","extrinsic"]].rename(columns={
        "ts":"target_ts","close_option":"future_option_close",
        "extrinsic":"future_extrinsic"
    })
    return pd.merge_asof(
        left.sort_values(["wanted_ts","ticker"]),
        future.sort_values(["target_ts","ticker"]),
        by="ticker",left_on="wanted_ts",right_on="target_ts",
        direction="forward",tolerance=tolerance
    ).reset_index(drop=True)

def build_feature_day(ticker,day,horizon_minutes=30):
    stock_key=f"{RAW_ROOT}/stocks/minute/{ticker}/{day:%Y}/{day:%m}/{day}.csv.gz"
    if not exists(stock_key):
        raise RuntimeError(f"Geen bestaande aandelen-minute-data voor {ticker} {day}.")
    stock=stock_features(get_df(stock_key))
    options=_load_option_day(ticker,day)
    if options.empty:
        raise RuntimeError(f"Geen bestaande optie-minute-data voor {ticker} {day}.")

    options=normalize_timestamp(options)
    for c in ["open","high","low","close","volume","transactions"]:
        if c in options:
            options[c]=pd.to_numeric(options[c],errors="coerce")
    options=options.dropna(subset=["ts","ticker","close"]).sort_values(["ticker","ts"])

    # Vectorized OCC parsing replaces a Python callback for every option-minute
    # row.  On dense option days this is the dominant CPU saving while keeping
    # the exact existing symbol contract.
    parsed=options["ticker"].astype(str).str.extract(
        r"^O:([A-Z0-9.\-]+?)(\d{6})([CP])(\d{8})$"
    )
    options["expiry"]=pd.to_datetime(parsed[1],format="%y%m%d",errors="coerce")
    options["option_type"]=np.where(parsed[2].eq("C"),"call",np.where(parsed[2].eq("P"),"put",None))
    options["strike"]=pd.to_numeric(parsed[3],errors="coerce")/1000.0

    # Join underlying stock features point-in-time.
    sf=stock.sort_values("ts")
    options=pd.merge_asof(
        options.sort_values("ts"),sf,left_on="ts",right_on="ts",
        direction="backward",tolerance=pd.Timedelta("2min"),
        suffixes=("_option","_stock")
    )

    options["dte"]=(options["expiry"].dt.tz_localize("UTC",nonexistent="NaT",ambiguous="NaT")
                    -options["ts"]).dt.total_seconds()/86400
    options["log_moneyness"]=np.log(
        options["close_stock"].replace(0,np.nan)/options["strike"].replace(0,np.nan)
    )
    options["is_call"]=(options["option_type"]=="call").astype(float)
    call_intr=(options["close_stock"]-options["strike"]).clip(lower=0)
    put_intr=(options["strike"]-options["close_stock"]).clip(lower=0)
    options["intrinsic"]=np.where(options["is_call"]==1,call_intr,put_intr)
    options["extrinsic"]=(options["close_option"]-options["intrinsic"]).clip(lower=0)

    # Option momentum and activity.
    for lag in [1,5,30]:
        options[f"option_return_{lag}m"]=options.groupby("ticker")["close_option"].pct_change(lag)
    if "volume" in options:
        options["option_volume"]=pd.to_numeric(options["volume"],errors="coerce")
    if "transactions" in options:
        options["option_transactions"]=pd.to_numeric(options["transactions"],errors="coerce")

    # Option trades minute summaries.
    trades=_load_trade_summary(ticker,day)
    if not trades.empty:
        trades["minute"]=pd.to_datetime(trades["minute"],utc=True,errors="coerce")
        options["minute"]=options["ts"].dt.floor("min")
        options=options.merge(
            trades.drop(columns=["underlying"],errors="ignore"),
            how="left",left_on=["ticker","minute"],right_on=["ticker","minute"]
        )

    # Daily OI/Greeks/IV snapshot if available for that date.
    oi=_oi_snapshot(ticker,day)
    if not oi.empty:
        rename={"contract":"ticker","open_interest":"oi_open_interest",
                "iv":"snapshot_iv","delta":"snapshot_delta",
                "gamma":"snapshot_gamma","theta":"snapshot_theta",
                "vega":"snapshot_vega"}
        oi=oi.rename(columns=rename)
        keep=["ticker"]+[c for c in rename.values() if c!="ticker" and c in oi]
        if "ticker" in oi:
            options=options.merge(oi[keep].drop_duplicates("ticker"),how="left",on="ticker")

    # SEC actuals + FMP estimates, joined at every market timestamp.  This
    # prevents an after-market result from appearing in earlier same-day rows
    # and prevents a historical FMP backfill from masquerading as a pre-event
    # estimate.
    earn=_earnings_records(ticker,day-timedelta(days=400),day+timedelta(days=400))
    options=attach_earnings_features(options,earn,"ts")
    options["eps_surprise_percent"]=pd.to_numeric(options.get("eps_surprise_pct"),errors="coerce")
    options["revenue_surprise_percent"]=pd.to_numeric(options.get("revenue_surprise_pct"),errors="coerce")

    # News/sentiment point-in-time. No future articles can leak backward.
    options=_news_asof_features(options,_load_sentiment(ticker,day))

    # Explicit point-in-time market-event features. These are intentionally
    # simple and auditable for a PWS: no future values are used.
    options["event_stock_breakout_up_30m"]=(pd.to_numeric(options.get("stock_return_30m"),errors="coerce")>=0.015).astype(float)
    options["event_stock_breakdown_30m"]=(pd.to_numeric(options.get("stock_return_30m"),errors="coerce")<=-0.015).astype(float)
    options["event_high_realized_vol"]=(pd.to_numeric(options.get("realized_vol_30m"),errors="coerce")>=0.70).astype(float)
    options["event_stock_volume_spike"]=(pd.to_numeric(options.get("stock_volume_z30"),errors="coerce")>=2.0).astype(float)
    options["event_option_jump_up"]=(pd.to_numeric(options.get("option_return_5m"),errors="coerce")>=0.10).astype(float)
    options["event_option_jump_down"]=(pd.to_numeric(options.get("option_return_5m"),errors="coerce")<=-0.10).astype(float)
    options["event_earnings_within_5d"]=(
        pd.to_numeric(options.get("days_to_earnings"),errors="coerce").between(0,5,inclusive="both")
    ).astype(float)

    # Reserved quote schema. Currently NaN when there is no quote subscription.
    options=ensure_quote_columns(options)

    # Future target: timestamp-based, not "30 rows later".
    # One grouped as-of join is substantially faster than one merge per
    # contract and preserves timestamp-based targets through missing minutes.
    d=_attach_future_option_targets(options,horizon_minutes)
    d["target_option_return_30m"]=d["future_option_close"]/d["close_option"].replace(0,np.nan)-1
    d["target_extrinsic_return_30m"]=d["future_extrinsic"]/d["extrinsic"].replace(0,np.nan)-1
    d["target_extrinsic_return_30m"]=d["target_extrinsic_return_30m"].clip(-3,3)

    # Drop raw bulky columns not needed in feature store.
    keep=[
        "ts","ticker","underlying","option_type","is_call","expiry","strike","dte",
        "close_option","close_stock","intrinsic","extrinsic","log_moneyness",
        "option_volume","option_transactions","option_return_1m","option_return_5m",
        "option_return_30m","stock_return_1m","stock_return_5m","stock_return_30m",
        "stock_return_60m","realized_vol_30m","realized_vol_60m","sma_10_ratio",
        "sma_30_ratio","macd_ratio","rsi14","stock_volume_z30","intraday_range",
        "trade_count","trade_volume","trade_dollar_volume","max_trade_size","trade_vwap",
        "oi_open_interest","snapshot_iv","snapshot_delta","snapshot_gamma",
        "snapshot_theta","snapshot_vega",
        "days_to_earnings","days_since_earnings","eps_surprise_percent",
        "revenue_surprise_percent",*EARNINGS_VECTOR_FEATURES,
        "sentiment_latest","sentiment_15m","sentiment_60m",
        "news_count_15m","news_count_60m","emotion_60m","surprise_60m",
        "source_quality_60m","event_importance_60m","bullish_signal_60m","bearish_signal_60m",
        "official_source_count_60m","professional_source_count_60m",
        "event_stock_breakout_up_30m","event_stock_breakdown_30m","event_high_realized_vol",
        "event_stock_volume_spike","event_option_jump_up","event_option_jump_down","event_earnings_within_5d",
        "earnings_news_60m","guidance_news_60m","analyst_news_60m","product_news_60m",
        "legal_regulatory_news_60m","mna_news_60m","management_news_60m","macro_news_60m",
        "bid","ask","midprice","spread","relative_spread","bid_size","ask_size",
        "target_option_return_30m","target_extrinsic_return_30m"
    ]
    keep=list(dict.fromkeys(keep))
    d=d[[c for c in keep if c in d.columns]]
    return d

def feature_key(ticker,day):
    return f"{FEATURE_ROOT}/{ticker}/{day:%Y}/{day:%m}/{day}.csv.gz"

def save_feature_day(ticker,day,horizon_minutes=30):
    d=build_feature_day(ticker,day,horizon_minutes)
    put_df(feature_key(ticker,day),d)
    return len(d)
