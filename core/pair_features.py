from datetime import timedelta
import numpy as np,pandas as pd
from core.config import FEATURE_ROOT,PAIR_ROOT,MACRO_ROOT
from core.storage import get_df,put_df,exists
from core.theoretical import theoretical_row
from core.earnings import EARNINGS_VECTOR_FEATURES

CONTEXT=[
"stock_return_1m","stock_return_5m","stock_return_30m","stock_return_60m",
"realized_vol_30m","realized_vol_60m","sma_10_ratio","sma_30_ratio","macd_ratio","rsi14",
"stock_volume_z30","intraday_range","days_to_earnings","days_since_earnings",
"eps_surprise_percent","revenue_surprise_percent","sentiment_latest","sentiment_15m",
"sentiment_60m","news_count_15m","news_count_60m","emotion_60m","surprise_60m",
"source_quality_60m","event_importance_60m","bullish_signal_60m","bearish_signal_60m",
"official_source_count_60m","professional_source_count_60m",
"event_stock_breakout_up_30m","event_stock_breakdown_30m","event_high_realized_vol",
"event_stock_volume_spike","event_option_jump_up","event_option_jump_down","event_earnings_within_5d",
"earnings_news_60m","guidance_news_60m","analyst_news_60m","product_news_60m",
"legal_regulatory_news_60m","mna_news_60m","management_news_60m","macro_news_60m"]
CONTEXT=list(dict.fromkeys(CONTEXT+EARNINGS_VECTOR_FEATURES))
OPT=["ticker","close_option","extrinsic","intrinsic","option_volume","option_transactions",
"option_return_1m","option_return_5m","option_return_30m","trade_count","trade_volume",
"trade_dollar_volume","oi_open_interest","snapshot_iv","snapshot_delta","snapshot_gamma",
"snapshot_theta","snapshot_vega"]

def feature_key(ticker,day):return f"{FEATURE_ROOT}/{ticker}/{day:%Y}/{day:%m}/{day}.csv.gz"
def pair_key(ticker,day,horizon):return f"{PAIR_ROOT}/h{horizon}/{ticker}/{day:%Y}/{day:%m}/{day}.csv.gz"

def load_rate_series():
    key=f"{MACRO_ROOT}/risk_free_3m.csv.gz"
    if not exists(key):return pd.DataFrame()
    d=get_df(key);d["date"]=pd.to_datetime(d["date"],errors="coerce").dt.date
    d["risk_free_rate"]=pd.to_numeric(d["risk_free_rate"],errors="coerce")
    return d.dropna(subset=["date"]).sort_values("date")
def rate_for_day(rates,day):
    if rates is None or rates.empty:return 0.0
    x=rates[rates["date"]<=day]
    return float(x.iloc[-1]["risk_free_rate"]) if not x.empty else 0.0

def build_pair_day(ticker,day,horizon_minutes=30,rates=None):
    fk=feature_key(ticker,day)
    if not exists(fk):raise RuntimeError(f"Feature Store ontbreekt: {ticker} {day}")
    d=get_df(fk)
    if d.empty:return d
    d["ts"]=pd.to_datetime(d["ts"],utc=True,errors="coerce")
    d["expiry"]=pd.to_datetime(d["expiry"],utc=True,errors="coerce")
    d["strike"]=pd.to_numeric(d["strike"],errors="coerce")
    d["close_option"]=pd.to_numeric(d["close_option"],errors="coerce")
    d["extrinsic"]=pd.to_numeric(d["extrinsic"],errors="coerce")
    d["close_stock"]=pd.to_numeric(d["close_stock"],errors="coerce")
    d=d.dropna(subset=["ts","expiry","strike","option_type","close_option","extrinsic","close_stock"])
    d["minute"]=d["ts"].dt.floor("min")
    for c in CONTEXT+OPT:
        if c in d and c!="ticker":
            d[c]=pd.to_numeric(d[c],errors="coerce")

    keycols=["minute","expiry","strike"]
    calls=d[d["option_type"].astype(str).str.lower()=="call"].copy()
    puts=d[d["option_type"].astype(str).str.lower()=="put"].copy()
    if calls.empty or puts.empty:return pd.DataFrame()

    ckeep=keycols+[c for c in OPT if c in calls]+[c for c in CONTEXT if c in calls]
    pkeep=keycols+[c for c in OPT if c in puts]
    calls=calls[ckeep].drop_duplicates(keycols,keep="last")
    puts=puts[pkeep].drop_duplicates(keycols,keep="last")
    calls=calls.rename(columns={c:f"call_{c}" for c in OPT if c in calls})
    puts=puts.rename(columns={c:f"put_{c}" for c in OPT if c in puts})
    x=calls.merge(puts,on=keycols,how="inner")
    if x.empty:return x

    stock=d[["minute","close_stock"]].dropna().groupby("minute",as_index=False)["close_stock"].last()
    x=x.merge(stock,on="minute",how="left")
    x["dte"]=(x["expiry"]-x["minute"]).dt.total_seconds()/86400
    sigma=pd.to_numeric(x["realized_vol_60m"],errors="coerce") if "realized_vol_60m" in x else pd.Series(np.nan,index=x.index)
    if sigma.isna().all() and "realized_vol_30m" in x:sigma=pd.to_numeric(x["realized_vol_30m"],errors="coerce")
    x["benchmark_sigma"]=sigma.clip(.08,3.0).fillna(.35)
    rate=rate_for_day(rates if rates is not None else load_rate_series(),day);x["risk_free_rate"]=rate

    vals=[theoretical_row(r.close_stock,r.strike,r.dte,rate,r.benchmark_sigma,r.call_extrinsic,r.put_extrinsic)
          for r in x.itertuples(index=False)]
    x=pd.concat([x,pd.DataFrame(vals,index=x.index)],axis=1)
    # Two point-in-time meanings of expectation value are kept explicitly:
    # (1) the model-theoretical discounted value and (2) the price/time value
    # actually observed in the market.  Realized future values remain targets
    # below and are never included in the same day's fusion vector.
    x["theoretical_call_expected_value"]=x["bs_call_price"]
    x["theoretical_put_expected_value"]=x["bs_put_price"]
    x["market_call_expected_value"]=pd.to_numeric(x.get("call_close_option"),errors="coerce")
    x["market_put_expected_value"]=pd.to_numeric(x.get("put_close_option"),errors="coerce")
    x["practical_call_time_value"]=pd.to_numeric(x.get("call_extrinsic"),errors="coerce")
    x["practical_put_time_value"]=pd.to_numeric(x.get("put_extrinsic"),errors="coerce")
    x["call_theory_market_gap"]=x["theoretical_call_expected_value"]-x["market_call_expected_value"]
    x["put_theory_market_gap"]=x["theoretical_put_expected_value"]-x["market_put_expected_value"]
    def col(name,default=0.0):
        return pd.to_numeric(x[name],errors="coerce") if name in x else pd.Series(default,index=x.index,dtype=float)
    c5=col("call_option_return_5m");p5=col("put_option_return_5m")
    x["call_put_return_divergence_5m"]=c5-p5
    x["call_up_put_down_5m"]=((c5>0)&(p5<0)).astype(int)
    x["call_down_put_up_5m"]=((c5<0)&(p5>0)).astype(int)
    x["both_options_up_5m"]=((c5>0)&(p5>0)).astype(int)
    x["both_options_down_5m"]=((c5<0)&(p5<0)).astype(int)
    x["call_put_volume_ratio"]=(col("call_option_volume")+1)/(col("put_option_volume")+1)
    x["call_put_trade_volume_ratio"]=(col("call_trade_volume")+1)/(col("put_trade_volume")+1)
    x["call_put_extrinsic_ratio"]=(col("call_extrinsic")+1e-8)/(col("put_extrinsic")+1e-8)
    T=x["dte"].clip(lower=0)/365
    x["put_call_parity_residual"]=col("call_close_option")-col("put_close_option")-(x["close_stock"]-x["strike"]*np.exp(-rate*T))

    # Future underlying.
    x["wanted_minute"]=x["minute"]+pd.Timedelta(minutes=horizon_minutes)
    fs=stock.rename(columns={"minute":"future_minute","close_stock":"future_stock_close"})
    x=pd.merge_asof(x.sort_values("wanted_minute"),fs.sort_values("future_minute"),
                    left_on="wanted_minute",right_on="future_minute",direction="forward",
                    tolerance=pd.Timedelta("5min"))
    x["future_stock_return"]=x["future_stock_close"]/x["close_stock"].replace(0,np.nan)-1
    x["future_stock_up"]=(x["future_stock_return"]>0).astype(float)

    # Future same strike/expiry call/put extrinsic.
    pieces=[]
    for (expiry,strike),g in x.groupby(["expiry","strike"],dropna=False):
        future=g[["minute","call_extrinsic","put_extrinsic"]].sort_values("minute").rename(
            columns={"minute":"pair_future_minute","call_extrinsic":"future_call_extrinsic",
                     "put_extrinsic":"future_put_extrinsic"})
        left=g.sort_values("wanted_minute")
        m=pd.merge_asof(left,future,left_on="wanted_minute",right_on="pair_future_minute",
                        direction="forward",tolerance=pd.Timedelta("5min"))
        pieces.append(m)
    x=pd.concat(pieces,ignore_index=True) if pieces else x.iloc[0:0]
    x["future_call_extrinsic_return"]=x["future_call_extrinsic"]/x["call_extrinsic"].replace(0,np.nan)-1
    x["future_put_extrinsic_return"]=x["future_put_extrinsic"]/x["put_extrinsic"].replace(0,np.nan)-1
    x["ticker"]=ticker;x["horizon_minutes"]=horizon_minutes
    return x

def save_pair_day(ticker,day,horizon_minutes=30,rates=None):
    d=build_pair_day(ticker,day,horizon_minutes,rates)
    put_df(pair_key(ticker,day,horizon_minutes),d)
    return len(d)
