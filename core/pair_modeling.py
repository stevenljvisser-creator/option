from __future__ import annotations
import pickle,math
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,brier_score_loss

from core.relationships import (
    discover_relationships,
    add_pattern_indicators,
    future_state_series,
    matched_relationships_for_row,
)
from core.quotes import QUOTE_FEATURES
from core.horizons import HORIZONS,DEFAULT_HORIZONS,horizon_label

STATE_NAMES=["call_up_put_down","call_down_put_up","both_up","both_down"]
STATE_LABELS={
    "call_up_put_down":"Call stijgt / Put daalt",
    "call_down_put_up":"Call daalt / Put stijgt",
    "both_up":"Call en put stijgen",
    "both_down":"Call en put dalen",
}

BASE=[
"close_stock","strike","dte","risk_free_rate","benchmark_sigma",
"call_extrinsic","put_extrinsic","bs_call_extrinsic","bs_put_extrinsic",
"call_expectation_gap","put_expectation_gap","put_call_parity_residual",
"call_option_return_1m","call_option_return_5m","call_option_return_30m",
"put_option_return_1m","put_option_return_5m","put_option_return_30m",
"call_put_return_divergence_5m","call_put_volume_ratio","call_put_trade_volume_ratio",
"call_put_extrinsic_ratio","stock_return_1m","stock_return_5m","stock_return_30m",
"stock_return_60m","realized_vol_30m","realized_vol_60m","sma_10_ratio","sma_30_ratio",
"macd_ratio","rsi14","stock_volume_z30","intraday_range"
]
NEWS=[
"days_to_earnings","days_since_earnings","eps_surprise_percent","revenue_surprise_percent",
"sentiment_latest","sentiment_15m","sentiment_60m","news_count_15m","news_count_60m",
"emotion_60m","surprise_60m","source_quality_60m","event_importance_60m",
"bullish_signal_60m","bearish_signal_60m","official_source_count_60m",
"professional_source_count_60m",
"earnings_news_60m","guidance_news_60m","analyst_news_60m",
"product_news_60m","legal_regulatory_news_60m","mna_news_60m","management_news_60m","macro_news_60m",
"event_stock_breakout_up_30m","event_stock_breakdown_30m","event_high_realized_vol",
"event_stock_volume_spike","event_option_jump_up","event_option_jump_down","event_earnings_within_5d"
]
NEWS=list(dict.fromkeys(NEWS+[
"earnings_event_today","earnings_time_bmo","earnings_time_amc","eps_estimated","eps_actual",
"eps_surprise","eps_surprise_pct","revenue_estimated","revenue_actual","revenue_surprise",
"revenue_surprise_pct","net_income","assets","liabilities","cash_flow",
"earnings_point_in_time_safe","earnings_data_age_days"
]))
OI=[
"call_option_volume","put_option_volume","call_trade_volume","put_trade_volume","call_trade_count",
"put_trade_count","call_oi_open_interest","put_oi_open_interest","call_snapshot_iv","put_snapshot_iv",
"call_snapshot_delta","put_snapshot_delta","call_snapshot_gamma","put_snapshot_gamma",
"call_snapshot_theta","put_snapshot_theta","call_snapshot_vega","put_snapshot_vega"
]
REL_PRIORS=[
"relationship_prior_call_up_put_down",
"relationship_prior_call_down_put_up",
"relationship_prior_both_up",
"relationship_prior_both_down",
"relationship_prior_stock_up",
"relationship_match_count",
"relationship_strength",
]
HORIZON_FEATURES=[
"horizon_equivalent_market_minutes",
"horizon_log_minutes",
"horizon_is_session",
]

def _notify(cb,p,msg):
    if cb:
        cb(float(p),str(msg))

def _reg_metrics(y,p):
    return {
        "mae":float(mean_absolute_error(y,p)),
        "rmse":float(mean_squared_error(y,p)**.5),
        "r2":float(r2_score(y,p)),
    }

def _ordered_probs(raw,classes):
    out=np.zeros((len(raw),len(STATE_NAMES)),dtype=float)
    lookup={str(c):i for i,c in enumerate(classes)}
    for j,state in enumerate(STATE_NAMES):
        if state in lookup:
            out[:,j]=raw[:,lookup[state]]
    sums=out.sum(axis=1)
    missing=sums<=0
    if missing.any():
        out[missing,:]=.25
        sums=out.sum(axis=1)
    return out/sums[:,None]

def _fit_calibrator(state_model,X,y):
    if len(X)<100 or pd.Series(y).nunique()<2:
        return None
    raw=_ordered_probs(state_model.predict_proba(X),state_model.classes_)
    cal=LogisticRegression(max_iter=400,C=1.0)
    cal.fit(raw,y)
    return cal

def _calibrated_probs(state_model,calibrator,X):
    raw=_ordered_probs(state_model.predict_proba(X),state_model.classes_)
    if calibrator is None:
        return raw
    p=calibrator.predict_proba(raw)
    return _ordered_probs(p,calibrator.classes_)

def _fit_binary_calibrator(model,X,y):
    y=pd.Series(y).astype(int)
    if len(X)<100 or y.nunique()<2:
        return None
    raw=model.predict_proba(X)[:,1].reshape(-1,1)
    cal=LogisticRegression(max_iter=300,C=1.0)
    cal.fit(raw,y)
    return cal

def _binary_prob(model,calibrator,X):
    raw=model.predict_proba(X)[:,1]
    if calibrator is None:
        return raw
    return calibrator.predict_proba(raw.reshape(-1,1))[:,1]

def _profit_labels(frame):
    """
    Gross profitability label with current data: future extrinsic value above
    current extrinsic value. Historical quote spread/slippage is unavailable,
    so this is explicitly NOT net profitability.
    """
    call=(pd.to_numeric(frame["future_call_extrinsic_return"],errors="coerce")>0).astype(int)
    put=(pd.to_numeric(frame["future_put_extrinsic_return"],errors="coerce")>0).astype(int)
    return call,put

def _wilson_interval(k,n,z=1.96):
    if n<=0:
        return (None,None)
    p=k/n
    den=1+z*z/n
    centre=(p+z*z/(2*n))/den
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return (max(0.0,centre-half),min(1.0,centre+half))

def _profit_evidence_bins(prob,actual,realized_return,bin_width=.10):
    prob=np.asarray(prob,dtype=float)
    actual=np.asarray(actual,dtype=int)
    ret=np.asarray(realized_return,dtype=float)
    rows=[]
    edges=np.arange(0,1.00001,bin_width)
    for i in range(len(edges)-1):
        lo=float(edges[i]);hi=float(edges[i+1])
        mask=(prob>=lo)&(prob<(hi if i<len(edges)-2 else hi+1e-12))
        if not mask.any():
            continue
        n=int(mask.sum())
        k=int(actual[mask].sum())
        ci_lo,ci_hi=_wilson_interval(k,n)
        vals=ret[mask]
        vals=vals[np.isfinite(vals)]
        pos=vals[vals>0];neg=vals[vals<=0]
        rows.append({
            "bin_low":lo,
            "bin_high":hi,
            "n":n,
            "mean_predicted_probability":float(np.mean(prob[mask])),
            "observed_positive_rate":float(k/n),
            "ci95_low":ci_lo,
            "ci95_high":ci_hi,
            "mean_realized_extrinsic_return":float(np.mean(vals)) if len(vals) else None,
            "mean_positive_return":float(np.mean(pos)) if len(pos) else None,
            "mean_nonpositive_return":float(np.mean(neg)) if len(neg) else None,
        })
    return rows

def _evidence_for_probability(prob,bins):
    if not bins:
        return {
            "n":0,"observed_positive_rate":None,"ci95_low":None,"ci95_high":None,
            "mean_realized_extrinsic_return":None,"mean_positive_return":None,
            "mean_nonpositive_return":None,
        }
    p=float(prob)
    for b in bins:
        lo=float(b["bin_low"]);hi=float(b["bin_high"])
        if p>=lo and (p<hi or (hi>=.999999 and p<=1.0)):
            return dict(b)
    return min(bins,key=lambda b:abs(float(b.get("mean_predicted_probability",.5))-p))

def _multiclass_brier(y_true,prob):
    lookup={s:i for i,s in enumerate(STATE_NAMES)}
    onehot=np.zeros_like(prob,dtype=float)
    for i,y in enumerate(y_true):
        if str(y) in lookup:
            onehot[i,lookup[str(y)]]=1.0
    return float(np.mean(np.sum((prob-onehot)**2,axis=1)))

def _multiclass_logloss(y_true,prob):
    lookup={s:i for i,s in enumerate(STATE_NAMES)}
    chosen=[]
    for i,y in enumerate(y_true):
        j=lookup.get(str(y))
        if j is not None:
            chosen.append(prob[i,j])
    if not chosen:
        return None
    return float(-np.mean(np.log(np.clip(np.asarray(chosen),1e-12,1.0))))

def _phase_by_time(d):
    x=d.copy().sort_values("minute")
    times=np.array(sorted(pd.Series(x["minute"].dropna().unique()).tolist()),dtype=object)
    if len(times)<20:
        n=len(x);a=int(n*.65);b=int(n*.80)
        phase=np.full(n,"final",dtype=object)
        phase[:a]="fit";phase[a:b]="cal"
        x["_phase"]=phase
        return x
    fit_cut=pd.Timestamp(times[max(1,int(len(times)*.65))-1])
    cal_cut=pd.Timestamp(times[max(2,int(len(times)*.80))-1])
    x["_phase"]=np.where(
        x["minute"]<=fit_cut,"fit",
        np.where(x["minute"]<=cal_cut,"cal","final")
    )
    return x

def _add_horizon_features(d,code):
    x=d.copy()
    spec=HORIZONS[code]
    x["target_horizon"]=code
    x["horizon_equivalent_market_minutes"]=float(spec["equivalent_market_minutes"])
    x["horizon_log_minutes"]=float(np.log1p(spec["equivalent_market_minutes"]))
    x["horizon_is_session"]=1.0 if spec["kind"]=="sessions" else 0.0
    return x

def prepare_multi_horizon_training(
    rows_by_h,
    relationship_bundles=None,
    callback=None,
    max_patterns=50,
):
    """
    Each horizon gets its own Relationship Discovery. The validated historical
    probabilities become empirical prior features before all horizons are stacked
    into one unified FINAL model.
    """
    relationship_bundles=dict(relationship_bundles or {})
    enhanced={}
    codes=[h for h in DEFAULT_HORIZONS if h in rows_by_h and rows_by_h[h] is not None and not rows_by_h[h].empty]
    if not codes:
        raise RuntimeError("Geen horizons met complete targets beschikbaar.")

    for i,code in enumerate(codes,1):
        d=rows_by_h[code].copy().sort_values("minute")
        _notify(
            callback,
            .48+.12*(i/max(1,len(codes))),
            f"{horizon_label(code)}: historische verbanden valideren…"
        )

        rel=relationship_bundles.get(code)
        if not rel:
            # Adaptive support, but never so low that isolated coincidences dominate.
            min_train=max(40,min(100,int(len(d)*.0025)))
            min_val=max(15,min(40,int(len(d)*.0010)))
            rel=discover_relationships(
                d,
                min_train_support=min_train,
                min_validation_support=min_val,
                max_order=3
            )
            relationship_bundles[code]=rel

        x,patterns=add_pattern_indicators(d,rel,max_patterns=max_patterns)
        x=_add_horizon_features(x,code)
        x=_phase_by_time(x)
        x["_pattern_count_used"]=len(patterns)
        enhanced[code]=x

    return enhanced,relationship_bundles

def train_multi_horizon_final(
    rows_by_h,
    relationship_bundles=None,
    callback=None,
):
    enhanced,relationship_bundles=prepare_multi_horizon_training(
        rows_by_h,relationship_bundles,callback
    )

    all_rows=pd.concat(list(enhanced.values()),ignore_index=True)
    all_rows=all_rows.dropna(subset=[
        "future_call_extrinsic_return",
        "future_put_extrinsic_return",
        "future_stock_return",
    ])

    pattern_cols=sorted([c for c in all_rows.columns if c.startswith("pattern_")])
    features=[
        c for c in (BASE+OI+NEWS+pattern_cols+REL_PRIORS+HORIZON_FEATURES)
        if c in all_rows.columns
    ]

    fit=all_rows[all_rows["_phase"]=="fit"].copy()
    cal=all_rows[all_rows["_phase"]=="cal"].copy()
    final=all_rows[all_rows["_phase"]=="final"].copy()

    if len(fit)<3000 or len(final)<800:
        raise RuntimeError(
            f"Te weinig multi-horizon data voor FINAL training "
            f"(fit={len(fit):,}, final={len(final):,})."
        )

    _notify(callback,.62,f"Unified FINAL model: {len(fit):,} fit-regels, {len(cal):,} kalibratie, {len(final):,} final holdout.")

    Xfit=fit[features].apply(pd.to_numeric,errors="coerce")
    Xcal=cal[features].apply(pd.to_numeric,errors="coerce")
    Xfinal=final[features].apply(pd.to_numeric,errors="coerce")

    y_call_fit=pd.to_numeric(fit["future_call_extrinsic_return"],errors="coerce").clip(-3,3)
    y_put_fit=pd.to_numeric(fit["future_put_extrinsic_return"],errors="coerce").clip(-3,3)
    y_stock_fit=(pd.to_numeric(fit["future_stock_return"],errors="coerce")>0).astype(int)
    y_state_fit=future_state_series(fit)

    # Regressors/direction use fit + calibration data, state raw model uses fit only
    # so calibration remains genuinely later.
    train80=pd.concat([fit,cal],ignore_index=True)
    X80=train80[features].apply(pd.to_numeric,errors="coerce")
    y_call80=pd.to_numeric(train80["future_call_extrinsic_return"],errors="coerce").clip(-3,3)
    y_put80=pd.to_numeric(train80["future_put_extrinsic_return"],errors="coerce").clip(-3,3)
    y_stock80=(pd.to_numeric(train80["future_stock_return"],errors="coerce")>0).astype(int)

    kw=dict(
        learning_rate=.06,max_iter=170,max_leaf_nodes=31,
        l2_regularization=1.0,random_state=42
    )

    _notify(callback,.66,"FINAL: toekomstige call-extrinsieke waarde trainen over alle termijnen…")
    call_model=HistGradientBoostingRegressor(**kw)
    call_model.fit(X80,y_call80)

    _notify(callback,.70,"FINAL: toekomstige put-extrinsieke waarde trainen over alle termijnen…")
    put_model=HistGradientBoostingRegressor(**kw)
    put_model.fit(X80,y_put80)

    _notify(callback,.74,"FINAL: aandelenrichting trainen over alle termijnen…")
    stock_model=HistGradientBoostingClassifier(**kw)
    stock_model.fit(X80,y_stock80)

    y_call_profit80,y_put_profit80=_profit_labels(train80)
    _notify(callback,.76,"FINAL: bruto winstkans voor call en put trainen…")
    call_profit_model=HistGradientBoostingClassifier(**kw)
    put_profit_model=HistGradientBoostingClassifier(**kw)
    call_profit_model.fit(X80,y_call_profit80)
    put_profit_model.fit(X80,y_put_profit80)

    state_mask=y_state_fit.notna()
    _notify(callback,.78,"FINAL: vier gezamenlijke call/put-uitkomsten trainen…")
    state_model=HistGradientBoostingClassifier(**kw)
    state_model.fit(Xfit.loc[state_mask],y_state_fit.loc[state_mask])

    # Separate probability calibration per horizon, because a 15-minute 70%
    # has a different empirical distribution than a one-week 70%.
    calibrators={}
    for code,x in enhanced.items():
        xc=x[x["_phase"]=="cal"]
        if xc.empty:
            calibrators[code]=None
            continue
        yc=future_state_series(xc)
        mask=yc.notna()
        calibrators[code]=_fit_calibrator(
            state_model,
            xc.loc[mask,features].apply(pd.to_numeric,errors="coerce"),
            yc.loc[mask]
        )

    call_profit_calibrators={}
    put_profit_calibrators={}
    for code,x in enhanced.items():
        xc=x[x["_phase"]=="cal"].copy()
        if xc.empty:
            call_profit_calibrators[code]=None
            put_profit_calibrators[code]=None
            continue
        Xc=xc[features].apply(pd.to_numeric,errors="coerce")
        yc_call,yc_put=_profit_labels(xc)
        call_profit_calibrators[code]=_fit_binary_calibrator(call_profit_model,Xc,yc_call)
        put_profit_calibrators[code]=_fit_binary_calibrator(put_profit_model,Xc,yc_put)

    _notify(callback,.84,"FINAL: iedere termijn evalueren op eigen onaangeraakte holdout…")
    metrics={}
    for code,x in enhanced.items():
        test=x[x["_phase"]=="final"].copy()
        if len(test)<100:
            metrics[code]={
                "status":"skipped",
                "label":horizon_label(code),
                "reason":"Te weinig final-holdout observaties."
            }
            continue

        Xt=test[features].apply(pd.to_numeric,errors="coerce")
        ycall=pd.to_numeric(test["future_call_extrinsic_return"],errors="coerce").clip(-3,3)
        yput=pd.to_numeric(test["future_put_extrinsic_return"],errors="coerce").clip(-3,3)
        ystock=(pd.to_numeric(test["future_stock_return"],errors="coerce")>0).astype(int)
        ystate=future_state_series(test)

        pc=call_model.predict(Xt)
        pp=put_model.predict(Xt)
        ps=stock_model.predict_proba(Xt)[:,1]
        pcall_profit=_binary_prob(call_profit_model,call_profit_calibrators.get(code),Xt)
        pput_profit=_binary_prob(put_profit_model,put_profit_calibrators.get(code),Xt)
        ycall_profit,yput_profit=_profit_labels(test)

        smask=ystate.notna()
        state_prob=_calibrated_probs(
            state_model,
            calibrators.get(code),
            Xt.loc[smask]
        )
        ysv=ystate.loc[smask].to_numpy()
        state_pred=np.asarray(STATE_NAMES)[np.argmax(state_prob,axis=1)]

        call_evidence=_profit_evidence_bins(
            pcall_profit,
            ycall_profit.to_numpy(),
            pd.to_numeric(test["future_call_extrinsic_return"],errors="coerce").to_numpy()
        )
        put_evidence=_profit_evidence_bins(
            pput_profit,
            yput_profit.to_numpy(),
            pd.to_numeric(test["future_put_extrinsic_return"],errors="coerce").to_numpy()
        )

        metrics[code]={
            "status":"trained",
            "label":horizon_label(code),
            "final_holdout_rows":int(len(test)),
            "call":_reg_metrics(ycall,pc),
            "put":_reg_metrics(yput,pp),
            "stock_direction_accuracy":float(np.mean((ps>=.5)==ystock.to_numpy())),
            "stock_direction_brier":float(brier_score_loss(ystock,ps)),
            "call_profit_probability_brier":float(brier_score_loss(ycall_profit,pcall_profit)),
            "put_profit_probability_brier":float(brier_score_loss(yput_profit,pput_profit)),
            "call_profit_probability_accuracy":float(np.mean((pcall_profit>=.5)==ycall_profit.to_numpy())),
            "put_profit_probability_accuracy":float(np.mean((pput_profit>=.5)==yput_profit.to_numpy())),
            "call_profit_evidence_bins":call_evidence,
            "put_profit_evidence_bins":put_evidence,
            "call_profit_base_rate":float(ycall_profit.mean()),
            "put_profit_base_rate":float(yput_profit.mean()),
            "state_accuracy":float(np.mean(state_pred==ysv)) if len(ysv) else None,
            "state_brier_multiclass":_multiclass_brier(ysv,state_prob) if len(ysv) else None,
            "state_log_loss":_multiclass_logloss(ysv,state_prob) if len(ysv) else None,
            "relationships":int(len((relationship_bundles.get(code) or {}).get("rows",[]))),
            "pattern_features":int(x["_pattern_count_used"].max()) if "_pattern_count_used" in x else 0,
        }

    overall={
        "status":"trained",
        "fit_rows":int(len(fit)),
        "calibration_rows":int(len(cal)),
        "final_rows":int(len(final)),
        "features":features,
        "horizons":[h for h in DEFAULT_HORIZONS if h in enhanced],
    }

    models={
        "features":features,
        "call_model":call_model,
        "put_model":put_model,
        "stock_model":stock_model,
        "call_profit_model":call_profit_model,
        "put_profit_model":put_profit_model,
        "call_profit_calibrators":call_profit_calibrators,
        "put_profit_calibrators":put_profit_calibrators,
        "profit_evidence":{
            code:{
                "call":(metrics.get(code) or {}).get("call_profit_evidence_bins",[]),
                "put":(metrics.get(code) or {}).get("put_profit_evidence_bins",[]),
                "call_base_rate":(metrics.get(code) or {}).get("call_profit_base_rate"),
                "put_base_rate":(metrics.get(code) or {}).get("put_profit_base_rate"),
                "call_brier":(metrics.get(code) or {}).get("call_profit_probability_brier"),
                "put_brier":(metrics.get(code) or {}).get("put_profit_probability_brier"),
                "final_holdout_rows":(metrics.get(code) or {}).get("final_holdout_rows"),
            } for code in enhanced
        },
        "state_model":state_model,
        "state_calibrators":calibrators,
        "state_names":STATE_NAMES,
    }

    _notify(callback,.94,"Multi-horizon FINAL model gereed; artifacts opslaan…")
    return {"overall":overall,"horizons":metrics},models,relationship_bundles

def predict_multi_horizon(
    current_df,
    model_bundle,
    horizons=None,
    threshold=.60,
):
    if not model_bundle:
        raise RuntimeError("Geen getraind multi-horizon FINAL model.")

    models=model_bundle.get("models") or {}
    required=["call_model","put_model","stock_model","state_model","call_profit_model","put_profit_model","features"]
    if not all(k in models for k in required):
        raise RuntimeError("Modelbundle bevat nog geen OptionEdge 15.0 clean multi-horizon FINAL model.")

    horizons=horizons or model_bundle.get("horizons") or DEFAULT_HORIZONS
    relationship_bundles=model_bundle.get("relationships") or {}
    features=models["features"]
    outputs={}

    for code in horizons:
        if code not in HORIZONS:
            continue
        rel=relationship_bundles.get(code) or {}
        d,_=add_pattern_indicators(current_df,rel,50)
        d=_add_horizon_features(d,code)
        for c in features:
            if c not in d:
                d[c]=np.nan
        X=d[features].apply(pd.to_numeric,errors="coerce")

        call_return=models["call_model"].predict(X)
        put_return=models["put_model"].predict(X)
        stock_prob=models["stock_model"].predict_proba(X)[:,1]
        call_profit_prob=_binary_prob(
            models["call_profit_model"],
            (models.get("call_profit_calibrators") or {}).get(code),
            X
        )
        put_profit_prob=_binary_prob(
            models["put_profit_model"],
            (models.get("put_profit_calibrators") or {}).get(code),
            X
        )
        state_prob=_calibrated_probs(
            models["state_model"],
            (models.get("state_calibrators") or {}).get(code),
            X
        )

        call_now=pd.to_numeric(d["call_extrinsic"],errors="coerce").to_numpy()
        put_now=pd.to_numeric(d["put_extrinsic"],errors="coerce").to_numpy()

        expected_call=call_now*(1+call_return)
        expected_put=put_now*(1+put_return)

        out=d.copy()
        out["horizon_code"]=code
        out["horizon_label"]=horizon_label(code)
        out["model_expected_call_extrinsic"]=expected_call
        out["model_expected_put_extrinsic"]=expected_put
        out["call_gross_edge"]=expected_call-call_now
        out["put_gross_edge"]=expected_put-put_now
        out["call_gross_extrinsic_edge_per_contract"]=out["call_gross_edge"]*100.0
        out["put_gross_extrinsic_edge_per_contract"]=out["put_gross_edge"]*100.0
        out["call_gross_edge_pct"]=np.divide(
            out["call_gross_edge"],call_now,
            out=np.full(len(out),np.nan),where=np.abs(call_now)>1e-9
        )
        out["put_gross_edge_pct"]=np.divide(
            out["put_gross_edge"],put_now,
            out=np.full(len(out),np.nan),where=np.abs(put_now)>1e-9
        )

        for j,state in enumerate(STATE_NAMES):
            out[f"prob_{state}"]=state_prob[:,j]
        out["prob_call_up"]=state_prob[:,0]+state_prob[:,2]
        out["prob_put_up"]=state_prob[:,1]+state_prob[:,2]
        out["prob_stock_up"]=stock_prob
        out["prob_call_gross_profit"]=call_profit_prob
        out["prob_put_gross_profit"]=put_profit_prob
        out["best_gross_profit_probability"]=np.maximum(call_profit_prob,put_profit_prob)
        out["best_profit_side"]=np.where(call_profit_prob>=put_profit_prob,"CALL","PUT")
        out["profit_probability_above_50"]=(out["best_gross_profit_probability"]>0.50)

        evidence=(models.get("profit_evidence") or {}).get(code,{})
        call_bins=evidence.get("call") or []
        put_bins=evidence.get("put") or []
        call_ev=[_evidence_for_probability(x,call_bins) for x in call_profit_prob]
        put_ev=[_evidence_for_probability(x,put_bins) for x in put_profit_prob]

        def add_evidence(prefix,items):
            out[f"{prefix}_evidence_n"]=[int(x.get("n",0) or 0) for x in items]
            out[f"{prefix}_historical_hit_rate"]=[x.get("observed_positive_rate") for x in items]
            out[f"{prefix}_hit_rate_ci95_low"]=[x.get("ci95_low") for x in items]
            out[f"{prefix}_hit_rate_ci95_high"]=[x.get("ci95_high") for x in items]
            out[f"{prefix}_historical_mean_return"]=[x.get("mean_realized_extrinsic_return") for x in items]
            out[f"{prefix}_historical_mean_positive_return"]=[x.get("mean_positive_return") for x in items]
            out[f"{prefix}_historical_mean_nonpositive_return"]=[x.get("mean_nonpositive_return") for x in items]

        add_evidence("call_profit",call_ev)
        add_evidence("put_profit",put_ev)

        out["best_profit_historical_hit_rate"]=np.where(
            call_profit_prob>=put_profit_prob,
            pd.to_numeric(out["call_profit_historical_hit_rate"],errors="coerce"),
            pd.to_numeric(out["put_profit_historical_hit_rate"],errors="coerce")
        )
        out["best_profit_evidence_n"]=np.where(
            call_profit_prob>=put_profit_prob,
            pd.to_numeric(out["call_profit_evidence_n"],errors="coerce"),
            pd.to_numeric(out["put_profit_evidence_n"],errors="coerce")
        )
        out["best_profit_ci95_low"]=np.where(
            call_profit_prob>=put_profit_prob,
            pd.to_numeric(out["call_profit_hit_rate_ci95_low"],errors="coerce"),
            pd.to_numeric(out["put_profit_hit_rate_ci95_low"],errors="coerce")
        )
        out["best_profit_ci95_high"]=np.where(
            call_profit_prob>=put_profit_prob,
            pd.to_numeric(out["call_profit_hit_rate_ci95_high"],errors="coerce"),
            pd.to_numeric(out["put_profit_hit_rate_ci95_high"],errors="coerce")
        )
        # A >50% probability means "more likely positive than non-positive".
        # We only call it empirically supported when edge is positive and enough
        # out-of-sample examples exist. A CI lower bound >50% is deliberately stricter.
        best_edge_pct=np.where(
            call_profit_prob>=put_profit_prob,
            pd.to_numeric(out["call_gross_edge_pct"],errors="coerce"),
            pd.to_numeric(out["put_gross_edge_pct"],errors="coerce")
        )
        out["paper_profit_case"]=(out["best_gross_profit_probability"]>0.50)&(best_edge_pct>0)
        out["paper_profit_empirically_supported"]=(
            out["paper_profit_case"]&
            (pd.to_numeric(out["best_profit_evidence_n"],errors="coerce")>=50)&
            (pd.to_numeric(out["best_profit_historical_hit_rate"],errors="coerce")>0.50)
        )
        out["paper_profit_strict_ci_supported"]=(
            out["paper_profit_case"]&
            (pd.to_numeric(out["best_profit_evidence_n"],errors="coerce")>=100)&
            (pd.to_numeric(out["best_profit_ci95_low"],errors="coerce")>0.50)
        )

        winner=np.argmax(state_prob,axis=1)
        out["most_likely_state"]=[STATE_NAMES[i] for i in winner]
        out["most_likely_state_label"]=out["most_likely_state"].map(STATE_LABELS)
        out["most_likely_state_probability"]=np.max(state_prob,axis=1)

        call_score=np.maximum(out["call_gross_edge_pct"].fillna(-999),0)*out["prob_call_gross_profit"]
        put_score=np.maximum(out["put_gross_edge_pct"].fillna(-999),0)*out["prob_put_gross_profit"]

        signals=[];scores=[]
        for cs,ps,pc,pp,ce,pe in zip(
            call_score,put_score,out["prob_call_gross_profit"],out["prob_put_gross_profit"],
            out["call_gross_edge_pct"],out["put_gross_edge_pct"]
        ):
            if np.isfinite(ce) and ce>0 and pc>=threshold and cs>=ps:
                signals.append("CALL_EDGE");scores.append(float(cs))
            elif np.isfinite(pe) and pe>0 and pp>=threshold and ps>cs:
                signals.append("PUT_EDGE");scores.append(float(ps))
            else:
                signals.append("NO_EDGE");scores.append(float(max(cs,ps,0)))

        out["paper_signal"]=signals
        out["paper_opportunity_score"]=scores
        out=out.sort_values(
            ["paper_opportunity_score","most_likely_state_probability"],
            ascending=[False,False]
        )
        outputs[code]=out

    return outputs

def explain_row(row,relationship_bundle=None,limit=8):
    return matched_relationships_for_row(
        row.to_dict() if hasattr(row,"to_dict") else row,
        relationship_bundle,
        max_patterns=50,
        limit=limit
    )

def serialize_bundle(x):
    return pickle.dumps(x,protocol=pickle.HIGHEST_PROTOCOL)
