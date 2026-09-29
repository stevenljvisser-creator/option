from itertools import combinations
import math
import numpy as np
import pandas as pd

STATE_NAMES=[
    "call_up_put_down",
    "call_down_put_up",
    "both_up",
    "both_down",
]

LABELS={
"call_up_put_down":"Call stijgt terwijl put daalt",
"call_down_put_up":"Call daalt terwijl put stijgt",
"both_options_up":"Call en put stijgen beide",
"both_options_down":"Call en put dalen beide",
"call_gap_positive":"Call-extrinsiek onder theoretische benchmark",
"call_gap_negative":"Call-extrinsiek boven theoretische benchmark",
"put_gap_positive":"Put-extrinsiek onder theoretische benchmark",
"put_gap_negative":"Put-extrinsiek boven theoretische benchmark",
"positive_news":"Positief nieuwssentiment",
"negative_news":"Negatief nieuwssentiment",
"high_news":"Veel nieuws in het voorafgaande uur",
"high_emotion":"Hoge emotie-intensiteit in nieuws",
"high_surprise":"Hoge surprise-score in nieuws",
"earnings_news":"Earningsnieuws in het voorafgaande uur",
"guidance_news":"Guidance-nieuws in het voorafgaande uur",
"analyst_news":"Analistennieuws in het voorafgaande uur",
"regulatory_news":"Juridisch/regulatoir nieuws in het voorafgaande uur",
"earnings_near":"Kwartaalcijfers binnen vijf dagen",
"call_volume_dominates":"Callvolume duidelijk hoger dan putvolume",
"put_volume_dominates":"Putvolume duidelijk hoger dan callvolume",
"stock_positive_momentum":"Aandeel had positieve 30m-momentum",
"stock_negative_momentum":"Aandeel had negatieve 30m-momentum",
"large_positive_pair_divergence":"Call-put divergentie sterk positief",
"large_negative_pair_divergence":"Call-put divergentie sterk negatief",
"parity_dislocation":"Grote afwijking van put-call parity",
}

def learn_condition_thresholds(df):
    """Learn distribution-dependent thresholds on discovery data only."""
    d=df.copy()
    def n(c):
        return pd.to_numeric(d[c],errors="coerce") if c in d else pd.Series(np.nan,index=d.index)
    out={}
    for name,col,q in [
        ("call_gap_hi","call_expectation_gap",.80),
        ("call_gap_lo","call_expectation_gap",.20),
        ("put_gap_hi","put_expectation_gap",.80),
        ("put_gap_lo","put_expectation_gap",.20),
        ("pair_div_hi","call_put_return_divergence_5m",.80),
        ("pair_div_lo","call_put_return_divergence_5m",.20),
        ("parity_abs_hi","put_call_parity_residual",.90),
    ]:
        x=n(col)
        if name=="parity_abs_hi":
            x=x.abs()
        out[name]=float(x.quantile(q)) if x.notna().any() else None
    return out

def add_condition_columns(df,thresholds=None):
    d=df.copy()
    thresholds=thresholds or learn_condition_thresholds(d)

    def n(c):
        return pd.to_numeric(d[c],errors="coerce") if c in d else pd.Series(np.nan,index=d.index)
    def th(name,fallback):
        v=thresholds.get(name)
        return fallback if v is None or not np.isfinite(v) else float(v)

    c5=n("call_option_return_5m");p5=n("put_option_return_5m")
    d["call_up_put_down"]=(c5>.01)&(p5<-.01)
    d["call_down_put_up"]=(c5<-.01)&(p5>.01)
    d["both_options_up"]=(c5>.01)&(p5>.01)
    d["both_options_down"]=(c5<-.01)&(p5<-.01)

    cg=n("call_expectation_gap");pg=n("put_expectation_gap")
    d["call_gap_positive"]=cg>=th("call_gap_hi",np.inf)
    d["call_gap_negative"]=cg<=th("call_gap_lo",-np.inf)
    d["put_gap_positive"]=pg>=th("put_gap_hi",np.inf)
    d["put_gap_negative"]=pg<=th("put_gap_lo",-np.inf)

    s=n("sentiment_60m")
    d["positive_news"]=s>=.20;d["negative_news"]=s<=-.20
    d["high_news"]=n("news_count_60m")>=2
    d["high_emotion"]=n("emotion_60m")>=.35
    d["high_surprise"]=n("surprise_60m")>=.35
    d["earnings_news"]=n("earnings_news_60m")>=1
    d["guidance_news"]=n("guidance_news_60m")>=1
    d["analyst_news"]=n("analyst_news_60m")>=1
    d["regulatory_news"]=n("legal_regulatory_news_60m")>=1

    e=n("days_to_earnings")
    d["earnings_near"]=(e>=0)&(e<=5)

    vr=n("call_put_volume_ratio")
    d["call_volume_dominates"]=vr>=2
    d["put_volume_dominates"]=vr<=.5

    m=n("stock_return_30m")
    d["stock_positive_momentum"]=m>=.005
    d["stock_negative_momentum"]=m<=-.005

    div=n("call_put_return_divergence_5m")
    d["large_positive_pair_divergence"]=div>=th("pair_div_hi",np.inf)
    d["large_negative_pair_divergence"]=div<=th("pair_div_lo",-np.inf)

    parity=n("put_call_parity_residual").abs()
    d["parity_dislocation"]=parity>=th("parity_abs_hi",np.inf)

    for c in LABELS:
        if c not in d:
            d[c]=False
        d[c]=d[c].fillna(False).astype(bool)
    return d

def future_state_series(df):
    call=pd.to_numeric(df["future_call_extrinsic_return"],errors="coerce")
    put=pd.to_numeric(df["future_put_extrinsic_return"],errors="coerce")
    state=pd.Series(index=df.index,dtype="object")
    state[(call>0)&(put<=0)]="call_up_put_down"
    state[(call<=0)&(put>0)]="call_down_put_up"
    state[(call>0)&(put>0)]="both_up"
    state[(call<=0)&(put<=0)]="both_down"
    return state

def _state_probs(df):
    state=future_state_series(df).dropna()
    n=len(state)
    if n==0:
        return {f"prob_{s}":None for s in STATE_NAMES}
    vc=state.value_counts()
    out={f"prob_{s}":float(vc.get(s,0)/n) for s in STATE_NAMES}
    out["prob_call_up"]=out["prob_call_up_put_down"]+out["prob_both_up"]
    out["prob_put_up"]=out["prob_call_down_put_up"]+out["prob_both_up"]
    return out

def _wilson(k,n,z=1.96):
    if n<=0:return np.nan,np.nan
    p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return max(0,center-half),min(1,center+half)

def _stats(d,baseline_stock_up):
    y=pd.to_numeric(d["future_stock_return"],errors="coerce").dropna()
    if y.empty:return {}
    n=len(y);up=int((y>0).sum());lo,hi=_wilson(up,n)
    idx=y.index
    call=pd.to_numeric(d.loc[idx,"future_call_extrinsic_return"],errors="coerce")
    put=pd.to_numeric(d.loc[idx,"future_put_extrinsic_return"],errors="coerce")
    out={
        "support":n,
        "prob_stock_up":float(up/n),
        "prob_stock_up_ci_low":float(lo),
        "prob_stock_up_ci_high":float(hi),
        "lift_vs_baseline":float(up/n-baseline_stock_up),
        "mean_future_stock_return":float(y.mean()),
        "median_future_stock_return":float(y.median()),
        "mean_future_call_extrinsic_return":float(call.mean()) if call.notna().any() else None,
        "mean_future_put_extrinsic_return":float(put.mean()) if put.notna().any() else None,
    }
    out.update(_state_probs(d.loc[idx]))
    return out

def discover_relationships(df,min_train_support=100,min_validation_support=40,max_order=3):
    raw=df.dropna(
        subset=["minute","future_stock_return","future_call_extrinsic_return","future_put_extrinsic_return"]
    ).sort_values("minute").copy()
    if len(raw)<1000:
        raise RuntimeError("Te weinig call/put-paarobservaties voor Relationship Discovery.")

    a=int(len(raw)*.60);b=int(len(raw)*.80)
    thresholds=learn_condition_thresholds(raw.iloc[:a])
    d=add_condition_columns(raw,thresholds)
    discovery=d.iloc[:a];validation=d.iloc[a:b];holdout=d.iloc[b:]

    bd=float((discovery["future_stock_return"]>0).mean())
    bv=float((validation["future_stock_return"]>0).mean())
    bh=float((holdout["future_stock_return"]>0).mean()) if len(holdout) else None

    rows=[];conds=list(LABELS)
    for order in range(1,max_order+1):
        for combo in combinations(conds,order):
            md=np.ones(len(discovery),dtype=bool)
            for c in combo:
                md &= discovery[c].to_numpy()
            if int(md.sum())<min_train_support:
                continue

            mv=np.ones(len(validation),dtype=bool)
            for c in combo:
                mv &= validation[c].to_numpy()
            if int(mv.sum())<min_validation_support:
                continue

            ds=_stats(discovery.loc[md],bd)
            vs=_stats(validation.loc[mv],bv)
            if not ds or not vs:
                continue

            direction=(
                "stijgend" if vs["mean_future_stock_return"]>0
                else "dalend" if vs["mean_future_stock_return"]<0
                else "neutraal"
            )

            # Score rewards validation support + departure from both stock baseline
            # and the most likely option-pair state.
            state_probs=[vs.get(f"prob_{s}") or 0.0 for s in STATE_NAMES]
            state_concentration=max(state_probs)-0.25
            score=abs(vs["lift_vs_baseline"])*math.sqrt(vs["support"]) + max(0,state_concentration)*math.sqrt(vs["support"])*0.5

            rows.append({
                "order":order,
                "conditions":list(combo),
                "description":" + ".join(LABELS[c] for c in combo),
                "validated_direction":direction,
                "discovery":ds,
                "validation":vs,
                "score":float(score),
            })

    rows.sort(key=lambda x:x["score"],reverse=True)

    return {
        "rows":rows,
        "baseline":{
            "discovery_prob_up":bd,
            "validation_prob_up":bv,
            "final_holdout_prob_up":bh,
            "discovery_rows":len(discovery),
            "validation_rows":len(validation),
            "final_holdout_rows":len(holdout),
            "discovery_state_probs":_state_probs(discovery),
            "validation_state_probs":_state_probs(validation),
            "final_holdout_state_probs":_state_probs(holdout),
        },
        "method":"60% discovery + 20% pattern validation + 20% untouched final model holdout",
        "condition_thresholds":thresholds,
        "final_holdout_start":str(holdout["minute"].min()) if len(holdout) else None,
    }

def add_pattern_indicators(df,bundle,max_patterns=50):
    """
    Adds:
    - binary indicators for the strongest validated historical relationships;
    - empirical prior probabilities based only on matching validated patterns.
    """
    thresholds=(bundle or {}).get("condition_thresholds") or None
    d=add_condition_columns(df,thresholds)
    patterns=(bundle or {}).get("rows",[])[:max_patterns]
    names=[]

    n=len(d)
    prior_names=[
        "call_up_put_down","call_down_put_up","both_up","both_down","stock_up"
    ]
    sums={name:np.zeros(n,dtype=float) for name in prior_names}
    weight_sum=np.zeros(n,dtype=float)
    match_count=np.zeros(n,dtype=float)
    strength=np.zeros(n,dtype=float)

    baseline=((bundle or {}).get("baseline",{}).get("validation_state_probs") or {})
    base_stock=((bundle or {}).get("baseline",{}).get("validation_prob_up"))
    defaults={
        "call_up_put_down":baseline.get("prob_call_up_put_down",0.25) or 0.25,
        "call_down_put_up":baseline.get("prob_call_down_put_up",0.25) or 0.25,
        "both_up":baseline.get("prob_both_up",0.25) or 0.25,
        "both_down":baseline.get("prob_both_down",0.25) or 0.25,
        "stock_up":0.5 if base_stock is None else float(base_stock),
    }

    for i,p in enumerate(patterns,1):
        name=f"pattern_{i:03d}"
        mask=np.ones(n,dtype=bool)
        for c in p.get("conditions",[]):
            if c not in d:
                mask &= False
            else:
                mask &= d[c].to_numpy()
        d[name]=mask.astype(int)
        names.append(name)

        v=p.get("validation",{}) or {}
        support=max(1,float(v.get("support") or 1))
        # Square-root weighting stops a very common pattern from dominating every prior.
        w=math.sqrt(support)
        if not mask.any():
            continue

        vals={
            "call_up_put_down":v.get("prob_call_up_put_down"),
            "call_down_put_up":v.get("prob_call_down_put_up"),
            "both_up":v.get("prob_both_up"),
            "both_down":v.get("prob_both_down"),
            "stock_up":v.get("prob_stock_up"),
        }
        for key,val in vals.items():
            if val is not None:
                sums[key][mask]+=float(val)*w
        weight_sum[mask]+=w
        match_count[mask]+=1
        strength[mask]+=abs(float(v.get("lift_vs_baseline") or 0.0))*w

    for key in prior_names:
        arr=np.full(n,defaults[key],dtype=float)
        has=weight_sum>0
        arr[has]=sums[key][has]/weight_sum[has]
        d[f"relationship_prior_{key}"]=arr

    d["relationship_match_count"]=match_count
    d["relationship_strength"]=np.divide(
        strength,weight_sum,out=np.zeros(n,dtype=float),where=weight_sum>0
    )
    return d,names

def matched_relationships_for_row(row,bundle,max_patterns=50,limit=8):
    if row is None or bundle is None:
        return []
    thresholds=(bundle or {}).get("condition_thresholds") or None
    frame=add_condition_columns(pd.DataFrame([row]),thresholds)
    out=[]
    for p in (bundle.get("rows",[])[:max_patterns]):
        if all(c in frame.columns and bool(frame.iloc[0][c]) for c in p.get("conditions",[])):
            v=p.get("validation",{}) or {}
            out.append({
                "description":p.get("description"),
                "support":v.get("support"),
                "prob_stock_up":v.get("prob_stock_up"),
                "prob_call_up_put_down":v.get("prob_call_up_put_down"),
                "prob_call_down_put_up":v.get("prob_call_down_put_up"),
                "prob_both_up":v.get("prob_both_up"),
                "prob_both_down":v.get("prob_both_down"),
                "score":p.get("score",0),
            })
    out.sort(key=lambda x:x.get("score",0),reverse=True)
    return out[:limit]
