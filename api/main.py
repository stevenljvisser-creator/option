from datetime import date,datetime,timezone
import json,os,hashlib,hmac,re,uuid
import pandas as pd
from fastapi import FastAPI,Depends,HTTPException,Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field
from sqlalchemy import select,desc
from sqlalchemy.orm import Session

from core.db import init_db,db_session,SessionLocal
from core.import_planner import build_plan,load_snapshot
from core.config import (
    PRO_NEWS_ROOT,PRO_NEWS_STATUS_ROOT,FEATURE_ROOT,PAIR_ROOT,PAIR_MODEL_ROOT,
    GPU_LLM_STATUS_ROOT,DEEP_RELATIONSHIP_ROOT,FUSION_MODEL_ROOT,
)
from core.horizons import DEFAULT_HORIZONS,HORIZONS,horizon_label
from core.models import AgentConfig,AgentJob,AgentCompanyProgress,AppSetting,ModelRun
from core.massive import MassiveREST,test_flat_files
from core.storage import test_connection as test_storage,get_json,put_json,exists
from core.storage_inventory import coverage_for_agent,DATASET_LABELS
from core.data_readiness import DEFAULT_TICKERS,readiness
from core.research_design import design_payload
from core.runtime_settings import (
    save_settings,get_settings,get_setting,setup_complete
)
from core.earnings import (
    FmpEarningsClient,SecEdgarClient,latest_earnings_run,load_processed_earnings,
)
from core.alpha_research import latest_research_suite
from core.paper_trading import (
    create_paper_signal,paper_trading_status,save_paper_signal,settle_paper_signal,
)
from core.trading_controls import (
    PortfolioState,RiskPolicy,TradeCandidate,default_safety_status,
    evaluate_trade,live_promotion_gate,
)
from core.monitoring import latest_monitoring_snapshot,monitor_performance,save_monitoring_snapshot

app=FastAPI(title="OptionEdge API",version="20.0")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request:Request, exc:Exception):
    # Show a useful error instead of only "Internal Server Error".
    return JSONResponse(
        status_code=500,
        content={"detail":f"{type(exc).__name__}: {str(exc)}"}
    )

class SetupPayload(BaseModel):
    storage_location:str="hel1"
    storage_endpoint:str|None=None
    storage_region:str|None=None
    storage_bucket:str
    storage_access_key:str
    storage_secret_key:str
    massive_api_key:str
    massive_flat_access_key:str
    massive_flat_secret_key:str
    fmp_api_key:str=""
    sec_user_agent:str=""
    import_chunk_rows:int=Field(default=300000,ge=50000,le=600000)
    admin_password:str=Field(min_length=8,max_length=128)

class SettingsUpdate(BaseModel):
    storage_location:str|None=None
    storage_endpoint:str|None=None
    storage_region:str|None=None
    storage_bucket:str|None=None
    storage_access_key:str|None=None
    storage_secret_key:str|None=None
    massive_api_key:str|None=None
    massive_flat_access_key:str|None=None
    massive_flat_secret_key:str|None=None
    fmp_api_key:str|None=None
    sec_user_agent:str|None=None
    import_chunk_rows:int|None=Field(default=None,ge=50000,le=600000)

class LoginPayload(BaseModel):
    password:str

class StartAgent(BaseModel):
    mode:str="missing_only"
    start_date:date
    end_date:date
    tickers:list[str]
    payload:dict={}

class TrainModelPayload(BaseModel):
    start_date:date
    end_date:date
    tickers:list[str]
    target:str="target_extrinsic_return_30m"
    horizon_minutes:int=30
    max_rows:int=Field(default=200000,ge=5000,le=1000000)
    use_all_available:bool=True


class PairTrainPayload(BaseModel):
    start_date:date
    end_date:date
    tickers:list[str]
    horizons:list[str]=Field(default_factory=lambda: DEFAULT_HORIZONS.copy())
    max_rows_per_horizon:int=Field(default=30000,ge=5000,le=80000)

class RelationshipPayload(BaseModel):
    start_date:date
    end_date:date
    tickers:list[str]
    horizons:list[str]=Field(default_factory=lambda: DEFAULT_HORIZONS.copy())
    max_rows_per_horizon:int=Field(default=30000,ge=5000,le=80000)
    min_train_support:int=Field(default=100,ge=20,le=10000)
    min_validation_support:int=Field(default=40,ge=10,le=5000)

class CoachPayload(BaseModel):
    message:str
    context:dict=Field(default_factory=dict)


class GPUNewsPayload(BaseModel):
    start_date:date
    end_date:date
    tickers:list[str]
    force:bool=False
    include_professional_media:bool=True
    include_company_official:bool=True
    include_sec:bool=True
    include_newswires:bool=True

class VectorBuildPayload(BaseModel):
    start_date:date
    end_date:date
    tickers:list[str]
    horizon_minutes:int=Field(default=30,ge=15,le=240)
    force:bool=False

class DeepRelationshipPayload(BaseModel):
    start_date:date
    end_date:date
    tickers:list[str]
    horizons:list[str]=Field(default_factory=lambda:["D1","D2","D3","W1"])
    targets:list[str]=Field(default_factory=lambda:[
        "future_call_extrinsic_return","future_put_extrinsic_return","future_stock_up"
    ])
    max_rows_per_horizon:int=Field(default=30000,ge=5000,le=100000)
    min_support:int=Field(default=80,ge=30,le=5000)

class FusionTrainPayload(BaseModel):
    start_date:date
    end_date:date
    tickers:list[str]
    horizons:list[str]=Field(default_factory=lambda:["D1","D2","D3","W1"])
    max_rows_per_horizon:int=Field(default=30000,ge=5000,le=80000)

class AlphaResearchPayload(BaseModel):
    start_date:date
    end_date:date
    tickers:list[str]
    horizons:list[str]=Field(default_factory=lambda:["D1","D2","D3","W1"])
    min_train_sessions:int=Field(default=80,ge=40,le=1000)
    test_sessions:int=Field(default=20,ge=5,le=250)
    n_splits:int=Field(default=5,ge=1,le=20)
    holdout_fraction:float=Field(default=.15,ge=.10,le=.35)
    commission_per_contract_side:float=Field(default=.65,ge=0,le=20)
    slippage_bps_per_side:float=Field(default=5.0,ge=0,le=500)
    fallback_roundtrip_spread_pct:float=Field(default=.08,ge=0,le=1)
    safety_margin_return:float=Field(default=.005,ge=0,le=.5)

class PaperSignalPayload(BaseModel):
    candidate:dict
    portfolio_state:dict
    model_id:str|None=None
    experiment_id:str|None=None
    persist:bool=True

class PaperSettlementPayload(BaseModel):
    signal_id:str
    exit_price:float=Field(ge=0)
    actual_cost:float=Field(default=0,ge=0)

class MonitoringPayload(BaseModel):
    records:list[dict]
    feature_reference:dict=Field(default_factory=dict)
    feature_columns:list[str]=Field(default_factory=list)

class AgentCfg(BaseModel):
    auto_enabled:bool=False
    interval_minutes:int=60
    default_start:date|None=None
    tickers:list[str]=[]

def _hash_password(password:str,salt_hex:str|None=None):
    salt=os.urandom(16) if not salt_hex else bytes.fromhex(salt_hex)
    digest=hashlib.pbkdf2_hmac("sha256",password.encode("utf-8"),salt,240000)
    return salt.hex(),digest.hex()

def _verify_password(password:str)->bool:
    salt=get_setting("ADMIN_SALT")
    expected=get_setting("ADMIN_HASH")
    if not salt or not expected:
        return False
    _,actual=_hash_password(password,salt)
    return hmac.compare_digest(actual,expected)

def _resolved_storage(location,endpoint=None,region=None):
    loc=(location or "").lower()
    if loc=="hel1":
        return "https://hel1.your-objectstorage.com","hel1"
    if loc=="nbg1":
        return "https://nbg1.your-objectstorage.com","nbg1"
    if not endpoint or not region:
        raise HTTPException(400,"Vul endpoint en region in bij 'custom'.")
    return endpoint.strip(),region.strip()

@app.on_event("startup")
def startup():
    init_db()
    db=SessionLocal()
    try:
        for a in ["stocks","options","news","google_finance","rich_news","option_trades","open_interest","earnings","sentiment","features","events","macro","pairs","vectors","relationships","deep_relationships","fusion_model","alpha_research","inventory"]:
            if not db.get(AgentConfig,a):
                db.add(AgentConfig(agent=a))

        # Legacy news pipeline is intentionally retired for OptionEdge 20.0.
        legacy_jobs=db.scalars(select(AgentJob).where(
            AgentJob.agent.in_(["news","google_finance","rich_news","sentiment"]),
            AgentJob.status.in_(["queued","running"])
        )).all()
        for j in legacy_jobs:
            j.status="stopped"
            j.finished_at=datetime.now(timezone.utc)
            j.message="Gestopt bij OptionEdge-upgrade: legacy nieuws is uitgesloten van het nieuwe model."
        db.commit()

        # On an existing configured installation, automatically build the first
        # read-only Hetzner inventory after upgrading to OptionEdge 20.0.
        if setup_complete() and not db.get(AppSetting,"LATEST_INVENTORY_AT"):
            active=db.scalar(select(AgentJob).where(
                AgentJob.agent=="inventory",
                AgentJob.status.in_(["queued","running"])
            ).order_by(desc(AgentJob.id)))
            if not active:
                db.add(AgentJob(
                    agent="inventory",mode="startup_scan",status="queued",
                    start_date=None,end_date=None,tickers=[],
                    message="Eerste directe Hetzner-inventarisatie na upgrade.",
                    payload={}
                ))
                db.commit()
    finally:
        db.close()

@app.get("/health")
def health():
    return {"ok":True,"version":"20.0","setup_complete":setup_complete()}

@app.get("/setup/status")
def setup_status():
    s=get_settings(mask_secrets=True)
    return {
        "configured":setup_complete(),
        "values":{
            "storage_endpoint":s.get("HETZNER_S3_ENDPOINT",""),
            "storage_region":s.get("HETZNER_S3_REGION",""),
            "storage_bucket":s.get("HETZNER_S3_BUCKET",""),
            "storage_access_key":s.get("HETZNER_S3_ACCESS_KEY",""),
            "storage_secret_key":s.get("HETZNER_S3_SECRET_KEY",""),
            "massive_api_key":s.get("MASSIVE_API_KEY",""),
            "massive_flat_access_key":s.get("MASSIVE_FLAT_ACCESS_KEY",""),
            "massive_flat_secret_key":s.get("MASSIVE_FLAT_SECRET_KEY",""),
            "fmp_api_key":s.get("FMP_API_KEY",""),
            "sec_user_agent":s.get("SEC_USER_AGENT",""),
            "import_chunk_rows":int(s.get("IMPORT_CHUNK_ROWS",300000) or 300000),
        }
    }

@app.post("/setup/save")
def setup_save(p:SetupPayload):
    if setup_complete():
        raise HTTPException(409,"OptionEdge is al ingesteld. Gebruik Instellingen om gegevens te wijzigen.")
    endpoint,region=_resolved_storage(p.storage_location,p.storage_endpoint,p.storage_region)
    salt,digest=_hash_password(p.admin_password)
    save_settings({
        "HETZNER_S3_ENDPOINT":endpoint,
        "HETZNER_S3_REGION":region,
        "HETZNER_S3_BUCKET":p.storage_bucket.strip(),
        "HETZNER_S3_ACCESS_KEY":p.storage_access_key.strip(),
        "HETZNER_S3_SECRET_KEY":p.storage_secret_key.strip(),
        "MASSIVE_API_KEY":p.massive_api_key.strip(),
        "MASSIVE_FLAT_ENDPOINT":"https://files.massive.com",
        "MASSIVE_FLAT_BUCKET":"flatfiles",
        "MASSIVE_FLAT_ACCESS_KEY":p.massive_flat_access_key.strip(),
        "MASSIVE_FLAT_SECRET_KEY":p.massive_flat_secret_key.strip(),
        "FMP_API_KEY":p.fmp_api_key.strip(),
        "SEC_USER_AGENT":p.sec_user_agent.strip(),
        "IMPORT_CHUNK_ROWS":str(p.import_chunk_rows),
        "ADMIN_SALT":salt,
        "ADMIN_HASH":digest,
    })
    return {"ok":True,"configured":setup_complete()}

@app.post("/setup/test")
def setup_test():
    if not setup_complete():
        raise HTTPException(400,"Sla eerst alle instellingen op.")
    result={}
    try:
        test_storage()
        result["hetzner"]={"ok":True,"message":"Object Storage bereikbaar."}
    except Exception as e:
        result["hetzner"]={"ok":False,"message":str(e)}
    try:
        MassiveREST().test_connection()
        result["massive_rest"]={"ok":True,"message":"Massive REST bereikbaar."}
    except Exception as e:
        result["massive_rest"]={"ok":False,"message":str(e)}
    try:
        test_flat_files()
        result["massive_flat"]={"ok":True,"message":"Massive Flat Files bereikbaar."}
    except Exception as e:
        result["massive_flat"]={"ok":False,"message":str(e)}
    try:
        SecEdgarClient().ticker_map()
        result["sec_edgar"]={"ok":True,"message":"SEC EDGAR Company Facts bereikbaar."}
    except Exception as e:
        result["sec_edgar"]={"ok":False,"message":str(e)}
    try:
        FmpEarningsClient().fetch_ticker("NVDA")
        result["fmp"]={"ok":True,"message":"FMP earnings bereikbaar (NVDA-test)."}
    except Exception as e:
        result["fmp"]={"ok":False,"message":str(e)}
    return {"ok":all(x["ok"] for x in result.values()),"checks":result}

@app.post("/auth/login")
def login(p:LoginPayload):
    return {"ok":_verify_password(p.password)}

@app.put("/settings")
def update_settings(p:SettingsUpdate):
    current=get_settings(mask_secrets=False)
    vals={}
    location=(p.storage_location or "").lower() if p.storage_location else None
    if location:
        endpoint,region=_resolved_storage(location,p.storage_endpoint,p.storage_region)
        vals["HETZNER_S3_ENDPOINT"]=endpoint
        vals["HETZNER_S3_REGION"]=region
    elif p.storage_endpoint:
        vals["HETZNER_S3_ENDPOINT"]=p.storage_endpoint.strip()
    if p.storage_region:
        vals["HETZNER_S3_REGION"]=p.storage_region.strip()
    if p.storage_bucket:
        vals["HETZNER_S3_BUCKET"]=p.storage_bucket.strip()
    if p.storage_access_key:
        vals["HETZNER_S3_ACCESS_KEY"]=p.storage_access_key.strip()
    if p.storage_secret_key:
        vals["HETZNER_S3_SECRET_KEY"]=p.storage_secret_key.strip()
    if p.massive_api_key:
        vals["MASSIVE_API_KEY"]=p.massive_api_key.strip()
    if p.massive_flat_access_key:
        vals["MASSIVE_FLAT_ACCESS_KEY"]=p.massive_flat_access_key.strip()
    if p.massive_flat_secret_key:
        vals["MASSIVE_FLAT_SECRET_KEY"]=p.massive_flat_secret_key.strip()
    if p.fmp_api_key:
        vals["FMP_API_KEY"]=p.fmp_api_key.strip()
    if p.sec_user_agent:
        vals["SEC_USER_AGENT"]=p.sec_user_agent.strip()
    if p.import_chunk_rows:
        vals["IMPORT_CHUNK_ROWS"]=str(p.import_chunk_rows)
    if vals:
        save_settings(vals)
    return {"ok":True}

@app.get("/agents/config")
def get_configs(db:Session=Depends(db_session)):
    return [{"agent":x.agent,"auto_enabled":x.auto_enabled,
             "interval_minutes":x.interval_minutes,
             "default_start":x.default_start,"tickers":x.tickers}
            for x in db.scalars(select(AgentConfig)).all()]

@app.put("/agents/{agent}/config")
def put_config(agent:str,p:AgentCfg,db:Session=Depends(db_session)):
    if not setup_complete():
        raise HTTPException(400,"Voltooi eerst de OptionEdge-installatie in de website.")
    x=db.get(AgentConfig,agent) or AgentConfig(agent=agent)
    x.auto_enabled=p.auto_enabled
    x.interval_minutes=p.interval_minutes
    x.default_start=p.default_start
    x.tickers=[t.strip().upper() for t in p.tickers if t.strip()]
    db.add(x);db.commit()
    return {"ok":True}

@app.post("/gpu-news/start")
def gpu_news_start(p:GPUNewsPayload):
    command_id=str(uuid.uuid4())
    cmd={
        "command_id":command_id,
        "action":"start",
        "created_at":datetime.now(timezone.utc).isoformat(),
        "start_date":str(p.start_date),
        "end_date":str(p.end_date),
        "tickers":[t.strip().upper() for t in p.tickers if t.strip()],
        "force":bool(p.force),
        "pipeline_version":"20.0",
        "embedding_model":"BAAI/bge-m3",
        "embedding_dimensions":1024,
        "llm_profile":"auto-20gb",
        "sources":{
            "professional_media":p.include_professional_media,
            "company_official":p.include_company_official,
            "sec":p.include_sec,
            "newswires":p.include_newswires,
        }
    }
    put_json(f"{GPU_LLM_STATUS_ROOT}/control.json",cmd)
    return {"ok":True,"command_id":command_id,"message":"Opdracht voor de 20 GB GPU-nieuwsagent opgeslagen in Hetzner."}

@app.post("/gpu-news/stop")
def gpu_news_stop():
    command_id=str(uuid.uuid4())
    put_json(f"{GPU_LLM_STATUS_ROOT}/control.json",{
        "command_id":command_id,
        "action":"stop",
        "created_at":datetime.now(timezone.utc).isoformat(),
    })
    return {"ok":True,"command_id":command_id}

@app.get("/gpu-news/status")
def gpu_news_status():
    status=None
    control=None
    try:
        status=get_json(f"{GPU_LLM_STATUS_ROOT}/status.json")
    except Exception:
        pass
    try:
        control=get_json(f"{GPU_LLM_STATUS_ROOT}/control.json")
    except Exception:
        pass
    # Backwards-compatible read-only fallback while the new GPU agent has not
    # written its first heartbeat yet.
    if status is None:
        try:
            status=get_json(f"{PRO_NEWS_STATUS_ROOT}/status.json")
            if status:
                status["legacy_pipeline"]=True
        except Exception:
            pass
    return {"status":status,"control":control}

@app.get("/research/design")
def research_design():
    return design_payload()


@app.get("/earnings/status")
def earnings_status():
    try:
        run=latest_earnings_run()
        read_error=None
    except Exception as exc:
        run=None;read_error=f"{type(exc).__name__}: {exc}"
    rows=(run or {}).get("ticker_status") or []
    errors=list((run or {}).get("errors") or [])
    if read_error:errors.append(read_error)
    return {
        "configured":{
            "sec_user_agent":bool(get_setting("SEC_USER_AGENT","") and "@" in get_setting("SEC_USER_AGENT","")),
            "fmp_api_key":bool(get_setting("FMP_API_KEY","")),
            "hetzner_storage":setup_complete(),
        },
        "last_successful_sec_import":(run or {}).get("last_successful_sec_import"),
        "last_successful_fmp_import":(run or {}).get("last_successful_fmp_import"),
        "earnings_events":int((run or {}).get("earnings_events",0) or 0),
        "tickers_with_quarterly_facts":int((run or {}).get("tickers_with_quarterly_facts",0) or 0),
        "errors":errors,"ticker_status":rows,
        "status_key":(run or {}).get("status_key"),
        "append_only":True,"legacy_data_preserved":True,
    }


@app.get("/earnings/data/{ticker}")
def earnings_data(ticker:str,start_date:date|None=None,end_date:date|None=None,limit:int=500):
    ticker=ticker.strip().upper();limit=max(1,min(5000,int(limit)))
    try:
        frame=load_processed_earnings(ticker)
    except Exception as exc:
        return {"ticker":ticker,"rows":[],"count":0,"error":f"{type(exc).__name__}: {exc}"}
    if not frame.empty:
        relevant=pd.to_datetime(frame.get("date"),errors="coerce").dt.date
        if start_date:frame=frame[relevant>=start_date]
        if end_date:
            relevant=pd.to_datetime(frame.get("date"),errors="coerce").dt.date
            frame=frame[relevant<=end_date]
    frame=frame.tail(limit).replace([float("inf"),float("-inf")],pd.NA)
    rows=json.loads(frame.to_json(orient="records",date_format="iso")) if not frame.empty else []
    return {"ticker":ticker,"rows":rows,"count":len(rows),"point_in_time_enforced":True}

@app.get("/data/readiness")
def data_readiness(tickers:str="",end_date:date|None=None):
    try:
        snapshot=get_json("market-data/v3/inventory/latest.json")
    except Exception:
        snapshot={}
    wanted=[x.strip().upper() for x in tickers.split(",") if x.strip()] or DEFAULT_TICKERS
    return readiness(snapshot,wanted,end_date)

@app.get("/project/status")
def project_status(db:Session=Depends(db_session)):
    try:
        snap=get_json("market-data/v3/inventory/latest.json")
    except Exception:
        snap={}
    ds=snap.get("datasets") or {}

    def has_dataset(name):
        return int((ds.get(name) or {}).get("objects",0) or 0)>0

    # Clean v4 stages are intentionally separate from rejected legacy news/features.
    model_setting=db.get(AppSetting,"LATEST_MULTI_HORIZON_MODEL_KEY")
    fusion_setting=db.get(AppSetting,"LATEST_FUSION_MODEL_KEY")
    model_ready=False
    model_version=None
    if model_setting and model_setting.value:
        try:
            import pickle
            from core.storage import get_bytes
            bundle=pickle.loads(get_bytes(model_setting.value))
            model_version=str(bundle.get("version",""))
            model_ready=(model_version in {"15.0","20.0"} and model_setting.value.startswith("market-data/v4/"))
        except Exception:
            pass
    fusion_ready=bool(
        fusion_setting and fusion_setting.value
        and fusion_setting.value.startswith(f"{FUSION_MODEL_ROOT}/")
        and exists(fusion_setting.value)
    )
    alpha_setting=db.get(AppSetting,"LATEST_ALPHA_RESEARCH_KEY")
    alpha_ready=bool(alpha_setting and alpha_setting.value and exists(alpha_setting.value))
    earnings_ready=has_dataset("earnings_processed")
    try:paper_state=paper_trading_status()
    except Exception:paper_state={"settled_trades":0}

    gpu=None
    try:
        gpu=get_json(f"{GPU_LLM_STATUS_ROOT}/status.json")
    except Exception:
        pass

    steps=[
        {"id":"market","label":"Brondata markt","done":has_dataset("stocks") and has_dataset("options"),
         "detail":"Aandelen- en option minute data aanwezig in Hetzner."},
        {"id":"trades","label":"Option trades","done":has_dataset("option_trades"),
         "detail":"Transactiedata voor activiteit/volume."},
        {"id":"professional_news","label":"Professioneel nieuws","done":has_dataset("professional_news"),
         "detail":"Nieuwe, schone nieuwslaag uit professionele/officiële bronnen. Legacy nieuws telt niet mee."},
        {"id":"news_language_model","label":"Nieuws-LLM + 1024D chunks","done":has_dataset("news_embeddings") and has_dataset("news_events"),
         "detail":"Volledige chunks, BGE-M3 embeddings en gestructureerde eventextractie op de 20 GB GPU."},
        {"id":"earnings","label":"SEC/FMP earningslaag","done":earnings_ready,
         "detail":"Officiële SEC-actuals en point-in-time FMP-consensus, append-only in Hetzner."},
        {"id":"clean_features","label":"Clean Feature Store","done":has_dataset("clean_features"),
         "detail":"Nieuwe features gebouwd zonder het afgekeurde legacy nieuws."},
        {"id":"events","label":"Event Engine","done":has_dataset("events"),
         "detail":"Prijs-, volatiliteits-, volume-, earnings- en professionele nieuwsevents samengevat."},
        {"id":"pairs","label":"Call/Put Pair Store","done":has_dataset("clean_pairs"),
         "detail":"Call/put-paren en theoretische expectation gaps."},
        {"id":"vectors","label":"384D dagelijkse fusievectoren","done":has_dataset("daily_vectors"),
         "detail":"Eén point-in-time vector per ticker-handelsdag, met ruwe 1024D nieuwschunks bewaard."},
        {"id":"deep_relationships","label":"Diepe verbandanalyse","done":has_dataset("deep_relationships"),
         "detail":"Dagelijkse discovery/confirmation met FDR en within-ticker controle."},
        {"id":"fusion_model","label":"Fusie- en dimensiemodel","done":fusion_ready,
         "detail":"128/256/384/512/768 validatie, nieuws-ablation, kalibratie en finale hold-out."},
        {"id":"model","label":"FINAL expectation model","done":model_ready,
         "detail":"Multi-horizon gekalibreerde paper-winstkansen en expectation-gap model."},
        {"id":"alpha_research","label":"Alpha Research Engine","done":alpha_ready,
         "detail":"Purged walk-forward, untouched holdout, kosten, ablaties, regimes en economische metrics."},
        {"id":"paper_trading","label":"Paper trading","done":int(paper_state.get("settled_trades",0) or 0)>=50,
         "detail":"Dezelfde signal/risk-pipeline, minimaal 50 afgewikkelde papertrades vóór live-kandidatuur."},
        {"id":"live_monitoring","label":"Live monitoring & kill switch","done":False,
         "detail":"Codepad aanwezig; operationele bewijsperiode, driftbaseline en brokeruitvoering ontbreken nog."},
        {"id":"limited_live","label":"Beperkte live trading","done":False,
         "detail":"Hard uitgeschakeld totdat alle kosten-, paper-, risico- en infrastructuurgates slagen."},
    ]
    done=sum(1 for x in steps if x["done"])
    pct=done/len(steps) if steps else 0
    next_steps=[x["label"] for x in steps if not x["done"]][:3]
    recent=db.scalars(
        select(AgentJob).order_by(desc(AgentJob.id)).limit(10)
    ).all()
    recent_jobs=[{
        "id":j.id,"agent":j.agent,"status":j.status,
        "message":j.message,"created_at":j.created_at,
        "finished_at":j.finished_at,
    } for j in recent]
    return {
        "progress":pct,
        "done_steps":done,
        "total_steps":len(steps),
        "steps":steps,
        "next_steps":next_steps,
        "model_version":model_version,
        "fusion_model_ready":fusion_ready,
        "fusion_model_key":fusion_setting.value if fusion_setting else None,
        "gpu_status":gpu,
        "legacy_news_excluded":True,
        "inventory_generated_at":snap.get("generated_at"),
        "recent_jobs":recent_jobs,
        "data_readiness":readiness(snap,DEFAULT_TICKERS),
    }

@app.post("/inventory/scan")
def inventory_scan(db:Session=Depends(db_session)):
    active=db.scalar(select(AgentJob).where(
        AgentJob.agent=="inventory",
        AgentJob.status.in_(["queued","running"])
    ).order_by(desc(AgentJob.id)))
    if active:
        return {"ok":True,"job_id":active.id,"status":active.status}

    job=AgentJob(
        agent="inventory",mode="read_only",status="queued",
        start_date=None,end_date=None,tickers=[],
        message="Hetzner Object Storage inventarisatie in wachtrij.",
        payload={}
    )
    db.add(job);db.commit();db.refresh(job)
    return {"ok":True,"job_id":job.id,"status":"queued"}

@app.get("/inventory/latest")
def inventory_latest(db:Session=Depends(db_session)):
    try:
        snapshot=get_json("market-data/v3/inventory/latest.json")
    except Exception:
        snapshot=None

    job=db.scalar(select(AgentJob).where(
        AgentJob.agent=="inventory"
    ).order_by(desc(AgentJob.id)))

    job_data=None
    if job:
        job_data={
            "id":job.id,
            "status":job.status,
            "progress":(
                float(job.current_fraction)
                if job.status=="running"
                else 1.0 if job.status=="done"
                else 0.0
            ),
            "current_item":job.current_item,
            "message":job.message,
            "created_at":job.created_at,
            "started_at":job.started_at,
            "finished_at":job.finished_at,
        }

    return {
        "snapshot":snapshot,
        "scan_job":job_data,
        "dataset_labels":DATASET_LABELS,
    }

@app.get("/inventory/coverage")
def inventory_coverage(
    agent:str,
    start_date:date,
    end_date:date,
    tickers:str="",
):
    try:
        snapshot=get_json("market-data/v3/inventory/latest.json")
    except Exception:
        return {"coverage":None}

    ticker_list=[x.strip().upper() for x in tickers.split(",") if x.strip()]
    return {
        "coverage":coverage_for_agent(
            snapshot,agent,ticker_list,start_date,end_date
        )
    }

def _queue_research_job(db:Session,agent:str,start_date:date,end_date:date,
                        tickers:list[str],payload:dict,mode:str="manual"):
    active=db.scalar(select(AgentJob).where(
        AgentJob.agent==agent,
        AgentJob.status.in_(["queued","running"])
    ).order_by(desc(AgentJob.id)))
    if active:
        raise HTTPException(409,f"{agent} draait al: job {active.id}")
    clean=[x.strip().upper() for x in tickers if x.strip()]
    if not clean:
        raise HTTPException(400,"Kies minimaal één ticker.")
    weekdays=sum(1 for x in pd.date_range(start_date,end_date,freq="D") if x.weekday()<5)
    job=AgentJob(
        agent=agent,mode=mode,status="queued",start_date=start_date,end_date=end_date,
        tickers=clean,total_units=max(1,weekdays*len(clean)),completed_units=0,
        current_fraction=0.0,message=f"{agent} staat in de wachtrij.",payload=payload,
    )
    db.add(job);db.commit();db.refresh(job)
    return job

def _latest_readiness(tickers:list[str],end_date:date):
    try:
        snapshot=get_json("market-data/v3/inventory/latest.json")
    except Exception:
        snapshot={}
    return readiness(snapshot,tickers,end_date)

@app.post("/vectors/build")
def build_vectors(p:VectorBuildPayload,db:Session=Depends(db_session)):
    gate=_latest_readiness(p.tickers,p.end_date)
    if not gate.get("vector_ready"):
        labels=", ".join(x.get("label",x.get("id","")) for x in gate.get("vector_blockers",[]))
        raise HTTPException(409,f"Vectorbouw geblokkeerd door ontbrekende bronlagen: {labels}. Scan/importeer data eerst.")
    job=_queue_research_job(
        db,"vectors",p.start_date,p.end_date,p.tickers,
        {"horizon_minutes":p.horizon_minutes,"force":p.force},
        mode="force" if p.force else "missing_only",
    )
    return {"ok":True,"job_id":job.id,"status":job.status}

@app.post("/deep-relationships/run")
def run_deep_relationships_endpoint(p:DeepRelationshipPayload,db:Session=Depends(db_session)):
    allowed_targets={"future_call_extrinsic_return","future_put_extrinsic_return","future_stock_up"}
    targets=[x for x in p.targets if x in allowed_targets]
    if not targets:
        raise HTTPException(400,"Kies minimaal één geldig target.")
    horizons=[x for x in ["D1","D2","D3","W1"] if x in p.horizons]
    if not horizons:
        raise HTTPException(400,"Kies minimaal één dagelijkse horizon: D1, D2, D3 of W1.")
    gate=_latest_readiness(p.tickers,p.end_date)
    if not gate.get("training_ready"):
        labels=", ".join(x.get("label",x.get("id","")) for x in gate.get("blockers",[]))
        raise HTTPException(409,f"Diepe analyse geblokkeerd door ontbrekende data: {labels}.")
    job=_queue_research_job(
        db,"deep_relationships",p.start_date,p.end_date,p.tickers,
        {"horizons":horizons,"targets":targets,
         "max_rows_per_horizon":p.max_rows_per_horizon,"min_support":p.min_support},
    )
    return {"ok":True,"job_id":job.id,"status":job.status}

@app.get("/deep-relationships/latest")
def latest_deep_relationships(db:Session=Depends(db_session)):
    setting=db.get(AppSetting,"LATEST_DEEP_RELATIONSHIP_KEY")
    if not setting or not setting.value:
        return {"bundle":None,"key":None}
    try:
        return {"bundle":get_json(setting.value),"key":setting.value}
    except Exception as exc:
        raise HTTPException(500,f"Diepe verbandanalyse kon niet worden gelezen: {exc}")

@app.post("/fusion-model/train")
def train_fusion_endpoint(p:FusionTrainPayload,db:Session=Depends(db_session)):
    active=db.scalar(select(AgentJob).where(
        AgentJob.agent=="fusion_model",AgentJob.status.in_(["queued","running"])
    ).order_by(desc(AgentJob.id)))
    if active:
        raise HTTPException(409,f"Fusietraining draait al: job {active.id}")
    tickers=[x.strip().upper() for x in p.tickers if x.strip()]
    horizons=[x for x in ["D1","D2","D3","W1"] if x in p.horizons]
    if not tickers or not horizons:
        raise HTTPException(400,"Kies minimaal één ticker en dagelijkse horizon.")
    gate=_latest_readiness(tickers,p.end_date)
    if not gate.get("training_ready"):
        labels=", ".join(x.get("label",x.get("id","")) for x in gate.get("blockers",[]))
        raise HTTPException(409,f"Fusietraining geblokkeerd door ontbrekende data: {labels}.")
    run=ModelRun(
        status="queued",stage="FUSION20",start_date=p.start_date,end_date=p.end_date,
        tickers=tickers,target="daily_option_expectation",horizon_minutes=0,
        max_rows=p.max_rows_per_horizon,progress=0.0,
        message="384D fusietraining in wachtrij.",metrics={},feature_sets={},
    )
    db.add(run);db.commit();db.refresh(run)
    job=_queue_research_job(
        db,"fusion_model",p.start_date,p.end_date,tickers,
        {"horizons":horizons,"max_rows_per_horizon":p.max_rows_per_horizon,
         "model_run_id":run.id},
    )
    return {"ok":True,"job_id":job.id,"model_run_id":run.id,"status":job.status}

@app.get("/fusion-model/latest")
def latest_fusion_model(db:Session=Depends(db_session)):
    run=db.scalar(select(ModelRun).where(ModelRun.stage=="FUSION20").order_by(desc(ModelRun.id)))
    if not run:
        return {"run":None}
    return {"run":{
        "id":run.id,"status":run.status,"stage":run.stage,"start_date":run.start_date,
        "end_date":run.end_date,"tickers":run.tickers,"progress":run.progress,
        "message":run.message,"metrics":run.metrics,"artifact_key":run.artifact_key,
        "created_at":run.created_at,"started_at":run.started_at,"finished_at":run.finished_at,
    }}


@app.post("/alpha-research/run")
def run_alpha_research(p:AlphaResearchPayload,db:Session=Depends(db_session)):
    tickers=[x.strip().upper() for x in p.tickers if x.strip()]
    horizons=[x for x in ["D1","D2","D3","W1"] if x in p.horizons]
    if not tickers or not horizons:
        raise HTTPException(400,"Kies minimaal één ticker en dagelijkse horizon.")
    active=db.scalar(select(AgentJob).where(
        AgentJob.agent=="alpha_research",AgentJob.status.in_(["queued","running"])
    ).order_by(desc(AgentJob.id)))
    if active:raise HTTPException(409,f"Alpha Research draait al: job {active.id}")
    run=ModelRun(
        status="queued",stage="ALPHA_RESEARCH",start_date=p.start_date,end_date=p.end_date,
        tickers=tickers,target="cost_aware_option_return",horizon_minutes=0,max_rows=0,
        progress=0.0,message="Alpha Research in wachtrij.",metrics={},feature_sets={},
    )
    db.add(run);db.commit();db.refresh(run)
    payload={
        "model_run_id":run.id,"horizons":horizons,
        "min_train_sessions":p.min_train_sessions,"test_sessions":p.test_sessions,
        "n_splits":p.n_splits,"holdout_fraction":p.holdout_fraction,
        "commission_per_contract_side":p.commission_per_contract_side,
        "slippage_bps_per_side":p.slippage_bps_per_side,
        "fallback_roundtrip_spread_pct":p.fallback_roundtrip_spread_pct,
        "safety_margin_return":p.safety_margin_return,
    }
    job=_queue_research_job(db,"alpha_research",p.start_date,p.end_date,tickers,payload)
    gate=_latest_readiness(tickers,p.end_date)
    return {
        "ok":True,"job_id":job.id,"model_run_id":run.id,"status":job.status,
        "all_imported_eligible_data":True,
        "readiness_warning":None if gate.get("training_ready") else gate.get("blockers"),
    }


@app.get("/alpha-research/latest")
def alpha_research_latest(db:Session=Depends(db_session)):
    run=db.scalar(select(ModelRun).where(ModelRun.stage=="ALPHA_RESEARCH").order_by(desc(ModelRun.id)))
    stored=None;error=None
    try:stored=latest_research_suite()
    except Exception as exc:error=f"{type(exc).__name__}: {exc}"
    return {
        "run":None if not run else {
            "id":run.id,"status":run.status,"progress":run.progress,"message":run.message,
            "metrics":run.metrics,"artifact_key":run.artifact_key,
            "created_at":run.created_at,"started_at":run.started_at,"finished_at":run.finished_at,
        },
        "suite":stored,"error":error,
        "live_trading_authorized":False,
    }


@app.get("/model/scorecard")
def model_scorecard(db:Session=Depends(db_session)):
    """Always-readable immutable performance history, independent of current imports."""
    runs=db.scalars(select(ModelRun).order_by(desc(ModelRun.id)).limit(50)).all()
    history=[{
        "id":row.id,"stage":row.stage,"status":row.status,"start_date":row.start_date,
        "end_date":row.end_date,"tickers":row.tickers,"target":row.target,
        "message":row.message,"metrics":row.metrics,"artifact_key":row.artifact_key,
        "data_policy":(row.metrics or {}).get("data_policy"),
        "data_usage":(row.metrics or {}).get("data_usage"),
        "created_at":row.created_at,"finished_at":row.finished_at,
    } for row in runs]
    try:
        alpha=latest_research_suite();alpha_error=None
    except Exception as exc:
        alpha=None;alpha_error=f"{type(exc).__name__}: {exc}"
    alpha_summary=[]
    for horizon,suite in ((alpha or {}).get("results") or {}).items():
        for experiment in suite.get("experiments") or []:
            holdout=experiment.get("holdout_metrics") or {}
            if not holdout:
                continue
            alpha_summary.append({
                "horizon":horizon,"experiment_id":experiment.get("experiment_id"),
                "model":experiment.get("model"),"feature_set":experiment.get("feature_set"),
                "target":experiment.get("target"),"EV_net":holdout.get("EV_net"),
                "expected_value_per_trade":holdout.get("expected_value_per_trade"),
                "net_return":holdout.get("net_return"),"sharpe":holdout.get("sharpe"),
                "sortino":holdout.get("sortino"),"max_drawdown":holdout.get("maximum_drawdown"),
                "profit_factor":holdout.get("profit_factor"),"trades":holdout.get("number_of_trades"),
                "costs_provisional":holdout.get("costs_provisional"),
                "leakage_valid":experiment.get("leakage_valid"),
            })
    # Keep the compact performance view available when the latest immutable
    # Object Storage manifest is temporarily unreadable. The full audit trail
    # still lives append-only in Object Storage; this is a display fallback.
    if not alpha_summary:
        latest_alpha=next((row for row in runs if row.stage=="ALPHA_RESEARCH"),None)
        if latest_alpha:
            alpha_summary=list((latest_alpha.metrics or {}).get("scorecard") or [])
    return {
        "available":bool(history or alpha_summary),"alpha_scorecard":alpha_summary,
        "model_run_history":history,"alpha_manifest_key":(alpha or {}).get("manifest_key"),
        "alpha_error":alpha_error,
        "message":(
            "Prestaties blijven zichtbaar uit opgeslagen modelruns; een onvolledige nieuwe import wist niets."
            if history or alpha_summary else
            "Nog geen modelrun. De scorecard blijft beschikbaar en toont resultaten zodra een run is opgeslagen."
        ),
    }


@app.get("/trading/safety-status")
def trading_safety_status():
    try:paper=paper_trading_status()
    except Exception as exc:paper={"error":f"{type(exc).__name__}: {exc}","settled_trades":0}
    try:experiment=latest_research_suite()
    except Exception:experiment=None
    try:monitoring=latest_monitoring_snapshot()
    except Exception as exc:monitoring={"error":f"{type(exc).__name__}: {exc}"}
    candidate=None
    for suite in ((experiment or {}).get("results") or {}).values():
        stable=set(suite.get("stable_candidates") or [])
        candidate=next((x for x in suite.get("experiments") or [] if x.get("experiment_id") in stable),candidate)
    status=default_safety_status()
    status["paper_trading"]=paper
    status["monitoring"]=monitoring
    status["live"]=live_promotion_gate(
        candidate,paper,data_healthy=bool(monitoring and not monitoring.get("kill_switch")),
        broker_adapter_configured=False,
    )
    return status


@app.post("/paper-trading/evaluate")
def paper_evaluate(p:PaperSignalPayload):
    try:
        candidate=TradeCandidate(**p.candidate)
        state=PortfolioState(**p.portfolio_state)
        signal=create_paper_signal(
            candidate,state,RiskPolicy(),model_id=p.model_id,experiment_id=p.experiment_id,
        )
        key=save_paper_signal(signal) if p.persist else None
        return {"ok":True,"signal":signal,"key":key}
    except (TypeError,ValueError) as exc:
        raise HTTPException(400,str(exc))


@app.post("/paper-trading/settle")
def paper_settle(p:PaperSettlementPayload):
    try:
        key,record=settle_paper_signal(p.signal_id,p.exit_price,p.actual_cost)
        return {"ok":True,"settlement":record,"key":key}
    except ValueError as exc:
        raise HTTPException(400,str(exc))


@app.get("/paper-trading/status")
def paper_status():
    try:return paper_trading_status()
    except Exception as exc:return {"signals":0,"settled_trades":0,"error":f"{type(exc).__name__}: {exc}"}


@app.post("/monitoring/evaluate")
def monitoring_evaluate(p:MonitoringPayload):
    snapshot=monitor_performance(pd.DataFrame(p.records),p.feature_reference,p.feature_columns)
    try:key=save_monitoring_snapshot(snapshot)
    except Exception as exc:raise HTTPException(500,f"Monitoring kon niet persistent worden opgeslagen: {exc}")
    return {"ok":True,"snapshot":snapshot,"key":key}


@app.get("/monitoring/latest")
def monitoring_latest():
    try:return {"snapshot":latest_monitoring_snapshot()}
    except Exception as exc:return {"snapshot":None,"error":f"{type(exc).__name__}: {exc}"}

def _latest_fusion_inputs(ticker:str):
    """Return the newest date with both a v20 vector and a clean Pair Store."""
    from pathlib import PurePosixPath
    from core.config import DAILY_VECTOR_ROOT
    from core.storage import list_keys

    wanted=ticker.strip().upper()
    vector_keys=sorted(
        (key for key in list_keys(f"{DAILY_VECTOR_ROOT}/{wanted}/") if key.endswith(".npz")),
        reverse=True,
    )
    for vector_key in vector_keys:
        try:
            day=date.fromisoformat(PurePosixPath(vector_key).stem)
        except ValueError:
            continue
        for source_horizon in [30,15,60,120,240]:
            pair_key=f"{PAIR_ROOT}/h{source_horizon}/{wanted}/{day:%Y}/{day:%m}/{day}.csv.gz"
            if exists(pair_key):
                return {
                    "ticker":wanted,"day":day,"vector_key":vector_key,
                    "pair_key":pair_key,"pair_horizon_minutes":source_horizon,
                }
    return None

@app.get("/fusion-model/status")
def fusion_model_status(ticker:str|None=None,db:Session=Depends(db_session)):
    setting=db.get(AppSetting,"LATEST_FUSION_MODEL_KEY")
    out={"ready":False,"version":"20.0","model_key":setting.value if setting else None,
         "ticker":ticker.strip().upper() if ticker else None,"as_of_date":None,
         "latest_vector_key":None,"pair_key":None,"reason":None}
    if not setting or not setting.value or not exists(setting.value):
        out["reason"]="Nog geen v20-fusiemodel. Bouw eerst dagvectoren en train daarna Fusie 2.0."
        return out
    if ticker:
        selected=_latest_fusion_inputs(ticker)
        if not selected:
            out["reason"]=(
                f"Geen handelsdag met zowel een v20-vector als een schone Pair Store voor "
                f"{ticker.strip().upper()}. Bouw of importeer eerst de ontbrekende laag."
            )
            return out
        out["as_of_date"]=str(selected["day"])
        out["latest_vector_key"]=selected["vector_key"]
        out["pair_key"]=selected["pair_key"]
    out["ready"]=True
    out["reason"]=(
        "V20-fusiemodel, dagelijkse vector en Pair Store voor dezelfde handelsdag zijn beschikbaar."
        if ticker else "V20-fusiemodel is beschikbaar."
    )
    return out

@app.get("/fusion-model/predict/{ticker}")
def predict_fusion_endpoint(
    ticker:str,horizons:str="D1,D2,D3,W1",top_n:int=20,
    db:Session=Depends(db_session),
):
    import pickle
    import numpy as np
    from core.storage import get_bytes,get_df
    from core.vector_fusion import load_raw_vector
    from core.fusion_modeling import predict_fusion_rows

    ticker=ticker.strip().upper();top_n=max(1,min(100,int(top_n)))
    setting=db.get(AppSetting,"LATEST_FUSION_MODEL_KEY")
    if not setting or not setting.value or not exists(setting.value):
        raise HTTPException(409,"Nog geen v20-fusiemodel. Train eerst Fusie 2.0.")
    bundle=pickle.loads(get_bytes(setting.value))
    if str(bundle.get("version",""))!="20.0":
        raise HTTPException(409,"Het nieuwste fusieartifact is niet versie 20.0; train opnieuw.")
    requested=[x.strip().upper() for x in horizons.split(",") if x.strip().upper() in ["D1","D2","D3","W1"]]
    trained=bundle.get("models") or {}
    requested=[x for x in requested if x in trained]
    if not requested:
        raise HTTPException(400,"Geen gevraagde dagelijkse horizon is in het fusiemodel getraind.")

    selected=_latest_fusion_inputs(ticker)
    if not selected:
        raise HTTPException(409,f"Geen datum met zowel een v20-vector als Pair Store gevonden voor {ticker}.")
    day=selected["day"];vector_key=selected["vector_key"];pair_key=selected["pair_key"]
    raw=load_raw_vector(ticker,day);pair_frame=get_df(pair_key)
    if pair_frame.empty:
        raise HTTPException(404,"De nieuwste bijpassende Pair Store is leeg.")

    results={};summary=[]
    wanted_columns=[
        "minute","expiry","strike","close_stock","call_ticker","put_ticker",
        "call_extrinsic","put_extrinsic","bs_call_extrinsic","bs_put_extrinsic",
        "call_expectation_gap","put_expectation_gap","put_call_parity_residual",
        "call_positive_probability","put_positive_probability",
        "expected_call_extrinsic_return","expected_put_extrinsic_return",
        "model_expected_call_extrinsic","model_expected_put_extrinsic",
        "call_gross_edge","put_gross_edge","best_side","best_positive_probability",
        "best_gross_edge","fusion_dimensions","expected_cost_return","net_expected_edge",
        "decision","decision_reasons","costs_provisional","risk_contracts",
    ]
    for code in requested:
        prediction=predict_fusion_rows(pair_frame,raw,trained[code])
        prediction=prediction.head(top_n).copy()
        metrics=((bundle.get("metrics") or {}).get("horizons") or {}).get(code,{})
        decisions=[]
        for _,row in prediction.iterrows():
            side=str(row.get("best_side") or "CALL").lower()
            price_value=pd.to_numeric(row.get(f"{side}_close_option",row.get(f"{side}_extrinsic")),errors="coerce")
            expected_value=pd.to_numeric(row.get(f"expected_{side}_extrinsic_return"),errors="coerce")
            age_value=pd.to_numeric(row.get("source_staleness_minutes"),errors="coerce")
            price=float(price_value) if pd.notna(price_value) and np.isfinite(price_value) else 0.0
            expected=float(expected_value) if pd.notna(expected_value) and np.isfinite(expected_value) else 0.0
            cost=.08+.001+(1.36/max(price*100,1e-9))
            return_metrics=metrics.get(f"final_holdout_{side}_return") or {}
            uncertainty=float(return_metrics.get("rmse") or .35)
            candidate=TradeCandidate(
                ticker=ticker,side=side.upper(),expected_return=expected,
                expected_cost_return=cost,uncertainty=uncertainty,option_price=max(price,.01),
                relative_spread=None,
                open_interest=int(pd.to_numeric(row.get(f"{side}_oi_open_interest"),errors="coerce"))
                    if pd.notna(pd.to_numeric(row.get(f"{side}_oi_open_interest"),errors="coerce")) else None,
                daily_volume=int(pd.to_numeric(row.get(f"{side}_option_volume"),errors="coerce"))
                    if pd.notna(pd.to_numeric(row.get(f"{side}_option_volume"),errors="coerce")) else None,
                data_age_minutes=float(age_value) if pd.notna(age_value) and np.isfinite(age_value) else 0.0,
            )
            decision=evaluate_trade(candidate,PortfolioState(equity=100000)).as_dict()
            decisions.append({
                "expected_cost_return":cost,"net_expected_edge":expected-cost,
                "decision":decision["decision"],"decision_reasons":decision["reasons"],
                "costs_provisional":True,"risk_contracts":decision["contracts"],
            })
        if decisions:
            prediction=pd.concat([prediction.reset_index(drop=True),pd.DataFrame(decisions)],axis=1)
        top=prediction.replace([np.inf,-np.inf],np.nan)
        rows=json.loads(top[[x for x in wanted_columns if x in top]].to_json(orient="records",date_format="iso"))
        results[code]={"label":horizon_label(code),"rows":rows,"validation":metrics}
        if rows:
            first=rows[0]
            summary.append({
                "horizon_code":code,"horizon":horizon_label(code),
                "best_side":first.get("best_side"),
                "positive_probability":first.get("best_positive_probability"),
                "gross_edge":first.get("best_gross_edge"),
                "expected_cost_return":first.get("expected_cost_return"),
                "net_expected_edge":first.get("net_expected_edge"),
                "decision":first.get("decision"),
                "decision_reasons":first.get("decision_reasons"),
                "strike":first.get("strike"),"expiry":first.get("expiry"),
                "selected_dimensions":first.get("fusion_dimensions"),
                "call_brier":(metrics.get("final_holdout_call") or {}).get("brier"),
                "put_brier":(metrics.get("final_holdout_put") or {}).get("brier"),
                "news_brier_gain_call":metrics.get("news_brier_gain_call"),
                "news_brier_gain_put":metrics.get("news_brier_gain_put"),
            })
    return {
        "ticker":ticker,"as_of_date":str(day),"vector_key":vector_key,"pair_key":pair_key,
        "model_key":setting.value,"requested_horizons":requested,
        "summary":summary,"horizons":results,
        "quotes_included":False,"net_profit_claim_allowed":False,
        "warning":"Kosten zijn voorlopig zolang historische bid/ask, diepte en echte fills ontbreken; NO_TRADE is dan fail-closed.",
    }

@app.post("/agents/{agent}/preview")
def preview_agent(agent:str,p:StartAgent):
    force=(p.mode=="force")
    try:
        plan=build_plan(
            agent,
            [t.strip().upper() for t in p.tickers if t.strip()],
            p.start_date,p.end_date,
            force=force,
            snapshot=load_snapshot()
        )
        return {"plan":plan}
    except Exception as exc:
        raise HTTPException(400,str(exc))

@app.post("/agents/{agent}/start")
def start(agent:str,p:StartAgent,db:Session=Depends(db_session)):
    if not setup_complete():
        raise HTTPException(400,"Voltooi eerst de OptionEdge-installatie in de website.")
    active=db.scalar(select(AgentJob).where(
        AgentJob.agent==agent,
        AgentJob.status.in_(["queued","running"])
    ).order_by(desc(AgentJob.id)))
    if active:
        raise HTTPException(409,f"Agent draait al: job {active.id}")
    ticks=[t.strip().upper() for t in p.tickers if t.strip()]
    force=(p.mode=="force")
    try:
        plan=build_plan(
            agent,ticks,p.start_date,p.end_date,
            force=force,snapshot=load_snapshot()
        )
    except Exception as exc:
        raise HTTPException(400,f"Importplan kon niet worden gemaakt: {exc}")

    payload=dict(p.payload or {})
    payload["import_plan"]=plan

    if plan.get("remaining_units",1)==0 and agent not in ["relationships","pair_model","model"]:
        return {
            "job_id":None,
            "status":"already_complete",
            "plan":plan,
            "message":"Alles in deze selectie staat al in Hetzner; er wordt niets opnieuw gedownload."
        }

    job=AgentJob(
        agent=agent,mode=p.mode,status="queued",
        start_date=p.start_date,end_date=p.end_date,
        tickers=ticks,
        total_units=int(plan.get("expected_units",0) or 0),
        completed_units=int(plan.get("already_present_units",0) or 0),
        message=(
            f"Gepland: {int(plan.get('already_present_units',0)):,} al aanwezig, "
            f"{int(plan.get('remaining_units',0)):,} nog te verwerken."
        ),
        payload=payload
    )
    db.add(job);db.commit();db.refresh(job)
    # Geen Redis/Celery meer: de dedicated OptionEdge worker leest queued jobs
    # rechtstreeks uit PostgreSQL. Hierdoor kan job-start niet falen door een broker.
    job.celery_task_id=""
    db.commit()
    return {"job_id":job.id,"status":"queued","plan":plan}

@app.post("/agents/{agent}/stop")
def stop(agent:str,db:Session=Depends(db_session)):
    job=db.scalar(select(AgentJob).where(
        AgentJob.agent==agent,
        AgentJob.status.in_(["queued","running"])
    ).order_by(desc(AgentJob.id)))
    if job:
        job.stop_requested=True
        db.commit()
    return {"ok":True}

@app.get("/agents/{agent}/latest")
def latest(agent:str,db:Session=Depends(db_session)):
    job=db.scalar(select(AgentJob).where(AgentJob.agent==agent).order_by(desc(AgentJob.id)))
    if not job:
        return {"job":None,"companies":[]}
    rows=db.scalars(select(AgentCompanyProgress).where(
        AgentCompanyProgress.job_id==job.id
    ).order_by(AgentCompanyProgress.ticker)).all()
    return {
        "job":{
            "id":job.id,"status":job.status,"mode":job.mode,
            "total_units":job.total_units,"completed_units":job.completed_units,
            "remaining_units":max(0,int(job.total_units or 0)-int(job.completed_units or 0)),
            "already_present_units":int((((job.payload or {}).get("import_plan") or {}).get("already_present_units",0)) or 0),
            "estimated_remaining_bytes":((job.payload or {}).get("import_plan") or {}).get("estimated_remaining_bytes"),
            "stored_dataset_bytes":((job.payload or {}).get("import_plan") or {}).get("stored_dataset_bytes"),
            "progress":(
                1.0 if job.status=="done" else job.current_fraction
                if job.agent in ["relationships","deep_relationships","model","pair_model","fusion_model","inventory"]
                else min(1.0,
                    sum((x.completed_units or 0)+(x.current_fraction or 0) for x in rows)
                    / max(1,sum((x.total_units or 0) for x in rows))
                ) if rows else min(1.0,(job.completed_units or 0)/max(1,job.total_units or 0))
            ),
            "current_item":job.current_item,"message":job.message,
            "start_date":job.start_date,"end_date":job.end_date
        },
        "companies":[
            {
                "ticker":x.ticker,"status":x.status,
                "completed_units":x.completed_units,"total_units":x.total_units,
                "remaining_units":max(0,int(x.total_units or 0)-int(x.completed_units or 0)),
                "progress":min(1.0,(x.completed_units+x.current_fraction)/max(1,x.total_units)),
                "current_item":x.current_item,"message":x.message
            } for x in rows
        ]
    }

def _job_progress_dict(job,db):
    rows=db.scalars(select(AgentCompanyProgress).where(
        AgentCompanyProgress.job_id==job.id
    )).all()
    total=max(0,int(job.total_units or 0))
    done=max(0,int(job.completed_units or 0))
    frac=float(job.current_fraction or 0)
    progress=(
        (1.0 if job.status=="done" else frac) if job.agent in ["relationships","deep_relationships","model","pair_model","fusion_model","inventory"]
        else min(
            1.0,
            sum((x.completed_units or 0)+(x.current_fraction or 0) for x in rows)
            / max(1,sum((x.total_units or 0) for x in rows))
        ) if rows else min(1.0,done/max(1,total))
    )
    plan=(job.payload or {}).get("import_plan") or {}
    return {
        "id":job.id,"agent":job.agent,"status":job.status,"mode":job.mode,
        "progress":progress,"completed_units":done,"total_units":total,
        "remaining_units":max(0,total-done),
        "already_present_units":int(plan.get("already_present_units",0) or 0),
        "estimated_remaining_bytes":plan.get("estimated_remaining_bytes"),
        "stored_dataset_bytes":plan.get("stored_dataset_bytes"),
        "current_fraction":frac,"current_item":job.current_item,
        "message":job.message,"start_date":job.start_date,"end_date":job.end_date,
        "tickers":job.tickers,
        "companies":[{
            "ticker":x.ticker,"status":x.status,
            "completed_units":x.completed_units,"total_units":x.total_units,
            "remaining_units":max(0,int(x.total_units or 0)-int(x.completed_units or 0)),
            "progress":min(1.0,(x.completed_units+x.current_fraction)/max(1,x.total_units)),
            "current_item":x.current_item,"message":x.message
        } for x in rows]
    }

@app.get("/agents/live")
def agents_live(db:Session=Depends(db_session)):
    jobs=db.scalars(select(AgentJob).where(
        AgentJob.status.in_(["queued","running"])
    ).order_by(AgentJob.id.asc())).all()
    recent=db.scalars(select(AgentJob).where(
        AgentJob.status.in_(["done","warning","error","stopped"])
    ).order_by(desc(AgentJob.id)).limit(12)).all()
    return {
        "active":[_job_progress_dict(j,db) for j in jobs],
        "recent":[_job_progress_dict(j,db) for j in recent]
    }

@app.get("/dashboard")
def dashboard(db:Session=Depends(db_session)):
    out={}
    for a in ["stocks","options","news","google_finance","rich_news","option_trades","open_interest","earnings","sentiment","features","events","macro","pairs","vectors","relationships","deep_relationships","fusion_model","inventory"]:
        j=db.scalar(select(AgentJob).where(AgentJob.agent==a).order_by(desc(AgentJob.id)))
        out[a]=None if not j else {
            "status":j.status,
            "progress":(j.current_fraction if j.agent in ["relationships","deep_relationships","model","pair_model","fusion_model","inventory"] else (j.completed_units+j.current_fraction*max(1,len(j.tickers)))/max(1,j.total_units)),
            "current_item":j.current_item,"message":j.message,"job_id":j.id
        }
    return {"agents":out}





@app.post("/relationships/run")
def run_relationship_discovery(p:RelationshipPayload,db:Session=Depends(db_session)):
    ticks=[t.strip().upper() for t in p.tickers if t.strip()]
    horizons=[h for h in DEFAULT_HORIZONS if h in p.horizons]
    if not ticks:
        raise HTTPException(400,"Kies minimaal één ticker.")
    if not horizons:
        raise HTTPException(400,"Kies minimaal één voorspellingstermijn.")

    job=AgentJob(
        agent="relationships",mode="manual",status="queued",
        start_date=p.start_date,end_date=p.end_date,tickers=ticks,
        payload={
            "horizons":horizons,
            "max_rows_per_horizon":p.max_rows_per_horizon,
            "min_train_support":p.min_train_support,
            "min_validation_support":p.min_validation_support,
        },
        message="Multi-horizon Relationship Discovery in wachtrij."
    )
    db.add(job);db.commit();db.refresh(job)
    return {"ok":True,"job_id":job.id,"horizons":horizons}

@app.get("/relationships/latest")
def latest_relationships(horizon:str|None=None,db:Session=Depends(db_session)):
    from core.storage import get_json
    code=None
    if horizon:
        h=str(horizon).upper()
        legacy={"15":"M15","30":"M30","60":"H1","120":"H2","240":"H4"}
        code=legacy.get(h,h)
        if code not in HORIZONS:
            raise HTTPException(400,f"Onbekende horizon: {horizon}")
    x=db.get(AppSetting,f"LATEST_RELATIONSHIP_KEY_{code}") if code else None
    x=x or db.get(AppSetting,"LATEST_RELATIONSHIP_KEY")
    if not x or not x.value:
        return {"bundle":None}
    return {
        "horizon":code,
        "label":horizon_label(code) if code else None,
        "key":x.value,
        "bundle":get_json(x.value)
    }

@app.get("/relationships/multi/latest")
def latest_multi_relationships(db:Session=Depends(db_session)):
    from core.storage import get_json
    x=db.get(AppSetting,"LATEST_MULTI_RELATIONSHIP_KEY")
    if not x or not x.value:
        return {"manifest":None}
    return {"key":x.value,"manifest":get_json(x.value)}

@app.post("/pair-model/train")
def train_pair_model_endpoint(p:PairTrainPayload,db:Session=Depends(db_session)):
    ticks=[t.strip().upper() for t in p.tickers if t.strip()]
    horizons=[h for h in DEFAULT_HORIZONS if h in p.horizons]
    if not ticks:
        raise HTTPException(400,"Kies minimaal één ticker.")
    if not horizons:
        raise HTTPException(400,"Kies minimaal één voorspellingstermijn.")

    mr=ModelRun(
        status="queued",stage="MULTI_FINAL",
        start_date=p.start_date,end_date=p.end_date,
        tickers=ticks,target="multi_horizon_call_put_extrinsic",
        horizon_minutes=0,max_rows=p.max_rows_per_horizon,
        progress=0.0,message="Wachtrij"
    )
    db.add(mr);db.commit();db.refresh(mr)

    job=AgentJob(
        agent="pair_model",mode="manual",status="queued",
        start_date=p.start_date,end_date=p.end_date,tickers=ticks,
        payload={
            "model_run_id":mr.id,
            "horizons":horizons,
            "max_rows_per_horizon":p.max_rows_per_horizon,
        },
        message="Handmatige multi-horizon FINAL training in wachtrij."
    )
    db.add(job);db.commit();db.refresh(job)
    return {
        "ok":True,
        "model_run_id":mr.id,
        "job_id":job.id,
        "horizons":horizons,
    }

@app.get("/pair-model/latest")
def latest_pair_model(db:Session=Depends(db_session)):
    x=db.scalar(
        select(ModelRun).where(
            ModelRun.stage.in_(["MULTI_FINAL","FINAL_GAP","PAIR_GAP"])
        ).order_by(desc(ModelRun.id))
    )
    if not x:
        return {"run":None}
    return {"run":{
        "id":x.id,"status":x.status,
        "start_date":x.start_date,"end_date":x.end_date,
        "tickers":x.tickers,
        "max_rows_per_horizon":x.max_rows,
        "progress":x.progress,"message":x.message,
        "metrics":x.metrics,"artifact_key":x.artifact_key,
        "created_at":x.created_at,"started_at":x.started_at,"finished_at":x.finished_at
    }}

@app.get("/final-model/status")
def final_model_status(
    ticker:str|None=None,
    horizons:str="ALL",
    db:Session=Depends(db_session)
):
    import pickle
    from core.storage import get_bytes,list_keys
    from core.config import PAIR_ROOT

    model_setting=(
        db.get(AppSetting,"LATEST_MULTI_HORIZON_MODEL_KEY")
        or db.get(AppSetting,"LATEST_PAIR_MODEL_KEY")
    )

    out={
        "ready":False,
        "api_version":"15.0",
        "model_key":model_setting.value if model_setting else None,
        "model_version":None,
        "trained_horizons":[],
        "pair_store_available":None,
        "pair_store_source":None,
        "reason":None,
    }

    if not model_setting or not model_setting.value:
        out["reason"]="Nog geen FINAL model getraind. Ga naar Training en train het multi-horizon FINAL model."
        return out

    try:
        bundle=pickle.loads(get_bytes(model_setting.value))
    except Exception as exc:
        out["reason"]=f"FINAL-modelartifact kan niet worden gelezen: {type(exc).__name__}: {exc}"
        return out

    out["model_version"]=str(bundle.get("version",""))
    out["trained_horizons"]=[
        h for h in (bundle.get("horizons") or [])
        if h in HORIZONS
    ]

    if horizons.upper()=="ALL":
        requested=out["trained_horizons"]
    else:
        requested=[]
        legacy={"15":"M15","30":"M30","60":"H1","120":"H2","240":"H4","1D":"D1","2D":"D2","3D":"D3","1W":"W1"}
        for raw in horizons.split(","):
            h=legacy.get(raw.strip().upper(),raw.strip().upper())
            if h in HORIZONS and h not in requested:
                requested.append(h)
    out["requested_horizons"]=requested
    out["missing_horizons"]=[
        h for h in requested if h not in out["trained_horizons"]
    ]

    if out["model_version"]!="15.0":
        out["reason"]=(
            "Het opgeslagen FINAL model is nog geen multi-horizon model. "
            "Train het model één keer opnieuw via Training."
        )
        return out

    if out["missing_horizons"]:
        out["reason"]=(
            "De gekozen termijn(en) zijn nog niet getraind: "+
            ", ".join(horizon_label(h) for h in out["missing_horizons"])+
            ". Train het multi-horizon FINAL model opnieuw met deze termijnen geselecteerd."
        )
        return out

    if ticker:
        ticker=ticker.strip().upper()
        source_key=None
        for old_h in [30,15,60,120,240]:
            keys=[
                k for k in list_keys(f"{PAIR_ROOT}/h{old_h}/{ticker}/")
                if k.endswith(".csv.gz")
            ]
            if keys:
                source_key=sorted(keys)[-1]
                break
        out["pair_store_available"]=bool(source_key)
        out["pair_store_source"]=source_key
        if not source_key:
            out["reason"]=(
                f"Geen Call/Put Pair Feature Store gevonden voor {ticker}. "
                "Start eerst Agents → Call/Put Pair Feature Store."
            )
            return out

    out["ready"]=True
    out["reason"]="FINAL model en Pair Store zijn beschikbaar."
    return out

@app.get("/final-model/predict/{ticker}")
def final_model_predict(
    ticker:str,
    horizons:str="ALL",
    top_n:int=20,
    threshold:float=0.60,
    db:Session=Depends(db_session)
):
    import pickle
    import numpy as np
    from core.storage import get_bytes,list_keys,get_df
    from core.config import PAIR_ROOT
    from core.pair_modeling import predict_multi_horizon,explain_row

    ticker=ticker.strip().upper()
    top_n=max(1,min(100,int(top_n)))
    threshold=max(0.50,min(0.95,float(threshold)))

    model_setting=(
        db.get(AppSetting,"LATEST_MULTI_HORIZON_MODEL_KEY")
        or db.get(AppSetting,"LATEST_PAIR_MODEL_KEY")
    )
    if not model_setting or not model_setting.value:
        raise HTTPException(
            409,
            "Nog geen multi-horizon FINAL model getraind. "
            "Ga naar Training → Train multi-horizon FINAL model."
        )

    bundle=pickle.loads(get_bytes(model_setting.value))
    if str(bundle.get("version",""))!="15.0":
        raise HTTPException(
            409,
            "Het nieuwste model is nog geen OptionEdge 15.0 clean multi-horizon model. "
            "Train het FINAL model één keer opnieuw."
        )

    trained=[h for h in bundle.get("horizons",[]) if h in HORIZONS]
    if horizons.upper()=="ALL":
        requested=trained
    else:
        requested=[]
        for raw in horizons.split(","):
            h=raw.strip().upper()
            legacy={"15":"M15","30":"M30","60":"H1","120":"H2","240":"H4","1D":"D1","2D":"D2","3D":"D3","1W":"W1"}
            h=legacy.get(h,h)
            if h in trained and h not in requested:
                requested.append(h)

    if not requested:
        raise HTTPException(400,"Geen van de gevraagde termijnen is in dit model getraind.")

    # Current pair features are horizon-independent. Reuse the newest available
    # pair file, preferring h30 so no historical data needs rebuilding.
    source_key=None
    for old_h in [30,15,60,120,240]:
        keys=[k for k in list_keys(f"{PAIR_ROOT}/h{old_h}/{ticker}/") if k.endswith(".csv.gz")]
        if keys:
            source_key=sorted(keys)[-1]
            break
    if not source_key:
        raise HTTPException(
            409,
            f"Geen Call/Put Pair Feature Store voor {ticker}. "
            "Draai eerst Agents → Call/Put Pair Feature Store."
        )

    frame=get_df(source_key)
    if frame.empty:
        raise HTTPException(404,"Het nieuwste call/put-pairbestand is leeg.")

    predictions=predict_multi_horizon(
        frame,bundle,
        horizons=requested,
        threshold=threshold
    )

    wanted=[
        "minute","expiry","strike","close_stock",
        "call_ticker","put_ticker",
        "call_extrinsic","put_extrinsic",
        "bs_call_extrinsic","bs_put_extrinsic",
        "call_expectation_gap","put_expectation_gap",
        "model_expected_call_extrinsic","model_expected_put_extrinsic",
        "call_gross_edge","put_gross_edge",
        "call_gross_extrinsic_edge_per_contract","put_gross_extrinsic_edge_per_contract",
        "call_gross_edge_pct","put_gross_edge_pct",
        "prob_call_up_put_down","prob_call_down_put_up",
        "prob_both_up","prob_both_down",
        "prob_call_up","prob_put_up","prob_stock_up",
        "prob_call_gross_profit","prob_put_gross_profit",
        "best_gross_profit_probability","best_profit_side",
        "profit_probability_above_50",
        "best_profit_historical_hit_rate","best_profit_evidence_n",
        "best_profit_ci95_low","best_profit_ci95_high",
        "call_profit_historical_hit_rate","call_profit_evidence_n",
        "call_profit_hit_rate_ci95_low","call_profit_hit_rate_ci95_high",
        "call_profit_historical_mean_return",
        "put_profit_historical_hit_rate","put_profit_evidence_n",
        "put_profit_hit_rate_ci95_low","put_profit_hit_rate_ci95_high",
        "put_profit_historical_mean_return",
        "paper_profit_case","paper_profit_empirically_supported",
        "paper_profit_strict_ci_supported",
        "most_likely_state","most_likely_state_label",
        "most_likely_state_probability",
        "relationship_match_count","relationship_strength",
        "relationship_prior_call_up_put_down",
        "relationship_prior_call_down_put_up",
        "relationship_prior_both_up","relationship_prior_both_down",
        "paper_signal","paper_opportunity_score",
        "days_to_earnings","sentiment_60m","news_count_60m",
        "earnings_news_60m","guidance_news_60m","analyst_news_60m",
    ]

    response_horizons={}
    summary=[]
    relationships=bundle.get("relationships") or {}

    for code in requested:
        pred=predictions.get(code)
        if pred is None or pred.empty:
            response_horizons[code]={
                "label":horizon_label(code),
                "rows":[],
                "top_relationships":[],
            }
            continue

        top=pred.head(top_n).copy()
        out=top[[c for c in wanted if c in top.columns]].copy()
        out=out.replace([np.inf,-np.inf],np.nan)
        rows=json.loads(out.to_json(orient="records",date_format="iso"))

        rels=explain_row(
            pred.iloc[0],
            relationships.get(code) or {},
            limit=8
        )

        r=pred.iloc[0]
        summary.append({
            "horizon_code":code,
            "horizon":horizon_label(code),
            "strike":r.get("strike"),
            "expiry":str(r.get("expiry")),
            "most_likely_state":r.get("most_likely_state_label"),
            "state_probability":r.get("most_likely_state_probability"),
            "prob_call_up":r.get("prob_call_up"),
            "prob_put_up":r.get("prob_put_up"),
            "prob_stock_up":r.get("prob_stock_up"),
            "paper_profit_probability":r.get("best_gross_profit_probability"),
            "paper_profit_side":r.get("best_profit_side"),
            "historical_hit_rate":r.get("best_profit_historical_hit_rate"),
            "evidence_n":r.get("best_profit_evidence_n"),
            "ci95_low":r.get("best_profit_ci95_low"),
            "ci95_high":r.get("best_profit_ci95_high"),
            "paper_profit_case":r.get("paper_profit_case"),
            "empirically_supported":r.get("paper_profit_empirically_supported"),
            "strict_ci_supported":r.get("paper_profit_strict_ci_supported"),
            "call_edge_pct":r.get("call_gross_edge_pct"),
            "put_edge_pct":r.get("put_gross_edge_pct"),
            "paper_signal":r.get("paper_signal"),
            "opportunity_score":r.get("paper_opportunity_score"),
        })

        hmetrics=((bundle.get("metrics") or {}).get("horizons") or {}).get(code,{})
        response_horizons[code]={
            "label":horizon_label(code),
            "rows":rows,
            "top_relationships":rels,
            "validation":{
                "final_holdout_rows":hmetrics.get("final_holdout_rows"),
                "call_profit_brier":hmetrics.get("call_profit_probability_brier"),
                "put_profit_brier":hmetrics.get("put_profit_probability_brier"),
                "call_profit_accuracy":hmetrics.get("call_profit_probability_accuracy"),
                "put_profit_accuracy":hmetrics.get("put_profit_probability_accuracy"),
                "call_base_rate":hmetrics.get("call_profit_base_rate"),
                "put_base_rate":hmetrics.get("put_profit_base_rate"),
            }
        }

    return {
        "ticker":ticker,
        "trained_horizons":trained,
        "requested_horizons":requested,
        "threshold":threshold,
        "model_key":model_setting.value,
        "source_pair_file":source_key,
        "quotes_included":False,
        "edge_type":"gross_paper_edge_before_spread_slippage",
        "summary":summary,
        "horizons":response_horizons,
    }

@app.post("/models/train")
def train_model(p:TrainModelPayload,db:Session=Depends(db_session)):
    """
    Always manual. Calling this endpoint never depends on auto-agent settings.
    Multiple requests may be queued; the dedicated model worker handles them.
    """
    tickers=[t.strip().upper() for t in p.tickers if t.strip()]
    if not tickers:
        raise HTTPException(400,"Kies minimaal één ticker.")
    mr=ModelRun(
        status="queued",stage="ALL",start_date=p.start_date,end_date=p.end_date,
        tickers=tickers,target=p.target,horizon_minutes=p.horizon_minutes,
        max_rows=p.max_rows,progress=0.0,message="Wachtrij"
    )
    db.add(mr);db.commit();db.refresh(mr)
    job=AgentJob(
        agent="model",mode="manual",status="queued",
        start_date=p.start_date,end_date=p.end_date,tickers=tickers,
        payload={
            "model_run_id":mr.id,"target":p.target,
            "horizon_minutes":p.horizon_minutes,"max_rows":p.max_rows
            ,"use_all_available":p.use_all_available
        },
        message="Handmatige training in wachtrij."
    )
    db.add(job);db.commit();db.refresh(job)
    return {"ok":True,"model_run_id":mr.id,"job_id":job.id}

@app.get("/models/latest")
def latest_model(db:Session=Depends(db_session)):
    x=db.scalar(select(ModelRun).order_by(desc(ModelRun.id)))
    if not x:return {"run":None}
    return {"run":{
        "id":x.id,"status":x.status,"stage":x.stage,"start_date":x.start_date,
        "end_date":x.end_date,"tickers":x.tickers,"target":x.target,
        "horizon_minutes":x.horizon_minutes,"max_rows":x.max_rows,
        "progress":x.progress,"message":x.message,"metrics":x.metrics,
        "artifact_key":x.artifact_key,"created_at":x.created_at,
        "started_at":x.started_at,"finished_at":x.finished_at
    }}

@app.get("/data/schema")
def data_schema():
    return {
        "active":[
            "stocks","option_minute_aggregates","option_trades","volume",
            "open_interest_snapshots","earnings",
            "professional_official_gpu_news","news_full_text_chunks","bge_m3_1024d_embeddings",
            "qwen_structured_event_extraction","news_source_quality","event_classification",
            "technical_price_events","risk_free_rate","clean_feature_store","call_put_pairs",
            "daily_384d_fusion_vectors","dimension_benchmark_128_256_384_512_768",
            "relationship_discovery","theoretical_expectation_gap","multi_horizon_final_probability_model"
            ,"sec_company_facts","fmp_point_in_time_earnings","purged_walk_forward_alpha_research",
            "economic_metrics","feature_ablation","regime_breakdowns","no_trade_decision",
            "independent_risk_manager","paper_trading","kill_switch","drift_monitoring"
        ],
        "excluded_from_new_model":[
            "legacy_market_data_news","legacy_google_finance_news",
            "legacy_news_enrichment","legacy_sentiment"
        ],
        "reserved":{
            "quotes":{
                "enabled":False,
                "columns":["bid","ask","midprice","spread","relative_spread","bid_size","ask_size"],
                "note":"Gereserveerd. Quotes kunnen later worden toegevoegd zonder overige data opnieuw te downloaden."
            }
        },
        "models":{
            "MULTI_FINAL":"verwachtingswaarde-model: voorspelt wanneer marktverwachting waarschijnlijk fout zit",
            "RELATIONSHIPS":"horizon-specifieke gevalideerde historische verbanden",
            "CALIBRATION":"aparte kanskalibratie per voorspellingstermijn",
            "EDGE":"verwachte toekomstige extrinsieke waarde minus actuele extrinsieke waarde",
            "QUOTES":"gereserveerd voor spread/slippage-correctie later",
            "FUSION384":"192 nieuws + 96 opties + 48 aandeel + 32 verwachting + 16 kwaliteit",
            "RAW_NEWS":"1024D BGE-M3 per chunk, target-vrij opgeslagen voor fold-specifieke pooling"
            ,"ALPHA":"verwachte opbrengst en onzekerheid, geselecteerd op netto out-of-sample EV",
            "RISK":"aparte positie-, exposure-, liquiditeits-, correlatie- en drawdownlaag",
            "EXECUTION":"live hard uit; eerst dezelfde pipeline in append-only paper trading"
        }
    }

@app.get("/server/status")
def server_status(db:Session=Depends(db_session)):
    from datetime import date,datetime,timezone
    workers={}
    for worker_name in ["MARKET","TRADES","FEATURES","NEWS","MODEL","INVENTORY"]:
        row=db.get(AppSetting,f"WORKER_HEARTBEAT_{worker_name}")
        age=None
        active=False
        if row and row.value:
            try:
                dt=datetime.fromisoformat(row.value)
                age=(datetime.now(timezone.utc)-dt).total_seconds()
                active=age < 30
            except Exception:
                pass
        workers[worker_name.lower()]={
            "active":active,
            "age_seconds":age
        }

    running=db.scalars(
        select(AgentJob).where(AgentJob.status=="running")
    ).all()
    queued=db.scalars(
        select(AgentJob).where(AgentJob.status=="queued")
    ).all()

    return {
        "version":"20.0",
        "setup_complete":setup_complete(),
        "workers":workers,
        "running_jobs":len(running),
        "queued_jobs":len(queued)
    }


@app.get("/agents/diagnostics")
def agent_diagnostics(db:Session=Depends(db_session)):
    from datetime import date,datetime,timezone
    result={"database":{"ok":True,"message":"PostgreSQL bereikbaar."}}

    try:
        test_flat_files()
        result["massive_flat"]={"ok":True,"message":"Massive Flat Files bereikbaar."}
    except Exception as e:
        result["massive_flat"]={"ok":False,"message":f"{type(e).__name__}: {e}"}

    try:
        from core.storage import test_connection
        test_connection()
        result["hetzner_storage"]={"ok":True,"message":"Hetzner Object Storage bereikbaar."}
    except Exception as e:
        result["hetzner_storage"]={"ok":False,"message":f"{type(e).__name__}: {e}"}

    for worker_name in ["MARKET","TRADES","FEATURES","NEWS","MODEL","INVENTORY"]:
        row=db.get(AppSetting,f"WORKER_HEARTBEAT_{worker_name}")
        ok=False
        age=None
        if row and row.value:
            try:
                dt=datetime.fromisoformat(row.value)
                age=(datetime.now(timezone.utc)-dt).total_seconds()
                ok=age<30
            except Exception:
                pass
        result[f"worker_{worker_name.lower()}"]={
            "ok":ok,
            "message":(
                f"Worker actief; laatste heartbeat {age:.0f} sec geleden."
                if ok and age is not None
                else "Geen recente heartbeat van deze worker."
            )
        }

    queued=db.scalars(select(AgentJob).where(AgentJob.status=="queued")).all()
    running=db.scalars(select(AgentJob).where(AgentJob.status=="running")).all()
    result["queue"]={
        "ok":True,
        "message":f"{len(queued)} queued, {len(running)} running."
    }

    return {
        "ok":all(v.get("ok",False) for k,v in result.items() if k!="queue"),
        "checks":result
    }


@app.post("/coach/chat")
def coach_chat(p:CoachPayload,db:Session=Depends(db_session)):
    """
    Operational v20 model coach. Only an explicit user instruction can queue a
    bounded, auditable job; the coach never silently rewrites model code.
    """
    msg=(p.message or "").strip()
    low=msg.lower()
    ctx=p.context or {}
    actions=[]
    reply=[]

    latest_fusion=db.scalar(
        select(ModelRun).where(
            ModelRun.stage=="FUSION20",ModelRun.status=="done"
        ).order_by(desc(ModelRun.id))
    )
    latest_legacy=db.scalar(select(ModelRun).order_by(desc(ModelRun.id)))
    latest=latest_fusion or latest_legacy
    metrics=(latest.metrics or {}) if latest else {}
    hm=(metrics.get("horizons") or {}) if isinstance(metrics,dict) else {}

    ticks=[str(x).strip().upper() for x in ctx.get("tickers",[]) if str(x).strip()]
    all_horizons=[h for h in ctx.get("horizons",DEFAULT_HORIZONS) if h in HORIZONS]
    daily_horizons=[h for h in ["D1","D2","D3","W1"] if h in all_horizons]
    try:
        start_day=date.fromisoformat(str(ctx.get("start_date")))
        end_day=date.fromisoformat(str(ctx.get("end_date")))
    except (TypeError,ValueError):
        start_day=end_day=None
    max_rows=max(5000,min(80000,int(ctx.get("max_rows_per_horizon",30000) or 30000)))

    # Set profit-probability threshold.
    m=re.search(r"(?:drempel|grens|minimum|kans).{0,20}?(\d{2})\s*%?",low)
    if m and any(w in low for w in ["drempel","grens","minimum","kans"]):
        pct=max(50,min(90,int(m.group(1))))
        s=db.get(AppSetting,"MODEL_PROFIT_THRESHOLD") or AppSetting(key="MODEL_PROFIT_THRESHOLD")
        s.value=str(pct/100.0);s.encrypted=False;db.add(s);db.commit()
        actions.append({"type":"setting","name":"profit_threshold","value":pct/100.0})
        reply.append(f"De modeldrempel is ingesteld op {pct}%. Nieuwe analyses gebruiken deze grens.")

    # Build vectors only after an explicit instruction and a successful source gate.
    if any(phrase in low for phrase in [
        "bouw vector","bouw de vector","vectoren bouwen","maak dagvector","maak de dagvector",
    ]):
        if not (ticks and start_day and end_day):
            reply.append("Voor vectorbouw ontbreken ticker- of datuminstellingen in de Modelcoach-context.")
        else:
            gate=_latest_readiness(ticks,end_day)
            if not gate.get("vector_ready"):
                labels=", ".join(x.get("label",x.get("id","")) for x in gate.get("vector_blockers",[]))
                reply.append(f"Vectorbouw is niet gestart. Importeer of voltooi eerst: {labels}.")
            else:
                try:
                    job=_queue_research_job(
                        db,"vectors",start_day,end_day,ticks,
                        {"horizon_minutes":30,"force":False},mode="coach",
                    )
                    actions.append({"type":"vectors","job_id":job.id})
                    reply.append(f"De ontbrekende dagelijkse vectoren zijn ingepland als job {job.id}.")
                except HTTPException as exc:
                    reply.append(str(exc.detail))

    # Fusion 2.0 is the default. The old model remains available only when the
    # user explicitly says legacy/intraday/old.
    wants_training=any(phrase in low for phrase in [
        "train opnieuw","hertrain","opnieuw trainen","train het model","train fusie",
        "train fusion","train 384",
    ])
    use_legacy=any(word in low for word in ["legacy","intraday","bestaande model","oude model"])
    if wants_training:
        horizons=all_horizons if use_legacy else daily_horizons
        if not (ticks and start_day and end_day and horizons):
            reply.append("Voor opnieuw trainen ontbreken ticker-, datum- of passende horizoninstellingen in de Modelcoach-context.")
        elif use_legacy:
            active=db.scalar(select(AgentJob).where(
                AgentJob.agent=="pair_model",AgentJob.status.in_(["queued","running"])
            ).order_by(desc(AgentJob.id)))
            if active:
                reply.append(f"Legacytraining draait al als job {active.id}; er is niets dubbel ingepland.")
            else:
                mr=ModelRun(
                    status="queued",stage="MULTI_FINAL",start_date=start_day,end_date=end_day,
                    tickers=ticks,target="multi_horizon_call_put_extrinsic",horizon_minutes=0,
                    max_rows=max_rows,progress=0.0,message="Legacytraining gestart via Modelcoach",
                )
                db.add(mr);db.flush()
                job=AgentJob(
                    agent="pair_model",mode="coach",status="queued",start_date=start_day,
                    end_date=end_day,tickers=ticks,payload={"model_run_id":mr.id,
                    "horizons":horizons,"max_rows_per_horizon":max_rows},
                    message="Expliciete legacytraining via Modelcoach.",
                )
                db.add(job);db.commit();db.refresh(job)
                actions.append({"type":"legacy_training","model_run_id":mr.id,"job_id":job.id})
                reply.append(f"De expliciet gevraagde legacytraining is ingepland als run {mr.id}.")
        else:
            gate=_latest_readiness(ticks,end_day)
            if not gate.get("training_ready"):
                labels=", ".join(x.get("label",x.get("id","")) for x in gate.get("blockers",[]))
                reply.append(f"Fusietraining is niet gestart. De harde datapoort blokkeert nog: {labels}.")
            else:
                active=db.scalar(select(AgentJob).where(
                    AgentJob.agent=="fusion_model",AgentJob.status.in_(["queued","running"])
                ).order_by(desc(AgentJob.id)))
                if active:
                    reply.append(f"Fusietraining draait al als job {active.id}; er is niets dubbel ingepland.")
                else:
                    mr=ModelRun(
                        status="queued",stage="FUSION20",start_date=start_day,end_date=end_day,
                        tickers=ticks,target="daily_option_expectation",horizon_minutes=0,
                        max_rows=max_rows,progress=0.0,message="Fusie 2.0 gestart via Modelcoach",
                        metrics={},feature_sets={},
                    )
                    db.add(mr);db.flush()
                    weekdays=sum(1 for x in pd.date_range(start_day,end_day,freq="D") if x.weekday()<5)
                    job=AgentJob(
                        agent="fusion_model",mode="coach",status="queued",start_date=start_day,
                        end_date=end_day,tickers=ticks,total_units=max(1,weekdays*len(ticks)),
                        completed_units=0,current_fraction=0.0,
                        payload={"model_run_id":mr.id,"horizons":horizons,
                                 "max_rows_per_horizon":max_rows},
                        message="Fusie 2.0-training via Modelcoach in wachtrij.",
                    )
                    db.add(job);db.commit();db.refresh(job)
                    actions.append({"type":"fusion_training","model_run_id":mr.id,"job_id":job.id})
                    reply.append(f"Fusie 2.0 is daadwerkelijk ingepland als run {mr.id} en job {job.id}.")

    # The v20 discovery/confirmation engine is the default relationship action.
    if "relationship" in low or "verband" in low:
        if any(w in low for w in ["opnieuw","zoek","herbereken","onderzoek","draai"]):
            horizons=all_horizons if use_legacy else daily_horizons
            if not (ticks and start_day and end_day and horizons):
                reply.append("Voor verbandanalyse ontbreken ticker-, datum- of horizoninstellingen.")
            elif use_legacy:
                job=AgentJob(
                    agent="relationships",mode="coach",status="queued",
                    start_date=start_day,end_date=end_day,
                    tickers=ticks,payload={
                        "horizons":horizons,
                        "max_rows_per_horizon":max_rows,
                        "min_train_support":100,
                        "min_validation_support":40,
                    },
                    message="Expliciete legacy Relationship Discovery via Modelcoach."
                )
                db.add(job);db.commit();db.refresh(job)
                actions.append({"type":"legacy_relationships","job_id":job.id})
                reply.append(f"De legacy-verbandanalyse is ingepland als job {job.id}.")
            else:
                gate=_latest_readiness(ticks,end_day)
                if not gate.get("training_ready"):
                    labels=", ".join(x.get("label",x.get("id","")) for x in gate.get("blockers",[]))
                    reply.append(f"De diepe verbandanalyse is niet gestart. De datapoort blokkeert nog: {labels}.")
                else:
                    try:
                        job=_queue_research_job(
                            db,"deep_relationships",start_day,end_day,ticks,
                            {"horizons":horizons,"targets":["future_call_extrinsic_return",
                             "future_put_extrinsic_return","future_stock_up"],
                             "max_rows_per_horizon":max_rows,"min_support":80},mode="coach",
                        )
                        actions.append({"type":"deep_relationships","job_id":job.id})
                        reply.append(f"De v20 discovery/confirmation-analyse is ingepland als job {job.id}.")
                    except HTTPException as exc:
                        reply.append(str(exc.detail))

    # Diagnostic answer from real last metrics.
    if not reply:
        if latest_fusion and hm:
            rows=[]
            for code,v in hm.items():
                if not isinstance(v,dict) or v.get("status")!="gereed":
                    continue
                rows.append((
                    code,
                    (v.get("final_holdout_call") or {}).get("brier"),
                    (v.get("final_holdout_put") or {}).get("brier"),
                    v.get("news_brier_gain_call"),v.get("news_brier_gain_put"),
                    v.get("selected_dimensions"),v.get("ticker_days",0),
                ))
            if rows:
                worst=max(rows,key=lambda r:max([x for x in r[1:3] if isinstance(x,(int,float))] or [0]))
                weak_news=[r[0] for r in rows if any(isinstance(x,(int,float)) and x<=0 for x in r[3:5])]
                reply.append(
                    f"Ik heb de echte laatste v20-holdoutmetrics bekeken. De meeste aandacht verdient "
                    f"{horizon_label(worst[0])}; de call/put-Brier is daar "
                    f"{worst[1] if worst[1] is not None else '—'} / {worst[2] if worst[2] is not None else '—'}, "
                    f"met {worst[6]} ticker-dagen en {worst[5]} geselecteerde dimensies."
                )
                if weak_news:
                    reply.append(
                        "De nieuws-ablation geeft nog geen positieve Brier-verbetering voor: "+
                        ", ".join(horizon_label(x) for x in weak_news)+
                        ". Controleer eerst brondekking, first-seen-tijden en eventextractie voordat je het model groter maakt."
                    )
                reply.append(
                    "Expliciete acties zijn: ‘bouw de dagvectoren’, ‘zoek de verbanden opnieuw’, "
                    "‘train Fusie 2.0 opnieuw’ of ‘train het legacy intradaymodel opnieuw’."
                )
            else:
                reply.append("Er zijn nog geen bruikbare v20-holdoutmetrics. Voltooi de datapoort en train daarna Fusie 2.0.")
        else:
            reply.append(
                "Er is nog geen afgeronde Fusie 2.0-run om te beoordelen. Ik kan na een expliciet verzoek "
                "dagvectoren bouwen, de v20-verbandanalyse uitvoeren of Fusie 2.0 trainen; de datapoort blijft actief."
            )

    return {"reply":" ".join(reply),"actions":actions,"latest_model_run_id":latest.id if latest else None}

@app.get("/live/stocks/{ticker}")
def live_stock(ticker:str,day:date):
    try:
        d=MassiveREST().aggregates(ticker.upper(),day,day)
        return {"rows":json.loads(d.to_json(orient="records",date_format="iso"))}
    except Exception as e:
        raise HTTPException(502,str(e))

@app.get("/live/options/{underlying}/chain")
def chain(underlying:str):
    try:
        d=MassiveREST().option_chain(underlying.upper())
        if d.empty:return {"rows":[]}
        s=pd.to_numeric(d["underlying_price"],errors="coerce")
        k=pd.to_numeric(d["strike"],errors="coerce")
        px=pd.to_numeric(d["mid"],errors="coerce")
        typ=d["type"].astype(str).str.lower()
        intr=pd.Series(0.0,index=d.index)
        intr.loc[typ=="call"]=(s-k).clip(lower=0).loc[typ=="call"]
        intr.loc[typ=="put"]=(k-s).clip(lower=0).loc[typ=="put"]
        d["intrinsic"]=intr
        d["expectation_value"]=(px-intr).clip(lower=0)
        d["spread_pct"]=(pd.to_numeric(d["ask"],errors="coerce")-pd.to_numeric(d["bid"],errors="coerce"))/px.replace(0,pd.NA)
        return {"rows":json.loads(d.to_json(orient="records",date_format="iso"))}
    except Exception as e:
        raise HTTPException(502,str(e))

@app.get("/live/options/{underlying}/{contract}")
def option_day(underlying:str,contract:str,day:date):
    try:
        r=MassiveREST()
        o=r.aggregates(contract,day,day)
        s=r.aggregates(underlying.upper(),day,day)
        return {
            "option":json.loads(o.to_json(orient="records",date_format="iso")),
            "stock":json.loads(s.to_json(orient="records",date_format="iso"))
        }
    except Exception as e:
        raise HTTPException(502,str(e))
