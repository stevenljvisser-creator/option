import os
import html
from datetime import date,timedelta
import requests
import pandas as pd
import streamlit as st

API=os.getenv("API_URL","http://api:8000")
APP_NAME="OptionEdge"
APP_VERSION="2.0"
ENGINE_NAME="Multimodal Expectation Intelligence"

DEFAULT=["NVDA","AMD","AVGO","AAPL","MSFT","GOOGL","META","AMZN","TSLA","JPM","BAC","GS","LLY","UNH","XOM","CVX","CAT","BA","WMT","COST","SPY","QQQ"]
HORIZON_LABELS={"M15":"15 min","M30":"30 min","H1":"1 uur","H2":"2 uur","H4":"4 uur","D1":"1 handelsdag","D2":"2 handelsdagen","D3":"3 handelsdagen","W1":"1 handelsweek"}
ALL_HORIZONS=list(HORIZON_LABELS)

AGENTS={
    "stocks":{"label":"Aandelenkoersen","icon":"📈","group":"Marktdata","help":"Historische minuutkoersen van het onderliggende aandeel."},
    "options":{"label":"Optieprijzen","icon":"🧩","group":"Marktdata","help":"Historische option minute bars; basis voor de werkelijk waargenomen extrinsieke waarde."},
    "option_trades":{"label":"Option trades","icon":"🔁","group":"Marktdata","help":"Werkelijke optietransacties, volume en activiteit."},
    "gpu_news":{"label":"Nieuws-LLM GPU","icon":"◉","group":"Nieuws & events","help":"Volledige teksten en revisies verzamelen, chunk-embeddings maken en per ticker gestructureerde events extraheren op Floris' 20-GB-GPU."},
    "earnings":{"label":"Kwartaalcijfers","icon":"📅","group":"Nieuws & events","help":"Earnings en verrassingen rond kwartaalcijfers."},
    "open_interest":{"label":"Open interest / IV / Greeks","icon":"🧮","group":"Optiecontext","help":"Actuele option-chain snapshots waar beschikbaar."},
    "features":{"label":"Clean Feature Store","icon":"🛠️","group":"Afgeleide data","help":"Berekent technische, nieuws-, event- en option-features. Gebruikt uitsluitend de nieuwe professionele nieuwslaag."},
    "events":{"label":"Event Engine","icon":"⚡","group":"Afgeleide data","help":"Vat koers-, volatiliteits-, volume-, earnings- en professionele nieuwsevents samen voor analyse en audit."},
    "macro":{"label":"Risicovrije rente","icon":"💵","group":"Afgeleide data","help":"Rente-input voor de theoretische optiebenchmark."},
    "pairs":{"label":"Call/Put Pair Store","icon":"🔗","group":"Afgeleide data","help":"Koppelt call en put met dezelfde strike/expiratie en berekent expectation gaps."},
}

LEGACY_AGENTS={
    "news":"Legacy nieuws — uitgesloten",
    "google_finance":"Legacy Google Finance — uitgesloten",
    "rich_news":"Legacy news enrichment — uitgesloten",
    "sentiment":"Legacy sentiment — uitgesloten",
}

ROUTES={
    "home":"Start",
    "method":"Wat onderzoeken we?",
    "data":"Data-overzicht",
    "vectors":"Dagvectoren",
    "agents":"Agent Control Center",
    "training":"Modeltraining",
    "forecast":"Kans & Edge",
    "relationships":"Historische verbanden",
    "coach":"Modelcoach",
    "news":"Nieuws & events",
    "tasks":"Lopende taken",
    "roadmap":"Roadmap",
    "live":"Live markt",
    "system":"Systeem",
}

st.set_page_config(page_title=f"{APP_NAME} — {ENGINE_NAME}",page_icon="◈",layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
:root{--oe-bg:#06111e;--oe-panel:#0b1b2b;--oe-panel2:#102436;--oe-cyan:#31e4d1;--oe-blue:#66b8ff;--oe-text:#f2f7fb;--oe-muted:#8ea6b8;--oe-border:rgba(117,170,194,.18);--oe-soft:rgba(49,228,209,.08);color-scheme:dark}
#MainMenu,footer,[data-testid="stHeader"]{visibility:hidden}
.stApp{background:radial-gradient(circle at 79% -15%,rgba(31,154,170,.14),transparent 37%),var(--oe-bg);color:var(--oe-text)}
.block-container{max-width:1560px;padding-top:1.45rem;padding-bottom:4rem;padding-left:2.2rem;padding-right:2.2rem}
[data-testid="stSidebar"]{background:#081624;border-right:1px solid var(--oe-border);min-width:285px!important}
[data-testid="stSidebar"]>div:first-child{padding-top:.8rem}
.oe-brand{padding:18px 17px;border-bottom:1px solid var(--oe-border);margin:0 -5px 15px}
.oe-brand h1{font-size:1.22rem;margin:0;letter-spacing:.02em;color:var(--oe-text);font-weight:780}.oe-brand .mark{color:var(--oe-cyan);margin-right:.4rem}.oe-brand .sub{font-size:.72rem;color:var(--oe-muted);margin-top:6px;letter-spacing:.08em;text-transform:uppercase}
.oe-hero{position:relative;overflow:hidden;padding:33px 35px;border:1px solid var(--oe-border);border-radius:18px;background:linear-gradient(135deg,rgba(14,43,62,.96),rgba(8,27,42,.90));margin-bottom:22px;box-shadow:0 24px 70px rgba(0,0,0,.18)}
.oe-hero:after{content:"";position:absolute;width:260px;height:260px;right:-90px;top:-120px;border:1px solid rgba(49,228,209,.24);border-radius:50%;box-shadow:0 0 70px rgba(49,228,209,.09)}
.oe-hero h1{font-size:clamp(2rem,4vw,3.55rem);line-height:1.02;max-width:1050px;letter-spacing:-.055em;margin:.3rem 0 .75rem;color:#f7fbfd}.oe-hero h1 em{font-style:normal;color:var(--oe-cyan)}
.oe-hero>div:last-child{font-size:1.03rem;color:#b6c8d4;max-width:970px;line-height:1.65}
.oe-kicker{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.16em;color:var(--oe-cyan)}
.oe-card{border:1px solid var(--oe-border);border-radius:15px;padding:19px 20px;min-height:144px;background:linear-gradient(150deg,rgba(15,38,56,.94),rgba(8,25,39,.78))}
.oe-card h3{font-size:1rem;margin:0 0 9px;color:#eef8fb}.oe-card p{font-size:.88rem;color:#9eb3c1;margin:0;line-height:1.6}
.oe-section{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.14em;color:#607f91;margin:17px 0 7px}
.oe-explain{border-left:3px solid var(--oe-cyan);padding:14px 17px;background:var(--oe-soft);border-radius:0 10px 10px 0;margin:12px 0 22px;color:#c5d7e0;line-height:1.55}
.oe-pill{display:inline-block;padding:.3rem .58rem;border:1px solid rgba(49,228,209,.25);border-radius:999px;color:var(--oe-cyan);background:rgba(49,228,209,.06);font:700 .69rem ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.04em}
.oe-ready{border:1px solid var(--oe-border);border-radius:14px;padding:17px 18px;background:rgba(7,24,37,.78);margin:.6rem 0 1rem}.oe-ready strong{color:var(--oe-cyan)}
div[data-testid="stMetric"]{border:1px solid var(--oe-border);border-radius:13px;padding:13px 14px;background:rgba(10,31,47,.72)}
div[data-testid="stMetric"] label{color:#819bad!important}div[data-testid="stMetricValue"]{color:#f3f9fb}
div.stButton>button{border-radius:9px;border:1px solid var(--oe-border);background:#0d2435;color:#dcebf1;min-height:2.6rem}
div.stButton>button:hover{border-color:var(--oe-cyan);color:var(--oe-cyan);background:#0c2937}
div.stButton>button[kind="primary"]{background:var(--oe-cyan);color:#042025;border-color:var(--oe-cyan);font-weight:750}
[data-testid="stSidebar"] div.stButton>button{border:none;background:transparent;text-align:left;justify-content:flex-start;width:100%;padding:.48rem .62rem;min-height:2.3rem;color:#91aabc}
[data-testid="stSidebar"] div.stButton>button:hover{background:var(--oe-soft);border:none;color:var(--oe-cyan)}
[data-testid="stDataFrame"],[data-testid="stExpander"],div[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--oe-border)!important}
h1,h2,h3{color:#f1f7fa!important}p,li{color:#b2c4cf}code{color:#78efe0!important;background:#0a2030!important}
div[data-baseweb="tab-list"]{gap:.35rem}button[data-baseweb="tab"]{background:#0a1c2b;border-radius:9px;color:#92aabc}
hr{border-color:var(--oe-border)!important}
</style>
""",unsafe_allow_html=True)

# ---------- HTTP ----------
def detail(r):
    try:
        x=r.json()
        return x.get("detail",x)
    except Exception:
        return r.text or f"HTTP {r.status_code}"

def get(path,params=None,timeout=120):
    r=requests.get(API+path,params=params,timeout=timeout)
    if not r.ok: raise RuntimeError(f"{r.status_code}: {detail(r)}")
    return r.json()

def post(path,data=None,timeout=120):
    r=requests.post(API+path,json=data,timeout=timeout)
    if not r.ok: raise RuntimeError(f"{r.status_code}: {detail(r)}")
    return r.json()

def put(path,data=None,timeout=120):
    r=requests.put(API+path,json=data,timeout=timeout)
    if not r.ok: raise RuntimeError(f"{r.status_code}: {detail(r)}")
    return r.json()

def human_bytes(n):
    n=float(n or 0)
    for unit in ["B","KB","MB","GB","TB"]:
        if n<1024 or unit=="TB": return f"{n:.2f} {unit}"
        n/=1024

def pct(x):
    try:
        if x is None or pd.isna(x): return "—"
        return f"{float(x):.1%}"
    except Exception:return "—"

# ---------- routing ----------
def q(name,default=None):
    try:
        x=st.query_params.get(name,default)
        return x[0] if isinstance(x,list) and x else x
    except Exception:return default

def go(page,agent=None):
    try:
        st.query_params["page"]=page
        if agent: st.query_params["agent"]=agent
        elif "agent" in st.query_params: del st.query_params["agent"]
    except Exception:
        st.session_state["page"]=page
        if agent: st.session_state["agent"]=agent
    st.rerun()

def route():
    page=q("page",st.session_state.get("page","home"))
    if page=="agent":
        a=q("agent",st.session_state.get("agent","stocks"))
        return "agent",a if a in AGENTS else "stocks"
    return (page if page in ROUTES else "home"),None

def nav(label,page,icon="",agent=None,key=None):
    if st.button(f"{icon}  {label}",key=key or f"nav_{page}_{agent or ''}",use_container_width=True):
        go(page,agent)

def hero(kicker,title,body):
    st.markdown(f'<div class="oe-hero"><div class="oe-kicker">{html.escape(str(kicker))}</div><h1>{title}</h1><div>{body}</div></div>',unsafe_allow_html=True)

def card(title,text):
    st.markdown(f'<div class="oe-card"><h3>{title}</h3><p>{text}</p></div>',unsafe_allow_html=True)

# ---------- setup/login ----------
def setup_page():
    hero("Eerste configuratie",f"{APP_NAME} {APP_VERSION}","De browser is alleen bediening. Data en instellingen blijven op de Hetzner-server.")
    c1,c2=st.columns(2)
    p1=c1.text_input("Kies wachtwoord",type="password")
    p2=c2.text_input("Herhaal wachtwoord",type="password")
    loc=st.radio("Object Storage-regio",["Helsinki (HEL1)","Nuremberg (NBG1)","Anders"],horizontal=True)
    endpoint=region=None
    if loc=="Anders":
        c1,c2=st.columns(2);endpoint=c1.text_input("S3 endpoint");region=c2.text_input("Region")
    bucket=st.text_input("Bucketnaam")
    c1,c2=st.columns(2);hkey=c1.text_input("S3 access key");hsecret=c2.text_input("S3 secret key",type="password")
    massive=st.text_input("Massive REST API key",type="password")
    c1,c2=st.columns(2);mfk=c1.text_input("Massive Flat Files access key");mfs=c2.text_input("Massive Flat Files secret key",type="password")
    c1,c2=st.columns(2)
    fmp=c1.text_input("FMP API key (earningsverwachtingen)",type="password")
    sec_agent=c2.text_input("SEC User-Agent (naam + e-mail)",placeholder="OptionEdge Naam naam@example.nl")
    chunk=st.select_slider("Chunkgrootte",[100000,150000,200000,250000,300000,350000,400000],value=300000)
    if st.button("OptionEdge activeren",type="primary",use_container_width=True):
        if len(p1 or "")<8: st.error("Wachtwoord minimaal 8 tekens.");return
        if p1!=p2: st.error("Wachtwoorden verschillen.");return
        if not all(str(x or "").strip() for x in [bucket,hkey,hsecret,massive,mfk,mfs]):
            st.error("Vul alle Object Storage- en Massive-velden in.");return
        location="hel1" if loc.startswith("Helsinki") else "nbg1" if loc.startswith("Nuremberg") else "custom"
        try:
            post("/setup/save",{"storage_location":location,"storage_endpoint":endpoint,"storage_region":region,"storage_bucket":bucket,"storage_access_key":hkey,"storage_secret_key":hsecret,"massive_api_key":massive,"massive_flat_access_key":mfk,"massive_flat_secret_key":mfs,"fmp_api_key":fmp,"sec_user_agent":sec_agent,"import_chunk_rows":int(chunk),"admin_password":p1})
            st.session_state["authenticated"]=True;st.rerun()
        except Exception as e: st.error(str(e))

def login_page():
    hero("Option expectation research",APP_NAME,f"{ENGINE_NAME} • cloudomgeving op Hetzner")
    pw=st.text_input("Wachtwoord",type="password")
    if st.button("Inloggen",type="primary",use_container_width=True):
        try:
            if post("/auth/login",{"password":pw}).get("ok"):
                st.session_state["authenticated"]=True;st.rerun()
            st.error("Onjuist wachtwoord.")
        except Exception as e: st.error(str(e))

setup={}
try: setup=get("/setup/status",timeout=20)
except Exception as e:
    st.error("OptionEdge API is niet bereikbaar.");st.code(str(e));st.stop()
if not setup.get("configured"): setup_page();st.stop()
if not st.session_state.get("authenticated",False): login_page();st.stop()

# ---------- sidebar ----------
page,agent_route=route()
with st.sidebar:
    st.markdown(f'<div class="oe-brand"><h1><span class="mark">◈</span>{APP_NAME}</h1><div class="sub">{ENGINE_NAME} · v{APP_VERSION}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="oe-section">Onderzoek</div>',unsafe_allow_html=True)
    nav("Overzicht","home","⌂");nav("Onderzoeksontwerp","method","◎");nav("Data & gereedheid","data","▦");nav("Dagvectoren","vectors","◇")
    st.markdown('<div class="oe-section">Analyse & model</div>',unsafe_allow_html=True)
    nav("Diepe verbanden","relationships","⌁");nav("Modeltraining","training","◫");nav("Kans & edge","forecast","◈");nav("Verbeteragent","coach","✦")
    st.markdown('<div class="oe-section">Operatie</div>',unsafe_allow_html=True)
    nav("Nieuws-LLM","news","◉");nav("Agents","agents","⚙");nav("Lopende taken","tasks","↻");nav("Roadmap","roadmap","↗");nav("Live markt","live","⌁");nav("Systeem","system","⚙")
    st.markdown('<div class="oe-section">Direct naar agent</div>',unsafe_allow_html=True)
    for k,a in AGENTS.items(): nav(a["label"],"agent",a["icon"],k,key=f"direct_{k}")
    st.divider()
    try:
        h=get("/health",timeout=8);st.caption(f"Engine {h.get('version','—')} · Hetzner persistent")
    except Exception:st.caption("Serverstatus onbekend")

# ---------- shared job UI ----------
def agent_name(key):
    a=AGENTS.get(key,{})
    return f"{a.get('icon','⚙')} {a.get('label',key)}"

def render_job(j,companies=True):
    total=int(j.get("total_units",0) or 0);done=int(j.get("completed_units",0) or 0)
    rem=max(0,int(j.get("remaining_units",total-done) or 0));pr=min(1,max(0,float(j.get("progress",0) or 0)))
    with st.container(border=True):
        c1,c2,c3,c4=st.columns([2.1,1,1,1])
        c1.markdown(f"**{agent_name(j.get('agent',''))}**  \n{j.get('current_item') or 'Wachten…'}")
        c2.metric("Gereed",f"{done:,}/{total:,}");c3.metric("Resterend",f"{rem:,}");c4.metric("Voortgang",f"{pr:.1%}")
        st.progress(pr,text=f"{pr:.1%} • {j.get('message') or j.get('status','')}")
        if j.get("estimated_remaining_bytes") is not None:
            st.caption(f"Geschat resterend volume: {human_bytes(j.get('estimated_remaining_bytes'))} • al aanwezig vóór start: {int(j.get('already_present_units',0) or 0):,}")
        if companies:
            df=pd.DataFrame(j.get("companies") or [])
            if not df.empty:
                cols=[x for x in ["ticker","status","completed_units","total_units","remaining_units","progress","current_item","message"] if x in df]
                st.dataframe(df[cols],use_container_width=True,hide_index=True,height=min(300,70+34*len(df)))

def live_all():
    try:
        active=get("/agents/live",timeout=25).get("active") or []
        gpu=(get("/gpu-news/status",timeout=15).get("status") or {})
        if gpu and gpu.get("state")=="running":
            active.append({"agent":"gpu_news","status":"running","completed_units":gpu.get("completed_units",0),"total_units":gpu.get("total_units",0),"remaining_units":gpu.get("remaining_units",0),"progress":gpu.get("progress",0),"current_item":gpu.get("current_item",""),"message":gpu.get("message",""),"companies":[]})
        if not active: st.success("Geen actieve jobs. Alle workers staan klaar.")
        for j in active: render_job(j,True)
    except Exception as e: st.error(str(e))

STATE_ORDER={"ontbreekt":0,"gedeeltelijk":1,"uitgesteld":2,"gereed":3}

def readiness_panel(compact=False):
    """Render the authoritative storage-backed data gate."""
    r=get("/data/readiness",timeout=35)
    score=float(r.get("readiness_score",0) or 0)
    blockers=int(r.get("blocker_count",0) or 0)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Verplichte data gereed",f"{score:.0%}")
    c2.metric("Blokkerende onderdelen",f"{blockers}")
    c3.metric("Onderzoeksdata vanaf",str(r.get("research_start","—")))
    c4.metric("Nieuwscontext vanaf",str(r.get("news_context_start","—")))
    st.progress(score,text="Training toegestaan" if r.get("training_ready") else "Training geblokkeerd: importeer eerst de ontbrekende verplichte data")
    if r.get("training_ready"):
        st.success("De verplichte data-poorten zijn aantoonbaar compleet volgens de laatste Object Storage-scan.")
    else:
        names=", ".join(str(x.get("label")) for x in (r.get("blockers") or [])[:7])
        st.error(f"Nog niet trainingsklaar. Blokkers: {names or 'onbekend'}.")
    if compact:
        return r
    rows=[]
    for x in r.get("rows") or []:
        rows.append({
            "Status":str(x.get("state","—")).upper(),"Volgorde":x.get("order"),
            "Groep":x.get("group"),"Onderdeel":x.get("label"),"Noodzaak":x.get("needed"),
            "Poort":x.get("gate"),"Dataset":x.get("dataset_found") or "—",
            "Objecten":int(x.get("objects",0) or 0),"Eerste":x.get("first_date"),"Laatste":x.get("last_date"),
            "Tickers":int(x.get("ticker_count",0) or 0),
            "Ontbrekende tickers":", ".join(x.get("missing_tickers") or []) or "—",
            "Waarom nodig":x.get("why"),
            "_rank":STATE_ORDER.get(str(x.get("state")),9),
        })
    df=pd.DataFrame(rows)
    if not df.empty:
        mode=st.radio("Sortering",["Wat ontbreekt eerst","Onderzoeksvolgorde","Datagroep"],horizontal=True,key="ready_sort")
        if mode=="Wat ontbreekt eerst":df=df.sort_values(["_rank","Volgorde","Onderdeel"])
        elif mode=="Datagroep":df=df.sort_values(["Groep","Volgorde"])
        else:df=df.sort_values(["Volgorde"])
        st.dataframe(df.drop(columns=["_rank","Volgorde"]),use_container_width=True,hide_index=True,height=620)
    st.warning(r.get("important_limit") or "Dekking van internetnieuws moet meetbaar worden gerapporteerd.")
    return r

def defaults(agent):
    start=date(2022,9,12);end=date.today()-timedelta(days=1)
    # SPY/QQQ are market ETFs and have no normal corporate quarterly earnings
    # event.  They stay in stocks/features/regime data, but do not create noisy
    # SEC/FMP earnings warnings in the company-earnings agent.
    ticks=DEFAULT if agent in ["stocks","features"] else [x for x in DEFAULT if x not in ["SPY","QQQ"]]
    if agent=="open_interest":start=end=date.today()
    return start,end,ticks

def render_standard_agent(agent):
    a=AGENTS[agent];hero(a["group"],f'{a["icon"]} {a["label"]}',a["help"])
    c1,c2=st.columns(2)
    if c1.button("← Alle agents",use_container_width=True):go("agents")
    if c2.button("Data-overzicht",use_container_width=True,key=f"data_{agent}"):go("data")
    start0,end0,ticks0=defaults(agent)
    c1,c2=st.columns(2);start=c1.date_input("Vanaf",start0,key=f"{agent}_s");end=c2.date_input("Tot",end0,key=f"{agent}_e")
    text=st.text_area("Tickers",value="\n".join(ticks0),height=130,key=f"{agent}_t")
    ticks=[x.strip().upper() for x in text.replace(",","\n").splitlines() if x.strip()]
    payload={}
    if agent=="features":
        payload["horizon_minutes"]=st.selectbox("Basis feature-horizon",[15,30,60],index=1,key="fh")
        payload["parallel_workers"]=st.selectbox(
            "Parallelle featuredagen",[1,2,3,4,6,8],index=3,key="feature_workers",
            help="Standaard 4. Dit versnelt S3 + pandas zonder data te samplen; verlaag bij weinig RAM."
        )
    if agent=="pairs":payload["horizon_minutes"]=30
    force=st.toggle("Bewust alles opnieuw verwerken",False,key=f"{agent}_force",help="Normaal UIT laten. Dan wordt alleen ontbrekende data verwerkt.")
    mode="force" if force else "missing_only"
    plan=None
    try:plan=post(f"/agents/{agent}/preview",{"mode":mode,"start_date":str(start),"end_date":str(end),"tickers":ticks,"payload":payload}).get("plan") or {}
    except Exception as e:st.warning(f"Importplan niet beschikbaar: {e}")
    if plan:
        expected=int(plan.get("expected_units",0) or 0);already=int(plan.get("already_present_units",0) or 0);remaining=int(plan.get("remaining_units",0) or 0)
        m=st.columns(4);m[0].metric("Totaal",f"{expected:,}");m[1].metric("Al in Hetzner",f"{already:,}");m[2].metric("Nog nodig",f"{remaining:,}");m[3].metric("Compleet",f"{already/expected:.1%}" if expected else "—")
        if expected:st.progress(min(1,already/expected),text=f"{already:,}/{expected:,} al aantoonbaar aanwezig")
        if plan.get("estimated_remaining_bytes") is not None:st.caption(f"Dataset in Hetzner: {human_bytes(plan.get('stored_dataset_bytes'))} • geschat nog: {human_bytes(plan.get('estimated_remaining_bytes'))}")
        if remaining==0 and not force:st.success("Deze selectie is al compleet. Er wordt niets opnieuw geladen.")
        for note in plan.get("notes") or []:st.caption("• "+str(note))
    c1,c2=st.columns([2,1])
    if c1.button("▶ Start ontbrekende data" if not force else "↻ Volledige herverwerking",type="primary",use_container_width=True,key=f"{agent}_start"):
        try:
            r=post(f"/agents/{agent}/start",{"mode":mode,"start_date":str(start),"end_date":str(end),"tickers":ticks,"payload":payload})
            st.success("Alles compleet; geen job gestart." if r.get("status")=="already_complete" else f"Job {r.get('job_id')} gestart.")
        except Exception as e:st.error(str(e))
    if c2.button("■ Stop",use_container_width=True,key=f"{agent}_stop"):
        try:post(f"/agents/{agent}/stop");st.info("Stopverzoek verstuurd.")
        except Exception as e:st.error(str(e))
    st.markdown("### Live voortgang")
    @st.fragment(run_every="2s")
    def one_live():
        try:
            r=get(f"/agents/{agent}/latest",timeout=25);j=r.get("job")
            if not j:st.info("Nog geen job uitgevoerd.");return
            j["agent"]=agent;j["companies"]=r.get("companies") or [];render_job(j,True)
        except Exception as e:st.error(str(e))
    one_live()

def render_gpu_agent():
    a=AGENTS["gpu_news"];hero(a["group"],f'{a["icon"]} {a["label"]}',a["help"])
    st.success("De nieuwe agent schrijft uitsluitend versievaste bronartikelen, chunks, 1024D embeddings en event-JSON naar `market-data/v5/`. Legacy nieuws wordt niet stil hergebruikt.")
    c1,c2,c3=st.columns(3)
    with c1:card("Taalmodel","Qwen3-14B-AWQ waar 20 GB VRAM dit toelaat; automatische fallback naar Qwen3-8B-AWQ. JSON-uitvoer wordt schema-gevalideerd.")
    with c2:card("Semantische laag","BAAI/BGE-M3 maakt per gededupliceerde chunk een 1024D embedding. De ruwe embedding blijft naast de dagelijkse vector bewaard.")
    with c3:card("Andere Hetzner-account","De GPU hoeft geen inkomende poort te openen. Hij leest opdrachten en schrijft resultaten via beperkte S3/Object-Storage-sleutels.")
    st.markdown("### Bron- en kwaliteitsbeleid")
    st.write("Officiële SEC- en bedrijfsbronnen, professionele financiële redacties, gelicentieerde feeds en newswires worden afzonderlijk gemeten. Per fetch worden publicatietijd, eerste waarneming, revisie, rechtenstatus, bronkwaliteit, hash, duplicaatcluster en foutreden opgeslagen.")
    st.warning("‘Alle nieuwsberichten op internet’ is niet aantoonbaar haalbaar. De agent toont daarom de gebruikte bronuniversums, gaten, mislukte downloads, paywalls en duplicaten — per ticker en datum.")
    c1,c2=st.columns(2)
    start=c1.date_input("Backfill vanaf",date(2022,9,12),key="gpu_s")
    end=c2.date_input("Backfill tot",date.today(),key="gpu_e")
    text=st.text_area("Tickers",value="\n".join([x for x in DEFAULT if x not in ["SPY","QQQ"]]),height=130,key="gpu_t")
    ticks=[x.strip().upper() for x in text.replace(",","\n").splitlines() if x.strip()]
    c1,c2=st.columns(2)
    force=c1.toggle("Bestaande professionele nieuwsdata opnieuw verwerken",False,key="gpu_force")
    all_sources=c2.toggle("Alle goedgekeurde broncategorieën",True,key="gpu_all")
    b1,b2=st.columns([2,1])
    if b1.button("▶ Start op GPU-computer Floris",type="primary",use_container_width=True):
        try:
            r=post("/gpu-news/start",{"start_date":str(start),"end_date":str(end),"tickers":ticks,"force":force,"include_professional_media":all_sources,"include_company_official":True,"include_sec":True,"include_newswires":all_sources})
            st.success(f"Opdracht {r.get('command_id')} in Hetzner gezet. Als de GPU-agent draait, pakt hij deze automatisch op.")
        except Exception as e:st.error(str(e))
    if b2.button("■ Stop GPU-agent",use_container_width=True):
        try:post("/gpu-news/stop",{});st.info("Stopopdracht geplaatst.")
        except Exception as e:st.error(str(e))
    st.markdown("### Live GPU-status")
    @st.fragment(run_every="2s")
    def gpu_live():
        try:
            r=get("/gpu-news/status",timeout=15);s=r.get("status") or {}
            if not s:st.warning("Nog geen heartbeat van de GPU-computer. Start `GPU_AGENT_FLORIS/START_GPU_AGENT.bat`.");return
            heartbeat=str(s.get("heartbeat","—")).replace("T"," ")[:19]
            m=st.columns(4);m[0].metric("Status",str(s.get("state","—")).upper());m[1].metric("Gereed",f"{int(s.get('completed_units',0)):,}/{int(s.get('total_units',0)):,}");m[2].metric("Resterend",f"{int(s.get('remaining_units',0)):,}");m[3].metric("Heartbeat",heartbeat)
            pr=min(1,max(0,float(s.get("progress",0) or 0)));st.progress(pr,text=f"{pr:.1%} • {s.get('message','')}")
            st.caption(f"Huidig: {s.get('current_item','—')}")
        except Exception as e:st.error(str(e))
    gpu_live()
    st.info("De map **GPU_AGENT_FLORIS_20GB** zit in het pakket. Op Ubuntu: `./install.sh`, `python configure.py`, daarna `./start.sh`. Object Storage is de koppeling met OptionEdge.")

# ---------- page functions ----------
def page_home():
    hero("OptionEdge · multimodale research","Wanneer zit de <em>optiemarkt</em> ernaast?","Eén controleerbare dagrepresentatie verbindt nieuwsbetekenis, optiegedrag, aandeelreactie en theoretische verwachting — zonder toekomstige informatie in de input te laten lekken.")
    st.markdown('<span class="oe-pill">384D PRIMAIR</span> &nbsp; <span class="oe-pill">1024D RUW NIEUWS</span> &nbsp; <span class="oe-pill">20 GB GPU</span> &nbsp; <span class="oe-pill">PURGED WALK-FORWARD</span>',unsafe_allow_html=True)
    st.markdown('<div class="oe-explain"><b>Centrale onderzoeksvraag</b><br>Welke point-in-time combinaties van nieuws, optie-oppervlak, aandeelbeweging en theoretisch-praktische verwachtingswaarde voorspellen latere afwijkingen — en blijven die patronen overeind in een volledig latere, onaangeraakte testperiode?</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:card("1 · Begrijp de dag","Een nieuws-LLM leest chunks per ticker, extraheert gebeurtenis, richting, causal chain, horizon, onzekerheid, novelty en relevantie.")
    with c2:card("2 · Verbind modaliteiten","192 nieuws + 96 opties + 48 aandeel + 32 verwachting + 16 kwaliteit/masks = één 384D ticker-dagvector.")
    with c3:card("3 · Bewijs generalisatie","Dimensie-ablaties, tijdgescheiden hypothesetoetsing, FDR-correctie, calibratie en een finale hold-out bepalen wat standhoudt.")

    st.markdown("### Data-gereedheid")
    try:readiness_panel(compact=True)
    except Exception as e:st.warning(f"Gereedheid kon niet worden geladen: {e}")

    st.markdown("### Projectvoortgang")
    try:
        ps=get("/project/status",timeout=25)
        st.progress(float(ps.get("progress",0) or 0),text=f"{int(ps.get('done_steps',0))}/{int(ps.get('total_steps',0))} hoofdfasen gereed • {float(ps.get('progress',0)):.0%}")
        rows=[]
        for x in ps.get("steps") or []:
            rows.append({"Status":"✓ Gereed" if x.get("done") else "○ Nog te doen","Onderdeel":x.get("label"),"Waarom":x.get("detail")})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        if ps.get("next_steps"):
            st.info("**Volgende stappen:** "+" → ".join(ps.get("next_steps")))
        if ps.get("legacy_news_excluded"):
            st.success("Legacy nieuws is uitgesloten van de nieuwe OptionEdge Feature Store en het nieuwe FINAL model.")
        recent=pd.DataFrame(ps.get("recent_jobs") or [])
        if not recent.empty:
            st.markdown("#### Wat is er onlangs gebeurd?")
            cols=[c for c in ["id","agent","status","message","created_at","finished_at"] if c in recent]
            st.dataframe(recent[cols],use_container_width=True,hide_index=True,height=280)
    except Exception as e:st.warning(f"Projectstatus kon niet worden geladen: {e}")

    st.markdown("### Direct verder")
    c1,c2,c3,c4=st.columns(4)
    if c1.button("▦ Controleer data",use_container_width=True):go("data")
    if c2.button("◉ Open nieuws-LLM",use_container_width=True):go("agent","gpu_news")
    if c3.button("◇ Bouw dagvectoren",use_container_width=True):go("vectors")
    if c4.button("⌁ Onderzoek verbanden",use_container_width=True):go("relationships")

    st.markdown("### Live jobs")
    st.caption("Alleen dit liveblok ververst; de hele pagina wordt niet steeds opnieuw geladen.")
    @st.fragment(run_every="2s")
    def home_jobs():live_all()
    home_jobs()

def page_method():
    hero("Research contract · v20","Van bronartikel naar <em>toetsbaar patroon</em>","De betekenis van iedere vectorcoördinaat, de as-of-grens, targets en validatie staan versie-vast. Een mooie backtest is niet genoeg: alleen informatie die op dat moment werkelijk beschikbaar was mag meedoen.")
    try:d=get("/research/design",timeout=25)
    except Exception as e:st.error(str(e));return
    st.info(d.get("dimension_decision"))
    layout=pd.DataFrame(d.get("vector_layout") or []).rename(columns={"label":"Blok","dimensions":"Dimensies","source":"Bronnen","meaning":"Betekenis","start":"Vanaf","stop":"Tot"})
    if not layout.empty:st.dataframe(layout[["Blok","Dimensies","Vanaf","Tot","Bronnen","Betekenis"]],use_container_width=True,hide_index=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("### Wat is input?")
        st.write("Nieuws dat vóór de dagelijkse as-of-grens first-seen was; tijdsynchrone optie-, aandeel- en benchmarkdata; theoretische waarde, actuele extrinsieke waarde en hun afwijking; plus expliciete dekking/masks.")
        st.markdown("### Wat is uitsluitend target?")
        for x in d.get("targets") or []:st.write(f"**{x.get('label')}** — {x.get('note')}")
    with c2:
        st.markdown("### Anti-lekkageregels")
        for x in d.get("leakage_rules") or []:st.write("• "+str(x))
    st.markdown("### Hoe verbanden worden onderzocht")
    methods=pd.DataFrame(d.get("relationship_methods") or []).rename(columns={"method":"Methode","question":"Vraag","guard":"Bescherming tegen schijnverband","status":"Implementatiestatus"})
    if not methods.empty:st.dataframe(methods,use_container_width=True,hide_index=True)
    st.warning("Zonder historische bid/ask-quotes zijn uitkomsten bruto/paper. Spread, slippage en uitvoerbaarheid moeten in een latere C2-vergelijking worden toegevoegd voordat ‘nettowinstkans’ mag worden geclaimd.")

def page_data():
    hero("Object Storage · bron van waarheid","Wat is er al — en wat <em>ontbreekt</em> nog?","Iedere geïmporteerde laag wordt netjes gesorteerd getoond. De gereedheidspoort vergelijkt de actuele Object Storage-inventaris met de vereisten voor vectorbouw, training en netto-uitkomst.")
    if st.button("↻ Scan Hetzner nu",type="primary"):
        try:post("/inventory/scan",{});st.success("Scan gestart.")
        except Exception as e:st.error(str(e))
    st.markdown("### Vereiste data en blokkers")
    try:readiness_panel(compact=False)
    except Exception as e:st.error(f"Gereedheidsanalyse mislukt: {e}")
    st.markdown("### SEC/FMP kwartaalcijfers")
    try:
        earnings=get("/earnings/status",timeout=30)
        m=st.columns(4)
        m[0].metric("Laatste SEC-import",str(earnings.get("last_successful_sec_import") or "Nog niet")[:19])
        m[1].metric("Laatste FMP-import",str(earnings.get("last_successful_fmp_import") or "Nog niet")[:19])
        m[2].metric("Earnings-events",int(earnings.get("earnings_events",0) or 0))
        m[3].metric("Tickers met SEC-cijfers",int(earnings.get("tickers_with_quarterly_facts",0) or 0))
        configured=earnings.get("configured") or {}
        if not configured.get("sec_user_agent") or not configured.get("fmp_api_key"):
            missing=[]
            if not configured.get("sec_user_agent"):missing.append("SEC User-Agent")
            if not configured.get("fmp_api_key"):missing.append("FMP API key")
            st.warning("Nog instellen via Systeem: "+", ".join(missing))
        status_rows=[]
        for row in earnings.get("ticker_status") or []:
            status_rows.append({
                "Ticker":row.get("ticker"),"Status":row.get("status"),
                "SEC-rijen":(row.get("sec") or {}).get("rows",0),
                "FMP-rijen":(row.get("fmp") or {}).get("rows",0),
                "Events":(row.get("processed") or {}).get("events",0),
                "Fouten":" | ".join(row.get("errors") or []),
                "Bijgewerkt":row.get("updated_at"),
            })
        if status_rows:
            st.dataframe(pd.DataFrame(status_rows).sort_values("Ticker"),use_container_width=True,hide_index=True)
        ticker=st.selectbox("Toon gecombineerde kwartaaldata",DEFAULT[:-2],key="earnings_ticker")
        quarterly=get(f"/earnings/data/{ticker}",{"limit":200},timeout=30)
        qdf=pd.DataFrame(quarterly.get("rows") or [])
        if not qdf.empty:
            preferred=["date","ticker","fiscal_year","fiscal_quarter","earnings_date","filing_date","form_type","earnings_time","eps_estimated","eps_actual","eps_surprise","eps_surprise_pct","revenue_estimated","revenue_actual","revenue_surprise","revenue_surprise_pct","net_income","assets","liabilities","cash_flow","estimate_pit_status"]
            st.dataframe(qdf[[c for c in preferred if c in qdf]].sort_values([c for c in ["date","fiscal_year","fiscal_quarter"] if c in qdf],ascending=False),use_container_width=True,hide_index=True)
        elif quarterly.get("error"):st.warning(quarterly.get("error"))
    except Exception as e:st.warning(f"Earningsstatus kon niet worden geladen: {e}")
    st.markdown("### Fysiek geïmporteerde datasets")
    @st.fragment(run_every="2s")
    def inv():
        try:
            r=get("/inventory/latest",timeout=25);j=r.get("scan_job") or {};snap=r.get("snapshot")
            if j.get("status") in ["queued","running"]:
                pr=min(1,max(0,float(j.get("progress",0) or 0)));st.progress(pr,text=f"Scan {pr:.1%} • {j.get('message','')}")
            if not snap:st.warning("Nog geen inventaris.");return
            m=st.columns(4);m[0].metric("Objecten",f"{int(snap.get('total_market_data_objects',0)):,}");m[1].metric("Opslag",human_bytes(snap.get("total_market_data_bytes",0)));m[2].metric("Datasets",len(snap.get("datasets") or {}));m[3].metric("Scan",str(snap.get("generated_at","—")).replace("T"," ")[:19])
            rows=[]
            for code,d in (snap.get("datasets") or {}).items():
                objects=int(d.get("objects",0) or 0)
                if objects<=0:continue
                if code in ["news_articles","news_chunks","news_embeddings","news_events","news_coverage","daily_vectors","daily_labels","deep_relationships","fusion_models","earnings_sec_raw","earnings_fmp_raw","earnings_processed","earnings_status","alpha_research","paper_trading","trading_control","monitoring"]:use="01 · Fusielaag v5"
                elif code in ["professional_news","clean_features","clean_pairs","clean_relationships","clean_pair_models","events"]:use="02 · Clean basis v4"
                elif code in ["stocks","options","option_trades","open_interest","earnings","macro","quotes","option_reference","corporate_actions","dividends","borrow"]:use="03 · Brondata"
                else:use="04 · Legacy / audit"
                rows.append({"Laag":use,"Code":code,"Dataset":d.get("label",code),"Objecten":objects,"Bytes":int(d.get("bytes",0) or 0),"Opslag":human_bytes(d.get("bytes",0)),"Datadagen":d.get("date_count",0),"Eerste":d.get("first_date"),"Laatste":d.get("last_date"),"Tickers":d.get("ticker_count",0),"Laatst gewijzigd":d.get("last_modified")})
            df=pd.DataFrame(rows)
            if df.empty:st.info("De scan heeft nog geen datasets met objecten gevonden.")
            else:
                groups=sorted(df["Laag"].unique())
                selected=st.multiselect("Toon lagen",groups,default=groups,key="imported_layers")
                view=df[df["Laag"].isin(selected)].sort_values(["Laag","Dataset","Eerste"],na_position="last")
                st.dataframe(view.drop(columns=["Bytes"]),use_container_width=True,hide_index=True,height=650)
                st.download_button("Download gesorteerde inventaris (CSV)",view.to_csv(index=False).encode("utf-8"),"optionedge_import_inventory.csv","text/csv")
        except Exception as e:st.error(str(e))
    inv()

def page_vectors():
    hero("Dagelijkse multimodale representatie","Eén ticker-dag, precies <em>384 dimensies</em>","De primaire vector wordt pas na de dagelijkse as-of-grens opgebouwd. Ruwe bronblokken blijven opgeslagen, zodat 128/256/384/512/768 onder exact dezelfde tijdsvalidatie eerlijk kunnen worden vergeleken.")
    try:d=get("/research/design",timeout=25)
    except Exception as e:st.error(str(e));return
    c1,c2,c3=st.columns(3)
    c1.metric("Primaire dimensies",d.get("primary_daily_dimensions",384))
    c2.metric("Ruwe nieuwsembedding",d.get("raw_news_dimensions",1024))
    c3.metric("Te vergelijken varianten"," / ".join(map(str,d.get("dimension_candidates") or [])))
    rows=[]
    colors={"news":"Cyaan","options":"Blauw","stock":"Groen","expectation":"Oranje","quality":"Grijs"}
    for x in d.get("vector_layout") or []:
        rows.append({"Coördinaten":f"{x.get('start')}–{int(x.get('stop',0))-1}","Blok":x.get("label"),"Dimensies":x.get("dimensions"),"Visuele groep":colors.get(x.get("code")),"Bron":x.get("source"),"Betekenis":x.get("meaning")})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.markdown("### Bouw ontbrekende vectoren")
    c1,c2=st.columns(2);start=c1.date_input("Vanaf",date(2022,9,16),key="vec_s");end=c2.date_input("Tot",date.today()-timedelta(days=1),key="vec_e")
    txt=st.text_area("Tickers",value="\n".join([x for x in DEFAULT if x not in ["SPY","QQQ"]]),height=120,key="vec_t")
    ticks=[x.strip().upper() for x in txt.replace(",","\n").splitlines() if x.strip()]
    c1,c2=st.columns(2);horizon=c1.selectbox("Bron Pair Store",[15,30,60,120,240],index=1,format_func=lambda x:f"h{x}");force=c2.toggle("Bestaande vectoren opnieuw bouwen",False)
    if st.button("◇ Start vectorbouw",type="primary",use_container_width=True):
        try:
            r=post("/vectors/build",{"start_date":str(start),"end_date":str(end),"tickers":ticks,"horizon_minutes":int(horizon),"force":force})
            st.success(f"Vectorjob {r.get('job_id')} gestart.")
        except Exception as e:st.error(str(e))
    st.markdown("### Live vectorjob")
    @st.fragment(run_every="2s")
    def vec_live():
        try:
            r=get("/agents/vectors/latest",timeout=25);j=r.get("job")
            if not j:st.info("Nog geen vectorjob uitgevoerd.");return
            j["agent"]="vectors";j["companies"]=r.get("companies") or [];render_job(j,True)
        except Exception as e:st.error(str(e))
    vec_live()
    st.info("384D is een vooraf gekozen hoofdvariant, niet automatisch de winnaar. De definitieve dimensie wordt uitsluitend gekozen op latere validatiedata; de finale testset blijft tot het einde onaangeraakt.")

def page_agents():
    hero("Orchestratie","Agent Control Center","Klik rechtstreeks op iedere agent. Onafhankelijke workers kunnen tegelijk draaien.")
    groups={}
    for k,a in AGENTS.items():groups.setdefault(a["group"],[]).append((k,a))
    for group,items in groups.items():
        st.markdown(f"### {group}")
        cols=st.columns(3)
        for i,(k,a) in enumerate(items):
            with cols[i%3]:
                with st.container(border=True):
                    st.markdown(f"**{a['icon']} {a['label']}**");st.caption(a["help"])
                    if st.button("Open agent →",key=f"open_{k}",use_container_width=True):go("agent",k)
    with st.expander("Legacy nieuwsagents"):
        st.warning("Deze agents/data blijven alleen voor audit en backwards compatibility bestaan. Het nieuwe OptionEdge-model gebruikt ze niet.")
        for k,v in LEGACY_AGENTS.items():st.write(f"• {v}")
    st.markdown("### Alle live agents")
    @st.fragment(run_every="2s")
    def alljobs():live_all()
    alljobs()

def page_training():
    hero("Alpha-first modelselectie","Bewijs <em>economische edge</em> — niet alleen accuracy","Iedere opgeslagen modelrun blijft zichtbaar. De primaire onderzoeksmotor gebruikt alle beschikbare, geschikte ticker-dagen, purged walk-forward, een onaangeraakte hold-out, realistische kostenaannames, NO_TRADE en exacte feature-ablaties.")
    st.markdown("### Permanente modelscorecard")
    try:
        score=get("/model/scorecard",timeout=40)
        st.info(score.get("message"))
        sdf=pd.DataFrame(score.get("alpha_scorecard") or [])
        if not sdf.empty:
            st.dataframe(sdf.sort_values(["horizon","EV_net"],ascending=[True,False]),use_container_width=True,hide_index=True,height=360)
        history=pd.DataFrame(score.get("model_run_history") or [])
        if not history.empty:
            with st.expander("Alle bewaarde modelruns",expanded=False):
                if "data_usage" in history:
                    history["Gebruikte regels"]=history["data_usage"].apply(
                        lambda x:(x or {}).get("loaded_rows") if isinstance(x,dict) else None
                    )
                cols=[c for c in ["id","stage","status","start_date","end_date","target","data_policy","Gebruikte regels","message","created_at","finished_at","artifact_key"] if c in history]
                st.dataframe(history[cols],use_container_width=True,hide_index=True)
        if score.get("alpha_error"):st.warning(score.get("alpha_error"))
    except Exception as e:st.warning(f"Scorecard kon niet worden geladen: {e}")
    try:
        ready=get("/data/readiness",timeout=25)
        allowed=bool(ready.get("training_ready"))
        if not allowed:st.error(f"Training is nog geblokkeerd door {ready.get('blocker_count',0)} verplichte dataonderdelen. Bekijk Data & gereedheid.")
        else:st.success("De opslagscan voldoet aan alle verplichte trainingspoorten.")
    except Exception as e:allowed=False;st.warning(str(e))

    alpha_tab,fusion_tab,legacy_tab=st.tabs(["Alpha Research · primair","Fusie 2.0 · benchmark","Bestaand intraday-model"])
    with alpha_tab:
        st.caption("Een onvolledige import verbergt oude resultaten niet. Een nieuwe run gebruikt alle geïmporteerde, geschikte ticker-dagen binnen de gekozen periode; één vooraf bepaalde near-ATM observatie per ticker-dag voorkomt pseudo-replicatie.")
        c1,c2=st.columns(2);a_start=c1.date_input("Vanaf",date(2022,9,16),key="ar_s");a_end=c2.date_input("Tot",date.today()-timedelta(days=1),key="ar_e")
        labels=[HORIZON_LABELS[x] for x in ["D1","D2","D3","W1"]]
        chosen=st.multiselect("Dagtermijnen",labels,default=labels,key="ar_h");a_h=[x for x in ["D1","D2","D3","W1"] if HORIZON_LABELS[x] in chosen]
        txt=st.text_area("Tickers",value="\n".join([x for x in DEFAULT if x not in ["SPY","QQQ"]]),height=120,key="ar_t");a_ticks=[x.strip().upper() for x in txt.replace(",","\n").splitlines() if x.strip()]
        c1,c2,c3=st.columns(3)
        commission=c1.number_input("Commissie per contract/zijde",min_value=0.0,value=.65,step=.05)
        slippage=c2.number_input("Slippage (bps/zijde)",min_value=0.0,value=5.0,step=1.0)
        spread=c3.number_input("Fallback roundtrip spread",min_value=0.0,max_value=1.0,value=.08,step=.01)
        if st.button("▶ Start Alpha Research",type="primary",use_container_width=True):
            try:
                r=post("/alpha-research/run",{"start_date":str(a_start),"end_date":str(a_end),"tickers":a_ticks,"horizons":a_h,"min_train_sessions":80,"test_sessions":20,"n_splits":5,"holdout_fraction":.15,"commission_per_contract_side":commission,"slippage_bps_per_side":slippage,"fallback_roundtrip_spread_pct":spread,"safety_margin_return":.005})
                st.success(f"Alpha-job {r.get('job_id')} gestart. Alle beschikbare geschikte data wordt gecatalogiseerd.")
                if r.get("readiness_warning"):st.warning("De run start, maar registreert ontbrekende datalagen expliciet.")
            except Exception as e:st.error(str(e))
        @st.fragment(run_every="3s")
        def alpha_live():
            try:
                result=get("/alpha-research/latest",timeout=40);run=result.get("run") or {}
                if run:
                    pr=min(1,max(0,float(run.get("progress",0) or 0)));st.progress(pr,text=f"{pr:.1%} · {run.get('status','')} · {run.get('message','')}")
                suite=result.get("suite") or {};rows=[];usage=[]
                for horizon,item in (suite.get("results") or {}).items():
                    u=item.get("data_usage") or {};usage.append({"Horizon":horizon,**u})
                    for exp in item.get("experiments") or []:
                        hold=exp.get("holdout_metrics") or {}
                        rows.append({"Horizon":horizon,"Model":exp.get("model"),"Features":exp.get("feature_set"),"Target":exp.get("target"),"EV netto/trade":hold.get("expected_value_per_trade"),"Net return":hold.get("net_return"),"Sharpe":hold.get("sharpe"),"Sortino":hold.get("sortino"),"Profit factor":hold.get("profit_factor"),"Max drawdown":hold.get("maximum_drawdown"),"Trades":hold.get("number_of_trades"),"Kosten voorlopig":hold.get("costs_provisional"),"Leakage geldig":exp.get("leakage_valid")})
                if rows:st.dataframe(pd.DataFrame(rows).sort_values(["Horizon","EV netto/trade"],ascending=[True,False]),use_container_width=True,hide_index=True,height=500)
                if usage:
                    with st.expander("Benutting van geïmporteerde data",expanded=True):st.dataframe(pd.DataFrame(usage),use_container_width=True,hide_index=True)
                for error in suite.get("errors") or []:st.warning(error)
            except Exception as e:st.warning(str(e))
        alpha_live()
    with fusion_tab:
        c1,c2=st.columns(2);start=c1.date_input("Vanaf",date(2022,9,16),key="fu_s");end=c2.date_input("Tot",date.today()-timedelta(days=1),key="fu_e")
        labels=[HORIZON_LABELS[x] for x in ["D1","D2","D3","W1"]]
        chosen=st.multiselect("Dagtermijnen",labels,default=labels,key="fu_h");h=[x for x in ["D1","D2","D3","W1"] if HORIZON_LABELS[x] in chosen]
        txt=st.text_area("Tickers",value="\n".join([x for x in DEFAULT if x not in ["SPY","QQQ"]]),height=120,key="fu_t");ticks=[x.strip().upper() for x in txt.replace(",","\n").splitlines() if x.strip()]
        st.success("Databeleid: alle beschikbare geschikte ticker-dagen en contractobservaties worden gebruikt; er is geen verborgen row-cap of representatieve sampling.")
        if st.button("▶ Train fusie- en dimensiemodel",type="primary",use_container_width=True,disabled=not allowed):
            try:
                r=post("/fusion-model/train",{"start_date":str(start),"end_date":str(end),"tickers":ticks,"horizons":h,"max_rows_per_horizon":30000})
                st.success(f"Fusierun {r.get('model_run_id')} gestart.")
            except Exception as e:st.error(str(e))
        @st.fragment(run_every="3s")
        def fusion_live():
            try:
                run=get("/fusion-model/latest",timeout=30).get("run")
                if not run:st.info("Nog geen fusietraining uitgevoerd.");return
                pr=min(1,max(0,float(run.get("progress",0) or 0)));st.progress(pr,text=f"{pr:.1%} · {run.get('status','')} · {run.get('message','')}")
                metrics=run.get("metrics") or {};summary=[];bench=[]
                for horizon,v in (metrics.get("horizons") or {}).items():
                    call=v.get("final_holdout_call") or {};putm=v.get("final_holdout_put") or {}
                    call_return=v.get("final_holdout_call_return") or {};put_return=v.get("final_holdout_put_return") or {}
                    summary.append({"Horizon":HORIZON_LABELS.get(horizon,horizon),"Status":v.get("status"),"Gekozen D":v.get("selected_dimensions"),"Ticker-dagen":v.get("ticker_days"),"Call Brier":call.get("brier"),"Put Brier":putm.get("brier"),"Call AUC":call.get("auc"),"Put AUC":putm.get("auc"),"Call return MAE":call_return.get("mae"),"Put return MAE":put_return.get("mae"),"Call return RMSE":call_return.get("rmse"),"Put return RMSE":put_return.get("rmse"),"Nieuws-gain call":v.get("news_brier_gain_call"),"Nieuws-gain put":v.get("news_brier_gain_put")})
                    for b in v.get("dimension_benchmark") or []:bench.append({"Horizon":HORIZON_LABELS.get(horizon,horizon),**b})
                if summary:st.markdown("#### Finale hold-out en nieuws-ablation");st.dataframe(pd.DataFrame(summary),use_container_width=True,hide_index=True)
                if bench:st.markdown("#### Dimensiekeuze op validatie");st.dataframe(pd.DataFrame(bench).sort_values(["Horizon","dimensions"]),use_container_width=True,hide_index=True)
            except Exception as e:st.error(str(e))
        fusion_live()
    with legacy_tab:
        st.caption("Behoud van het bestaande intraday Expectation Gap-model voor vergelijking en backwards compatibility. Ook dit trainingspad gebruikt nu standaard alle beschikbare geschikte data; de oude sampling-code blijft alleen als expliciete debugmodus bestaan.")
        c1,c2=st.columns(2);start2=c1.date_input("Vanaf",date(2022,9,12),key="tr_s");end2=c2.date_input("Tot",date.today()-timedelta(days=1),key="tr_e")
        labs=st.multiselect("Termijnen",list(HORIZON_LABELS.values()),default=list(HORIZON_LABELS.values()),key="tr_h");h2=[c for c,l in HORIZON_LABELS.items() if l in labs]
        txt2=st.text_area("Tickers",value="\n".join([x for x in DEFAULT if x not in ["SPY","QQQ"]]),height=100,key="tr_t");ticks2=[x.strip().upper() for x in txt2.replace(",","\n").splitlines() if x.strip()]
        if st.button("Train bestaand intraday-model",use_container_width=True):
            try:
                r=post("/pair-model/train",{"start_date":str(start2),"end_date":str(end2),"tickers":ticks2,"horizons":h2,"max_rows_per_horizon":30000});st.success(f"Modelrun {r.get('model_run_id')} gestart.")
            except Exception as e:st.error(str(e))

def page_forecast():
    hero("Modeluitkomst","Kans, verwachting & <em>edge</em>","Gebruik standaard het nieuwe dagfusiemodel. Het bestaande intradaymodel blijft beschikbaar als controlemodel en voor de kortere termijnen.")
    model_choice=st.radio(
        "Modelweergave",["Fusie 2.0 · nieuws + opties + aandeel + verwachting","Bestaand intradaymodel"],
        horizontal=True,key="forecast_model_choice",
    )
    if model_choice.startswith("Fusie"):
        c1,c2,c3=st.columns([1,2,1])
        ticker=c1.text_input("Ticker","NVDA",key="fusion_fc_t").upper()
        daily_codes=["D1","D2","D3","W1"]
        selected=c2.multiselect("Dagtermijnen",[HORIZON_LABELS[x] for x in daily_codes],default=[HORIZON_LABELS["D1"],HORIZON_LABELS["D3"]],key="fusion_fc_h")
        horizons=[x for x in daily_codes if HORIZON_LABELS[x] in selected]
        topn=c3.selectbox("Optieparen",[5,10,20,30],index=0,key="fusion_fc_n")
        ready=False
        try:
            status=get("/fusion-model/status",{"ticker":ticker},timeout=30)
            if status.get("ready"):ready=True;st.success("V20-fusiemodel, dagvector en bijpassende Pair Store kunnen worden gebruikt.")
            else:st.warning(status.get("reason") or "Fusiemodel nog niet gereed.")
        except Exception as e:st.error(str(e))
        if st.button("◈ Bereken v20-kans en verwachtingswaarde",type="primary",use_container_width=True,disabled=not ready or not horizons):
            try:
                st.session_state["fusion_fc_result"]=get(
                    f"/fusion-model/predict/{ticker}",
                    {"horizons":",".join(horizons),"top_n":int(topn)},timeout=180,
                )
            except Exception as e:st.error(str(e))
        result=st.session_state.get("fusion_fc_result")
        if result and result.get("ticker")==ticker:
            st.caption(f"As-of handelsdag: {result.get('as_of_date','—')} · historische quotes inbegrepen: NEE")
            summary=pd.DataFrame(result.get("summary") or [])
            if not summary.empty:
                summary=summary.rename(columns={
                    "horizon":"Termijn","best_side":"Beste zijde","positive_probability":"P positieve bruto-uitkomst",
                    "gross_edge":"Verwachte bruto edge per aandeel","strike":"Strike","expiry":"Expiratie",
                    "expected_cost_return":"Verwachte kosten (rendement)","net_expected_edge":"Netto verwachte edge",
                    "decision":"Beslissing","decision_reasons":"Redenen NO_TRADE",
                    "selected_dimensions":"Gekozen dimensies","call_brier":"Call Brier","put_brier":"Put Brier",
                    "news_brier_gain_call":"Nieuws-gain call","news_brier_gain_put":"Nieuws-gain put",
                })
                columns=[x for x in ["Termijn","Beste zijde","P positieve bruto-uitkomst","Verwachte bruto edge per aandeel","Verwachte kosten (rendement)","Netto verwachte edge","Beslissing","Redenen NO_TRADE","Strike","Expiratie","Gekozen dimensies","Call Brier","Put Brier","Nieuws-gain call","Nieuws-gain put"] if x in summary]
                st.dataframe(summary[columns],use_container_width=True,hide_index=True)
            available=result.get("requested_horizons") or []
            if available:
                code=st.selectbox("Detailtermijn",available,format_func=lambda x:HORIZON_LABELS.get(x,x),key="fusion_fc_detail")
                payload=(result.get("horizons") or {}).get(code,{})
                df=pd.DataFrame(payload.get("rows") or [])
                if not df.empty:
                    row=df.iloc[0];side=str(row.get("best_side","—"));probability=float(row.get("best_positive_probability",0) or 0)
                    expected=row.get("model_expected_call_extrinsic") if side=="CALL" else row.get("model_expected_put_extrinsic")
                    current=row.get("call_extrinsic") if side=="CALL" else row.get("put_extrinsic")
                    edge=row.get("best_gross_edge")
                    metrics=payload.get("validation") or {};side_metrics=metrics.get("final_holdout_call") if side=="CALL" else metrics.get("final_holdout_put")
                    cols=st.columns(5)
                    cols[0].metric("P positieve bruto-uitkomst",pct(probability),side)
                    cols[1].metric("Actuele tijdwaarde",f"${float(current):.3f}" if pd.notna(current) else "—")
                    cols[2].metric("Model-verwachte tijdwaarde",f"${float(expected):.3f}" if pd.notna(expected) else "—")
                    cols[3].metric("Bruto edge / contract",f"${float(edge)*100:.2f}" if pd.notna(edge) else "—")
                    cols[4].metric("Final hold-out Brier",f"{float((side_metrics or {}).get('brier')):.4f}" if (side_metrics or {}).get("brier") is not None else "—")
                    if row.get("decision")=="NO_TRADE":st.warning("NO_TRADE — "+", ".join(row.get("decision_reasons") or []))
                    else:st.success("TRADE-kandidaat volgens de research-risklaag; dit is geen live order.")
                    st.markdown("#### Hoogst gerangschikte call/put-paren")
                    show=[x for x in ["minute","expiry","strike","close_stock","best_side","best_positive_probability","expected_cost_return","net_expected_edge","decision","decision_reasons","call_positive_probability","put_positive_probability","call_extrinsic","put_extrinsic","model_expected_call_extrinsic","model_expected_put_extrinsic","call_gross_edge","put_gross_edge","fusion_dimensions"] if x in df]
                    st.dataframe(df[show],use_container_width=True,hide_index=True,height=500)
            st.warning(result.get("warning") or "Bruto/paper-resultaat; quotes, spread en slippage ontbreken.")
        return

    st.markdown("### Bestaand intradaymodel")
    c1,c2,c3=st.columns([1,2,1]);ticker=c1.text_input("Ticker","NVDA",key="fc_t").upper();labs=c2.multiselect("Termijnen",list(HORIZON_LABELS.values()),default=["1 uur","1 handelsdag"],key="fc_h");h=[c for c,l in HORIZON_LABELS.items() if l in labs];topn=c3.selectbox("Paren",[5,10,20,30],index=0,key="fc_n")
    ready=False
    try:
        s=get("/final-model/status",{"ticker":ticker,"horizons":",".join(h) if h else "ALL"},timeout=25)
        if s.get("ready"):ready=True;st.success("FINAL model en clean Pair Store zijn beschikbaar.")
        else:st.warning(s.get("reason") or "Model nog niet klaar.")
    except Exception as e:st.error(str(e))
    if st.button("◈ Bereken kans & edge",type="primary",use_container_width=True,disabled=not ready):
        try:st.session_state["fc_result"]=get(f"/final-model/predict/{ticker}",{"horizons":",".join(h),"top_n":topn,"threshold":.60},timeout=120)
        except Exception as e:st.error(str(e))
    r=st.session_state.get("fc_result")
    if r:
        summary=pd.DataFrame(r.get("summary") or [])
        if not summary.empty:
            summary=summary.rename(columns={"horizon":"Termijn","paper_profit_probability":"P positieve paper-uitkomst","paper_profit_side":"Beste zijde","historical_hit_rate":"Historische hit-rate","evidence_n":"Historische N","ci95_low":"95% CI laag","ci95_high":"95% CI hoog","call_edge_pct":"Call edge","put_edge_pct":"Put edge","most_likely_state":"Meest waarschijnlijk","state_probability":"P state"})
            cols=[c for c in ["Termijn","Beste zijde","P positieve paper-uitkomst","Historische hit-rate","Historische N","95% CI laag","95% CI hoog","Call edge","Put edge","Meest waarschijnlijk","P state"] if c in summary]
            st.dataframe(summary[cols],use_container_width=True,hide_index=True)
        available=r.get("requested_horizons") or []
        if available:
            code=st.selectbox("Detailtermijn",available,format_func=lambda x:HORIZON_LABELS.get(x,x),key="fc_detail")
            d=(r.get("horizons") or {}).get(code,{})
            df=pd.DataFrame(d.get("rows") or [])
            if not df.empty:
                x=df.iloc[0];best=float(x.get("best_gross_profit_probability",0) or 0);side=str(x.get("best_profit_side","—"))
                m=st.columns(5);m[0].metric("Paper-winstkans",pct(best),side);m[1].metric("Historische hit-rate",pct(x.get("best_profit_historical_hit_rate")));m[2].metric("Vergelijkbare gevallen",f"{int(x.get('best_profit_evidence_n',0) or 0):,}");m[3].metric("95% interval",f"{pct(x.get('best_profit_ci95_low'))} – {pct(x.get('best_profit_ci95_high'))}");edge=x.get("call_gross_edge_pct") if side=="CALL" else x.get("put_gross_edge_pct");m[4].metric("Bruto edge",pct(edge))
                if best>.5:st.info(">50% betekent hier dat een positieve **bruto/paper-uitkomst** volgens het gekalibreerde model waarschijnlijker is dan een niet-positieve uitkomst.")
                if bool(x.get("paper_profit_strict_ci_supported",False)):st.success("Sterk statistisch ondersteund: ook de ondergrens van het 95%-interval ligt boven 50% met voldoende historische N.")
                elif bool(x.get("paper_profit_empirically_supported",False)):st.success("Empirisch ondersteund: modelkans >50%, positieve edge en historische hit-rate >50% met voldoende N.")
                st.warning("Zonder historische bid/ask quotes zijn spread, slippage en fills nog niet in deze paper-winstkans verwerkt.")
                st.markdown("#### Alle onderzochte paren")
                showcols=[c for c in ["minute","expiry","strike","close_stock","best_profit_side","best_gross_profit_probability","best_profit_historical_hit_rate","best_profit_evidence_n","call_gross_edge_pct","put_gross_edge_pct","prob_call_up_put_down","prob_call_down_put_up","prob_both_up","prob_both_down"] if c in df]
                st.dataframe(df[showcols],use_container_width=True,hide_index=True,height=480)

def page_relationships():
    hero("Hypothese → onafhankelijke bevestiging","Zoek <em>diep</em>, toets streng","Tot 1.200 numerieke signalen worden alleen in het vroege discovery-deel verkend. Geselecteerde hypothesen gaan daarna naar een latere confirmation-periode met Benjamini–Hochberg FDR-correctie, effectgrootte en minimale support.")
    st.warning("Dit onderzoekt voorspellende samenhang, geen bewezen causaliteit. Sterke correlatie kan nog steeds door marktregime, earnings of een andere confounder worden veroorzaakt.")
    run_tab,result_tab,classic_tab=st.tabs(["Nieuwe diepe analyse","Bevestigde patronen","Bestaande relationship engine"])
    with run_tab:
        c1,c2=st.columns(2);start=c1.date_input("Discovery start",date(2022,9,16),key="rel_s");end=c2.date_input("Confirmation einde",date.today()-timedelta(days=1),key="rel_e")
        txt=st.text_area("Tickers",value="\n".join([x for x in DEFAULT if x not in ["SPY","QQQ"]]),height=110,key="rel_t");ticks=[x.strip().upper() for x in txt.replace(",","\n").splitlines() if x.strip()]
        daily_relation_codes=["D1","D2","D3","W1"]
        labs=st.multiselect("Dagelijkse horizonnen",[HORIZON_LABELS[x] for x in daily_relation_codes],default=[HORIZON_LABELS[x] for x in daily_relation_codes],key="rel_h");h=[c for c in daily_relation_codes if HORIZON_LABELS[c] in labs]
        target_labels={"Call extrinsiek rendement":"future_call_extrinsic_return","Put extrinsiek rendement":"future_put_extrinsic_return","Aandeel omhoog":"future_stock_up"}
        selected=st.multiselect("Targets",list(target_labels),default=list(target_labels),key="rel_targets");targets=[target_labels[x] for x in selected]
        c1,c2=st.columns(2);support=c1.select_slider("Minimale support",[40,60,80,100,150,200,300],value=80);maxrows=c2.select_slider("Max observaties per horizon",[10000,20000,30000,50000,80000],value=30000)
        if st.button("⌁ Start diepe verbandanalyse",type="primary",use_container_width=True):
            try:
                r=post("/deep-relationships/run",{"start_date":str(start),"end_date":str(end),"tickers":ticks,"horizons":h,"targets":targets,"max_rows_per_horizon":int(maxrows),"min_support":int(support)})
                st.success(f"Analysejob {r.get('job_id')} gestart.")
            except Exception as e:st.error(str(e))
        @st.fragment(run_every="3s")
        def deep_live():
            try:
                r=get("/agents/deep_relationships/latest",timeout=25);j=r.get("job")
                if not j:st.info("Nog geen diepe analyse uitgevoerd.");return
                j["agent"]="deep_relationships";j["companies"]=[];render_job(j,False)
            except Exception as e:st.error(str(e))
        deep_live()
    with result_tab:
        try:
            b=get("/deep-relationships/latest",timeout=35).get("bundle")
            if not b:st.info("Nog geen diepe analyseresultaten.")
            else:
                st.caption(f"Run: {b.get('created_at','—')} · {b.get('start_date','—')} t/m {b.get('end_date','—')}")
                options=[]
                analyses=b.get("analyses") or {}
                for horizon,targets_data in analyses.items():
                    for target in targets_data:options.append((horizon,target))
                if options:
                    choice=st.selectbox("Resultaat",options,format_func=lambda x:f"{HORIZON_LABELS.get(x[0],x[0])} · {x[1]}")
                    result=(analyses.get(choice[0]) or {}).get(choice[1]) or {};summary=result.get("summary") or {}
                    m=st.columns(4);m[0].metric("Observaties",summary.get("rows",0));m[1].metric("Geteste features",summary.get("tested_features",0));m[2].metric("Bevestigd",summary.get("confirmed_hypotheses",0));m[3].metric("Causaliteitsclaim","NEE")
                    confirmed=pd.DataFrame(result.get("confirmed") or [])
                    if confirmed.empty:st.info("Geen verband doorstond in deze run zowel richtingstabiliteit als FDR ≤ 0,05 en minimale support.")
                    else:st.dataframe(confirmed,use_container_width=True,hide_index=True,height=620)
        except Exception as e:st.error(str(e))
    with classic_tab:
        lab=st.selectbox("Termijn",list(HORIZON_LABELS.values()),index=2,key="classic_rel");code={v:k for k,v in HORIZON_LABELS.items()}[lab]
        try:
            b=get("/relationships/latest",{"horizon":code},timeout=30).get("bundle")
            rows=[]
            for i,x in enumerate((b or {}).get("rows") or [],1):
                v=x.get("validation") or {};rows.append({"#":i,"Verband":x.get("description"),"N validatie":v.get("support"),"P aandeel↑":v.get("prob_stock_up"),"P call↑/put↓":v.get("prob_call_up_put_down"),"P call↓/put↑":v.get("prob_call_down_put_up"),"Lift":v.get("lift_vs_baseline")})
            if rows:st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,height=500)
            else:st.info("Nog geen bestaand Relationship Discovery-resultaat voor deze termijn.")
        except Exception as e:st.error(str(e))

def page_coach():
    hero("Auditable AI-assistent","Modelcoach","Bespreek echte metrics en laat reproduceerbare verbeteracties uitvoeren.")
    if "coach_history" not in st.session_state:st.session_state["coach_history"]=[]
    c1,c2=st.columns(2);start=c1.date_input("Actieperiode vanaf",date(2022,9,12),key="co_s");end=c2.date_input("Tot",date.today()-timedelta(days=1),key="co_e")
    labs=st.multiselect("Termijnen",list(HORIZON_LABELS.values()),default=list(HORIZON_LABELS.values()),key="co_h");h=[c for c,l in HORIZON_LABELS.items() if l in labs]
    txt=st.text_area("Tickers voor acties",value="\n".join([x for x in DEFAULT if x not in ["SPY","QQQ"]]),height=100,key="co_t");ticks=[x.strip().upper() for x in txt.replace(",","\n").splitlines() if x.strip()]
    st.caption("Voorbeelden: “Welke termijn is slecht gekalibreerd?”, “Bouw de dagvectoren”, “Train Fusie 2.0 opnieuw”, “Zoek de verbanden opnieuw”. Zeg expliciet ‘legacy intraday’ om het oude model te gebruiken.")
    for item in st.session_state["coach_history"]:
        with st.chat_message(item["role"]):st.write(item["content"])
    prompt=st.chat_input("Vraag of verbeter het model…")
    if prompt:
        st.session_state["coach_history"].append({"role":"user","content":prompt})
        try:
            rr=post("/coach/chat",{"message":prompt,"context":{"start_date":str(start),"end_date":str(end),"tickers":ticks,"horizons":h,"max_rows_per_horizon":30000}})
            text=rr.get("reply","");acts=rr.get("actions") or []
            if acts:text+="\n\nUitgevoerde acties: "+", ".join(str(x) for x in acts)
            st.session_state["coach_history"].append({"role":"assistant","content":text})
        except Exception as e:st.session_state["coach_history"].append({"role":"assistant","content":f"Actie mislukt: {e}"})
        st.rerun()

def page_news():
    hero("Floris GPU · 20 GB VRAM","Van artikel naar <em>marktbetekenis</em>","De taalmodellaag analyseert nieuws per ticker, bewaart brontekst en revisies waar dit juridisch en technisch mag, maakt overlappende chunks, embedt ze in 1024D en extraheert gestructureerde gebeurtenissen.")
    st.success("**Clean-news policy:** de fusielaag gebruikt alleen versievaste v5-artikelen, chunks, embeddings en eventextracties. Oud nieuws blijft uitsluitend voor audit behouden.")
    c1,c2,c3=st.columns(3)
    with c1:card("Brondekking","SEC, investor relations, professionele financiële media, newswires en gelicentieerde feeds blijven afzonderlijke bronuniversums met dekking en fouten.")
    with c2:card("Qwen event-JSON","Eventtype, tickers, richting, impact, tijdshorizon, onzekerheid, novelty, causal chain, feiten versus verwachting en relevante citatiespans.")
    with c3:card("BGE-M3 embeddings","1024 dimensies per gededupliceerde chunk. Dagelijkse attention-pooling gebruikt kwaliteit, relevantie, importance, novelty en recency.")
    if st.button("◉ Open Nieuws-LLM GPU Agent",type="primary",use_container_width=True):go("agent","gpu_news")
    st.markdown("### Wat wordt per artikel opgeslagen?")
    cols=["canonical_url","source_domain","source_tier","published_utc","first_seen_utc","available_utc","point_in_time_basis","revision_id","content_hash","language","rights_status","fetch_status","duplicate_cluster","ticker_relevance","event_json","chunk_ids","embedding_model"]
    st.code("  ·  ".join(cols))
    st.warning("Een brede crawl is geen bewijs van volledige internetdekking. OptionEdge rapporteert precies welke bronnen en perioden zijn doorzocht, en toont robots/paywall/licentie/timeout/parse-fouten als ontbrekende data.")
    st.markdown("### GPU-status")
    @st.fragment(run_every="2s")
    def nlive():
        try:
            s=get("/gpu-news/status",timeout=15).get("status") or {}
            if not s:st.warning("Nog geen GPU heartbeat.");return
            pr=min(1,max(0,float(s.get("progress",0) or 0)));st.progress(pr,text=f"{pr:.1%} • {s.get('state','')} • {s.get('message','')}")
            st.caption(f"Heartbeat: {s.get('heartbeat','—')} • huidig: {s.get('current_item','—')}")
        except Exception as e:st.error(str(e))
    nlive()

def page_tasks():
    hero("Workers & GPU-heartbeat","Lopende <em>taken</em>","Alle actieve imports, vectorbouw, analyses en modelruns op één plek. Alleen dit statusblok ververst; invoer op andere pagina's blijft staan.")
    @st.fragment(run_every="2s")
    def jobs():live_all()
    jobs()
    st.markdown("### Laatste projecttaken")
    try:
        recent=pd.DataFrame(get("/project/status",timeout=25).get("recent_jobs") or [])
        if recent.empty:st.info("Nog geen taken geregistreerd.")
        else:st.dataframe(recent.sort_values("id",ascending=False),use_container_width=True,hide_index=True)
    except Exception as e:st.error(str(e))

def page_roadmap():
    hero("Van data naar aantoonbare edge","Roadmap met <em>go/no-go-poorten</em>","De interface markeert niet alleen wat gebouwd is, maar ook wat nog niet mag worden geconcludeerd. Elke fase heeft een expliciete uitvoer en stopcriterium.")
    rows=[
        {"Fase":"1 · Datakwaliteit","Uitvoer":"Object Storage-scan, coverage, SEC/FMP-status, schema's en gaten","Poort":"Verplichte bronnen aantoonbaar betrouwbaar"},
        {"Fase":"2 · Timestamps & leakage","Uitvoer":"as_of, first_seen, filing/eventtijd, revisies en automatische audits","Poort":"Iedere feature was werkelijk beschikbaar op t"},
        {"Fase":"3 · Betrouwbare features","Uitvoer":"Versnelde Clean Feature Store, earnings, nieuws/events en Pair Store","Poort":"Reproduceerbaar; targets fysiek/logisch gescheiden"},
        {"Fase":"4 · Simpele baselines","Uitvoer":"Historical mean en transparante lineaire benchmark","Poort":"Complexiteit moet aantoonbaar beter zijn"},
        {"Fase":"5 · Alpha research","Uitvoer":"Alle geschikte ticker-dagen, exacte ablaties, regimes en hypotheses","Poort":"Stabiele voorspellende waarde, geen dataminingclaim"},
        {"Fase":"6 · Walk-forward","Uitvoer":"Purged folds, embargo en één onaangeraakte eindhold-out","Poort":"Geen overlap of leakage; kandidaat vooraf gekozen"},
        {"Fase":"7 · Realistische kosten","Uitvoer":"Entry/exit quotes, diepte, spread, slippage, fees, impact en fills","Poort":"Positieve EV na niet-voorlopige kosten"},
        {"Fase":"8 · Informatiefusie","Uitvoer":"Nieuws/opties/earnings/theorie-ablaties en dimensievergelijking","Poort":"Toegevoegde laag verbetert meerdere OOS-perioden"},
        {"Fase":"9 · Risicomodel","Uitvoer":"Sizing, exposure, correlatie, liquiditeit, drawdown en NO_TRADE","Poort":"Alle limieten en kill switch slagen"},
        {"Fase":"10 · Paper trading","Uitvoer":"Dezelfde signal/risk-pipeline zonder echt geld","Poort":"Backtest, voorspeld en paper blijven consistent"},
        {"Fase":"11 · Live monitoring","Uitvoer":"PnL, calibratie, feature/prediction/modeldrift en execution quality","Poort":"Stabiele bewijsperiode en automatische stop"},
        {"Fase":"12 · Beperkt live","Uitvoer":"Kleine exposure via gecontroleerde brokeradapter","Poort":"Alle promotiegates bewezen; anders live uit"},
    ]
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,height=620)
    st.markdown("### Huidige voortgang")
    try:
        ps=get("/project/status",timeout=25);r=[]
        for x in ps.get("steps") or []:r.append({"Status":"GEREED" if x.get("done") else "NOG NODIG","Onderdeel":x.get("label"),"Controle":x.get("detail")})
        st.dataframe(pd.DataFrame(r),use_container_width=True,hide_index=True)
    except Exception as e:st.error(str(e))

def page_live():
    hero("Actuele markt","Live inspectie — uitvoering <em>uitgeschakeld</em>","Marktdata kan worden bekeken, maar echte orders blijven hard geblokkeerd totdat out-of-sample EV, kosten, drawdown, paper trading, datastabiliteit en alle risicogates aantoonbaar slagen.")
    try:
        safety=get("/trading/safety-status",timeout=30);live=safety.get("live") or {};paper=safety.get("paper_trading") or {}
        st.error("LIVE TRADING UIT" if not live.get("live_execution_enabled") else "Live status onbekend")
        c1,c2,c3=st.columns(3);c1.metric("Paper-signalen",paper.get("signals",0));c2.metric("Afgewikkeld",paper.get("settled_trades",0));c3.metric("Paper net PnL",paper.get("net_pnl",0))
        checks=pd.DataFrame([{"Gate":name,"Status":"GEREED" if ok else "GEBLOKKEERD"} for name,ok in (live.get("checks") or {}).items()])
        if not checks.empty:st.dataframe(checks,use_container_width=True,hide_index=True)
    except Exception as e:st.warning(f"Veiligheidsstatus niet beschikbaar: {e}")
    c0,cx=st.columns([1,1])
    ticker=c0.text_input("Ticker","NVDA",key="live_t").upper()
    live_day=cx.date_input("Handelsdag",date.today(),key="live_day")
    c1,c2=st.columns(2)
    if c1.button("Aandeel ophalen",use_container_width=True):
        try:
            d=get(f"/live/stocks/{ticker}",{"day":str(live_day)},timeout=30);st.dataframe(pd.DataFrame(d.get("rows") or []),use_container_width=True)
        except Exception as e:st.error(str(e))
    if c2.button("Option chain ophalen",use_container_width=True):
        try:
            d=get(f"/live/options/{ticker}/chain",timeout=45);st.dataframe(pd.DataFrame(d.get("rows") or d),use_container_width=True)
        except Exception as e:st.error(str(e))

def page_system():
    hero("Beheer","Systeem","Server, workers, verbindingen en diagnostiek.")
    try:
        s=get("/server/status",timeout=20);st.json(s)
    except Exception as e:st.error(str(e))
    c1,c2=st.columns(2)
    if c1.button("Test verbindingen",type="primary",use_container_width=True):
        try:st.json(post("/setup/test",{}))
        except Exception as e:st.error(str(e))
    if c2.button("Open Data-overzicht",use_container_width=True):go("data")
    st.markdown("### Earnings-bronnen")
    try:
        settings=get("/setup/status",timeout=20).get("values") or {}
        c1,c2=st.columns(2)
        fmp=c1.text_input("Nieuwe FMP API key",type="password",placeholder="ingesteld" if settings.get("fmp_api_key") else "ontbreekt",key="sys_fmp")
        sec=c2.text_input("SEC User-Agent",value="" if str(settings.get("sec_user_agent","")).startswith("••") else settings.get("sec_user_agent","") or "",placeholder="OptionEdge Naam naam@example.nl",key="sys_sec")
        if st.button("Earnings-instellingen opslaan",use_container_width=True):
            put("/settings",{"fmp_api_key":fmp or None,"sec_user_agent":sec or None})
            st.success("SEC/FMP-instellingen opgeslagen.")
    except Exception as e:st.warning(str(e))
    try:
        d=get("/agents/diagnostics",timeout=45);st.markdown("### Diagnostiek");st.json(d)
    except Exception as e:st.warning(str(e))

# ---------- dispatch ----------
if page=="home":page_home()
elif page=="method":page_method()
elif page=="data":page_data()
elif page=="vectors":page_vectors()
elif page=="agents":page_agents()
elif page=="training":page_training()
elif page=="forecast":page_forecast()
elif page=="relationships":page_relationships()
elif page=="coach":page_coach()
elif page=="news":page_news()
elif page=="tasks":page_tasks()
elif page=="roadmap":page_roadmap()
elif page=="live":page_live()
elif page=="system":page_system()
elif page=="agent":
    if agent_route=="gpu_news":render_gpu_agent()
    else:render_standard_agent(agent_route)
