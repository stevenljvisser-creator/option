import os,time
from datetime import datetime,timezone
from sqlalchemy import select
from core.db import SessionLocal,init_db
from core.models import AgentJob,AppSetting,ModelRun
from worker.tasks import run_agent_job,check_auto_agents

POLL_SECONDS=2
AUTO_CHECK_SECONDS=60
WORKER_NAME=os.getenv("WORKER_NAME","market")
ALLOWED_AGENTS=[x.strip() for x in os.getenv("WORKER_AGENTS","stocks,options").split(",") if x.strip()]
AUTO_SCHEDULER=os.getenv("AUTO_SCHEDULER","0")=="1"

def heartbeat():
    db=SessionLocal()
    try:
        key=f"WORKER_HEARTBEAT_{WORKER_NAME.upper()}"
        x=db.get(AppSetting,key) or AppSetting(key=key)
        x.value=datetime.now(timezone.utc).isoformat();x.encrypted=False
        db.add(x);db.commit()
    finally:db.close()

def recover():
    db=SessionLocal()
    try:
        rows=db.scalars(select(AgentJob).where(
            AgentJob.status=="running",AgentJob.agent.in_(ALLOWED_AGENTS)
        )).all()
        for j in rows:
            j.status="queued";j.current_fraction=0
            j.message=f"Worker {WORKER_NAME} herstart; job opnieuw in wachtrij."
            if j.agent in ["model","pair_model","fusion_model","alpha_research"]:
                run_id=int((j.payload or {}).get("model_run_id",0) or 0)
                mr=db.get(ModelRun,run_id) if run_id else None
                if mr:
                    mr.status="queued"
                    mr.message="Modelworker herstart; training opnieuw in wachtrij."
        db.commit()
    finally:db.close()

def next_job():
    db=SessionLocal()
    try:
        j=db.scalar(select(AgentJob).where(
            AgentJob.status=="queued",AgentJob.agent.in_(ALLOWED_AGENTS)
        ).order_by(AgentJob.id.asc()))
        return j.id if j else None
    finally:db.close()

def main():
    init_db();recover()
    last_auto=0;last_hb=0
    print(f"Worker {WORKER_NAME}: {ALLOWED_AGENTS}",flush=True)
    while True:
        t=time.time()
        if t-last_hb>=10:
            try:heartbeat()
            except Exception as e:print("heartbeat:",e,flush=True)
            last_hb=t
        if AUTO_SCHEDULER and t-last_auto>=AUTO_CHECK_SECONDS:
            try:check_auto_agents()
            except Exception as e:print("auto:",e,flush=True)
            last_auto=t
        jid=next_job()
        if jid:
            run_agent_job(jid);heartbeat();continue
        time.sleep(POLL_SECONDS)

if __name__=="__main__":main()
