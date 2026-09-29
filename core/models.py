from datetime import datetime,timezone,date
from sqlalchemy import String,Integer,Float,Boolean,DateTime,Date,Text,JSON,UniqueConstraint
from sqlalchemy.orm import Mapped,mapped_column
from core.db import Base

def utcnow(): return datetime.now(timezone.utc)

class AgentConfig(Base):
    __tablename__="agent_configs"
    agent:Mapped[str]=mapped_column(String(30),primary_key=True)
    auto_enabled:Mapped[bool]=mapped_column(Boolean,default=False)
    interval_minutes:Mapped[int]=mapped_column(Integer,default=60)
    default_start:Mapped[date|None]=mapped_column(Date,nullable=True)
    tickers:Mapped[list]=mapped_column(JSON,default=list)
    updated_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=utcnow,onupdate=utcnow)

class AgentCursor(Base):
    __tablename__="agent_cursors"
    agent:Mapped[str]=mapped_column(String(30),primary_key=True)
    last_successful_date:Mapped[date|None]=mapped_column(Date,nullable=True)
    updated_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=utcnow,onupdate=utcnow)

class AgentJob(Base):
    __tablename__="agent_jobs"
    id:Mapped[int]=mapped_column(Integer,primary_key=True,autoincrement=True)
    agent:Mapped[str]=mapped_column(String(30),index=True)
    mode:Mapped[str]=mapped_column(String(20),default="full")
    status:Mapped[str]=mapped_column(String(20),default="queued",index=True)
    start_date:Mapped[date|None]=mapped_column(Date,nullable=True)
    end_date:Mapped[date|None]=mapped_column(Date,nullable=True)
    tickers:Mapped[list]=mapped_column(JSON,default=list)
    total_units:Mapped[int]=mapped_column(Integer,default=0)
    completed_units:Mapped[int]=mapped_column(Integer,default=0)
    current_fraction:Mapped[float]=mapped_column(Float,default=0.0)
    current_item:Mapped[str]=mapped_column(String(255),default="")
    message:Mapped[str]=mapped_column(Text,default="")
    stop_requested:Mapped[bool]=mapped_column(Boolean,default=False)
    celery_task_id:Mapped[str]=mapped_column(String(80),default="")
    payload:Mapped[dict]=mapped_column(JSON,default=dict)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=utcnow)
    started_at:Mapped[datetime|None]=mapped_column(DateTime(timezone=True),nullable=True)
    finished_at:Mapped[datetime|None]=mapped_column(DateTime(timezone=True),nullable=True)
    updated_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=utcnow,onupdate=utcnow)

class AgentCompanyProgress(Base):
    __tablename__="agent_company_progress"
    id:Mapped[int]=mapped_column(Integer,primary_key=True,autoincrement=True)
    job_id:Mapped[int]=mapped_column(Integer,index=True)
    ticker:Mapped[str]=mapped_column(String(32),index=True)
    status:Mapped[str]=mapped_column(String(20),default="queued")
    total_units:Mapped[int]=mapped_column(Integer,default=0)
    completed_units:Mapped[int]=mapped_column(Integer,default=0)
    current_fraction:Mapped[float]=mapped_column(Float,default=0.0)
    current_item:Mapped[str]=mapped_column(String(255),default="")
    message:Mapped[str]=mapped_column(Text,default="")
    updated_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=utcnow,onupdate=utcnow)
    __table_args__=(UniqueConstraint("job_id","ticker",name="uq_job_ticker"),)


class ModelRun(Base):
    __tablename__="model_runs"
    id:Mapped[int]=mapped_column(Integer,primary_key=True,autoincrement=True)
    status:Mapped[str]=mapped_column(String(20),default="queued",index=True)
    stage:Mapped[str]=mapped_column(String(20),default="C1")
    start_date:Mapped[date|None]=mapped_column(Date,nullable=True)
    end_date:Mapped[date|None]=mapped_column(Date,nullable=True)
    tickers:Mapped[list]=mapped_column(JSON,default=list)
    target:Mapped[str]=mapped_column(String(64),default="target_extrinsic_return_30m")
    horizon_minutes:Mapped[int]=mapped_column(Integer,default=30)
    max_rows:Mapped[int]=mapped_column(Integer,default=200000)
    progress:Mapped[float]=mapped_column(Float,default=0.0)
    message:Mapped[str]=mapped_column(Text,default="")
    metrics:Mapped[dict]=mapped_column(JSON,default=dict)
    feature_sets:Mapped[dict]=mapped_column(JSON,default=dict)
    artifact_key:Mapped[str]=mapped_column(Text,default="")
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=utcnow)
    started_at:Mapped[datetime|None]=mapped_column(DateTime(timezone=True),nullable=True)
    finished_at:Mapped[datetime|None]=mapped_column(DateTime(timezone=True),nullable=True)

class AppSetting(Base):

    __tablename__="app_settings"
    key:Mapped[str]=mapped_column(String(80),primary_key=True)
    value:Mapped[str]=mapped_column(Text,default="")
    encrypted:Mapped[bool]=mapped_column(Boolean,default=False)
    updated_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=utcnow,onupdate=utcnow)
