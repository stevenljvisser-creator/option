from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from core.config import DATABASE_URL

engine=create_engine(DATABASE_URL,pool_pre_ping=True,future=True)
SessionLocal=sessionmaker(bind=engine,autoflush=False,expire_on_commit=False)

class Base(DeclarativeBase): pass


# Existing OptionEdge/MarketScope installations can be several schema generations
# behind the current Python models. SQLAlchemy create_all() creates missing tables,
# but deliberately does NOT add new columns to tables that already exist.  Keep
# this upgrade additive and idempotent: never drop/rename columns or touch data.
_POSTGRES_ADD_COLUMNS={
    "agent_configs":{
        "auto_enabled":"BOOLEAN DEFAULT FALSE",
        "interval_minutes":"INTEGER DEFAULT 60",
        "default_start":"DATE",
        "tickers":"JSONB DEFAULT '[]'::jsonb",
        "updated_at":"TIMESTAMPTZ DEFAULT NOW()",
    },
    "agent_cursors":{
        "last_successful_date":"DATE",
        "updated_at":"TIMESTAMPTZ DEFAULT NOW()",
    },
    "agent_jobs":{
        "agent":"VARCHAR(30) DEFAULT 'unknown'",
        "mode":"VARCHAR(20) DEFAULT 'full'",
        "status":"VARCHAR(20) DEFAULT 'queued'",
        "start_date":"DATE",
        "end_date":"DATE",
        "tickers":"JSONB DEFAULT '[]'::jsonb",
        "total_units":"INTEGER DEFAULT 0",
        "completed_units":"INTEGER DEFAULT 0",
        "current_fraction":"DOUBLE PRECISION DEFAULT 0",
        "current_item":"VARCHAR(255) DEFAULT ''",
        "message":"TEXT DEFAULT ''",
        "stop_requested":"BOOLEAN DEFAULT FALSE",
        "celery_task_id":"VARCHAR(80) DEFAULT ''",
        "payload":"JSONB DEFAULT '{}'::jsonb",
        "created_at":"TIMESTAMPTZ DEFAULT NOW()",
        "started_at":"TIMESTAMPTZ",
        "finished_at":"TIMESTAMPTZ",
        "updated_at":"TIMESTAMPTZ DEFAULT NOW()",
    },
    "agent_company_progress":{
        "job_id":"INTEGER DEFAULT 0",
        "ticker":"VARCHAR(32) DEFAULT ''",
        "status":"VARCHAR(20) DEFAULT 'queued'",
        "total_units":"INTEGER DEFAULT 0",
        "completed_units":"INTEGER DEFAULT 0",
        "current_fraction":"DOUBLE PRECISION DEFAULT 0",
        "current_item":"VARCHAR(255) DEFAULT ''",
        "message":"TEXT DEFAULT ''",
        "updated_at":"TIMESTAMPTZ DEFAULT NOW()",
    },
    "model_runs":{
        "status":"VARCHAR(20) DEFAULT 'queued'",
        "stage":"VARCHAR(20) DEFAULT 'C1'",
        "start_date":"DATE",
        "end_date":"DATE",
        "tickers":"JSONB DEFAULT '[]'::jsonb",
        "target":"VARCHAR(64) DEFAULT 'target_extrinsic_return_30m'",
        "horizon_minutes":"INTEGER DEFAULT 30",
        "max_rows":"INTEGER DEFAULT 200000",
        "progress":"DOUBLE PRECISION DEFAULT 0",
        "message":"TEXT DEFAULT ''",
        "metrics":"JSONB DEFAULT '{}'::jsonb",
        "feature_sets":"JSONB DEFAULT '{}'::jsonb",
        "artifact_key":"TEXT DEFAULT ''",
        "created_at":"TIMESTAMPTZ DEFAULT NOW()",
        "started_at":"TIMESTAMPTZ",
        "finished_at":"TIMESTAMPTZ",
    },
    "app_settings":{
        "value":"TEXT DEFAULT ''",
        "encrypted":"BOOLEAN DEFAULT FALSE",
        "updated_at":"TIMESTAMPTZ DEFAULT NOW()",
    },
}

_POSTGRES_INDEXES=(
    "CREATE INDEX IF NOT EXISTS ix_agent_jobs_agent ON agent_jobs (agent)",
    "CREATE INDEX IF NOT EXISTS ix_agent_jobs_status ON agent_jobs (status)",
    "CREATE INDEX IF NOT EXISTS ix_agent_company_progress_job_id ON agent_company_progress (job_id)",
    "CREATE INDEX IF NOT EXISTS ix_agent_company_progress_ticker ON agent_company_progress (ticker)",
    "CREATE INDEX IF NOT EXISTS ix_model_runs_status ON model_runs (status)",
)

# Portable equivalents are used only by local diagnostics/tests. Production is
# PostgreSQL, where JSONB and TIMESTAMPTZ are retained above.
_SQLITE_ADD_COLUMNS={
    table:{
        column:(
            ddl.replace("JSONB DEFAULT '[]'::jsonb","TEXT DEFAULT '[]'")
               .replace("JSONB DEFAULT '{}'::jsonb","TEXT DEFAULT '{}'")
               .replace("TIMESTAMPTZ DEFAULT NOW()","DATETIME")
               .replace("TIMESTAMPTZ","DATETIME")
               .replace("DOUBLE PRECISION","FLOAT")
               .replace("BOOLEAN DEFAULT FALSE","BOOLEAN DEFAULT 0")
        )
        for column,ddl in columns.items()
    }
    for table,columns in _POSTGRES_ADD_COLUMNS.items()
}


def _ensure_additive_schema(conn):
    """Add current model columns without deleting or rewriting existing data."""
    schema=inspect(conn)
    existing_tables=set(schema.get_table_names())
    if conn.dialect.name=="postgresql":
        for table,columns in _POSTGRES_ADD_COLUMNS.items():
            if table not in existing_tables:
                continue
            for column,ddl in columns.items():
                conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS "{column}" {ddl}'))
        for statement in _POSTGRES_INDEXES:
            conn.execute(text(statement))
        return

    if conn.dialect.name=="sqlite":
        # SQLite lacks ADD COLUMN IF NOT EXISTS on common deployed versions.
        # Re-inspect each table and add only missing columns.
        for table,columns in _SQLITE_ADD_COLUMNS.items():
            if table not in existing_tables:
                continue
            present={x["name"] for x in inspect(conn).get_columns(table)}
            for column,ddl in columns.items():
                if column not in present:
                    conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {ddl}'))
                    present.add(column)


def init_db():
    from core import models  # noqa: F401 - registers metadata
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        _ensure_additive_schema(conn)


def db_session():
    db=SessionLocal()
    try: yield db
    finally: db.close()
