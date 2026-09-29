from sqlalchemy import create_engine, inspect, text

from core.db import _ensure_additive_schema


def test_additive_schema_upgrade_preserves_old_rows_and_adds_current_columns():
    engine=create_engine("sqlite+pysqlite:///:memory:",future=True)
    with engine.begin() as conn:
        # Simulate an older OptionEdge/MarketScope database: tables exist, but
        # several columns introduced by newer code are absent.
        conn.execute(text("""
            CREATE TABLE agent_jobs (
                id INTEGER PRIMARY KEY,
                agent VARCHAR(30),
                status VARCHAR(20),
                message TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE model_runs (
                id INTEGER PRIMARY KEY,
                status VARCHAR(20),
                stage VARCHAR(20),
                metrics TEXT
            )
        """))
        conn.execute(text("INSERT INTO agent_jobs(id,agent,status,message) VALUES (7,'features','done','keep me')"))
        conn.execute(text("INSERT INTO model_runs(id,status,stage,metrics) VALUES (3,'done','C1','{}')"))

        _ensure_additive_schema(conn)
        # Idempotency matters because API + workers can start concurrently.
        _ensure_additive_schema(conn)

        job_cols={c["name"] for c in inspect(conn).get_columns("agent_jobs")}
        model_cols={c["name"] for c in inspect(conn).get_columns("model_runs")}
        assert {"payload","current_fraction","tickers","created_at","updated_at"} <= job_cols
        assert {"start_date","end_date","tickers","feature_sets","artifact_key","finished_at"} <= model_cols

        row=conn.execute(text("SELECT id,agent,status,message FROM agent_jobs WHERE id=7")).one()
        assert tuple(row)==(7,"features","done","keep me")
        run=conn.execute(text("SELECT id,status,stage,metrics FROM model_runs WHERE id=3")).one()
        assert tuple(run)==(3,"done","C1","{}")
