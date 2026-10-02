"""
Database engine, seeding and guarded raw-SQL execution.

Author: Roshitha K

* Tries PostgreSQL first; if it is unreachable, falls back to a local SQLite file
  so the demo always runs.
* Seeds 100 historical sensor logs when the sensor table is empty.
* `execute_raw_sql` accepts exactly one statement, is read-only by default
  (rolled back after execution) and only writes when `allow_write=True`.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

import config
from analytics.stream_simulator import StreamSimulator
from database.models import Base, SensorLog, UserActivity

log = logging.getLogger(__name__)

_READ_ONLY_STARTS = ("select", "with", "explain", "show")
_WRITE_WORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|attach|detach|pragma|replace|grant|revoke|vacuum|copy)\b",
    re.IGNORECASE,
)


@dataclass
class SQLResult:
    ok: bool
    message: str
    dataframe: Optional[pd.DataFrame] = None
    rowcount: Optional[int] = None
    elapsed_ms: float = 0.0


class DatabaseManager:
    def __init__(self, primary_url: Optional[str] = None, fallback_url: Optional[str] = None,
                 seed: bool = True) -> None:
        self.engine, self.backend = self._connect(
            primary_url or config.DATABASE_URL, fallback_url or config.FALLBACK_DATABASE_URL
        )
        Base.metadata.create_all(self.engine)
        self._sessions = sessionmaker(bind=self.engine, expire_on_commit=False)
        if seed:
            self.seed_if_empty()

    # -- connection ----------------------------------------------------------
    @staticmethod
    def _make_engine(url: str) -> Engine:
        if url.startswith("sqlite"):
            return create_engine(url, connect_args={"check_same_thread": False})
        return create_engine(
            url, pool_pre_ping=True,
            connect_args={"connect_timeout": config.DB_CONNECT_TIMEOUT_S},
        )

    def _connect(self, primary: str, fallback: str):
        try:
            engine = self._make_engine(primary)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine, engine.dialect.name
        except Exception as exc:  # noqa: BLE001 - any driver/network error triggers fallback
            log.warning("Primary database unavailable (%s); using fallback %s", type(exc).__name__, fallback)
        engine = self._make_engine(fallback)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine, engine.dialect.name

    def session(self) -> Session:
        return self._sessions()

    # -- seeding / ingestion -------------------------------------------------
    def seed_if_empty(self, n_rows: int = config.SEED_ROW_COUNT) -> int:
        with self.session() as s:
            count = s.scalar(select(func.count()).select_from(SensorLog))
        if count:
            return 0
        self.insert_logs(StreamSimulator(seed=42).generate_history(n_rows))
        log.info("Seeded %d historical sensor logs", n_rows)
        return n_rows

    def insert_logs(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cols = ["timestamp", "zone_id", "traffic_density", "aqi", "power_consumption", "noise_level"]
        df[cols].to_sql(SensorLog.__tablename__, self.engine, if_exists="append", index=False)
        return len(df)

    def count_logs(self) -> int:
        with self.session() as s:
            return int(s.scalar(select(func.count()).select_from(SensorLog)) or 0)

    # -- queries -------------------------------------------------------------
    def fetch_recent(self, limit: int = 800, zones: Optional[list] = None) -> pd.DataFrame:
        stmt = select(
            SensorLog.id, SensorLog.timestamp, SensorLog.zone_id, SensorLog.traffic_density,
            SensorLog.aqi, SensorLog.power_consumption, SensorLog.noise_level,
        )
        if zones:
            stmt = stmt.where(SensorLog.zone_id.in_(zones))
        stmt = stmt.order_by(SensorLog.timestamp.desc(), SensorLog.id.desc()).limit(limit)
        df = pd.read_sql(stmt, self.engine)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values(["timestamp", "id"]).reset_index(drop=True)

    def log_activity(self, username: str, role: Optional[str], action: str, details: str = "") -> None:
        with self.session() as s:
            s.add(UserActivity(username=username, role=role, action=action, details=details[:4000]))
            s.commit()

    def fetch_activity(self, limit: int = 200, action_prefix: Optional[str] = None) -> pd.DataFrame:
        stmt = select(UserActivity.timestamp, UserActivity.username, UserActivity.role,
                      UserActivity.action, UserActivity.details)
        if action_prefix:
            stmt = stmt.where(UserActivity.action.like(f"{action_prefix}%"))
        stmt = stmt.order_by(UserActivity.id.desc()).limit(limit)
        return pd.read_sql(stmt, self.engine)

    def table_names(self) -> list:
        return sorted(inspect(self.engine).get_table_names())

    def preview_table(self, table: str, limit: int = 50) -> pd.DataFrame:
        if table not in self.table_names():  # whitelist: never interpolate unknown names
            raise ValueError(f"Unknown table: {table!r}")
        quoted = self.engine.dialect.identifier_preparer.quote(table)
        return pd.read_sql(text(f"SELECT * FROM {quoted} LIMIT :n"), self.engine, params={"n": int(limit)})

    # -- raw SQL -------------------------------------------------------------
    @staticmethod
    def _validate(sql: str) -> str:
        cleaned = (sql or "").strip().rstrip(";").strip()
        if not cleaned:
            raise ValueError("Empty SQL statement.")
        if ";" in cleaned:
            raise ValueError("Only a single SQL statement is allowed.")
        if "--" in cleaned or "/*" in cleaned:
            raise ValueError("SQL comments are not allowed.")
        return cleaned

    def execute_raw_sql(self, sql: str, params: Optional[dict] = None,
                        allow_write: bool = False) -> SQLResult:
        """Run one statement. Returns a DataFrame for row-returning queries,
        otherwise an execution status with the affected row count."""
        start = perf_counter()
        try:
            cleaned = self._validate(sql)
            if not allow_write:
                first = cleaned.split(None, 1)[0].lower()
                if first not in _READ_ONLY_STARTS or _WRITE_WORDS.search(cleaned):
                    raise ValueError("Read-only mode: enable 'Allow write statements' to run this.")
            ctx = self.engine.begin() if allow_write else self.engine.connect()
            with ctx as conn:
                result = conn.execute(text(cleaned), params or {})
                if result.returns_rows:
                    df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
                    outcome = SQLResult(True, f"{len(df)} row(s) returned.", df, len(df))
                else:
                    outcome = SQLResult(True, f"Statement executed; {result.rowcount} row(s) affected.",
                                        None, result.rowcount)
                if not allow_write:
                    conn.rollback()
            outcome.elapsed_ms = (perf_counter() - start) * 1000
            return outcome
        except Exception as exc:  # noqa: BLE001 - surface any DB error to the admin UI
            return SQLResult(False, f"{type(exc).__name__}: {exc}", None, None,
                             (perf_counter() - start) * 1000)
