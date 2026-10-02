"""
SQLAlchemy 2.x ORM models.

Author: Roshitha K

Timestamps are naive UTC for portability between PostgreSQL TIMESTAMP and SQLite.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Naive UTC 'now' (datetime.utcnow is deprecated in Python 3.12)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class SensorLog(Base):
    """One IoT telemetry reading for one city zone."""

    __tablename__ = "sensor_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    zone_id: Mapped[str] = mapped_column(String(32), nullable=False)
    traffic_density: Mapped[float] = mapped_column(Float, nullable=False)  # vehicles/min
    aqi: Mapped[float] = mapped_column(Float, nullable=False)
    power_consumption: Mapped[float] = mapped_column(Float, nullable=False)  # MW
    noise_level: Mapped[float] = mapped_column(Float, nullable=False)  # dB

    __table_args__ = (Index("ix_sensor_logs_zone_ts", "zone_id", "timestamp"),)


class UserActivity(Base):
    """Audit trail of logins, page views and SQL executed from the admin panel."""

    __tablename__ = "user_activity"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    username: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    details: Mapped[str] = mapped_column(Text, nullable=True)
