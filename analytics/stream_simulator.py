"""
Real-time urban IoT telemetry simulator.

Author: Roshitha K

Research context
----------------
Urban computing (Zheng et al., 2014, ACM TIST) frames smart-city analytics as a
loop of sensing -> data management -> analytics -> services. Real-time sensing
of traffic, air quality, energy and noise lets planners act on current
conditions rather than on static historical reports (Batty, 2013; Kitchin, 2014).

This module plays the *sensing* layer: it emits one reading per zone per batch
with a diurnal (morning/evening rush) pattern, zone-specific baselines and
physically plausible couplings (more traffic -> worse AQI and louder streets).
The data is synthetic; nothing here is measured from a real city.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

import numpy as np
import pandas as pd

import config

SENSOR_COLUMNS = ["timestamp", "zone_id", "traffic_density", "aqi", "power_consumption", "noise_level"]


@dataclass(frozen=True)
class ZoneProfile:
    base_traffic: float      # vehicles/min at the diurnal reference level
    aqi_base: float
    aqi_traffic_coef: float  # AQI points per vehicle/min
    power_base: float        # MW
    power_amp: float         # MW added at peak demand
    noise_base: float        # dB
    noise_traffic_coef: float


ZONE_PROFILES = {
    "Downtown":    ZoneProfile(60, 55, 0.55, 42.0, 18.0, 52, 0.30),
    "Industrial":  ZoneProfile(40, 80, 0.45, 75.0, 25.0, 58, 0.25),
    "Residential": ZoneProfile(30, 38, 0.50, 28.0, 14.0, 44, 0.28),
    "Port":        ZoneProfile(45, 70, 0.50, 55.0, 20.0, 55, 0.27),
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def diurnal_factor(ts: datetime) -> float:
    """Smooth 0..~0.75 demand curve with morning, midday and evening bumps."""
    h = ts.hour + ts.minute / 60.0
    return (
        0.55 * math.exp(-((h - 8.5) ** 2) / (2 * 1.8 ** 2))
        + 0.65 * math.exp(-((h - 17.5) ** 2) / (2 * 2.0 ** 2))
        + 0.25 * math.exp(-((h - 13.0) ** 2) / (2 * 3.0 ** 2))
    )


class StreamSimulator:
    """Generates telemetry batches. Pass `seed` for reproducible output."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def _reading(self, zone: str, ts: datetime) -> dict:
        p = ZONE_PROFILES[zone]
        d = diurnal_factor(ts)
        traffic = max(0.0, p.base_traffic * (0.4 + d) + self.rng.normal(0, p.base_traffic * 0.06))
        aqi = float(np.clip(p.aqi_base + p.aqi_traffic_coef * traffic + self.rng.normal(0, 4), 5, 400))
        power = max(1.0, p.power_base + p.power_amp * d + self.rng.normal(0, 0.03 * p.power_base))
        noise = float(np.clip(p.noise_base + p.noise_traffic_coef * traffic + self.rng.normal(0, 1.5), 30, 110))
        return {
            "timestamp": ts,
            "zone_id": zone,
            "traffic_density": round(traffic, 2),
            "aqi": round(aqi, 1),
            "power_consumption": round(power, 2),
            "noise_level": round(noise, 1),
        }

    def generate_batch(self, ts: Optional[datetime] = None) -> pd.DataFrame:
        """One reading per zone, all sharing the same timestamp."""
        ts = (ts or _utcnow()).replace(microsecond=0)
        return pd.DataFrame([self._reading(z, ts) for z in config.ZONES], columns=SENSOR_COLUMNS)

    def generate_history(
        self, n_rows: int, end: Optional[datetime] = None, step_minutes: int = 15
    ) -> pd.DataFrame:
        """`n_rows` ascending readings (zones interleaved), ending at `end`."""
        if n_rows <= 0:
            return pd.DataFrame(columns=SENSOR_COLUMNS)
        end = (end or _utcnow()).replace(microsecond=0)
        zones = config.ZONES
        steps = math.ceil(n_rows / len(zones))
        rows = []
        for i in range(steps):
            ts = end - timedelta(minutes=step_minutes * (steps - 1 - i))
            rows.extend(self._reading(z, ts) for z in zones)
        return pd.DataFrame(rows[-n_rows:], columns=SENSOR_COLUMNS).reset_index(drop=True)

    def stream(self, interval_s: float = config.STREAM_INTERVAL_S,
               max_batches: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Blocking generator yielding a fresh batch every `interval_s` seconds."""
        n = 0
        while max_batches is None or n < max_batches:
            yield self.generate_batch()
            n += 1
            if max_batches is None or n < max_batches:
                time.sleep(interval_s)
