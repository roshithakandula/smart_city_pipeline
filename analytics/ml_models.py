"""
PyTorch regression: estimate traffic congestion (vehicles/min) from
AQI, power consumption and noise level.

Author: Roshitha K

Research context
----------------
Predictive trend modelling replaces static historical reporting: instead of a
monthly PDF describing last month's congestion, a model fed by live telemetry
estimates current/near-term load (Batty, 2013; Zheng et al., 2014). This MLP is
deliberately small (3 -> 32 -> 16 -> 1) so it trains in seconds on CPU.

Evaluation uses a chronological split (last 20% of time as validation) and
reports MAE / RMSE / R^2 against a predict-the-training-mean baseline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
from analytics.stream_simulator import StreamSimulator

log = logging.getLogger(__name__)

FEATURES = ["aqi", "power_consumption", "noise_level"]
TARGET = "traffic_density"


class CongestionNet(nn.Module):
    def __init__(self, n_in: int = len(FEATURES), hidden: Sequence[int] = (32, 16), dropout: float = 0.05):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class Normalizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Normalizer":
        x = df[FEATURES].to_numpy(dtype=np.float64)
        y = df[TARGET].to_numpy(dtype=np.float64)
        return cls(x.mean(0), np.where(x.std(0) < 1e-8, 1.0, x.std(0)),
                   float(y.mean()), float(y.std()) if y.std() > 1e-8 else 1.0)

    def x(self, arr: np.ndarray) -> np.ndarray:
        return ((arr - self.x_mean) / self.x_std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(),
                "y_mean": self.y_mean, "y_std": self.y_std}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(np.array(d["x_mean"]), np.array(d["x_std"]), float(d["y_mean"]), float(d["y_std"]))


def bootstrap_training_frame(db_df: pd.DataFrame, min_rows: int = config.MIN_TRAIN_ROWS,
                             synthetic_rows: int = 2400) -> Tuple[pd.DataFrame, bool]:
    """Return (frame, augmented). If the database holds fewer than `min_rows`
    rows, older synthetic history is prepended so a first model can be trained."""
    if len(db_df) >= min_rows:
        return db_df, False
    end = pd.to_datetime(db_df["timestamp"]).min().to_pydatetime() if len(db_df) else None
    if end is not None:
        end = end - pd.Timedelta(minutes=15).to_pytimedelta()
    synth = StreamSimulator(seed=7).generate_history(synthetic_rows, end=end)
    frame = pd.concat([synth, db_df[synth.columns]], ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True), True


class CongestionPredictor:
    def __init__(self, checkpoint_path: Path = config.CHECKPOINT_PATH) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model: Optional[CongestionNet] = None
        self.norm: Optional[Normalizer] = None
        self.metrics: Dict[str, float] = {}

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.norm is not None

    # -- training ------------------------------------------------------------
    def train(self, df: pd.DataFrame, epochs: int = 150, lr: float = 3e-3, batch_size: int = 64,
              val_frac: float = 0.2, patience: int = 25, seed: int = 42) -> Dict[str, float]:
        df = df.dropna(subset=FEATURES + [TARGET])
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp", kind="stable")
        if len(df) < 50:
            raise ValueError(f"Need at least 50 rows to train, got {len(df)}.")
        n_val = max(10, int(len(df) * val_frac))
        train_df, val_df = df.iloc[:-n_val], df.iloc[-n_val:]

        torch.manual_seed(seed)
        np.random.seed(seed)
        norm = Normalizer.fit(train_df)

        def tensors(frame: pd.DataFrame):
            x = torch.from_numpy(norm.x(frame[FEATURES].to_numpy(dtype=np.float64)))
            y = torch.from_numpy(((frame[TARGET].to_numpy(dtype=np.float64) - norm.y_mean) / norm.y_std).astype(np.float32))
            return x, y

        xtr, ytr = tensors(train_df)
        xva, yva = tensors(val_df)
        gen = torch.Generator().manual_seed(seed)
        loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True, generator=gen)

        model = CongestionNet()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        best_val, best_state, stale, last_epoch = float("inf"), None, 0, 0
        for epoch in range(1, epochs + 1):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(xva), yva).item()
            last_epoch = epoch
            if val_loss < best_val - 1e-6:
                best_val, stale = val_loss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            pred = model(xva).numpy() * norm.y_std + norm.y_mean
        true = val_df[TARGET].to_numpy(dtype=np.float64)
        err = pred - true
        ss_res, ss_tot = float((err ** 2).sum()), float(((true - true.mean()) ** 2).sum())
        base_err = true - norm.y_mean
        self.model, self.norm = model, norm
        self.metrics = {
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "baseline_mae": float(np.abs(base_err).mean()),
            "baseline_rmse": float(np.sqrt((base_err ** 2).mean())),
            "best_val_loss": float(best_val),
            "epochs_run": last_epoch,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
        }
        self.save()
        return self.metrics

    # -- inference -----------------------------------------------------------
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError("Model is not trained or loaded.")
        x = torch.from_numpy(self.norm.x(df[FEATURES].to_numpy(dtype=np.float64)))
        self.model.eval()
        with torch.no_grad():
            out = self.model(x).numpy() * self.norm.y_std + self.norm.y_mean
        return np.clip(out, 0.0, None)

    def predict(self, aqi: float, power_consumption: float, noise_level: float) -> float:
        row = pd.DataFrame([{"aqi": aqi, "power_consumption": power_consumption, "noise_level": noise_level}])
        return float(self.predict_batch(row)[0])

    # -- persistence ---------------------------------------------------------
    def save(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "normalizer": self.norm.to_dict(),
                    "metrics": self.metrics, "features": FEATURES}, self.checkpoint_path)

    def load(self) -> bool:
        if not self.checkpoint_path.exists():
            return False
        try:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            if ckpt.get("features") != FEATURES:
                log.warning("Checkpoint feature list differs from code; retrain required.")
                return False
            model = CongestionNet()
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self.model, self.norm, self.metrics = model, Normalizer.from_dict(ckpt["normalizer"]), ckpt["metrics"]
            return True
        except Exception as exc:  # noqa: BLE001 - corrupt/incompatible checkpoint -> retrain
            log.warning("Could not load checkpoint: %s", exc)
            return False


if __name__ == "__main__":  # python -m analytics.ml_models
    frame, _ = bootstrap_training_frame(StreamSimulator(seed=1).generate_history(0))
    p = CongestionPredictor()
    print(p.train(frame))
