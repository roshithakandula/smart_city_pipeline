"""
Central configuration for the GenAI & Smart City Real-Time Analytics Pipeline.

Author: Roshitha K

Every value can be overridden with an environment variable so the same code
runs on a laptop (SQLite fallback) and in a deployment (PostgreSQL).
All timestamps used across the project are naive UTC.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

APP_TITLE = "GenAI & Smart City Real-Time Analytics Pipeline"
AUTHOR = "Roshitha K"

# --- Database ---------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/smart_city",
)
FALLBACK_DATABASE_URL = os.getenv(
    "FALLBACK_DATABASE_URL", "sqlite:///smart_city_fallback.db"
)
DB_CONNECT_TIMEOUT_S = int(os.getenv("DB_CONNECT_TIMEOUT_S", "3"))
SEED_ROW_COUNT = int(os.getenv("SEED_ROW_COUNT", "100"))

# --- Security ---------------------------------------------------------------
# If JWT_SECRET is not provided a random per-process secret is used, which is
# safe but invalidates all sessions on restart. Set it explicitly in production.
JWT_SECRET = os.getenv("JWT_SECRET") or secrets.token_urlsafe(48)
JWT_SECRET_IS_EPHEMERAL = "JWT_SECRET" not in os.environ or not os.environ["JWT_SECRET"]
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
BCRYPT_ROUNDS = int(os.getenv("BCRYPT_ROUNDS", "12"))

# --- Domain -----------------------------------------------------------------
ZONES = ["Downtown", "Industrial", "Residential", "Port"]
STREAM_INTERVAL_S = int(os.getenv("STREAM_INTERVAL_S", "5"))

# --- ML ---------------------------------------------------------------------
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", BASE_DIR / "analytics" / "checkpoints"))
CHECKPOINT_PATH = CHECKPOINT_DIR / "congestion_mlp.pt"
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "400"))

# --- RAG --------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", BASE_DIR / "genai" / "faiss_index"))
EXTRA_REPORTS_DIR = Path(os.getenv("EXTRA_REPORTS_DIR", BASE_DIR / "genai" / "reports"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.12"))
