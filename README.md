# GenAI & Smart City Real-Time Analytics Pipeline

**Author: Roshitha K**

A Streamlit application that simulates real-time urban IoT telemetry (traffic, AQI, power, noise) for four
zones (Downtown, Industrial, Residential, Port), stores it in PostgreSQL (SQLite fallback), forecasts
congestion with a PyTorch MLP, and answers questions over urban research reports with a LangChain + FAISS
RAG assistant. Login uses bcrypt + JWT with role-based access control.

All data is **synthetic**. The built-in report corpus consists of project-authored, paraphrased summaries of
themes from the cited literature (Zheng et al. 2014; Batty 2013; Kitchin 2014; Zanella et al. 2014;
Lewis et al. 2020). The "30% manual reporting reduction" is a **design target**, not a measured result.

## Structure
```
app.py                       Streamlit UI, session handling, 4 pages
config.py                    All settings (env-overridable)
auth/authenticator.py        bcrypt hashing, JWT, RBAC
database/models.py           SensorLog, UserActivity (SQLAlchemy 2.x)
database/db_manager.py       Engine (PostgreSQL -> SQLite fallback), seeding, guarded raw SQL
analytics/stream_simulator.py  Real-time telemetry generator
analytics/ml_models.py       PyTorch regression (AQI, power, noise -> traffic)
genai/rag_pipeline.py        LangChain + FAISS + MiniLM RAG with citations
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501. Demo logins: `admin / admin123` (Admin), `analyst / user123` (User).

* **Database**: by default the app tries `postgresql+psycopg2://postgres:postgres@localhost:5432/smart_city`.
  If unreachable it uses `sqlite:///smart_city_fallback.db`. Tables are created and **100 historical
  sensor logs are seeded automatically** when `sensor_logs` is empty.
  Quick PostgreSQL: `docker run -d --name sc-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=smart_city -p 5432:5432 postgres:16`
* **Model**: on first visit to *PyTorch Predictions* a model trains in seconds and is saved to
  `analytics/checkpoints/congestion_mlp.pt`. Train from CLI: `python -m analytics.ml_models`.
* **RAG**: the first question downloads `all-MiniLM-L6-v2` (~90 MB) and builds the FAISS index in
  `genai/faiss_index/`. CLI: `python -m genai.rag_pipeline "How does real-time data help planners?"`.
  Add your own `.txt`/`.md` reports to `genai/reports/`; the index rebuilds automatically.

## Roles
| Page | Admin | User/Analyst |
|---|---|---|
| Executive Analytics | yes | yes |
| PyTorch Predictions | yes | yes |
| GenAI Assistant | yes | yes |
| Database Management | yes | no |

## Configuration (env vars)
`DATABASE_URL`, `FALLBACK_DATABASE_URL`, `DB_CONNECT_TIMEOUT_S`, `SEED_ROW_COUNT`, `JWT_SECRET`,
`JWT_EXPIRE_MINUTES`, `BCRYPT_ROUNDS`, `STREAM_INTERVAL_S`, `CHECKPOINT_DIR`, `MIN_TRAIN_ROWS`,
`EMBEDDING_MODEL`, `FAISS_INDEX_DIR`, `EXTRA_REPORTS_DIR`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`,
`RAG_TOP_K`, `RAG_MIN_SIMILARITY`. Defaults are in `config.py`.

## Security notes
* Set `JWT_SECRET` to a long random value; otherwise a random per-process secret is used.
* Change the demo passwords in `auth/authenticator.py` (or replace the in-memory user store with a DB table).
* The SQL editor is admin-only, single-statement, read-only by default (rolled back), and audit-logged.
  It is still a powerful tool: use a least-privilege database user in any shared deployment.
* `allow_dangerous_deserialization=True` is used for FAISS loading; it is safe only because the index is
  built locally by this code. Never load an index file from an untrusted source.

## Known limitations
* Telemetry and model training data are simulated; model metrics describe the workflow, not a real city.
* Answers are extractive (stitched from retrieved text), not LLM-generated; `RAG_MIN_SIMILARITY` is a
  heuristic threshold you may need to tune.
* The model uses a random-free chronological split but a single seed; no cross-validation.
* Python 3.10-3.12 targeted. `requirements.txt` has lower bounds only; see Verification status below.

## Verification status (honest)
The authoring sandbox had no network access, so only these were run: `python -m compileall -q .` (exit 0)
and the simulator (4 rows/batch, 100 seeded rows with 25 per zone, ascending timestamps). **Not run:**
pip install of the full stack, SQLAlchemy/PostgreSQL, bcrypt/JWT login, PyTorch training, FAISS/MiniLM
retrieval, and Streamlit itself. Run locally to confirm:
```bash
python -m compileall -q .
python -m analytics.ml_models          # prints MAE/RMSE/R2 and baseline
python -m genai.rag_pipeline "How does real-time data change city planning?"   # answer with [n] citations
streamlit run app.py & sleep 8; curl -s localhost:8501/_stcore/health          # expect: ok
```
