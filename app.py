"""
GenAI & Smart City Real-Time Analytics Pipeline - Streamlit application.

Author: Roshitha K

Run with:  streamlit run app.py

Pages (RBAC enforced in the sidebar AND inside each page function):
  Executive Analytics, PyTorch Predictions, GenAI Assistant, Database Management (admin only).
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from analytics.ml_models import FEATURES, CongestionPredictor, bootstrap_training_frame
from analytics.stream_simulator import StreamSimulator
from auth.authenticator import (PAGE_DASHBOARD, PAGE_DB, PAGE_GENAI, PAGE_PREDICT,
                                ROLE_ADMIN, ROLE_USER, Authenticator)
from database.db_manager import DatabaseManager
from genai.rag_pipeline import RAGAssistant

st.set_page_config(page_title=config.APP_TITLE, page_icon="🏙️", layout="wide")


# --------------------------------------------------------------------------- resources
@st.cache_resource
def get_db() -> DatabaseManager:
    return DatabaseManager()


@st.cache_resource
def get_auth() -> Authenticator:
    return Authenticator()


@st.cache_resource
def get_simulator() -> StreamSimulator:
    return StreamSimulator()


@st.cache_resource
def get_rag() -> RAGAssistant:
    return RAGAssistant()


@st.cache_resource
def get_predictor() -> CongestionPredictor:
    predictor = CongestionPredictor()
    if not predictor.load():
        train_predictor(predictor)
    return predictor


def train_predictor(predictor: CongestionPredictor, epochs: int = 150) -> bool:
    frame, augmented = bootstrap_training_frame(get_db().fetch_recent(limit=20000))
    predictor.train(frame, epochs=epochs)
    return augmented


# --------------------------------------------------------------------------- login
def login_page() -> None:
    st.title("🏙️ " + config.APP_TITLE)
    st.caption("Real-time IoT telemetry · PyTorch forecasting · RAG over urban reports")
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        with st.form("login_form"):
            st.subheader("Sign in")
            role_label = st.selectbox("Role", ["Admin", "User / Analyst"])
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
        with st.expander("Demo accounts"):
            st.code("admin / admin123\nanalyst / user123")
        if config.JWT_SECRET_IS_EPHEMERAL:
            st.caption("JWT_SECRET is not set: sessions use a random per-process secret.")
    if submitted:
        role = ROLE_ADMIN if role_label == "Admin" else ROLE_USER
        result = get_auth().authenticate(username, password, role)
        get_db().log_activity(username or "?", result.role, "login_ok" if result.ok else "login_failed")
        if result.ok:
            st.session_state["token"] = result.token
            st.rerun()
        else:
            with mid:
                st.error(result.message)


# --------------------------------------------------------------------------- dashboard
KPI_SPECS = [
    ("Traffic density", "traffic_density", "veh/min", "inverse"),
    ("AQI", "aqi", "", "inverse"),
    ("Power load", "power_consumption", "MW", "off"),
    ("Noise", "noise_level", "dB", "inverse"),
]


def render_kpis(df: pd.DataFrame) -> None:
    # One row per timestamp (mean across the selected zones) so latest vs previous is like-for-like.
    agg = df.groupby("timestamp", as_index=False)[[k[1] for k in KPI_SPECS]].mean().sort_values("timestamp")
    latest = agg.iloc[-1]
    prev = agg.iloc[-2] if len(agg) > 1 else None
    for col, (label, key, unit, color) in zip(st.columns(len(KPI_SPECS)), KPI_SPECS):
        delta = None if prev is None else f"{latest[key] - prev[key]:+.1f}"
        col.metric(label, f"{latest[key]:.1f} {unit}".strip(), delta, delta_color=color)
    st.caption(f"Latest timestamp (UTC): {latest['timestamp']:%Y-%m-%d %H:%M:%S} · zone-averaged")


def page_dashboard(user: dict) -> None:
    db, sim = get_db(), get_simulator()
    st.title("📊 Executive Analytics")
    c1, c2, c3 = st.columns([2, 1, 1])
    zones = c1.multiselect("Zones", config.ZONES, default=config.ZONES)
    window = c2.slider("History window (rows)", 100, 2000, 600, step=100)
    live = c3.toggle(f"Live stream (every {config.STREAM_INTERVAL_S}s)")
    if c3.button("Ingest one batch now"):
        db.insert_logs(sim.generate_batch())
    if not zones:
        st.info("Select at least one zone.")
        return

    @st.fragment(run_every=config.STREAM_INTERVAL_S if live else None)
    def panel() -> None:
        if live:
            db.insert_logs(sim.generate_batch())
        df = db.fetch_recent(limit=window, zones=zones)
        if df.empty:
            st.warning("No sensor data yet.")
            return
        render_kpis(df)
        left, right = st.columns(2)
        left.plotly_chart(px.line(df, x="timestamp", y="traffic_density", color="zone_id",
                                  title="Traffic density by zone (veh/min)"))
        right.plotly_chart(px.line(df, x="timestamp", y="aqi", color="zone_id", title="AQI by zone"))
        left, right = st.columns(2)
        left.plotly_chart(px.scatter(df, x="traffic_density", y="aqi", color="zone_id", opacity=0.7,
                                     title="Traffic vs AQI"))
        zone_avg = df.groupby("zone_id", as_index=False)[[k[1] for k in KPI_SPECS]].mean()
        right.plotly_chart(px.bar(zone_avg, x="zone_id", y=["traffic_density", "aqi"], barmode="group",
                                  title="Zone-wise averages"))
        st.subheader("Recent readings")
        st.dataframe(df.sort_values("timestamp", ascending=False).head(100))

    panel()


# --------------------------------------------------------------------------- predictions
def page_predict(user: dict) -> None:
    st.title("🧠 PyTorch Predictive Model")
    st.caption("MLP regression: AQI + power load + noise → estimated traffic congestion (vehicles/min).")
    with st.spinner("Loading model (trains once on first run)…"):
        predictor = get_predictor()
    c1, c2, c3 = st.columns(3)
    aqi = c1.slider("AQI", 0, 300, 90)
    power = c2.slider("Power consumption (MW)", 10, 120, 55)
    noise = c3.slider("Noise level (dB)", 35, 100, 65)
    estimate = predictor.predict(aqi, power, noise)
    left, right = st.columns([1, 1])
    fig = go.Figure(go.Indicator(mode="gauge+number", value=estimate,
                                 title={"text": "Estimated congestion (veh/min)"},
                                 gauge={"axis": {"range": [0, 120]}}))
    left.plotly_chart(fig)
    sweep = pd.DataFrame({"aqi": list(range(0, 301, 5))})
    sweep["power_consumption"], sweep["noise_level"] = power, noise
    sweep["estimate"] = predictor.predict_batch(sweep)
    right.plotly_chart(px.line(sweep, x="aqi", y="estimate", title="Sensitivity to AQI (others fixed)"))

    st.subheader("Model card")
    m = predictor.metrics
    if m:
        a, b, c, d = st.columns(4)
        a.metric("Validation MAE", f"{m['mae']:.2f}", f"baseline {m['baseline_mae']:.2f}", delta_color="off")
        b.metric("Validation RMSE", f"{m['rmse']:.2f}", f"baseline {m['baseline_rmse']:.2f}", delta_color="off")
        c.metric("R²", f"{m['r2']:.3f}")
        d.metric("Epochs run", f"{m['epochs_run']}")
        st.caption(f"Chronological split: {m['train_rows']} train / {m['val_rows']} validation rows. "
                   f"Baseline = predict the training mean. Inputs: {', '.join(FEATURES)}. "
                   "Trained on synthetic telemetry, so it demonstrates the workflow, not real-city accuracy.")
    with st.expander("Retrain on current database"):
        epochs = st.slider("Max epochs", 20, 400, 150, step=10)
        if st.button("Retrain"):
            with st.spinner("Training…"):
                augmented = train_predictor(predictor, epochs)
            if augmented:
                st.info("Database had fewer than the minimum rows; synthetic history was added for training.")
            st.rerun()


# --------------------------------------------------------------------------- GenAI
def page_genai(user: dict) -> None:
    st.title("💬 GenAI Assistant")
    st.caption("Ask about smart-city development, big data, AQI trends and automated reporting. "
               "Answers are extracted from indexed reports and cited.")
    history = st.session_state.setdefault("chat", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**[{s['n']}] {s['report_id']}** — {s['title']}  \n"
                                    f"*{s['citation']}* · similarity {s['similarity']}")
    question = st.chat_input("e.g. How does real-time data change city planning?")
    if question:
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching reports (first run downloads the embedding model)…"):
                answer = get_rag().query(question)
            st.markdown(answer.answer)
        history.append({"role": "assistant", "content": answer.answer, "sources": answer.sources})
        get_db().log_activity(user["sub"], user["role"], "genai_query", question)
        st.rerun()


# --------------------------------------------------------------------------- database admin
DEFAULT_SQL = ("SELECT zone_id, COUNT(*) AS readings, AVG(traffic_density) AS avg_traffic, AVG(aqi) AS avg_aqi\n"
               "FROM sensor_logs GROUP BY zone_id ORDER BY zone_id")


def page_db(user: dict) -> None:
    if not Authenticator.can_access(user["role"], PAGE_DB):  # defence in depth
        st.error("Admin access required.")
        return
    db = get_db()
    st.title("🗄️ Database Management")
    st.caption(f"Backend: **{db.backend}** · sensor rows: **{db.count_logs()}**")
    t1, t2, t3 = st.tabs(["SQL editor", "Table previewer", "Execution log"])
    with t1:
        sql = st.text_area("SQL (single statement)", DEFAULT_SQL, height=150)
        allow_write = st.checkbox("Allow write statements (INSERT/UPDATE/DELETE/DDL)")
        if st.button("Run", type="primary"):
            result = db.execute_raw_sql(sql, allow_write=allow_write)
            db.log_activity(user["sub"], user["role"], "sql_ok" if result.ok else "sql_error",
                            f"{sql}\n-> {result.message}")
            (st.success if result.ok else st.error)(f"{result.message} ({result.elapsed_ms:.1f} ms)")
            if result.dataframe is not None:
                st.dataframe(result.dataframe)
    with t2:
        table = st.selectbox("Table", db.table_names())
        limit = st.slider("Rows", 5, 500, 50)
        if table:
            st.dataframe(db.preview_table(table, limit))
    with t3:
        st.dataframe(db.fetch_activity(limit=200))


# --------------------------------------------------------------------------- shell
PAGES = {PAGE_DASHBOARD: page_dashboard, PAGE_PREDICT: page_predict, PAGE_GENAI: page_genai, PAGE_DB: page_db}


def main() -> None:
    claims = get_auth().verify_token(st.session_state.get("token"))
    if claims is None:
        st.session_state.pop("token", None)
        login_page()
        return
    with st.sidebar:
        st.markdown(f"### 🏙️ Smart City Analytics\n**Developed by {config.AUTHOR}**")
        st.divider()
        st.write(f"Signed in as **{claims['sub']}** ({claims['role']})")
        allowed = list(Authenticator.allowed_pages(claims["role"]))
        page = st.radio("Navigate", allowed)
        if st.button("Log out"):
            get_db().log_activity(claims["sub"], claims["role"], "logout")
            for key in ("token", "chat"):
                st.session_state.pop(key, None)
            st.rerun()
    if not Authenticator.can_access(claims["role"], page):
        st.error("You do not have access to this page.")
        return
    PAGES[page](claims)


main()
