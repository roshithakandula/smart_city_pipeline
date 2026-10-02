"""
Retrieval-Augmented Generation (RAG) over urban research reports.

Author: Roshitha K

Research context
----------------
RAG (Lewis et al., 2020, NeurIPS) grounds answers in retrieved documents rather
than model memory, which suits unstructured municipal reports that change often
and must be cited. This module embeds the corpus with a local
sentence-transformers model (all-MiniLM-L6-v2), indexes it with FAISS through
LangChain, and answers by *extractive synthesis*: the most query-relevant
sentences of the top-k chunks are stitched together with numbered citations.
No external LLM or API key is needed, and nothing is generated that is not in
the retrieved text.

IMPORTANT - about the corpus
----------------------------
The built-in documents are project-authored SUMMARIES of themes from the cited
literature. They are paraphrased, are not verbatim excerpts, and contain no
quantitative findings attributed to those papers. The "30% manual reporting
reduction" is this project's design target, not a published result. Drop your
own real reports (.txt / .md) into genai/reports/ and they are indexed too.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import config

log = logging.getLogger(__name__)

_NOTE = "Project-authored summary of themes in the cited work (paraphrased, not verbatim). "

BUILTIN_REPORTS: List[Dict[str, str]] = [
    {
        "id": "UC-2014",
        "title": "Urban Computing: Concepts, Methodologies, and Applications",
        "citation": "Zheng, Capra, Wolfson & Yang (2014), ACM Transactions on Intelligent Systems and Technology",
        "text": _NOTE + (
            "Urban computing is described as acquiring, integrating and analysing large, heterogeneous data "
            "from sensors, vehicles, buildings and people in urban spaces. Its goal is to address problems such as "
            "traffic congestion, rising energy consumption and air pollution. A typical framework has four layers: "
            "urban sensing, data management, data analytics and service providing. Real-time sensing feeds the "
            "analytics layer, so decisions can follow current conditions instead of lagging reports. Data "
            "heterogeneity and spatio-temporal correlation between traffic, air quality and energy use are "
            "recurring technical challenges."
        ),
    },
    {
        "id": "BIGDATA-2013",
        "title": "Big Data, Smart Cities and City Planning",
        "citation": "Batty (2013), Dialogues in Human Geography",
        "text": _NOTE + (
            "Big data from sensors and digital infrastructure changes how cities are understood. Instead of "
            "relying only on slow, periodic surveys and long-term strategic plans, planners can observe the city "
            "in near real time and shift attention toward short-term operational management. The work argues that "
            "real-time analytics complements, rather than replaces, long-term planning, and that models of "
            "flows such as traffic and energy become more useful when they are continuously updated with data."
        ),
    },
    {
        "id": "REALTIME-2014",
        "title": "The Real-Time City? Big Data and Smart Urbanism",
        "citation": "Kitchin (2014), GeoJournal",
        "text": _NOTE + (
            "Real-time data streams promise more responsive urban management, but the paper stresses critical "
            "limits. Sensor data can be incomplete or biased, analytics can encourage a purely technocratic view "
            "of the city, and large-scale data collection raises privacy, surveillance and governance concerns. "
            "Good smart-city practice therefore pairs real-time dashboards with transparency, data quality "
            "checks and human judgement about what the numbers mean."
        ),
    },
    {
        "id": "IOT-2014",
        "title": "Internet of Things for Smart Cities",
        "citation": "Zanella, Bui, Castellani, Vangelista & Zorzi (2014), IEEE Internet of Things Journal",
        "text": _NOTE + (
            "Urban IoT deployments connect heterogeneous devices such as traffic counters, air-quality monitors, "
            "noise meters and smart meters through shared communication technologies and platforms. Reported "
            "services include structural health monitoring, waste management, air quality monitoring, noise "
            "monitoring, traffic congestion monitoring and smart lighting. A common platform for real-time "
            "telemetry ingestion lets different city departments reuse the same data instead of building "
            "isolated systems."
        ),
    },
    {
        "id": "RAG-2020",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "citation": "Lewis et al. (2020), Advances in Neural Information Processing Systems",
        "text": _NOTE + (
            "Retrieval-augmented generation combines a retriever with a text generator so answers are based on "
            "retrieved documents as well as the model's built-in knowledge. For municipal reports this means "
            "unstructured documents such as incident summaries, planning memos and audit reports can be searched "
            "semantically and cited, and the knowledge base can be updated by adding documents rather than "
            "retraining a model. Grounding in retrieved passages also makes it easier to check where an answer "
            "came from."
        ),
    },
    {
        "id": "AQI-TRAFFIC",
        "title": "Air Quality Index Trends and Traffic Emissions in Urban Zones",
        "citation": "General background; see also WHO Global Air Quality Guidelines (2021)",
        "text": _NOTE + (
            "The Air Quality Index (AQI) converts pollutant concentrations such as PM2.5 and nitrogen dioxide "
            "into a single scale that communicates health risk. In dense urban zones, road traffic is a major "
            "source of these pollutants, so AQI often rises during morning and evening rush hours and falls "
            "overnight. Industrial and port areas can show a higher baseline AQI from stationary and shipping "
            "sources, while residential areas usually show lower baselines. The 2021 WHO guideline update "
            "tightened recommended pollutant limits, increasing the importance of continuous monitoring and "
            "trend analysis instead of occasional spot checks."
        ),
    },
    {
        "id": "REPORT-CYCLE",
        "title": "Reducing Manual Reporting Cycles with Automated Analytics Pipelines",
        "citation": "Project design note (this repository); a design target, not a published study",
        "text": _NOTE.replace("of the cited work", "of this project's goals") + (
            "A recurring operational cost in city analytics is the manual reporting cycle: exporting sensor "
            "data, cleaning it in spreadsheets, building charts and writing summaries. This pipeline targets a "
            "reduction of roughly 30 percent in manual reporting effort by automating ingestion, aggregation, "
            "dashboarding and question answering. The 30 percent figure is a design objective; to claim it, "
            "teams should time the old manual cycle first and compare it with the automated workflow."
        ),
    },
    {
        "id": "PREDICTIVE-VS-STATIC",
        "title": "Predictive Trend Modelling versus Static Historical Reporting",
        "citation": "Project design note (this repository)",
        "text": _NOTE.replace("of the cited work", "of this project's approach") + (
            "Static historical reports describe what already happened and arrive after the decision window has "
            "closed. Predictive models trained on streaming telemetry estimate current or upcoming conditions, "
            "for example congestion inferred from air quality, power demand and noise. Predictions should be "
            "validated chronologically, compared with a simple baseline such as the historical mean, and shown "
            "with their error so operators know how much to trust them."
        ),
    },
]

_STOPWORDS = frozenset(
    "a an the and or of to in on for with by is are was were be been it its this that these those what which "
    "who how why when where does do did can could should would about into from as at not no than then there "
    "their they them have has had i you we me my our your".split()
)


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS and len(t) > 2}


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


@dataclass
class RAGAnswer:
    answer: str
    sources: List[dict] = field(default_factory=list)
    found: bool = True


class RAGAssistant:
    """LangChain + FAISS retrieval assistant with extractive, cited answers."""

    def __init__(self, index_dir: Path = config.FAISS_INDEX_DIR, embedding_model: str = config.EMBEDDING_MODEL,
                 extra_dir: Path = config.EXTRA_REPORTS_DIR, rebuild: bool = False) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.extra_dir = Path(extra_dir)
        self.rebuild = rebuild
        self._store = None
        self.n_documents = 0
        self.n_chunks = 0

    # -- corpus ----------------------------------------------------------------
    def load_corpus(self) -> List[Dict[str, str]]:
        docs = list(BUILTIN_REPORTS)
        if self.extra_dir.is_dir():
            for path in sorted(self.extra_dir.glob("*")):
                if path.suffix.lower() in (".txt", ".md") and path.is_file():
                    body = path.read_text(encoding="utf-8", errors="replace").strip()
                    if body:
                        docs.append({"id": path.stem.upper(), "title": path.stem.replace("_", " "),
                                     "citation": f"User-supplied file {path.name}", "text": body})
        return docs

    @staticmethod
    def _fingerprint(docs: List[Dict[str, str]], model: str) -> str:
        h = hashlib.sha256(model.encode())
        for d in docs:
            h.update(d["id"].encode())
            h.update(d["text"].encode())
        h.update(f"{config.RAG_CHUNK_SIZE}:{config.RAG_CHUNK_OVERLAP}".encode())
        return h.hexdigest()

    # -- index -----------------------------------------------------------------
    def _embeddings(self):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:  # older installs
            from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=self.embedding_model,
                                     encode_kwargs={"normalize_embeddings": True})

    def build_or_load(self) -> None:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

        docs = self.load_corpus()
        self.n_documents = len(docs)
        fp = self._fingerprint(docs, self.embedding_model)
        fp_file = self.index_dir / "corpus.sha256"
        embeddings = self._embeddings()

        if not self.rebuild and fp_file.exists() and fp_file.read_text().strip() == fp \
                and (self.index_dir / "index.faiss").exists():
            # Safe: the index is built locally by this code, never downloaded.
            self._store = FAISS.load_local(str(self.index_dir), embeddings, allow_dangerous_deserialization=True)
            self.n_chunks = self._store.index.ntotal
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=config.RAG_CHUNK_SIZE,
                                                  chunk_overlap=config.RAG_CHUNK_OVERLAP)
        lc_docs = []
        for d in docs:
            for i, chunk in enumerate(splitter.split_text(d["text"])):
                lc_docs.append(Document(page_content=chunk, metadata={
                    "report_id": d["id"], "title": d["title"], "citation": d["citation"], "chunk": i}))
        self._store = FAISS.from_documents(lc_docs, embeddings)
        self.n_chunks = len(lc_docs)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.index_dir))
        fp_file.write_text(fp)

    # -- query -----------------------------------------------------------------
    def query(self, question: str, k: int = config.RAG_TOP_K) -> RAGAnswer:
        question = (question or "").strip()
        if not question:
            return RAGAnswer("Please ask a question.", [], False)
        if self._store is None:
            self.build_or_load()
        hits = self._store.similarity_search_with_score(question, k=k)
        # FAISS IndexFlatL2 returns squared L2; for unit vectors cos = 1 - d/2.
        scored = [(doc, max(0.0, 1.0 - float(dist) / 2.0)) for doc, dist in hits]
        scored = [(d, s) for d, s in scored if s >= config.RAG_MIN_SIMILARITY]
        if not scored:
            return RAGAnswer("I could not find relevant information in the indexed urban reports.", [], False)

        q_tokens = _tokens(question)
        # One source entry per report (best chunk first), numbered for citation.
        order: List[str] = []
        sources: Dict[str, dict] = {}
        for doc, sim in scored:
            rid = doc.metadata["report_id"]
            if rid not in sources:
                order.append(rid)
                sources[rid] = {"n": len(order), "report_id": rid, "title": doc.metadata["title"],
                                "citation": doc.metadata["citation"], "similarity": round(sim, 3),
                                "snippet": doc.page_content}
        picks = []
        for doc, sim in scored:
            n = sources[doc.metadata["report_id"]]["n"]
            for idx, sent in enumerate(_sentences(doc.page_content)):
                overlap = len(q_tokens & _tokens(sent))
                picks.append((overlap, sim, -idx, n, sent))
        picks.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        chosen, seen = [], set()
        for overlap, _sim, _neg, n, sent in picks:
            if sent in seen:
                continue
            seen.add(sent)
            chosen.append((n, sent))
            if len(chosen) == 3:
                break
        chosen.sort(key=lambda t: t[0])
        body = " ".join(f"{sent} [{n}]" for n, sent in chosen)
        return RAGAnswer(body, [sources[r] for r in order], True)


if __name__ == "__main__":  # python -m genai.rag_pipeline "your question"
    import sys
    q = " ".join(sys.argv[1:]) or "How does real-time sensing help reduce manual reporting?"
    rag = RAGAssistant()
    ans = rag.query(q)
    print(ans.answer)
    for s in ans.sources:
        print(f"[{s['n']}] {s['report_id']} - {s['citation']} (sim={s['similarity']})")
    print(f"documents={rag.n_documents} chunks={rag.n_chunks}")
