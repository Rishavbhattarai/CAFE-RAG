"""Phase 7: Streamlit UI for CAFE-RAG.

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import chromadb
import streamlit as st

from cafe_rag import db, retrieval
from cafe_rag.index import CHROMA_PATH, COLLECTION_NAME

DB_PATH = REPO_ROOT / "data" / "cafe_rag.db"
DATAVERSE_URL = "https://dataverse.harvard.edu/dataset.xhtml?persistentId="

EXAMPLE_QUERIES = [
    "daily heat exposure data for US counties, 2010 onward",
    "flooding linked to waterborne disease in South Asia",
    "air pollution and cardiovascular mortality",
    "drought impacts on food security in sub-Saharan Africa",
]


@st.cache_resource
def get_connection():
    return db.connect(DB_PATH)


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(COLLECTION_NAME)


def render_result(rank: int, result: dict) -> None:
    title = result["title"] or "(untitled dataset)"
    doi_url = DATAVERSE_URL + result["persistent_id"]

    st.markdown(f"**{rank}. [{title}]({doi_url})**  \n`{result['persistent_id']}` &nbsp;·&nbsp; score {result['score']}")

    domains = (result["hazard_domains"] or []) + (result["health_domains"] or [])
    if domains:
        st.markdown(" ".join(f"`{d}`" for d in domains))

    if result["matched_signals"]:
        st.caption("Matched: " + ", ".join(result["matched_signals"]))

    if result["related_datasets"]:
        with st.expander(f"Joinable datasets ({len(result['related_datasets'])})"):
            for rel in result["related_datasets"]:
                rel_url = DATAVERSE_URL + rel["persistent_id"]
                st.markdown(
                    f"- [{rel['title'] or '(untitled)'}]({rel_url}) "
                    f"-- {rel['edge_type']}, weight {rel['weight']}"
                )

    st.divider()


def main() -> None:
    st.set_page_config(page_title="CAFE-RAG", page_icon="🌍", layout="centered")
    st.title("🌍 CAFE-RAG")
    st.caption(
        "Agentic, retrieval-augmented search over 818 Harvard Dataverse "
        "climate-health datasets (CAFE, CIESIN, HELD collections)."
    )

    conn = get_connection()
    collection = get_collection()

    with st.sidebar:
        st.header("About")
        st.write(
            "Ask in plain language. The system extracts hazard/health domain, "
            "geography, and time-range intent from your query, runs hybrid "
            "keyword + semantic search, and suggests joinable datasets from "
            "a knowledge graph of shared geography, time, and source."
        )
        k = st.slider("Number of results", min_value=3, max_value=20, value=10)
        st.caption(
            "Eval (44-query synthetic gold set): recall@5 0.705, "
            "recall@10 0.773, MRR 0.635."
        )
        st.markdown(
            "[GitHub repo](https://github.com/Rishavbhattarai/CAFE-RAG) "
            "&middot; source code"
        )

    st.subheader("Try:")
    cols = st.columns(2)
    clicked_example = None
    for i, example in enumerate(EXAMPLE_QUERIES):
        if cols[i % 2].button(example, use_container_width=True):
            clicked_example = example

    query = st.text_input(
        "Search query",
        value=clicked_example or st.session_state.get("query", ""),
        placeholder="e.g. daily heat exposure data for US counties, 2010 onward",
        label_visibility="collapsed",
    )
    st.session_state["query"] = query

    if not query:
        return

    with st.spinner("Extracting intent, searching, expanding graph..."):
        results = retrieval.search(conn, collection, query, k=k)

    if not results:
        st.warning("No results found.")
        return

    st.subheader(f"{len(results)} results")
    for i, result in enumerate(results, start=1):
        render_result(i, result)


if __name__ == "__main__":
    main()
