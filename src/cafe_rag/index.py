"""Phase 3 entry point: build the search index over structured_datasets
(schema.py) -- an FTS5 keyword index and a Chroma semantic vector index,
both keyed by persistent_id. This is the retrieval backbone Phase 5
(hybrid RAG + agentic retrieval) queries against.

Resumable: FTS5 only reprocesses rows missing from `dataset_fts`; the
Chroma pass only embeds ids not already present in the collection.

Usage:
    python -m cafe_rag.index
    python -m cafe_rag.index --skip-fts
    python -m cafe_rag.index --skip-vectors
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
import ollama
from tqdm import tqdm

from . import db

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "cafe_rag.db"
CHROMA_PATH = REPO_ROOT / "data" / "chroma"

EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "cafe_rag_datasets"


def _embed_text(sd: dict) -> str:
    """Same field set used for LLM classification in extract_llm.py, so the
    semantic index and the classifier see a consistent view of each dataset."""
    parts = [f"Title: {sd.get('title') or '(none)'}"]
    if sd.get("description"):
        parts.append(f"Description: {sd['description'][:2000]}")
    if sd.get("keywords"):
        parts.append(f"Keywords: {', '.join(sd['keywords'])}")
    if sd.get("subjects"):
        parts.append(f"Subjects: {', '.join(sd['subjects'])}")
    return "\n".join(parts)


def build_fts_index(conn) -> None:
    pending = db.datasets_ready_to_index(conn)
    print(f"{len(pending)} datasets to add to FTS5 index")
    all_structured = dict(db.all_structured_datasets(conn))
    for persistent_id in tqdm(pending, desc="fts indexing"):
        db.upsert_fts_dataset(conn, persistent_id, all_structured[persistent_id])
    print(f"done. dataset_fts total = {db.fts_dataset_count(conn)}")


def build_vector_index(conn) -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    all_structured = db.all_structured_datasets(conn)
    existing = set(collection.get(ids=[pid for pid, _ in all_structured])["ids"])
    pending = [(pid, sd) for pid, sd in all_structured if pid not in existing]

    print(f"{len(pending)} datasets to embed into Chroma")
    failures = 0
    for persistent_id, sd in tqdm(pending, desc="vector indexing"):
        try:
            text = _embed_text(sd)
            response = ollama.embed(model=EMBED_MODEL, input=text)
            embedding = response["embeddings"][0]
            collection.add(
                ids=[persistent_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "title": sd.get("title") or "",
                    "hazard_domains": ",".join(sd.get("hazard_domains") or []),
                    "health_domains": ",".join(sd.get("health_domains") or []),
                }],
            )
        except Exception as exc:  # noqa: BLE001 - log and keep going
            failures += 1
            tqdm.write(f"FAILED {persistent_id}: {exc}")

    print(f"done. chroma collection count = {collection.count()}, failures this run = {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the search index (Phase 3).")
    parser.add_argument("--skip-fts", action="store_true", help="skip the FTS5 keyword index")
    parser.add_argument("--skip-vectors", action="store_true", help="skip the Chroma vector index")
    args = parser.parse_args()

    conn = db.connect(DB_PATH)
    if not args.skip_fts:
        build_fts_index(conn)
    if not args.skip_vectors:
        build_vector_index(conn)


if __name__ == "__main__":
    main()
