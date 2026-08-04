"""Phase 5 entry point: hybrid RAG + agentic retrieval over the Phase 3/4
indexes.

Pipeline for a natural-language query:
  1. LLM extracts structured intent (extract_query.py): hazard/health
     domains, geographic hints, year range.
  2. Hybrid search: FTS5 keyword search + Chroma semantic search, merged
     by Reciprocal Rank Fusion (RRF).
  3. Filter-match boost: results whose metadata actually matches the
     extracted intent get a score bump (soft, not a hard filter -- see
     schema.QueryFilters for why).
  4. Graph expansion: each top result gets a few "you can join this with"
     suggestions pulled from Phase 4's knowledge graph.

Usage:
    python -m cafe_rag.retrieval "daily heat exposure data for US counties, 2010 onward"
    python -m cafe_rag.retrieval "flooding linked to waterborne disease in South Asia" --k 5
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import chromadb
import ollama

from . import db
from .extract_query import extract_filters
from .index import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL
from .schema import QueryFilters

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "cafe_rag.db"

RRF_K = 60
CANDIDATE_POOL = 30
FILTER_BOOST = 0.02  # added per matched signal, small relative to 1/RRF_K ~ 0.016
GRAPH_EXPANSION_MIN_WEIGHT = 0.6
GRAPH_EXPANSION_PER_RESULT = 2


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)


def _fts_candidates(conn, query: str, pool: int) -> list[str]:
    try:
        hits = db.search_fts(conn, _fts_query(query), limit=pool)
    except Exception:
        return []
    return [pid for pid, _ in hits]


def _chroma_candidates(collection, query: str, pool: int) -> list[str]:
    embedding = ollama.embed(model=EMBED_MODEL, input=query)["embeddings"][0]
    result = collection.query(query_embeddings=[embedding], n_results=pool)
    return result["ids"][0]


def _rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, pid in enumerate(ranked, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return scores


def _matched_signals(sd: dict, filters: QueryFilters) -> list[str]:
    signals = []
    for domain in filters.hazard_domains:
        if domain in (sd.get("hazard_domains") or []):
            signals.append(f"hazard_domain:{domain}")
    for domain in filters.health_domains:
        if domain in (sd.get("health_domains") or []):
            signals.append(f"health_domain:{domain}")
    geo_coverage = [g.lower() for g in (sd.get("geographic_coverage") or [])]
    for hint in filters.geographic_hints:
        if any(hint.lower() in g or g in hint.lower() for g in geo_coverage):
            signals.append(f"geography:{hint}")
    if filters.year_min or filters.year_max:
        for r in sd.get("temporal_coverage") or []:
            years = re.findall(r"\d{4}", f"{r.get('start') or ''} {r.get('end') or ''}")
            if not years:
                continue
            start, end = int(min(years)), int(max(years))
            lo = filters.year_min or start
            hi = filters.year_max or end
            if start <= hi and lo <= end:
                signals.append("temporal_range")
                break
    return signals


def search(conn, collection, query: str, k: int = 10) -> list[dict]:
    filters = extract_filters(query)

    fts_ids = _fts_candidates(conn, query, CANDIDATE_POOL)
    chroma_ids = _chroma_candidates(collection, query, CANDIDATE_POOL)
    fused = _rrf_fuse([fts_ids, chroma_ids])

    structured = dict(db.all_structured_datasets(conn))

    scored = []
    for pid, base_score in fused.items():
        sd = structured.get(pid)
        if not sd:
            continue
        signals = _matched_signals(sd, filters)
        score = base_score + FILTER_BOOST * len(signals)
        scored.append((pid, score, sd, signals))

    scored.sort(key=lambda row: row[1], reverse=True)
    top = scored[:k]

    results = []
    for pid, score, sd, signals in top:
        related_raw = db.get_related_datasets(
            conn, pid, min_weight=GRAPH_EXPANSION_MIN_WEIGHT, limit=GRAPH_EXPANSION_PER_RESULT
        )
        related = [
            {
                "persistent_id": rel_pid,
                "title": (structured.get(rel_pid) or {}).get("title"),
                "edge_type": edge_type,
                "weight": round(weight, 2),
            }
            for rel_pid, edge_type, weight, _detail in related_raw
        ]
        results.append({
            "persistent_id": pid,
            "title": sd.get("title"),
            "score": round(score, 4),
            "matched_signals": signals,
            "hazard_domains": sd.get("hazard_domains"),
            "health_domains": sd.get("health_domains"),
            "related_datasets": related,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid RAG retrieval over CAFE-RAG (Phase 5).")
    parser.add_argument("query", help="natural-language search query")
    parser.add_argument("--k", type=int, default=10, help="number of results to return")
    args = parser.parse_args()

    conn = db.connect(DB_PATH)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    results = search(conn, collection, args.query, k=args.k)
    for i, r in enumerate(results, start=1):
        print(f"{i}. [{r['score']}] {r['title']} ({r['persistent_id']})")
        if r["matched_signals"]:
            print(f"   matched: {', '.join(r['matched_signals'])}")
        for rel in r["related_datasets"]:
            print(f"   + joinable ({rel['edge_type']}, w={rel['weight']}): {rel['title']} ({rel['persistent_id']})")


if __name__ == "__main__":
    main()
