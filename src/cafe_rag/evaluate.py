"""Phase 6 entry point: evaluate retrieval.py against the Phase 6 gold
query set (gold_set.py) with recall@k and MRR.

Caveat carried over from gold_set.py: this measures whether retrieval finds
the document a query was generated *from*, not full relevance judgment
against the whole corpus. Treat these numbers as a lower bound on
real-world usefulness, not a ceiling -- see gold_set.py's docstring.

Usage:
    python -m cafe_rag.evaluate
    python -m cafe_rag.evaluate --k-values 1 3 5 10 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from tqdm import tqdm

from . import db, retrieval
from .index import CHROMA_PATH, COLLECTION_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "cafe_rag.db"
GOLD_SET_PATH = REPO_ROOT / "data" / "gold_queries.json"
EVAL_RESULTS_PATH = REPO_ROOT / "data" / "eval_results.json"

DEFAULT_K_VALUES = [1, 3, 5, 10]
MAX_K = 20  # results pool depth; also the MRR cutoff


def _first_relevant_rank(results: list[dict], relevant_ids: set[str]) -> int | None:
    for rank, r in enumerate(results, start=1):
        if r["persistent_id"] in relevant_ids:
            return rank
    return None


def evaluate(conn, collection, gold: list[dict], k_values: list[int]) -> dict:
    per_query = []
    for entry in tqdm(gold, desc="evaluating"):
        results = retrieval.search(conn, collection, entry["query"], k=MAX_K)
        relevant_ids = set(entry["relevant_persistent_ids"])
        rank = _first_relevant_rank(results, relevant_ids)
        per_query.append({
            "query": entry["query"],
            "source_persistent_id": entry["source_persistent_id"],
            "rank": rank,
            "reciprocal_rank": (1.0 / rank) if rank else 0.0,
        })

    n = len(per_query)
    metrics = {
        f"recall@{k}": sum(1 for q in per_query if q["rank"] and q["rank"] <= k) / n
        for k in k_values
    }
    metrics["mrr"] = sum(q["reciprocal_rank"] for q in per_query) / n
    metrics["not_found_in_top"] = sum(1 for q in per_query if q["rank"] is None)
    metrics["n_queries"] = n

    return {"metrics": metrics, "per_query": per_query}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against the gold query set (Phase 6).")
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    args = parser.parse_args()

    gold = json.loads(GOLD_SET_PATH.read_text())
    conn = db.connect(DB_PATH)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    result = evaluate(conn, collection, gold, args.k_values)

    EVAL_RESULTS_PATH.write_text(json.dumps(result, indent=2))

    print(f"\n{result['metrics']['n_queries']} gold queries evaluated (pool depth = {MAX_K})\n")
    for k in args.k_values:
        print(f"recall@{k}:  {result['metrics'][f'recall@{k}']:.3f}")
    print(f"MRR:        {result['metrics']['mrr']:.3f}")
    print(f"not found in top {MAX_K}: {result['metrics']['not_found_in_top']} / {result['metrics']['n_queries']}")
    print(f"\nfull results written to {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
