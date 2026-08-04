"""Phase 6 gold-set generation: build a synthetic query set for evaluating
retrieval.py, since no human relevance-judgment log exists.

Method: sample N structured_datasets, and for each have the LLM write a
realistic natural-language query a researcher would type to find that
specific dataset. The source dataset (plus any near-duplicate datasets
sharing the same title+description, since 33 of 818 datasets fall into 11
such groups) is the query's relevant set.

Caveat, worth stating up front: this measures whether retrieval can find
the document a query was generated *from* -- not full relevance judgment
against the whole corpus. It's a standard substitute when no query log
exists, but it will not catch cases where a genuinely better dataset exists
and outranks the source. Treat recall@k/MRR from this harness as a lower
bound on real-world usefulness, not a ceiling.

Usage:
    python -m cafe_rag.gold_set --n 50
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import ollama
from pydantic import BaseModel
from tqdm import tqdm

from . import db

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "cafe_rag.db"
GOLD_SET_PATH = REPO_ROOT / "data" / "gold_queries.json"

MODEL = "gemma4"
MIN_DESCRIPTION_LEN = 30
SEED = 42

SYSTEM_PROMPT = """You write realistic search queries for a climate-and-health dataset
discovery tool used by researchers. Given one dataset's title, description, keywords, and
subjects, write ONE short natural-language query (5-20 words) that a researcher would type
to try to find this specific dataset -- the way they'd actually search, not a copy of the
title. Focus on what the dataset is *about* (topic, hazard, health outcome, place, time
period) rather than its exact name.
"""


class GeneratedQuery(BaseModel):
    query: str


def _duplicate_groups(rows: list[tuple[str, dict]]) -> dict[str, list[str]]:
    """persistent_id -> list of persistent_ids (including itself) sharing the
    same title+description, so gold-set relevance isn't unfairly strict."""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pid, sd in rows:
        key = (sd.get("title") or "", (sd.get("description") or "")[:200])
        groups[key].append(pid)

    by_pid: dict[str, list[str]] = {}
    for members in groups.values():
        for pid in members:
            by_pid[pid] = members
    return by_pid


def _dataset_prompt(sd: dict) -> str:
    parts = [f"Title: {sd.get('title') or '(none)'}"]
    if sd.get("description"):
        parts.append(f"Description: {sd['description'][:1500]}")
    if sd.get("keywords"):
        parts.append(f"Keywords: {', '.join(sd['keywords'])}")
    if sd.get("subjects"):
        parts.append(f"Subjects: {', '.join(sd['subjects'])}")
    return "\n".join(parts)


def generate_query(sd: dict, model: str = MODEL) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _dataset_prompt(sd)},
        ],
        format=GeneratedQuery.model_json_schema(),
        options={"temperature": 0.7},
    )
    parsed = json.loads(response["message"]["content"])
    return parsed["query"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Phase 6 gold query set.")
    parser.add_argument("--n", type=int, default=50, help="number of gold queries to generate")
    args = parser.parse_args()

    conn = db.connect(DB_PATH)
    rows = db.all_structured_datasets(conn)
    dup_groups = _duplicate_groups(rows)

    eligible = [(pid, sd) for pid, sd in rows if len(sd.get("description") or "") >= MIN_DESCRIPTION_LEN]
    print(f"{len(eligible)} datasets eligible (description >= {MIN_DESCRIPTION_LEN} chars)")

    random.seed(SEED)
    sample = random.sample(eligible, min(args.n, len(eligible)))

    gold = []
    failures = 0
    for pid, sd in tqdm(sample, desc="generating gold queries"):
        try:
            query = generate_query(sd)
            gold.append({
                "query": query,
                "source_persistent_id": pid,
                "source_title": sd.get("title"),
                "relevant_persistent_ids": dup_groups.get(pid, [pid]),
            })
        except Exception as exc:  # noqa: BLE001 - log and keep going
            failures += 1
            tqdm.write(f"FAILED {pid}: {exc}")

    GOLD_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLD_SET_PATH.write_text(json.dumps(gold, indent=2))
    print(f"done. {len(gold)} gold queries written to {GOLD_SET_PATH}, failures = {failures}")


if __name__ == "__main__":
    main()
