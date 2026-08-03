"""Phase 2 entry point: structure harvested datasets into the normalized
schema (schema.py) -- deterministic extraction plus an LLM classification
pass for hazard/health domains -- and store in `structured_datasets`.

Resumable: only processes datasets that don't already have a structured
row, so it's safe to Ctrl-C and restart.

Usage:
    python -m cafe_rag.structure
    python -m cafe_rag.structure --limit 20   # smoke test on a small batch
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tqdm import tqdm

from . import db
from .extract_deterministic import extract_deterministic
from .extract_llm import classify

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "cafe_rag.db"


def structure_one(conn: sqlite3.Connection, persistent_id: str) -> None:
    metadata_json, = conn.execute(
        "SELECT metadata_json FROM raw_datasets WHERE persistent_id = ?", (persistent_id,)
    ).fetchone()
    metadata = json.loads(metadata_json)
    collections = [
        r[0] for r in conn.execute(
            "SELECT collection FROM dataset_collections WHERE persistent_id = ?", (persistent_id,)
        ).fetchall()
    ]

    sd = extract_deterministic(persistent_id, collections, metadata)
    cls = classify(sd)
    sd.hazard_domains = cls.hazard_domains
    sd.health_domains = cls.health_domains
    sd.units_mentioned = cls.units_mentioned

    db.upsert_structured_dataset(conn, persistent_id, sd.model_dump(mode="json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Structure harvested datasets (Phase 2).")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N pending datasets")
    args = parser.parse_args()

    conn = db.connect(DB_PATH)
    pending = db.datasets_ready_to_structure(conn)
    if args.limit:
        pending = pending[: args.limit]

    print(f"{len(pending)} datasets to structure")
    failures = 0
    for persistent_id in tqdm(pending, desc="structuring"):
        try:
            structure_one(conn, persistent_id)
        except Exception as exc:  # noqa: BLE001 - log and keep going
            failures += 1
            tqdm.write(f"FAILED {persistent_id}: {exc}")

    print(f"done. structured_datasets total = {db.structured_dataset_count(conn)}, failures this run = {failures}")


if __name__ == "__main__":
    main()
