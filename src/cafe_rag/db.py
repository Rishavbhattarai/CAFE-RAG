"""SQLite storage for raw harvested dataset metadata (Phase 1 output).

Downstream phases (LLM structuring, FTS5/Chroma indexing, knowledge graph)
read from `raw_datasets` rather than re-hitting the Dataverse API.

A dataset can be linked into more than one Dataverse collection (confirmed:
GRDI and EnvClim, which the project brief attributes to CIESIN, are also
returned under a CAFE subtree search). `raw_datasets` is keyed by
persistent_id (one row per unique dataset); `dataset_collections` records
every collection a dataset was observed under, so that membership isn't
lost when the same dataset resurfaces under a second collection.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_datasets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    persistent_id   TEXT NOT NULL UNIQUE,
    name            TEXT,
    url             TEXT,
    published_at    TEXT,
    search_json     TEXT NOT NULL,
    metadata_json   TEXT,
    harvested_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_collections (
    persistent_id   TEXT NOT NULL,
    collection      TEXT NOT NULL,
    first_seen_at   TEXT NOT NULL,
    PRIMARY KEY (persistent_id, collection)
);

CREATE INDEX IF NOT EXISTS idx_dataset_collections_collection ON dataset_collections(collection);

CREATE TABLE IF NOT EXISTS structured_datasets (
    persistent_id   TEXT PRIMARY KEY,
    structured_json TEXT NOT NULL,
    structured_at   TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS dataset_fts USING fts5(
    persistent_id UNINDEXED,
    title,
    description,
    keywords,
    subjects
);

CREATE TABLE IF NOT EXISTS dataset_edges (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    persistent_id_a   TEXT NOT NULL,
    persistent_id_b   TEXT NOT NULL,
    edge_type         TEXT NOT NULL,
    weight            REAL NOT NULL,
    detail            TEXT,
    created_at        TEXT NOT NULL,
    UNIQUE(persistent_id_a, persistent_id_b, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_edges_a ON dataset_edges(persistent_id_a);
CREATE INDEX IF NOT EXISTS idx_edges_b ON dataset_edges(persistent_id_b);
CREATE INDEX IF NOT EXISTS idx_edges_type ON dataset_edges(edge_type);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def upsert_search_hit(conn: sqlite3.Connection, collection: str, hit: dict) -> None:
    """Insert/update a dataset from a /api/search hit (metadata_json left null
    until fetched separately by `upsert_dataset_metadata`), and record that
    it was observed under `collection`."""
    persistent_id = hit.get("global_id")
    if not persistent_id:
        return
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO raw_datasets (persistent_id, name, url, published_at, search_json, harvested_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(persistent_id) DO UPDATE SET
            search_json = excluded.search_json,
            name = excluded.name,
            url = excluded.url,
            published_at = excluded.published_at,
            harvested_at = excluded.harvested_at
        """,
        (
            persistent_id,
            hit.get("name"),
            hit.get("url"),
            hit.get("published_at"),
            json.dumps(hit),
            now,
        ),
    )
    conn.execute(
        """
        INSERT INTO dataset_collections (persistent_id, collection, first_seen_at)
        VALUES (?, ?, ?)
        ON CONFLICT(persistent_id, collection) DO NOTHING
        """,
        (persistent_id, collection, now),
    )
    conn.commit()


def upsert_dataset_metadata(conn: sqlite3.Connection, persistent_id: str, metadata: dict) -> None:
    conn.execute(
        "UPDATE raw_datasets SET metadata_json = ? WHERE persistent_id = ?",
        (json.dumps(metadata), persistent_id),
    )
    conn.commit()


def datasets_missing_metadata(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT persistent_id FROM raw_datasets WHERE metadata_json IS NULL"
    ).fetchall()
    return [r[0] for r in rows]


def count_by_collection(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT collection, COUNT(*) FROM dataset_collections GROUP BY collection"
    ).fetchall()
    return dict(rows)


def total_dataset_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM raw_datasets").fetchone()[0]


def datasets_ready_to_structure(conn: sqlite3.Connection) -> list[str]:
    """Datasets with real (non-restricted) metadata that haven't been
    through Phase 2 structuring yet."""
    rows = conn.execute(
        """
        SELECT r.persistent_id FROM raw_datasets r
        LEFT JOIN structured_datasets s ON s.persistent_id = r.persistent_id
        WHERE r.metadata_json IS NOT NULL
          AND s.persistent_id IS NULL
          AND json_extract(r.metadata_json, '$._restricted') IS NULL
        """
    ).fetchall()
    return [r[0] for r in rows]


def upsert_structured_dataset(conn: sqlite3.Connection, persistent_id: str, structured: dict) -> None:
    conn.execute(
        """
        INSERT INTO structured_datasets (persistent_id, structured_json, structured_at)
        VALUES (?, ?, ?)
        ON CONFLICT(persistent_id) DO UPDATE SET
            structured_json = excluded.structured_json,
            structured_at = excluded.structured_at
        """,
        (persistent_id, json.dumps(structured), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def structured_dataset_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM structured_datasets").fetchone()[0]


def all_structured_datasets(conn: sqlite3.Connection) -> list[tuple[str, dict]]:
    rows = conn.execute("SELECT persistent_id, structured_json FROM structured_datasets").fetchall()
    return [(pid, json.loads(sj)) for pid, sj in rows]


def datasets_ready_to_index(conn: sqlite3.Connection) -> list[str]:
    """Structured datasets that haven't been added to the FTS5 index yet."""
    rows = conn.execute(
        """
        SELECT s.persistent_id FROM structured_datasets s
        LEFT JOIN dataset_fts f ON f.persistent_id = s.persistent_id
        WHERE f.persistent_id IS NULL
        """
    ).fetchall()
    return [r[0] for r in rows]


def upsert_fts_dataset(conn: sqlite3.Connection, persistent_id: str, sd: dict) -> None:
    """Idempotent: FTS5 has no natural conflict target, so delete-then-insert."""
    conn.execute("DELETE FROM dataset_fts WHERE persistent_id = ?", (persistent_id,))
    conn.execute(
        "INSERT INTO dataset_fts (persistent_id, title, description, keywords, subjects) VALUES (?, ?, ?, ?, ?)",
        (
            persistent_id,
            sd.get("title") or "",
            sd.get("description") or "",
            " ".join(sd.get("keywords") or []),
            " ".join(sd.get("subjects") or []),
        ),
    )
    conn.commit()


def fts_dataset_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM dataset_fts").fetchone()[0]


def replace_edges(conn: sqlite3.Connection, edge_type: str, edges: list[tuple[str, str, float, str | None]]) -> None:
    """Full rebuild of one edge type: the pairwise computation is cheap
    enough (~sub-second for 818 datasets) that incremental resumability
    isn't worth the complexity -- just recompute and replace."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM dataset_edges WHERE edge_type = ?", (edge_type,))
    conn.executemany(
        """
        INSERT INTO dataset_edges (persistent_id_a, persistent_id_b, edge_type, weight, detail, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [(a, b, edge_type, weight, detail, now) for a, b, weight, detail in edges],
    )
    conn.commit()


def edge_count(conn: sqlite3.Connection, edge_type: str | None = None) -> int:
    if edge_type:
        return conn.execute(
            "SELECT COUNT(*) FROM dataset_edges WHERE edge_type = ?", (edge_type,)
        ).fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM dataset_edges").fetchone()[0]


def get_related_datasets(
    conn: sqlite3.Connection,
    persistent_id: str,
    edge_type: str | None = None,
    min_weight: float = 0.0,
    limit: int = 20,
) -> list[tuple[str, str, float, str | None]]:
    """Datasets linked to `persistent_id`, best (highest-weight) first.
    Returns (related_persistent_id, edge_type, weight, detail) tuples."""
    query = """
        SELECT
            CASE WHEN persistent_id_a = ? THEN persistent_id_b ELSE persistent_id_a END,
            edge_type, weight, detail
        FROM dataset_edges
        WHERE (persistent_id_a = ? OR persistent_id_b = ?)
          AND weight >= ?
    """
    params: list = [persistent_id, persistent_id, persistent_id, min_weight]
    if edge_type:
        query += " AND edge_type = ?"
        params.append(edge_type)
    query += " ORDER BY weight DESC LIMIT ?"
    params.append(limit)
    return conn.execute(query, params).fetchall()


def search_fts(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[tuple[str, float]]:
    """Keyword search over the FTS5 index. Returns (persistent_id, bm25_rank)
    pairs, best match first (bm25 is negative; lower/more-negative is better)."""
    rows = conn.execute(
        """
        SELECT persistent_id, bm25(dataset_fts) AS rank
        FROM dataset_fts
        WHERE dataset_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]
