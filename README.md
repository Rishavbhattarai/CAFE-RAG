# CAFE-RAG: An Agentic Discovery Layer for Climate-Health Data

Retrieval-augmented, agentic search over the Harvard Dataverse CAFE, CIESIN,
and HELD climate-health collections (1,000+ datasets, generalist repository,
no faceted search). Ask in plain language, get back ranked datasets with
DOIs, variable-level detail, and a runnable code snippet.

## Status

Phase 1 (metadata harvesting), Phase 2 (LLM structuring), Phase 3
(FTS5 + Chroma search index), and Phase 4 (knowledge graph) done for all
818 structured datasets. See project plan for the full 7-phase roadmap.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 1: Harvest

```bash
# confirm HELD's collection alias first (uncomment it in config.py once known)
python scripts/find_collection_alias.py "Harvard Environment and Law Data"

# harvest all configured collections into data/cafe_rag.db
python -m cafe_rag.harvest

# useful flags
python -m cafe_rag.harvest --collections CAFE CIESIN
python -m cafe_rag.harvest --skip-details   # search hits only, fast pass
python -m cafe_rag.harvest --force          # re-fetch, ignore cache
```

Output:
- `data/raw/search/<collection>/start_*.json` — cached raw search pages
- `data/raw/datasets/<doi>.json` — cached raw per-dataset metadata
- `data/cafe_rag.db` — SQLite `raw_datasets` table (search + metadata JSON)

The harvester is resumable: interrupt it any time, re-run, and it skips
anything already cached on disk.

## Phase 2: Structure

```bash
python -m cafe_rag.structure                # process all pending datasets
python -m cafe_rag.structure --limit 20      # smoke test on a small batch
```

Extracts the normalized `StructuredDataset` schema (schema.py) per dataset:
deterministic fields pulled straight from Dataverse's structured metadata,
plus an LLM classification pass (Ollama, `gemma4`) for hazard/health domains
and units mentioned. Resumable — only processes datasets missing a
`structured_datasets` row.

## Phase 3: Index

```bash
python -m cafe_rag.index                # build both indexes
python -m cafe_rag.index --skip-vectors # FTS5 only (fast, no LLM calls)
python -m cafe_rag.index --skip-fts     # Chroma only
```

Builds the search backbone Phase 5 (RAG + agentic retrieval) will query:
- **FTS5** (`dataset_fts` table in `cafe_rag.db`) — keyword/lexical search
  over title, description, keywords, subjects.
- **Chroma** (`data/chroma/`, collection `cafe_rag_datasets`) — semantic
  search via `nomic-embed-text` embeddings (Ollama, local, 768-dim) over
  the same fields used for Phase 2's LLM classification prompt.

Both passes are resumable: FTS5 skips rows already in `dataset_fts`, Chroma
skips ids already present in the collection.

## Phase 4: Knowledge graph

```bash
python -m cafe_rag.graph                                  # rebuild all edge types
python -m cafe_rag.graph --edge-types same_geography same_source
```

Links datasets so the system can answer "what can I join to this?" Not
incrementally resumable like Phases 1-3 -- pairwise computation over 818
datasets is cheap (a few seconds), so each run recomputes and replaces each
edge type from scratch. Edge types, stored in `dataset_edges` with a 0-1
weight:

- `same_geography` -- shared `geographic_coverage` tag (tags used by more
  than 25 datasets, like "United States" or "Global", are skipped so the
  graph doesn't collapse into a near-complete clique)
- `same_source` -- identical `source_data` string (e.g. both derived from
  "NHANES — CDC")
- `overlapping_time` -- intersecting year ranges, weighted by the overlap's
  fraction of the shorter range
- `bbox_overlap` -- intersecting geographic bounding boxes, weighted by
  intersection area as a fraction of the smaller box

Query with `db.get_related_datasets(conn, persistent_id, edge_type=None,
min_weight=0.0, limit=20)`.

## Repo layout

```
src/cafe_rag/
  config.py               collections to harvest, API settings
  dataverse_client.py     thin wrapper over the Dataverse Search + Native APIs
  db.py                   SQLite schema and upsert/query helpers
  harvest.py              Phase 1 CLI entry point
  schema.py                StructuredDataset schema + hazard/health vocab (Phase 2)
  extract_deterministic.py Phase 2: pull structured fields straight from metadata
  extract_llm.py            Phase 2: LLM hazard/health domain classification
  structure.py              Phase 2 CLI entry point
  index.py                  Phase 3 CLI entry point (FTS5 + Chroma)
  graph.py                   Phase 4 CLI entry point (knowledge graph)
scripts/
  find_collection_alias.py   look up an unknown collection's alias
data/                  gitignored: raw cache, sqlite db, chroma store
```
