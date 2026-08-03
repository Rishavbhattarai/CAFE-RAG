"""Phase 1 entry point: harvest all dataset metadata from configured
Dataverse collections into data/raw/ (cache) and data/cafe_rag.db (SQLite).

Resumable by design: every search page and every dataset metadata response
is cached to disk keyed by its identifier, and re-runs skip anything already
cached unless --force is passed. Safe to Ctrl-C and restart.

Usage:
    python -m cafe_rag.harvest
    python -m cafe_rag.harvest --collections CAFE CIESIN
    python -m cafe_rag.harvest --force
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from . import db
from .config import COLLECTIONS
from .dataverse_client import DataverseAPIError, DataverseClient

# How many *consecutive* failures to tolerate before assuming we've been
# rate-limited/blocked (observed: AWS ELB starts returning blanket 403s for
# every request, including ones that succeeded moments earlier) and bailing
# out early rather than burning through the rest of the queue uselessly.
CONSECUTIVE_FAILURE_BAILOUT = 15

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
DB_PATH = REPO_ROOT / "data" / "cafe_rag.db"


def _safe_name(persistent_id: str) -> str:
    return persistent_id.replace("/", "_").replace(":", "_")


def harvest_search_pages(client: DataverseClient, conn, collection: str, force: bool) -> None:
    """Replay any cached search pages from disk first (zero network cost),
    then hit the API only for pages that aren't cached yet -- either because
    this collection has never been harvested, or because it's grown since
    the last run."""
    page_dir = RAW_DIR / "search" / collection
    page_dir.mkdir(parents=True, exist_ok=True)

    cached_pages = sorted(page_dir.glob("start_*.json")) if not force else []
    total_count = None
    next_start = 0
    for cache_file in cached_pages:
        page_raw = json.loads(cache_file.read_text())
        for hit in page_raw.get("data", {}).get("items", []):
            db.upsert_search_hit(conn, collection, hit)
        total_count = page_raw.get("data", {}).get("total_count", total_count)
        next_start += len(page_raw.get("data", {}).get("items", []))

    if total_count is not None and next_start >= total_count:
        return  # fully cached, no network needed

    for page in client.iter_search_pages(collection, start=next_start):
        cache_file = page_dir / f"start_{page.start:06d}.json"
        cache_file.write_text(json.dumps(page.raw, indent=2))
        for hit in page.raw.get("data", {}).get("items", []):
            db.upsert_search_hit(conn, collection, hit)


def _status_code(exc: Exception) -> int | None:
    match = re.match(r"^(\d{3}) for ", str(exc))
    return int(match.group(1)) if match else None


def harvest_dataset_metadata(client: DataverseClient, conn, force: bool) -> None:
    dataset_dir = RAW_DIR / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    pending = db.datasets_missing_metadata(conn) if not force else [
        r[0] for r in conn.execute("SELECT persistent_id FROM raw_datasets").fetchall()
    ]

    consecutive_failures = 0
    for persistent_id in tqdm(pending, desc="dataset metadata"):
        cache_file = dataset_dir / f"{_safe_name(persistent_id)}.json"
        if not force and cache_file.exists():
            metadata = json.loads(cache_file.read_text())
        else:
            try:
                metadata = client.get_dataset_metadata(persistent_id)
            except DataverseAPIError as exc:
                consecutive_failures += 1
                if _status_code(exc) == 401:
                    # Genuinely and permanently inaccessible to an anonymous
                    # caller (observed: DesignSafe-sourced datasets that
                    # require an authenticated/permitted account). Record a
                    # sentinel so future runs don't keep re-requesting it.
                    db.upsert_dataset_metadata(conn, persistent_id, {"_restricted": True, "reason": str(exc)})
                    tqdm.write(f"RESTRICTED (permanent) {persistent_id}")
                    consecutive_failures = 0
                else:
                    tqdm.write(f"FAILED {persistent_id}: {exc}")
                if consecutive_failures >= CONSECUTIVE_FAILURE_BAILOUT:
                    tqdm.write(
                        f"{consecutive_failures} consecutive failures -- likely rate-limited/blocked. "
                        "Stopping early; re-run later to pick up where this left off."
                    )
                    return
                continue
            except Exception as exc:  # noqa: BLE001 - unexpected, log and keep going
                tqdm.write(f"FAILED {persistent_id}: {exc}")
                consecutive_failures += 1
                if consecutive_failures >= CONSECUTIVE_FAILURE_BAILOUT:
                    tqdm.write(f"{consecutive_failures} consecutive failures -- stopping early.")
                    return
                continue
            consecutive_failures = 0
            cache_file.write_text(json.dumps(metadata, indent=2))
        db.upsert_dataset_metadata(conn, persistent_id, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest Dataverse collection metadata (Phase 1).")
    parser.add_argument("--collections", nargs="*", default=COLLECTIONS)
    parser.add_argument("--force", action="store_true", help="re-fetch even if cached")
    parser.add_argument("--skip-details", action="store_true", help="only harvest search hits, skip per-dataset metadata calls")
    args = parser.parse_args()

    client = DataverseClient()
    conn = db.connect(DB_PATH)

    for collection in args.collections:
        print(f"== {collection} ==")
        try:
            harvest_search_pages(client, conn, collection, args.force)
        except Exception as exc:  # noqa: BLE001 - log and continue with next collection
            print(f"FAILED search phase for {collection}: {exc}")

    if not args.skip_details:
        harvest_dataset_metadata(client, conn, args.force)

    print("Datasets harvested per collection:")
    for collection, count in db.count_by_collection(conn).items():
        print(f"  {collection}: {count}")


if __name__ == "__main__":
    main()
