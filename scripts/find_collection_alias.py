"""One-off helper: look up a Dataverse collection's alias by name.

The project brief names a "HELD" (Harvard Environment and Law Data)
collection but its alias isn't confirmed. Run this, eyeball the results,
and update COLLECTIONS in src/cafe_rag/config.py.

Usage:
    python scripts/find_collection_alias.py "Harvard Environment and Law Data"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cafe_rag.dataverse_client import DataverseClient  # noqa: E402


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "HELD"
    client = DataverseClient()
    results = client.resolve_collection_alias(query)
    if not results:
        print(f"No dataverse collections found for query: {query!r}")
        return
    for item in results:
        print(f"name={item.get('name')!r}  identifier={item.get('identifier')!r}  url={item.get('url')}")


if __name__ == "__main__":
    main()
