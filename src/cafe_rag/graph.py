"""Phase 4 entry point: build a knowledge graph linking structured_datasets
by shared geography, overlapping time period, and shared original source --
so the system can answer "what can I join to this?"

Not incrementally resumable like Phases 1-3: with 818 datasets the full
pairwise computation is sub-second to a few seconds per edge type, so each
run just recomputes and replaces each edge type wholesale (db.replace_edges).

Edge types:
- same_geography: datasets tagged with the same geographic_coverage value.
  Values used by more than GEO_GROUP_CAP datasets are skipped -- broad tags
  like "United States" or "Global" would otherwise create a near-complete
  graph instead of a useful one. weight = 1.0 (categorical match).
- same_source: datasets whose `source_data` string matches exactly (e.g.
  two datasets both derived from "NHANES -- CDC"). weight = 1.0.
- overlapping_time: temporal_coverage year ranges intersect. weight = the
  overlap's fraction of the *shorter* of the two ranges, so a narrow range
  fully contained in a broad one scores higher than two ranges that barely
  touch at the edges.
- bbox_overlap: geographic_bounding_box rectangles intersect (standard
  axis-aligned bounding box test). weight = intersection area as a fraction
  of the smaller box's area.

Usage:
    python -m cafe_rag.graph
    python -m cafe_rag.graph --edge-types same_geography same_source
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from . import db

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "cafe_rag.db"

GEO_GROUP_CAP = 25
EDGE_TYPES = ["same_geography", "same_source", "overlapping_time", "bbox_overlap"]


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def build_same_geography(rows: list[tuple[str, dict]]) -> list[tuple[str, str, float, str | None]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for pid, sd in rows:
        for g in sd.get("geographic_coverage") or []:
            groups[g].add(pid)

    edges: dict[tuple[str, str], set[str]] = defaultdict(set)
    for value, members in groups.items():
        if len(members) > GEO_GROUP_CAP:
            continue
        for a, b in combinations(sorted(members), 2):
            edges[_pair(a, b)].add(value)

    return [(a, b, 1.0, ", ".join(sorted(values))) for (a, b), values in edges.items()]


def build_same_source(rows: list[tuple[str, dict]]) -> list[tuple[str, str, float, str | None]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for pid, sd in rows:
        source = sd.get("source_data")
        if source:
            groups[source].add(pid)

    edges: list[tuple[str, str, float, str | None]] = []
    for source, members in groups.items():
        for a, b in combinations(sorted(members), 2):
            edges.append((a, b, 1.0, source))
    return edges


def _year_range(sd: dict) -> tuple[int, int] | None:
    years = []
    for r in sd.get("temporal_coverage") or []:
        for key in ("start", "end"):
            value = r.get(key)
            if value:
                match = re.search(r"\d{4}", value)
                if match:
                    years.append(int(match.group()))
    return (min(years), max(years)) if years else None


def build_overlapping_time(rows: list[tuple[str, dict]]) -> list[tuple[str, str, float, str | None]]:
    ranged = [(pid, _year_range(sd)) for pid, sd in rows]
    ranged = [(pid, r) for pid, r in ranged if r]

    edges = []
    for (pid_a, (a_start, a_end)), (pid_b, (b_start, b_end)) in combinations(ranged, 2):
        overlap_start = max(a_start, b_start)
        overlap_end = min(a_end, b_end)
        if overlap_start > overlap_end:
            continue
        overlap_years = overlap_end - overlap_start + 1
        shorter_range = min(a_end - a_start, b_end - b_start) + 1
        weight = min(overlap_years / shorter_range, 1.0)
        a, b = _pair(pid_a, pid_b)
        detail = f"{overlap_start}-{overlap_end}"
        edges.append((a, b, weight, detail))
    return edges


def _bbox_area(b: dict) -> float:
    return (b["east"] - b["west"]) * (b["north"] - b["south"])


def build_bbox_overlap(rows: list[tuple[str, dict]]) -> list[tuple[str, str, float, str | None]]:
    boxed = []
    for pid, sd in rows:
        b = sd.get("geographic_bounding_box")
        if b and all(b.get(k) is not None for k in ("west", "east", "north", "south")):
            boxed.append((pid, b))

    edges = []
    for (pid_a, box_a), (pid_b, box_b) in combinations(boxed, 2):
        west = max(box_a["west"], box_b["west"])
        east = min(box_a["east"], box_b["east"])
        south = max(box_a["south"], box_b["south"])
        north = min(box_a["north"], box_b["north"])
        if west > east or south > north:
            continue
        intersection_area = (east - west) * (north - south)
        smaller_area = min(_bbox_area(box_a), _bbox_area(box_b))
        if smaller_area <= 0:
            continue
        weight = min(intersection_area / smaller_area, 1.0)
        a, b = _pair(pid_a, pid_b)
        edges.append((a, b, weight, None))
    return edges


BUILDERS = {
    "same_geography": build_same_geography,
    "same_source": build_same_source,
    "overlapping_time": build_overlapping_time,
    "bbox_overlap": build_bbox_overlap,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the knowledge graph (Phase 4).")
    parser.add_argument("--edge-types", nargs="+", choices=EDGE_TYPES, default=EDGE_TYPES)
    args = parser.parse_args()

    conn = db.connect(DB_PATH)
    rows = db.all_structured_datasets(conn)

    for edge_type in args.edge_types:
        edges = BUILDERS[edge_type](rows)
        db.replace_edges(conn, edge_type, edges)
        print(f"{edge_type}: {len(edges)} edges")

    print(f"done. dataset_edges total = {db.edge_count(conn)}")


if __name__ == "__main__":
    main()
