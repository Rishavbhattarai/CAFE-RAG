"""Phase 5 query understanding: pull structured intent out of a
natural-language search query -- hazard/health domains (fixed vocab, same
as Phase 2's classifier), free-text geography hints, and a year range --
so retrieval.py can boost matching datasets rather than relying on lexical/
semantic search alone.
"""
from __future__ import annotations

import json

import ollama

from .schema import HAZARD_DOMAINS, HEALTH_DOMAINS, QueryFilters

MODEL = "gemma4"

SYSTEM_PROMPT = f"""You extract structured search intent from a researcher's natural-language
query over a climate-and-health dataset catalog. Return JSON with:
- hazard_domains: zero or more from {HAZARD_DOMAINS}, only if clearly implied by the query
- health_domains: zero or more from {HEALTH_DOMAINS}, only if clearly implied by the query
- geographic_hints: short place names mentioned or clearly implied (e.g. "United States", \
"South Asia", "Kenya", "Western US"). Empty list if the query is not geography-specific.
- year_min / year_max: integers if the query names or implies a time range (e.g. "since 2010" \
-> year_min=2010, "1990s" -> year_min=1990, year_max=1999). Null if no time range is implied.

Do not guess fields that aren't actually implied by the query text.
"""

JSON_SCHEMA = QueryFilters.model_json_schema()


def extract_filters(query: str, model: str = MODEL) -> QueryFilters:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        format=JSON_SCHEMA,
        options={"temperature": 0.0},
    )
    parsed = json.loads(response["message"]["content"])

    hazard = [h for h in parsed.get("hazard_domains", []) if h in HAZARD_DOMAINS]
    health = [h for h in parsed.get("health_domains", []) if h in HEALTH_DOMAINS]

    return QueryFilters(
        hazard_domains=hazard,
        health_domains=health,
        geographic_hints=parsed.get("geographic_hints", []),
        year_min=parsed.get("year_min"),
        year_max=parsed.get("year_max"),
    )
