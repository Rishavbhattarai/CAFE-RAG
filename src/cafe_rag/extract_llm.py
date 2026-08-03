"""LLM classification step (Phase 2): normalize a dataset's free-text
keywords/description into the fixed hazard/health domain vocabularies in
schema.py, and pull out any explicit units of measurement mentioned in the
description. Everything else in StructuredDataset is deterministic --  this
step only fills in what genuinely isn't structured anywhere in the source
metadata.
"""
from __future__ import annotations

import json

import ollama

from .schema import HAZARD_DOMAINS, HEALTH_DOMAINS, LLMClassification, StructuredDataset

MODEL = "gemma4"

SYSTEM_PROMPT = f"""You classify climate-and-health research datasets for a discovery search index.

Given a dataset's title, description, keywords, and subject tags, return JSON with:
- hazard_domains: zero or more from {HAZARD_DOMAINS}
- health_domains: zero or more from {HEALTH_DOMAINS}
- units_mentioned: short strings for any explicit units of measurement named in the text \
(e.g. "degrees C", "mm precipitation", "cases per 100,000", "micrograms per cubic meter"). \
Empty list if none are explicitly stated -- do not guess.

Only select hazard/health domains that are clearly relevant to the dataset's actual content, \
not just tangentially mentioned. A dataset can have zero, one, or multiple domains in each list. \
If the dataset is not really about a climate hazard or a health outcome (e.g. it's raw census \
or economic data used only as a covariate), it's fine to return empty lists.
"""

JSON_SCHEMA = LLMClassification.model_json_schema()


def _dataset_prompt(sd: StructuredDataset) -> str:
    parts = [f"Title: {sd.title or '(none)'}"]
    if sd.description:
        # descriptions can be long/HTML-laden; truncate for the classifier
        parts.append(f"Description: {sd.description[:2000]}")
    if sd.keywords:
        parts.append(f"Keywords: {', '.join(sd.keywords)}")
    if sd.subjects:
        parts.append(f"Subjects: {', '.join(sd.subjects)}")
    return "\n".join(parts)


def classify(sd: StructuredDataset, model: str = MODEL) -> LLMClassification:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _dataset_prompt(sd)},
        ],
        format=JSON_SCHEMA,
        options={"temperature": 0.0},
    )
    content = response["message"]["content"]
    parsed = json.loads(content)

    # Guard against the model inventing categories outside the fixed vocab.
    hazard = [h for h in parsed.get("hazard_domains", []) if h in HAZARD_DOMAINS]
    health = [h for h in parsed.get("health_domains", []) if h in HEALTH_DOMAINS]
    units = parsed.get("units_mentioned", [])

    return LLMClassification(hazard_domains=hazard, health_domains=health, units_mentioned=units)
