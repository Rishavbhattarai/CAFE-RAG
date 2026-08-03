"""Normalized dataset schema (Phase 2 output).

Two-part design: most of these fields are already present as *structured*
custom metadata on CAFE datasets (spatial reference system, spatial file
type, geographic coverage, time period covered) -- pulled directly, no LLM
needed. Only `hazard_domains` / `health_domains` require classification,
since they only exist implicitly in free-text keywords/descriptions.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class TemporalCoverage(BaseModel):
    start: str | None = None
    end: str | None = None


class GeographicBoundingBox(BaseModel):
    west: float | None = None
    east: float | None = None
    north: float | None = None
    south: float | None = None


class StructuredDataset(BaseModel):
    persistent_id: str
    title: str | None = None
    description: str | None = None
    collections: list[str] = Field(default_factory=list)

    keywords: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)

    # Deterministic, pulled straight from Dataverse's structured fields.
    spatial_reference_system: str | None = None
    spatial_file_type: str | None = None
    geographic_units: list[str] = Field(default_factory=list)
    geographic_coverage: list[str] = Field(default_factory=list)  # country/state/city strings
    geographic_bounding_box: GeographicBoundingBox | None = None
    temporal_coverage: list[TemporalCoverage] = Field(default_factory=list)
    source_data: str | None = None
    derived_from_existing_dataset: str | None = None

    # LLM-classified: CAFE's keywords use NIEHS's Climate Change & Human
    # Health glossary, which doesn't map 1:1 onto a clean hazard/health
    # taxonomy, so this step normalizes them into a fixed vocabulary.
    hazard_domains: list[str] = Field(default_factory=list)
    health_domains: list[str] = Field(default_factory=list)
    units_mentioned: list[str] = Field(default_factory=list)


# Controlled vocabularies for the LLM classification step. Deliberately
# broad-but-fixed -- an open-ended LLM taxonomy would make cross-dataset
# filtering (the whole point of Phase 2/3) useless.
HAZARD_DOMAINS = [
    "heat", "cold", "drought", "flood", "wildfire", "hurricane_storm",
    "sea_level_rise", "air_quality", "precipitation_extreme", "wind",
    "landslide", "earthquake", "general_climate", "other_hazard",
]

HEALTH_DOMAINS = [
    "mortality", "cardiovascular", "respiratory", "infectious_disease",
    "mental_health", "heat_illness", "injury", "maternal_child_health",
    "chronic_disease", "food_water_security", "occupational_health",
    "health_access_equity", "general_public_health", "other_health",
]


class LLMClassification(BaseModel):
    hazard_domains: list[str] = Field(default_factory=list)
    health_domains: list[str] = Field(default_factory=list)
    units_mentioned: list[str] = Field(default_factory=list)
