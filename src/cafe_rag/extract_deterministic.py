"""Deterministic extraction of already-structured fields from a Dataverse
dataset metadata blob -- no LLM involved. See schema.py for field meanings.
"""
from __future__ import annotations

from .schema import GeographicBoundingBox, StructuredDataset, TemporalCoverage


def _fields_by_name(block: dict) -> dict:
    return {f["typeName"]: f for f in block.get("fields", [])}


def _primitive(compound_value: dict, key: str) -> str | None:
    entry = compound_value.get(key)
    return entry.get("value") if entry else None


def extract_deterministic(persistent_id: str, collections: list[str], metadata: dict) -> StructuredDataset:
    data = metadata.get("data", {})
    blocks = data.get("latestVersion", {}).get("metadataBlocks", {})

    citation = _fields_by_name(blocks.get("citation", {}))
    geospatial = _fields_by_name(blocks.get("geospatial", {}))
    cafe_location = _fields_by_name(blocks.get("customCAFEDataLocation", {}))
    cafe_sources = _fields_by_name(blocks.get("customCAFEDataSources", {}))

    title = citation.get("title", {}).get("value")

    description = None
    ds_desc = citation.get("dsDescription", {}).get("value") or []
    if ds_desc:
        description = ds_desc[0].get("dsDescriptionValue", {}).get("value")

    keywords = [
        kw.get("keywordValue", {}).get("value")
        for kw in citation.get("keyword", {}).get("value", [])
        if kw.get("keywordValue", {}).get("value")
    ]

    subjects = citation.get("subject", {}).get("value") or []
    if isinstance(subjects, str):
        subjects = [subjects]

    spatial_reference_system = None
    crs_field = cafe_location.get("cafeSpatialReferenceSystem", {}).get("value")
    if isinstance(crs_field, dict):
        spatial_reference_system = _primitive(crs_field, "cafeSpatialReferenceSystemName")

    spatial_file_type = cafe_location.get("cafeSpatialFileType", {}).get("value")
    if isinstance(spatial_file_type, list):
        spatial_file_type = ", ".join(spatial_file_type)

    geographic_units = geospatial.get("geographicUnit", {}).get("value") or []
    if isinstance(geographic_units, str):
        geographic_units = [geographic_units]

    geographic_coverage: list[str] = []
    for entry in geospatial.get("geographicCoverage", {}).get("value", []) or []:
        parts = [
            _primitive(entry, key)
            for key in ("country", "state", "city", "otherGeographicCoverage")
        ]
        combined = ", ".join(p for p in parts if p)
        if combined:
            geographic_coverage.append(combined)

    bbox = None
    bbox_entries = geospatial.get("geographicBoundingBox", {}).get("value")
    if isinstance(bbox_entries, list) and bbox_entries:
        entry = bbox_entries[0]
        bbox = GeographicBoundingBox(
            west=_float_or_none(_primitive(entry, "westLongitude")),
            east=_float_or_none(_primitive(entry, "eastLongitude")),
            north=_float_or_none(_primitive(entry, "northLatitude")),
            south=_float_or_none(_primitive(entry, "southLatitude")),
        )
    elif isinstance(bbox_entries, dict):
        bbox = GeographicBoundingBox(
            west=_float_or_none(_primitive(bbox_entries, "westLongitude")),
            east=_float_or_none(_primitive(bbox_entries, "eastLongitude")),
            north=_float_or_none(_primitive(bbox_entries, "northLatitude")),
            south=_float_or_none(_primitive(bbox_entries, "southLatitude")),
        )

    temporal_coverage = [
        TemporalCoverage(
            start=_primitive(entry, "timePeriodCoveredStart"),
            end=_primitive(entry, "timePeriodCoveredEnd"),
        )
        for entry in citation.get("timePeriodCovered", {}).get("value", []) or []
    ]

    source_entries = cafe_sources.get("cafeSourceData", {}).get("value") or []
    if isinstance(source_entries, dict):
        source_entries = [source_entries]
    source_summaries = []
    for entry in source_entries:
        title = _primitive(entry, "cafeSourceDataTitle")
        institution = _primitive(entry, "cafeSourceDataInstitution")
        bits = [b for b in (title, institution) if b]
        if bits:
            source_summaries.append(" — ".join(bits))
    source_data = "; ".join(source_summaries) or None

    derived_from = cafe_sources.get("cafeDerivedFromExistingDataset", {}).get("value")

    return StructuredDataset(
        persistent_id=persistent_id,
        title=title,
        description=description,
        collections=collections,
        keywords=keywords,
        subjects=subjects,
        spatial_reference_system=spatial_reference_system,
        spatial_file_type=spatial_file_type,
        geographic_units=geographic_units,
        geographic_coverage=geographic_coverage,
        geographic_bounding_box=bbox,
        temporal_coverage=temporal_coverage,
        source_data=source_data,
        derived_from_existing_dataset=derived_from,
    )


def _float_or_none(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
