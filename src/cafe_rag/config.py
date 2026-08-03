"""Static configuration for the Phase 1 Dataverse harvester."""

DATAVERSE_BASE_URL = "https://dataverse.harvard.edu"

# Collection aliases to harvest, all confirmed via
# scripts/find_collection_alias.py against the live Dataverse search API.
COLLECTIONS = [
    "CAFE",
    "CIESIN",
    "cafe-extracted-data",
    "held-extracted-data",
]

SEARCH_PAGE_SIZE = 100  # Dataverse API caps per_page at 1000, but smaller
# pages make caching/resuming more granular and are gentler on the API.

REQUEST_TIMEOUT_SECS = 60
# 2026-08-02: a harvest run at 0.2s got IP-blocked by Dataverse's AWS edge
# layer (blanket 403s, including on previously-successful requests) after
# ~2,100 requests. Backing this off; if it still gets blocked, wait for the
# block to clear (observed AWS WAF rate-limit blocks are typically transient)
# before re-running rather than hammering it further.
REQUEST_SLEEP_SECS = 1.5
