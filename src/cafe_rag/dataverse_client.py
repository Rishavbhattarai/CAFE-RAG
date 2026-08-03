"""Thin client over the Harvard Dataverse public Search API and Native API.

Docs: https://guides.dataverse.org/en/latest/api/search.html
      https://guides.dataverse.org/en/latest/api/native-api.html
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Iterator

import requests

from .config import (
    DATAVERSE_BASE_URL,
    REQUEST_SLEEP_SECS,
    REQUEST_TIMEOUT_SECS,
    SEARCH_PAGE_SIZE,
)


class DataverseAPIError(RuntimeError):
    pass


@dataclass
class SearchPage:
    collection: str
    start: int
    per_page: int
    raw: dict


class DataverseClient:
    def __init__(self, base_url: str = DATAVERSE_BASE_URL, session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        # requests' own `timeout` resets on every byte received, so a slowly
        # trickling response can stall well past REQUEST_TIMEOUT_SECS. Running
        # the call in a worker and bounding it with future.result() enforces
        # a real wall-clock deadline; a timed-out thread is abandoned (not
        # killed -- Python can't do that) but the bounded pool keeps that cheap.
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _get(self, path: str, params: dict, max_retries: int = 5) -> dict:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            future = self._executor.submit(
                self.session.get, url, params=params, timeout=REQUEST_TIMEOUT_SECS
            )
            try:
                resp = future.result(timeout=REQUEST_TIMEOUT_SECS + 5)
            except FutureTimeoutError:
                last_exc = TimeoutError(f"hard wall-clock timeout waiting on {url}")
                time.sleep(min(2 ** attempt, 30))
                continue
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
                continue
            time.sleep(REQUEST_SLEEP_SECS)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                last_exc = DataverseAPIError(f"{resp.status_code} for {resp.url}: {resp.text[:300]}")
                time.sleep(min(2 ** attempt, 30))
                continue
            raise DataverseAPIError(f"{resp.status_code} for {resp.url}: {resp.text[:300]}")
        raise DataverseAPIError(f"exhausted {max_retries} retries for {url}: {last_exc}")

    def iter_search_pages(
        self, collection_alias: str, per_page: int = SEARCH_PAGE_SIZE, start: int = 0
    ) -> Iterator[SearchPage]:
        """Paginate /api/search for all datasets in a collection (subtree),
        optionally resuming from a given offset instead of page 0."""
        while True:
            body = self._get(
                "/api/search",
                {
                    "q": "*",
                    "subtree": collection_alias,
                    "type": "dataset",
                    "per_page": per_page,
                    "start": start,
                },
            )
            data = body.get("data", {})
            yield SearchPage(collection=collection_alias, start=start, per_page=per_page, raw=body)

            items = data.get("items", [])
            total = data.get("total_count", 0)
            start += len(items)
            if not items or start >= total:
                break

    def get_dataset_metadata(self, persistent_id: str) -> dict:
        """Fetch full metadata for a dataset by its DOI/persistentId."""
        return self._get(
            "/api/datasets/:persistentId/",
            {"persistentId": persistent_id},
        )

    def resolve_collection_alias(self, query: str) -> list[dict]:
        """Best-effort lookup of dataverse (collection) aliases matching a name.

        Useful for confirming an unknown alias like HELD before harvesting it.
        """
        body = self._get(
            "/api/search",
            {"q": query, "type": "dataverse", "per_page": 20},
        )
        return body.get("data", {}).get("items", [])
