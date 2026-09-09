"""CDC Chronic Disease Indicators client (Socrata).

Fetches one year at a time. Socrata paginates with $limit/$offset, and a year
of CDI runs to roughly 90,000 rows, so a year is several pages.

Fetching per-year rather than per-page-across-all-years is deliberate: the
landing layer replaces a whole year at once, so a year is the unit that can
succeed or fail atomically. A run interrupted between years leaves complete
partitions behind it and an unadvanced watermark.
"""
from __future__ import annotations

import logging
from typing import Any, Iterator, Sequence

from src.ingest import config
from src.ingest.http import Fetcher, fetch_json

logger = logging.getLogger(__name__)

SOURCE_NAME = "cdi"


def _default_fetcher(url: str, params: dict[str, Any]) -> Any:
    return fetch_json(url, params)


def fetch_year(
    year: int,
    *,
    fetcher: Fetcher | None = None,
    page_size: int = config.CDI_PAGE_SIZE,
    select: str = config.CDI_SELECT,
    endpoint: str = config.CDI_ENDPOINT,
) -> list[dict]:
    """Every CDI record for one year, across as many pages as it takes."""
    fetch = fetcher or _default_fetcher
    records: list[dict] = []
    offset = 0

    while True:
        page = fetch(
            endpoint,
            {
                "$select": select,
                "$where": f"yearstart = {year}",
                "$order": ":id",  # stable paging; without it Socrata may repeat rows
                "$limit": page_size,
                "$offset": offset,
            },
        )
        if not isinstance(page, list):
            raise TypeError(
                f"Expected a JSON array from {endpoint}, got {type(page).__name__}. "
                f"This usually means an error object: {str(page)[:200]}"
            )

        records.extend(page)
        logger.debug("year=%s offset=%s fetched=%s", year, offset, len(page))

        if len(page) < page_size:
            break
        offset += page_size

    logger.info("CDI year %s: %s records", year, f"{len(records):,}")
    return records


def iter_years(
    years: Sequence[int], *, fetcher: Fetcher | None = None
) -> Iterator[tuple[int, list[dict]]]:
    """Yield (year, records) so the caller can land each year as it arrives."""
    for year in years:
        yield year, fetch_year(year, fetcher=fetcher)


def latest_year(
    *,
    fetcher: Fetcher | None = None,
    endpoint: str = config.CDI_ENDPOINT,
) -> int:
    """The most recent year the resource currently publishes.

    Asked rather than assumed: hardcoding the latest year means the pipeline
    silently stops picking up new data the moment CDC publishes another.
    """
    fetch = fetcher or _default_fetcher
    result = fetch(endpoint, {"$select": "max(yearstart) as maxy"})
    if not isinstance(result, list) or not result:
        raise ValueError(f"Could not determine latest year from {endpoint}: {result!r}")
    value = result[0].get("maxy")
    if value is None:
        raise ValueError(f"Response carried no max(yearstart): {result!r}")
    return int(value)


def earliest_year(
    *,
    fetcher: Fetcher | None = None,
    endpoint: str = config.CDI_ENDPOINT,
) -> int:
    """The earliest year the resource currently publishes."""
    fetch = fetcher or _default_fetcher
    result = fetch(endpoint, {"$select": "min(yearstart) as miny"})
    if not isinstance(result, list) or not result:
        raise ValueError(f"Could not determine earliest year from {endpoint}: {result!r}")
    value = result[0].get("miny")
    if value is None:
        raise ValueError(f"Response carried no min(yearstart): {result!r}")
    return int(value)
