"""Census ACS 5-year client for adult population denominators.

This is the join that turns a prevalence rate into a count of people, which
is the whole point of the excess-burden metric. Without it the pipeline
produces percentages nobody can act on.

The API requires a key. Requests without one return HTTP 302 to a "Missing
Key" HTML page rather than a JSON error, so a naive client fails with a
confusing JSON parse error instead of a useful message. This module checks
for the key first and says exactly what to do about it.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Sequence

from src.ingest import config
from src.ingest.http import Fetcher, MissingCredentialError, fetch_json

logger = logging.getLogger(__name__)

SOURCE_NAME = "acs"

KEY_SIGNUP_URL = "https://api.census.gov/data/key_signup.html"


def require_api_key(env: dict[str, str] | None = None) -> str:
    """Return the Census API key, or explain precisely how to get one."""
    environ = env if env is not None else os.environ
    key = environ.get(config.ACS_API_KEY_ENV, "").strip()
    if not key:
        raise MissingCredentialError(
            f"{config.ACS_API_KEY_ENV} is not set. The Census API rejects "
            f"unauthenticated requests with a redirect to a 'Missing Key' page.\n"
            f"  1. Request a free key: {KEY_SIGNUP_URL}\n"
            f"  2. export {config.ACS_API_KEY_ENV}=<your key>\n"
            f"CDI ingestion does not need a key and runs without this."
        )
    return key


def _default_fetcher(url: str, params: dict[str, Any]) -> Any:
    return fetch_json(url, params)


def parse_response(payload: Any) -> list[dict]:
    """Convert the ACS array-of-arrays into records.

    ACS returns a header row followed by value rows, e.g.
        [["NAME", "S0101_C01_026E", "state"],
         ["Alabama", "3966465", "01"]]
    which is not the shape anything downstream wants.
    """
    if not isinstance(payload, list) or len(payload) < 1:
        raise ValueError(f"Unexpected ACS payload shape: {str(payload)[:200]}")
    header, *rows = payload
    if not isinstance(header, list):
        raise ValueError(f"ACS header row is not a list: {str(header)[:200]}")
    return [dict(zip(header, row)) for row in rows]


def fetch_year(
    year: int,
    *,
    api_key: str | None = None,
    fetcher: Fetcher | None = None,
    variable: str = config.ACS_ADULT_POP_VARIABLE,
) -> list[dict]:
    """Adult (18+) population by state for one ACS 5-year vintage."""
    fetch = fetcher or _default_fetcher
    key = api_key if api_key is not None else require_api_key()

    payload = fetch(
        config.ACS_ENDPOINT.format(year=year),
        {"get": f"NAME,{variable}", "for": "state:*", "key": key},
    )
    records = parse_response(payload)

    # Stamp the vintage: ACS 5-year estimates are labelled by their end year,
    # and a record without it cannot be joined to a CDI year.
    for record in records:
        record["acs_year"] = str(year)
    logger.info("ACS %s: %s states", year, len(records))
    return records


def iter_years(
    years: Sequence[int],
    *,
    api_key: str | None = None,
    fetcher: Fetcher | None = None,
):
    """Yield (year, records) so the caller can land each vintage as it arrives."""
    key = api_key if api_key is not None else require_api_key()
    for year in years:
        yield year, fetch_year(year, api_key=key, fetcher=fetcher)
