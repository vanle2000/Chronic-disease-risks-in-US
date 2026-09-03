"""Ingestion constants.

Every value here was verified against the live APIs on 2026-09-03. Where a
value replaces something the v1 code assumed, the reason is recorded, because
the v1 assumption is what a reader will otherwise expect.
"""
from __future__ import annotations

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STATE_DIR = DATA_DIR / "_state"

# ---------------------------------------------------------------------------
# CDC Chronic Disease Indicators (Socrata)
# ---------------------------------------------------------------------------
# The v1 README used resource id g4ie-h725. That resource now returns
# {"error": true, "message": "You must be logged in to access this resource"}.
# hksd-2xuw is the live public replacement (catalogue-verified, updated
# 2026-06-04). Building against g4ie-h725 would fail on the first call.
CDI_DOMAIN = "data.cdc.gov"
CDI_RESOURCE_ID = "hksd-2xuw"
CDI_ENDPOINT = f"https://{CDI_DOMAIN}/resource/{CDI_RESOURCE_ID}.json"

# The API serves lowercase field names; the CSV export v1 was written against
# uses CamelCase (YearStart, LocationAbbr, DataValueAlt). We keep the API's
# names verbatim through the landing layer -- renaming belongs in dbt staging,
# not in ingestion, so that landed Parquet is a faithful record of the source.
CDI_COLUMNS = (
    "yearstart",
    "yearend",
    "locationabbr",
    "locationdesc",
    "locationid",
    "datasource",
    "topic",
    "topicid",
    "question",
    "questionid",
    "datavalueunit",
    "datavaluetype",
    "datavaluetypeid",
    "datavalue",
    "datavaluealt",
    "stratificationcategory1",
    "stratificationcategoryid1",
    "stratification1",
    "stratificationid1",
    "geolocation",
)

# Socrata's :@computed_region_* columns are internal spatial joins that change
# without notice. Selecting explicitly excludes them.
CDI_SELECT = ",".join(CDI_COLUMNS)

# Socrata caps a single page at 50k rows. 25k keeps each response near 10MB.
CDI_PAGE_SIZE = 25_000

# The live resource covers 2015-2023. Earlier history lives only in the
# login-gated g4ie-h725 resource and is therefore not reachable.
CDI_EARLIEST_YEAR = 2015

# ---------------------------------------------------------------------------
# Census ACS 5-year
# ---------------------------------------------------------------------------
# The API returns HTTP 302 -> "Missing Key" without a key. It is free and
# instant from https://api.census.gov/data/key_signup.html, read from the
# CENSUS_API_KEY environment variable.
ACS_ENDPOINT = "https://api.census.gov/data/{year}/acs/acs5/subject"

# S0101_C01_026E is "Total population; Estimate; SELECTED AGE CATEGORIES -
# 18 years and over". This is the adult denominator, matching CDI's adult
# indicators. Using B01001_001E (all ages) would inflate every rate's
# denominator by the under-18 population and understate excess burden.
ACS_ADULT_POP_VARIABLE = "S0101_C01_026E"
ACS_API_KEY_ENV = "CENSUS_API_KEY"

# acs5 is published with a lag; 2015-2019 and 2020+ are both available, but
# 2015-2019 5-year estimates overlap heavily and should not be treated as
# independent observations. Recorded here so the caveat travels with the code.
ACS_EARLIEST_YEAR = 2015

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0
USER_AGENT = "preventable-burden/0.1 (portfolio project; contact via GitHub)"
