"""Tests for the CDI and ACS clients.

Every test here runs offline. The clients take their transport as an argument
precisely so that CI reports on this code rather than on whether CDC and the
Census Bureau happen to be up.

The CDI fixture is a real response recorded from hksd-2xuw on 2026-09-03. The
ACS fixture is synthetic, built to the documented array-of-arrays contract,
because the API now requires a key and none is available in CI.
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ingest import acs, socrata  # noqa: E402
from src.ingest.http import (  # noqa: E402
    FetchError,
    MissingCredentialError,
    fetch_json,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CDI
# ---------------------------------------------------------------------------

class TestCdiFetchYear:
    def test_returns_records_from_a_single_page(self):
        rows = _load("cdi_sample.json")
        captured = []

        def fake(url, params):
            captured.append(params)
            return rows

        result = socrata.fetch_year(2021, fetcher=fake, page_size=1000)
        assert len(result) == len(rows)
        assert result[0]["questionid"] == rows[0]["questionid"]

    def test_filters_on_the_requested_year(self):
        captured = []

        def fake(url, params):
            captured.append(params)
            return []

        socrata.fetch_year(2019, fetcher=fake, page_size=1000)
        assert captured[0]["$where"] == "yearstart = 2019"

    def test_orders_by_id_for_stable_paging(self):
        """Without a stable sort, Socrata can repeat or drop rows across pages."""
        captured = []

        def fake(url, params):
            captured.append(params)
            return []

        socrata.fetch_year(2021, fetcher=fake)
        assert captured[0]["$order"] == ":id"

    def test_excludes_computed_region_columns(self):
        captured = []

        def fake(url, params):
            captured.append(params)
            return []

        socrata.fetch_year(2021, fetcher=fake)
        select = captured[0]["$select"]
        assert "questionid" in select
        assert ":@computed_region" not in select

    def test_pages_until_a_short_page_arrives(self):
        page_size = 2
        pages = [
            [{"i": 0}, {"i": 1}],
            [{"i": 2}, {"i": 3}],
            [{"i": 4}],  # short page ends the loop
        ]
        calls = []

        def fake(url, params):
            calls.append(params["$offset"])
            index = params["$offset"] // page_size
            return pages[index] if index < len(pages) else []

        result = socrata.fetch_year(2021, fetcher=fake, page_size=page_size)
        assert len(result) == 5
        assert calls == [0, 2, 4]

    def test_exact_multiple_of_page_size_makes_one_final_call(self):
        page_size = 2
        pages = {0: [{"i": 0}, {"i": 1}], 2: []}

        def fake(url, params):
            return pages.get(params["$offset"], [])

        result = socrata.fetch_year(2021, fetcher=fake, page_size=page_size)
        assert len(result) == 2

    def test_error_object_instead_of_array_is_reported_clearly(self):
        """Socrata returns a dict, not a list, when a resource is gated."""
        def fake(url, params):
            return {"error": True, "message": "You must be logged in"}

        with pytest.raises(TypeError, match="Expected a JSON array"):
            socrata.fetch_year(2021, fetcher=fake)


class TestCdiYearBounds:
    def test_latest_year_parses_the_aggregate(self):
        result = socrata.latest_year(fetcher=lambda u, p: [{"maxy": "2023"}])
        assert result == 2023

    def test_earliest_year_parses_the_aggregate(self):
        result = socrata.earliest_year(fetcher=lambda u, p: [{"miny": "2015"}])
        assert result == 2015

    def test_empty_response_raises(self):
        with pytest.raises(ValueError, match="Could not determine latest year"):
            socrata.latest_year(fetcher=lambda u, p: [])

    def test_missing_field_raises(self):
        with pytest.raises(ValueError, match="carried no max"):
            socrata.latest_year(fetcher=lambda u, p: [{}])


class TestCdiFixtureShape:
    """Guards the assumptions the rest of the pipeline is built on."""

    def test_fixture_uses_lowercase_api_field_names(self):
        row = _load("cdi_sample.json")[0]
        assert "yearstart" in row
        assert "YearStart" not in row, "v1 assumed CamelCase; the API is lowercase"

    def test_geolocation_is_a_geojson_object_not_wkt(self):
        row = _load("cdi_sample.json")[0]
        assert isinstance(row["geolocation"], dict)
        assert row["geolocation"]["type"] == "Point"

    def test_stable_id_columns_are_present(self):
        """Panel selection must key on ids, not on rewordable question text."""
        row = _load("cdi_sample.json")[0]
        for column in ("questionid", "topicid", "datavaluetypeid",
                       "stratificationid1", "locationid"):
            assert column in row


# ---------------------------------------------------------------------------
# ACS
# ---------------------------------------------------------------------------

class TestAcsApiKey:
    def test_missing_key_names_the_variable_and_the_signup_url(self):
        with pytest.raises(MissingCredentialError) as exc:
            acs.require_api_key(env={})
        message = str(exc.value)
        assert "CENSUS_API_KEY" in message
        assert acs.KEY_SIGNUP_URL in message

    def test_blank_key_is_treated_as_missing(self):
        with pytest.raises(MissingCredentialError):
            acs.require_api_key(env={"CENSUS_API_KEY": "   "})

    def test_present_key_is_returned(self):
        assert acs.require_api_key(env={"CENSUS_API_KEY": "abc123"}) == "abc123"

    def test_message_says_cdi_is_unaffected(self):
        """An operator should not think a missing key blocks the whole run."""
        with pytest.raises(MissingCredentialError, match="CDI ingestion does not need"):
            acs.require_api_key(env={})


class TestAcsParsing:
    def test_header_row_becomes_keys(self):
        payload = _load("acs_sample.json")
        records = acs.parse_response(payload)
        assert len(records) == len(payload) - 1
        assert records[0]["NAME"] == "Alabama"
        assert records[0]["state"] == "01"

    def test_adult_population_variable_is_carried_through(self):
        records = acs.parse_response(_load("acs_sample.json"))
        assert "S0101_C01_026E" in records[0]

    def test_header_only_payload_yields_no_records(self):
        assert acs.parse_response([["NAME", "state"]]) == []

    def test_non_list_payload_raises(self):
        with pytest.raises(ValueError, match="Unexpected ACS payload"):
            acs.parse_response({"error": "Missing Key"})

    def test_empty_payload_raises(self):
        with pytest.raises(ValueError, match="Unexpected ACS payload"):
            acs.parse_response([])


class TestAcsFetchYear:
    def test_stamps_the_vintage_year(self):
        records = acs.fetch_year(
            2022, api_key="k", fetcher=lambda u, p: _load("acs_sample.json")
        )
        assert all(r["acs_year"] == "2022" for r in records)

    def test_requests_all_states_with_the_key(self):
        captured = []

        def fake(url, params):
            captured.append((url, params))
            return _load("acs_sample.json")

        acs.fetch_year(2022, api_key="secret", fetcher=fake)
        url, params = captured[0]
        assert "2022" in url
        assert params["for"] == "state:*"
        assert params["key"] == "secret"
        assert acs.config.ACS_ADULT_POP_VARIABLE in params["get"]


# ---------------------------------------------------------------------------
# HTTP retry behaviour
# ---------------------------------------------------------------------------

class TestRetries:
    def test_client_error_is_not_retried(self, monkeypatch):
        """A 404 fails identically on every attempt; retrying just delays it."""
        import urllib.error
        import io

        attempts = []

        def boom(request, timeout=None):
            attempts.append(1)
            raise urllib.error.HTTPError(
                request.full_url, 404, "Not Found", {}, io.BytesIO(b"nope")
            )

        monkeypatch.setattr("urllib.request.urlopen", boom)
        with pytest.raises(FetchError, match="not retryable"):
            fetch_json("https://example.test/x", sleep=lambda s: None)
        assert len(attempts) == 1

    def test_server_error_is_retried_to_the_limit(self, monkeypatch):
        import urllib.error
        import io

        attempts = []

        def boom(request, timeout=None):
            attempts.append(1)
            raise urllib.error.HTTPError(
                request.full_url, 503, "Unavailable", {}, io.BytesIO(b"busy")
            )

        monkeypatch.setattr("urllib.request.urlopen", boom)
        with pytest.raises(FetchError, match="Giving up"):
            fetch_json("https://example.test/x", max_retries=3, sleep=lambda s: None)
        assert len(attempts) == 3

    def test_rate_limit_is_retried(self, monkeypatch):
        import urllib.error
        import io

        attempts = []

        def boom(request, timeout=None):
            attempts.append(1)
            raise urllib.error.HTTPError(
                request.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"slow down")
            )

        monkeypatch.setattr("urllib.request.urlopen", boom)
        with pytest.raises(FetchError):
            fetch_json("https://example.test/x", max_retries=2, sleep=lambda s: None)
        assert len(attempts) == 2

    def test_backoff_grows_exponentially(self, monkeypatch):
        import urllib.error
        import io

        slept = []

        def boom(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, 500, "Error", {}, io.BytesIO(b"")
            )

        monkeypatch.setattr("urllib.request.urlopen", boom)
        with pytest.raises(FetchError):
            fetch_json(
                "https://example.test/x", max_retries=4, backoff=1.0,
                sleep=slept.append,
            )
        assert slept == [1.0, 2.0, 4.0]
