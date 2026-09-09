"""Tests for the watermark contract.

These matter more than their size suggests: every other ingestion module
trusts resolve_range to decide what to fetch, and advance to decide what has
already landed. A defect here silently skips years.
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ingest.watermark import (  # noqa: E402
    WatermarkError,
    YearRange,
    advance,
    last_year,
    new_batch_id,
    read,
    resolve_range,
)

SOURCE = "cdi"
EARLIEST = 2015
LATEST = 2023


@pytest.fixture()
def state_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    directory = tmp_path / "_state"
    directory.mkdir()
    return directory


class TestYearRange:
    def test_inclusive_of_both_ends(self):
        assert YearRange(2015, 2017).years() == [2015, 2016, 2017]

    def test_single_year(self):
        assert YearRange(2020, 2020).years() == [2020]

    def test_rejects_inverted_range(self):
        with pytest.raises(WatermarkError, match="after end year"):
            YearRange(2023, 2015)


class TestBatchId:
    def test_is_unique_across_calls(self):
        assert len({new_batch_id() for _ in range(50)}) == 50

    def test_sorts_chronologically(self):
        first, second = new_batch_id(), new_batch_id()
        assert first[:16] <= second[:16]


class TestReadMissingOrCorrupt:
    def test_absent_file_reads_as_empty(self, state_dir):
        assert read(state_dir) == {}

    def test_corrupt_json_reads_as_empty_rather_than_raising(self, state_dir):
        (state_dir / "watermark.json").write_text("{not json", encoding="utf-8")
        assert read(state_dir) == {}

    def test_non_object_json_reads_as_empty(self, state_dir):
        (state_dir / "watermark.json").write_text("[1, 2, 3]", encoding="utf-8")
        assert read(state_dir) == {}

    def test_last_year_is_none_when_unset(self, state_dir):
        assert last_year(SOURCE, state_dir) is None


class TestResolveRangeIncremental:
    def test_first_run_takes_the_full_available_history(self, state_dir):
        result = resolve_range(
            SOURCE, earliest_year=EARLIEST, latest_available=LATEST, state_dir=state_dir
        )
        assert (result.start, result.end) == (EARLIEST, LATEST)

    def test_refetches_the_last_ingested_year(self, state_dir):
        # CDC revises the most recent year after publication, so an
        # incremental run must not treat it as final.
        advance(SOURCE, YearRange(2015, 2022), batch_id="b1", rows_written=10,
                state_dir=state_dir)
        result = resolve_range(
            SOURCE, earliest_year=EARLIEST, latest_available=LATEST, state_dir=state_dir
        )
        assert result.start == 2022, "should re-fetch the revisable latest year"
        assert result.end == LATEST

    def test_caps_start_at_latest_available(self, state_dir):
        # If the source retracts a year, the watermark must not produce an
        # inverted range.
        advance(SOURCE, YearRange(2015, 2023), batch_id="b1", rows_written=10,
                state_dir=state_dir)
        result = resolve_range(
            SOURCE, earliest_year=EARLIEST, latest_available=2021, state_dir=state_dir
        )
        assert (result.start, result.end) == (2021, 2021)


class TestResolveRangeBackfill:
    def test_explicit_range_ignores_the_watermark(self, state_dir):
        advance(SOURCE, YearRange(2015, 2023), batch_id="b1", rows_written=10,
                state_dir=state_dir)
        result = resolve_range(
            SOURCE, earliest_year=EARLIEST, latest_available=LATEST,
            backfill_from=2016, backfill_to=2018, state_dir=state_dir,
        )
        assert (result.start, result.end) == (2016, 2018)

    def test_open_ended_from_defaults_to_latest(self, state_dir):
        result = resolve_range(
            SOURCE, earliest_year=EARLIEST, latest_available=LATEST,
            backfill_from=2020, state_dir=state_dir,
        )
        assert (result.start, result.end) == (2020, LATEST)

    def test_open_ended_to_defaults_to_earliest(self, state_dir):
        result = resolve_range(
            SOURCE, earliest_year=EARLIEST, latest_available=LATEST,
            backfill_to=2017, state_dir=state_dir,
        )
        assert (result.start, result.end) == (EARLIEST, 2017)

    def test_rejects_years_before_the_source_exists(self, state_dir):
        # 2001 is in the login-gated legacy resource, not this one. Silently
        # clamping would produce a run that looks successful and is not.
        with pytest.raises(WatermarkError, match="only reaches back to"):
            resolve_range(
                SOURCE, earliest_year=EARLIEST, latest_available=LATEST,
                backfill_from=2001, state_dir=state_dir,
            )

    def test_rejects_years_the_source_has_not_published(self, state_dir):
        with pytest.raises(WatermarkError, match="publishes only through"):
            resolve_range(
                SOURCE, earliest_year=EARLIEST, latest_available=LATEST,
                backfill_to=2030, state_dir=state_dir,
            )


class TestAdvance:
    def test_persists_and_reloads(self, state_dir):
        advance(SOURCE, YearRange(2015, 2023), batch_id="b1", rows_written=42,
                state_dir=state_dir)
        assert last_year(SOURCE, state_dir) == 2023
        entry = read(state_dir)[SOURCE]
        assert entry["last_batch_id"] == "b1"
        assert entry["last_rows_written"] == 42
        assert entry["last_range"] == "2015-2023"

    def test_never_moves_backwards(self, state_dir):
        # A backfill of old years run after recent years have landed must not
        # convince the next incremental run that the gap is missing.
        advance(SOURCE, YearRange(2015, 2023), batch_id="b1", rows_written=10,
                state_dir=state_dir)
        advance(SOURCE, YearRange(2015, 2017), batch_id="b2", rows_written=10,
                state_dir=state_dir)
        assert last_year(SOURCE, state_dir) == 2023

    def test_sources_are_independent(self, state_dir):
        advance("cdi", YearRange(2015, 2023), batch_id="b1", rows_written=1,
                state_dir=state_dir)
        advance("acs", YearRange(2015, 2021), batch_id="b2", rows_written=1,
                state_dir=state_dir)
        assert last_year("cdi", state_dir) == 2023
        assert last_year("acs", state_dir) == 2021

    def test_creates_the_state_directory(self, tmp_path):
        nested = tmp_path / "does" / "not" / "exist"
        advance(SOURCE, YearRange(2015, 2016), batch_id="b1", rows_written=1,
                state_dir=nested)
        assert (nested / "watermark.json").exists()

    def test_leaves_no_temp_file_behind(self, state_dir):
        advance(SOURCE, YearRange(2015, 2016), batch_id="b1", rows_written=1,
                state_dir=state_dir)
        assert list(state_dir.glob("*.tmp")) == []

    def test_output_is_valid_json(self, state_dir):
        advance(SOURCE, YearRange(2015, 2016), batch_id="b1", rows_written=1,
                state_dir=state_dir)
        payload = json.loads((state_dir / "watermark.json").read_text(encoding="utf-8"))
        assert payload[SOURCE]["last_year_ingested"] == 2016


class TestFailedRunSemantics:
    def test_watermark_unchanged_when_advance_is_not_called(self, state_dir):
        """A run that dies before advance() must be repeated, not skipped."""
        advance(SOURCE, YearRange(2015, 2020), batch_id="b1", rows_written=10,
                state_dir=state_dir)
        before = read(state_dir)

        resolved = resolve_range(
            SOURCE, earliest_year=EARLIEST, latest_available=LATEST, state_dir=state_dir
        )
        assert resolved.end == LATEST  # a run would have been attempted
        # ... and it dies here, so advance() never runs.

        assert read(state_dir) == before
        assert resolve_range(
            SOURCE, earliest_year=EARLIEST, latest_available=LATEST, state_dir=state_dir
        ) == resolved, "the next run must retry the same range"
