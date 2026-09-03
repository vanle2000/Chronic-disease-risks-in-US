"""Tests for the landing layer.

The property that matters here is idempotency: re-running a year must produce
the same data, and must be provable rather than asserted.
"""
import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ingest.land import (  # noqa: E402
    BATCH_ID_COL,
    INGESTED_AT_COL,
    ROW_HASH_COL,
    SOURCE_URL_COL,
    landed_years,
    partition_dir,
    partition_fingerprint,
    read_partition,
    row_hash,
    write_partition,
)

SOURCE = "cdi"
URL = "https://data.cdc.gov/resource/hksd-2xuw.json"


def _records(n: int = 3) -> list[dict]:
    return [
        {
            "yearstart": "2021",
            "locationabbr": ["AR", "CA", "TX"][i % 3],
            "questionid": "AST01",
            "datavaluealt": str(10.0 + i),
            "geolocation": {"type": "Point", "coordinates": [-92.2, 34.7]},
        }
        for i in range(n)
    ]


@pytest.fixture()
def raw_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "raw"


class TestRowHash:
    def test_is_stable_across_calls(self):
        record = _records(1)[0]
        assert row_hash(record) == row_hash(dict(record))

    def test_ignores_key_order(self):
        a = {"x": 1, "y": 2}
        b = {"y": 2, "x": 1}
        assert row_hash(a) == row_hash(b)

    def test_changes_when_a_value_changes(self):
        a = _records(1)[0]
        b = dict(a, datavaluealt="99.9")
        assert row_hash(a) != row_hash(b)

    def test_excludes_provenance_columns(self):
        """Re-ingesting unchanged data must hash identically."""
        bare = _records(1)[0]
        stamped = dict(
            bare,
            **{
                BATCH_ID_COL: "b1",
                INGESTED_AT_COL: "2026-09-03T00:00:00+00:00",
                SOURCE_URL_COL: URL,
            },
        )
        assert row_hash(bare) == row_hash(stamped)


class TestWritePartition:
    def test_writes_expected_row_count(self, raw_dir):
        n = write_partition(_records(5), source=SOURCE, year=2021,
                            batch_id="b1", source_url=URL, raw_dir=raw_dir)
        assert n == 5
        assert len(read_partition(SOURCE, 2021, raw_dir)) == 5

    def test_stamps_every_provenance_column(self, raw_dir):
        write_partition(_records(2), source=SOURCE, year=2021,
                        batch_id="b1", source_url=URL, raw_dir=raw_dir)
        frame = read_partition(SOURCE, 2021, raw_dir)
        for column in (BATCH_ID_COL, INGESTED_AT_COL, SOURCE_URL_COL, ROW_HASH_COL):
            assert column in frame.columns
            assert frame[column].notna().all()
        assert (frame[BATCH_ID_COL] == "b1").all()

    def test_preserves_source_columns_verbatim(self, raw_dir):
        """Landing must not rename or coerce; dbt staging owns that."""
        write_partition(_records(1), source=SOURCE, year=2021,
                        batch_id="b1", source_url=URL, raw_dir=raw_dir)
        frame = read_partition(SOURCE, 2021, raw_dir)
        assert "yearstart" in frame.columns
        assert "YearStart" not in frame.columns
        assert frame.loc[0, "questionid"] == "AST01"

    def test_serialises_nested_geolocation(self, raw_dir):
        """CDI's geolocation is a GeoJSON object, not a WKT string."""
        write_partition(_records(1), source=SOURCE, year=2021,
                        batch_id="b1", source_url=URL, raw_dir=raw_dir)
        frame = read_partition(SOURCE, 2021, raw_dir)
        assert isinstance(frame.loc[0, "geolocation"], str)
        assert "Point" in frame.loc[0, "geolocation"]

    def test_partitions_by_year(self, raw_dir):
        write_partition(_records(1), source=SOURCE, year=2021,
                        batch_id="b1", source_url=URL, raw_dir=raw_dir)
        write_partition(_records(1), source=SOURCE, year=2022,
                        batch_id="b2", source_url=URL, raw_dir=raw_dir)
        assert partition_dir(SOURCE, 2021, raw_dir).exists()
        assert partition_dir(SOURCE, 2022, raw_dir).exists()
        assert landed_years(SOURCE, raw_dir) == [2021, 2022]

    def test_empty_year_lands_an_empty_partition(self, raw_dir):
        """'We looked and found nothing' must differ from 'we never looked'."""
        n = write_partition([], source=SOURCE, year=2024,
                            batch_id="b1", source_url=URL, raw_dir=raw_dir)
        assert n == 0
        assert partition_dir(SOURCE, 2024, raw_dir).exists()
        assert read_partition(SOURCE, 2024, raw_dir).empty
        assert 2024 in landed_years(SOURCE, raw_dir)


class TestIdempotency:
    def test_rerun_produces_the_same_fingerprint(self, raw_dir):
        write_partition(_records(4), source=SOURCE, year=2021,
                        batch_id="batch-one", source_url=URL, raw_dir=raw_dir)
        first = partition_fingerprint(SOURCE, 2021, raw_dir)

        write_partition(_records(4), source=SOURCE, year=2021,
                        batch_id="batch-two", source_url=URL, raw_dir=raw_dir)
        second = partition_fingerprint(SOURCE, 2021, raw_dir)

        assert first == second, "identical source data must fingerprint identically"

    def test_rerun_replaces_rather_than_appends(self, raw_dir):
        write_partition(_records(4), source=SOURCE, year=2021,
                        batch_id="batch-one", source_url=URL, raw_dir=raw_dir)
        write_partition(_records(4), source=SOURCE, year=2021,
                        batch_id="batch-two", source_url=URL, raw_dir=raw_dir)

        frame = read_partition(SOURCE, 2021, raw_dir)
        assert len(frame) == 4, "a re-run must not double the partition"
        assert (frame[BATCH_ID_COL] == "batch-two").all()
        assert len(list(partition_dir(SOURCE, 2021, raw_dir).glob("*.parquet"))) == 1

    def test_fingerprint_changes_when_source_data_changes(self, raw_dir):
        write_partition(_records(4), source=SOURCE, year=2021,
                        batch_id="b1", source_url=URL, raw_dir=raw_dir)
        before = partition_fingerprint(SOURCE, 2021, raw_dir)

        changed = _records(4)
        changed[0]["datavaluealt"] = "999.9"
        write_partition(changed, source=SOURCE, year=2021,
                        batch_id="b2", source_url=URL, raw_dir=raw_dir)

        assert partition_fingerprint(SOURCE, 2021, raw_dir) != before

    def test_fingerprint_ignores_ingest_timestamp(self, raw_dir):
        """Two runs seconds apart must not look like a data change."""
        write_partition(_records(2), source=SOURCE, year=2021,
                        batch_id="b1", source_url=URL, raw_dir=raw_dir)
        first = partition_fingerprint(SOURCE, 2021, raw_dir)
        write_partition(_records(2), source=SOURCE, year=2021,
                        batch_id="b2", source_url="https://elsewhere.example",
                        raw_dir=raw_dir)
        assert partition_fingerprint(SOURCE, 2021, raw_dir) == first


class TestReadMissing:
    def test_absent_partition_reads_as_empty(self, raw_dir):
        assert read_partition(SOURCE, 1999, raw_dir).empty

    def test_absent_source_has_no_landed_years(self, raw_dir):
        assert landed_years("nonexistent", raw_dir) == []

    def test_fingerprint_of_absent_partition_is_none(self, raw_dir):
        assert partition_fingerprint(SOURCE, 1999, raw_dir) is None

    def test_unparseable_partition_dir_is_ignored(self, raw_dir):
        (raw_dir / SOURCE / "year=notanumber").mkdir(parents=True)
        write_partition(_records(1), source=SOURCE, year=2021,
                        batch_id="b1", source_url=URL, raw_dir=raw_dir)
        assert landed_years(SOURCE, raw_dir) == [2021]
