"""Write fetched records to partitioned Parquet with provenance metadata.

Landing is deliberately dumb. It does not rename columns, coerce types,
filter rows, or drop anything. Landed Parquet is a faithful record of what
the source returned, so that a question about a number in a dashboard can be
traced to the exact bytes that produced it. Interpretation belongs in dbt.

Each year is written as a complete replacement of its partition. CDI publishes
a year in full rather than as a stream of changes, so replace is both correct
and what makes a backfill idempotent: re-running a range produces the same
data, which `partition_fingerprint` can prove.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import pathlib
import shutil
from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

from src.ingest import config

logger = logging.getLogger(__name__)

# Provenance columns. Prefixed so they cannot collide with a source field, and
# excluded from the row hash so that re-ingesting unchanged data is detectable
# as a no-op rather than looking like a change.
BATCH_ID_COL = "_batch_id"
INGESTED_AT_COL = "_ingested_at"
SOURCE_URL_COL = "_source_url"
ROW_HASH_COL = "_row_hash"

METADATA_COLUMNS = (BATCH_ID_COL, INGESTED_AT_COL, SOURCE_URL_COL, ROW_HASH_COL)


def row_hash(record: Mapping[str, object]) -> str:
    """Content hash of one source record.

    Keys are sorted and values serialised canonically so the hash depends on
    content alone -- not on dict ordering, which varies between JSON parses.
    Metadata columns are excluded by construction: they are stamped after
    hashing, so an unchanged row hashes identically on every re-ingest.
    """
    payload = {k: v for k, v in sorted(record.items()) if k not in METADATA_COLUMNS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def partition_dir(
    source: str, year: int, raw_dir: pathlib.Path | None = None
) -> pathlib.Path:
    return (raw_dir or config.RAW_DIR) / source / f"year={year}"


def write_partition(
    records: Sequence[Mapping[str, object]],
    *,
    source: str,
    year: int,
    batch_id: str,
    source_url: str,
    raw_dir: pathlib.Path | None = None,
) -> int:
    """Replace one year's partition. Returns the number of rows written.

    Writing zero rows is a legitimate outcome -- a year the source has not
    published yet -- and produces an empty partition rather than an absent one,
    so "we looked and there was nothing" is distinguishable from "we never
    looked".
    """
    directory = partition_dir(source, year, raw_dir)
    ingested_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    stamped = []
    for record in records:
        row = dict(record)
        row[ROW_HASH_COL] = row_hash(record)
        row[BATCH_ID_COL] = batch_id
        row[INGESTED_AT_COL] = ingested_at
        row[SOURCE_URL_COL] = source_url
        stamped.append(row)

    frame = pd.DataFrame(stamped)
    if frame.empty:
        # An empty frame has no columns, which Parquet cannot round-trip into
        # anything readable. Give it the metadata schema so downstream reads
        # of an empty partition behave like reads of a populated one.
        frame = pd.DataFrame(columns=list(METADATA_COLUMNS))

    # Nested values (CDI's geolocation is a GeoJSON object) have no scalar
    # Parquet type. Serialise them so the landed file stays flat and readable
    # by DuckDB without a struct-aware reader.
    for column in frame.columns:
        if frame[column].map(lambda v: isinstance(v, (dict, list))).any():
            frame[column] = frame[column].map(
                lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
            )

    # Replace rather than merge: a partial previous run must not leave orphan
    # rows behind a successful one.
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    output = directory / f"{batch_id}.parquet"
    frame.to_parquet(output, index=False)

    logger.info("Landed %s rows to %s", f"{len(frame):,}", output)
    return len(frame)


def read_partition(
    source: str, year: int, raw_dir: pathlib.Path | None = None
) -> pd.DataFrame:
    """Read one landed partition, or an empty frame if it does not exist."""
    directory = partition_dir(source, year, raw_dir)
    files = sorted(directory.glob("*.parquet")) if directory.exists() else []
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def partition_fingerprint(
    source: str, year: int, raw_dir: pathlib.Path | None = None
) -> str | None:
    """A hash of a partition's content, ignoring when it was ingested.

    This is the idempotency check. Two runs over the same year produce
    different batch ids and timestamps but must produce the same fingerprint.
    If they do not, the source changed or the pipeline is not deterministic --
    and the difference is worth knowing about either way.
    """
    frame = read_partition(source, year, raw_dir)
    if frame.empty or ROW_HASH_COL not in frame.columns:
        return None
    joined = "\n".join(sorted(frame[ROW_HASH_COL].astype(str)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def landed_years(
    source: str, raw_dir: pathlib.Path | None = None
) -> list[int]:
    """Years with a landed partition, ascending."""
    base = (raw_dir or config.RAW_DIR) / source
    if not base.exists():
        return []
    years = []
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("year="):
            try:
                years.append(int(child.name.removeprefix("year=")))
            except ValueError:
                logger.warning("Ignoring unparseable partition directory %s", child)
    return sorted(years)


def iter_records(frame: pd.DataFrame) -> Iterable[dict]:
    """Rows as plain dicts, for callers that would rather not touch pandas."""
    return (row.to_dict() for _, row in frame.iterrows())
