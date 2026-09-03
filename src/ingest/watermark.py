"""High-water marks for incremental ingestion.

The watermark answers one question: which years should the next run fetch?

Two rules shape the design.

1. The watermark advances only after a run succeeds. A run that dies halfway
   leaves it untouched, so the next run repeats the same range rather than
   skipping the years that never landed.

2. An incremental run re-fetches the most recent year already ingested. CDC
   revises the latest year after first publication, so treating it as final
   would freeze the first draft of the newest data permanently.

State lives in a JSON file under data/_state/, which .gitignore excludes.
It is derived state: committing it would create merge conflicts and would
misrepresent what a fresh clone actually holds.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import pathlib
import uuid

from src.ingest import config

logger = logging.getLogger(__name__)

WATERMARK_FILENAME = "watermark.json"


class WatermarkError(RuntimeError):
    """Raised when a requested range cannot be satisfied."""


@dataclasses.dataclass(frozen=True)
class YearRange:
    """An inclusive range of years to fetch."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise WatermarkError(
                f"start year {self.start} is after end year {self.end}"
            )

    def years(self) -> list[int]:
        return list(range(self.start, self.end + 1))

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"


def new_batch_id() -> str:
    """A batch id that sorts chronologically and never collides.

    Sortable because every landed row carries it, and reading a partition in
    batch order is how you reconstruct what a given run wrote.
    """
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def watermark_path(state_dir: pathlib.Path | None = None) -> pathlib.Path:
    return (state_dir or config.STATE_DIR) / WATERMARK_FILENAME


def read(state_dir: pathlib.Path | None = None) -> dict:
    """Return the watermark document, or an empty one if absent or corrupt.

    A corrupt file is treated as absent rather than fatal: the recovery is a
    full re-ingest, which is safe because landing is idempotent. Failing hard
    here would leave an operator with a broken pipeline and no obvious fix.
    """
    path = watermark_path(state_dir)
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Watermark at %s unreadable (%s); treating as empty", path, exc)
        return {}
    if not isinstance(document, dict):
        logger.warning("Watermark at %s is not an object; treating as empty", path)
        return {}
    return document


def last_year(source: str, state_dir: pathlib.Path | None = None) -> int | None:
    """Latest year successfully ingested for ``source``, or None."""
    entry = read(state_dir).get(source)
    if not isinstance(entry, dict):
        return None
    value = entry.get("last_year_ingested")
    return value if isinstance(value, int) else None


def resolve_range(
    source: str,
    *,
    earliest_year: int,
    latest_available: int,
    backfill_from: int | None = None,
    backfill_to: int | None = None,
    state_dir: pathlib.Path | None = None,
) -> YearRange:
    """Decide which years the next run should fetch.

    An explicit backfill range wins outright and ignores the watermark; that
    is the whole point of asking for one. Otherwise the range runs from the
    last ingested year (re-fetched, per rule 2) through the latest year the
    source currently offers.
    """
    if backfill_from is not None or backfill_to is not None:
        start = backfill_from if backfill_from is not None else earliest_year
        end = backfill_to if backfill_to is not None else latest_available
        if start < earliest_year:
            raise WatermarkError(
                f"{source}: requested backfill from {start}, but the source "
                f"only reaches back to {earliest_year}"
            )
        if end > latest_available:
            raise WatermarkError(
                f"{source}: requested backfill to {end}, but the source "
                f"currently publishes only through {latest_available}"
            )
        return YearRange(start, end)

    previous = last_year(source, state_dir)
    start = earliest_year if previous is None else min(previous, latest_available)
    return YearRange(start, latest_available)


def advance(
    source: str,
    year_range: YearRange,
    *,
    batch_id: str,
    rows_written: int,
    state_dir: pathlib.Path | None = None,
) -> dict:
    """Record a successful run. Call this only after landing succeeds.

    The stored year never moves backwards: a backfill of 2015-2017 run after
    2023 has already landed must not convince the next incremental run that
    2018 onwards is missing.
    """
    directory = state_dir or config.STATE_DIR
    directory.mkdir(parents=True, exist_ok=True)

    document = read(state_dir)
    previous = last_year(source, state_dir)
    document[source] = {
        "last_year_ingested": max(year_range.end, previous or year_range.end),
        "last_run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "last_batch_id": batch_id,
        "last_range": str(year_range),
        "last_rows_written": rows_written,
    }

    path = watermark_path(state_dir)
    # Write to a sibling temp file and replace, so an interrupted write cannot
    # leave a half-serialised watermark that the next run has to recover from.
    temp_path = path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)

    logger.info(
        "Watermark for %s advanced to %s (batch %s, %s rows)",
        source,
        document[source]["last_year_ingested"],
        batch_id,
        f"{rows_written:,}",
    )
    return document
