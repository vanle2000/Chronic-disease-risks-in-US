#!/usr/bin/env python
"""Ingest CDI and ACS into the raw Parquet landing zone.

    python scripts/run_ingest.py                          # incremental, both sources
    python scripts/run_ingest.py --source cdi             # one source
    python scripts/run_ingest.py --backfill-from 2015 --backfill-to 2018
    python scripts/run_ingest.py --dry-run                # resolve ranges, fetch nothing

Exit codes: 0 success, 1 ingestion failure, 2 bad arguments or missing
credentials. CI distinguishes "the pipeline broke" from "you invoked it wrong".
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ingest import acs, config, land, socrata, watermark  # noqa: E402
from src.ingest.http import FetchError, MissingCredentialError  # noqa: E402

logger = logging.getLogger("ingest")

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest CDC CDI and Census ACS into data/raw/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", choices=("cdi", "acs", "all"), default="all",
        help="Which source to ingest (default: all).",
    )
    parser.add_argument(
        "--backfill-from", type=int, metavar="YYYY",
        help="Start of an explicit backfill range. Ignores the watermark.",
    )
    parser.add_argument(
        "--backfill-to", type=int, metavar="YYYY",
        help="End of an explicit backfill range. Ignores the watermark.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print the year ranges without fetching or writing.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Debug logging.",
    )
    return parser.parse_args(argv)


def ingest_cdi(args: argparse.Namespace) -> int:
    """Returns rows written. Advances the watermark only on full success."""
    latest = socrata.latest_year()
    earliest = socrata.earliest_year()
    logger.info("CDI publishes %s-%s", earliest, latest)

    year_range = watermark.resolve_range(
        socrata.SOURCE_NAME,
        earliest_year=earliest,
        latest_available=latest,
        backfill_from=args.backfill_from,
        backfill_to=args.backfill_to,
    )
    logger.info("CDI range to ingest: %s", year_range)
    if args.dry_run:
        return 0

    batch_id = watermark.new_batch_id()
    total = 0
    for year, records in socrata.iter_years(year_range.years()):
        total += land.write_partition(
            records,
            source=socrata.SOURCE_NAME,
            year=year,
            batch_id=batch_id,
            source_url=config.CDI_ENDPOINT,
        )

    watermark.advance(
        socrata.SOURCE_NAME, year_range, batch_id=batch_id, rows_written=total
    )
    return total


def ingest_acs(args: argparse.Namespace) -> int:
    """Returns rows written. Advances the watermark only on full success."""
    api_key = acs.require_api_key()

    # ACS vintages are bounded by what CDI needs; pulling denominators for
    # years with no numerator is wasted work.
    latest = socrata.latest_year()
    year_range = watermark.resolve_range(
        acs.SOURCE_NAME,
        earliest_year=config.ACS_EARLIEST_YEAR,
        latest_available=latest,
        backfill_from=args.backfill_from,
        backfill_to=args.backfill_to,
    )
    logger.info("ACS range to ingest: %s", year_range)
    if args.dry_run:
        return 0

    batch_id = watermark.new_batch_id()
    total = 0
    for year, records in acs.iter_years(year_range.years(), api_key=api_key):
        total += land.write_partition(
            records,
            source=acs.SOURCE_NAME,
            year=year,
            batch_id=batch_id,
            source_url=config.ACS_ENDPOINT.format(year=year),
        )

    watermark.advance(
        acs.SOURCE_NAME, year_range, batch_id=batch_id, rows_written=total
    )
    return total


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.backfill_from and args.backfill_to and args.backfill_from > args.backfill_to:
        logger.error(
            "--backfill-from %s is after --backfill-to %s",
            args.backfill_from, args.backfill_to,
        )
        return EXIT_USAGE

    sources = ("cdi", "acs") if args.source == "all" else (args.source,)
    written: dict[str, int] = {}

    for source in sources:
        try:
            written[source] = (
                ingest_cdi(args) if source == "cdi" else ingest_acs(args)
            )
        except MissingCredentialError as exc:
            logger.error("%s: %s", source, exc)
            # A missing key is an operator problem, not a pipeline fault. If
            # CDI already succeeded, keep that work and report the partial run.
            return EXIT_USAGE
        except (FetchError, watermark.WatermarkError, ValueError, TypeError) as exc:
            logger.error("%s ingestion failed: %s", source, exc)
            logger.error("Watermark not advanced; the next run repeats this range.")
            return EXIT_FAILURE

    verb = "Would ingest" if args.dry_run else "Ingested"
    for source, rows in written.items():
        logger.info("%s %s: %s rows", verb, source, f"{rows:,}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
