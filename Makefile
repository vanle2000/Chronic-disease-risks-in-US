.PHONY: install test lint clean ingest ingest-cdi ingest-acs backfill dry-run fingerprint

install:
	pip install -r requirements.txt

# --- ingestion -------------------------------------------------------------
# Incremental by default: re-fetches the most recent landed year, because CDC
# revises it after first publication, then pulls everything newer.
ingest:
	python scripts/run_ingest.py

ingest-cdi:
	python scripts/run_ingest.py --source cdi

ingest-acs:
	python scripts/run_ingest.py --source acs

# Replay an explicit range, ignoring the watermark.
#   make backfill FROM=2015 TO=2018
backfill:
	@if [ -z "$(FROM)" ]; then echo "usage: make backfill FROM=2015 [TO=2023]"; exit 2; fi
	python scripts/run_ingest.py --backfill-from $(FROM) $(if $(TO),--backfill-to $(TO),)

# Resolve the year ranges without fetching or writing anything.
dry-run:
	python scripts/run_ingest.py --dry-run

# Content hash per landed partition, ignoring ingest timestamps. Two runs over
# the same source data must print identical hashes.
fingerprint:
	@python -c "import sys; sys.path.insert(0,'.'); \
	from src.ingest import land; \
	[print(f'{s:4s} {y}  {land.partition_fingerprint(s,y)}') \
	 for s in ('cdi','acs') for y in land.landed_years(s)]"

# --- quality ---------------------------------------------------------------
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ tests/ scripts/ --max-line-length=100

clean:
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -f data/processed/*.parquet

# Removes landed data AND the watermark, so the next run re-ingests from
# scratch. Separate from `clean` because it throws away 27MB and ~50s of work.
clean-raw:
	rm -rf data/raw data/_state
