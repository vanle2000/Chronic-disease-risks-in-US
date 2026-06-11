"""Run the SQL analytics pipeline and export Tableau extracts.

Usage:
    python scripts/run_sql_pipeline.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "analytics" / "chronic_disease.duckdb"
SQL_FILES = [
    PROJECT_ROOT / "sql" / "01_create_analytics_views.sql",
    PROJECT_ROOT / "sql" / "02_export_tableau_extracts.sql",
]


def run_sql_file(sql_file: pathlib.Path) -> None:
    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_file}")

    command = ["duckdb", str(DB_PATH), "-c", f".read {sql_file}"]
    print(f"Running {sql_file.relative_to(PROJECT_ROOT)}")
    subprocess.run(command, check=True)


def main() -> None:
    (PROJECT_ROOT / "data" / "analytics").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "tableau").mkdir(parents=True, exist_ok=True)

    parquet_path = PROJECT_ROOT / "data" / "processed" / "cdi_processed.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            "Processed parquet not found. Run python src/data/preprocessing.py first."
        )

    for sql_file in SQL_FILES:
        run_sql_file(sql_file)

    print("SQL pipeline complete")
    print(f"DuckDB database: {DB_PATH}")
    print(f"Tableau extracts: {PROJECT_ROOT / 'tableau'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        raise
