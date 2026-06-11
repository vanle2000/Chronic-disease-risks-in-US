"""Preprocess the CDC Chronic Disease Indicators dataset.

This file creates the cleaned parquet layer that feeds SQL, Tableau, notebooks,
and machine learning.
"""

from __future__ import annotations

import logging
import pathlib
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_FILE_NAME = "U.S._Chronic_Disease_Indicators__CDI_.csv"
PROCESSED_FILE_NAME = "cdi_processed.parquet"

MORTALITY_SOURCES = {
    "NVSS, Mortality",
    "NVSS",
    "Death Certificate",
}

DROP_COLS = [
    "LocationDesc",
    "DataValueFootnoteSymbol",
    "DatavalueFootnote",
    "LowConfidenceLimit",
    "HighConfidenceLimit",
    "Response",
    "ResponseID",
]

ENCODE_COLUMNS = [
    "LocationAbbr",
    "DataValueType",
    "StratificationCategory1",
    "Stratification1",
]


def load_raw(path: pathlib.Path | None = None) -> pd.DataFrame:
    """Load the raw CDC file from data/raw."""
    file_path = path or RAW_DIR / RAW_FILE_NAME

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            f"Place the CDC CSV at data/raw/{RAW_FILE_NAME}."
        )

    logger.info("Loading raw data from %s", file_path)
    df = pd.read_csv(file_path, low_memory=False)
    logger.info("Loaded %s rows and %s columns", df.shape[0], df.shape[1])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw fields and remove records that cannot support analysis."""
    df = df.copy()

    for col in ["DataValue", "DataValueAlt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "DataValueAlt" not in df.columns:
        raise KeyError("DataValueAlt is required for analysis")

    before = len(df)
    df = df.dropna(subset=["DataValueAlt"])
    logger.info("Dropped %s rows with missing DataValueAlt", before - len(df))

    if "GeoLocation" in df.columns:
        df = df.dropna(subset=["GeoLocation"])

    drop_cols = [col for col in DROP_COLS if col in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.dropna(axis=1, how="all")

    assert len(df) > 100, "Dataset unexpectedly small after cleaning"
    return df


def parse_geolocation(df: pd.DataFrame) -> pd.DataFrame:
    """Extract longitude and latitude from CDC WKT POINT values."""
    df = df.copy()

    if "GeoLocation" not in df.columns:
        df["Longitude"] = np.nan
        df["Latitude"] = np.nan
        return df

    coords = df["GeoLocation"].astype(str).str.extract(r"POINT \(([^ ]+) ([^ ]+)\)")
    df["Longitude"] = pd.to_numeric(coords[0], errors="coerce")
    df["Latitude"] = pd.to_numeric(coords[1], errors="coerce")
    df = df.drop(columns=["GeoLocation"])
    return df


def impute_data_value_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing DataValueUnit using the mode inside Topic and Question groups."""
    df = df.copy()

    required = {"Topic", "Question", "DataValueUnit"}
    if not required.issubset(df.columns):
        return df

    modes = (
        df.dropna(subset=["DataValueUnit"])
        .groupby(["Topic", "Question"])["DataValueUnit"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else np.nan)
    )

    missing_mask = df["DataValueUnit"].isna()
    keys = list(zip(df.loc[missing_mask, "Topic"], df.loc[missing_mask, "Question"]))
    df.loc[missing_mask, "DataValueUnit"] = [modes.get(key, np.nan) for key in keys]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create analytics and baseline modeling features."""
    df = df.copy()

    if "DataSource" in df.columns:
        df["is_mortality"] = df["DataSource"].isin(MORTALITY_SOURCES).astype(int)
    else:
        df["is_mortality"] = 0

    if {"YearStart", "YearEnd"}.issubset(df.columns):
        df["YearStart"] = pd.to_numeric(df["YearStart"], errors="coerce")
        df["YearEnd"] = pd.to_numeric(df["YearEnd"], errors="coerce")
        df["disease_duration"] = (df["YearEnd"] - df["YearStart"]).clip(lower=0)
    else:
        df["disease_duration"] = 0

    df["value_normalized"] = df["DataValueAlt"]

    df["risk_level"] = pd.cut(
        df["DataValueAlt"],
        bins=[-np.inf, 100_000, 250_000, 500_000, np.inf],
        labels=["Low", "Moderate", "High", "Very High"],
    )

    if "Topic" in df.columns:
        topic_dummies = pd.get_dummies(df["Topic"], prefix="topic")
        df = pd.concat([df, topic_dummies], axis=1)

    if {"YearStart", "LocationAbbr"}.issubset(df.columns):
        df = df.sort_values(["YearStart", "LocationAbbr"]).reset_index(drop=True)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple integer encodings for baseline sklearn models."""
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()

    for col in ENCODE_COLUMNS:
        if col in df.columns:
            encoder = LabelEncoder()
            df[f"{col}_enc"] = encoder.fit_transform(df[col].astype(str))

    return df


def save_processed(df: pd.DataFrame, name: str = PROCESSED_FILE_NAME) -> pathlib.Path:
    """Save the cleaned dataset as parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / name
    df.to_parquet(output_path, index=False)
    logger.info("Saved processed data to %s", output_path)
    return output_path


def run_pipeline(path: pathlib.Path | None = None) -> pd.DataFrame:
    """Run the full preprocessing workflow."""
    raw = load_raw(path)
    df = clean(raw)
    df = parse_geolocation(df)
    df = impute_data_value_unit(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    save_processed(df)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    processed = run_pipeline()
    print(processed.shape)
    print(processed["risk_level"].value_counts(dropna=False))
