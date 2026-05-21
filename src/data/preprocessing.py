"""Load, clean, and feature-engineer the CDC Chronic Disease Indicators dataset."""

import logging
import pathlib
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = pathlib.Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = pathlib.Path(__file__).parents[2] / "data" / "processed"

MORTALITY_SOURCES = {"NVSS, Mortality", "NVSS", "Death Certificate"}

DROP_COLS = [
    "LocationDesc",
    "DataValueFootnoteSymbol",
    "DatavalueFootnote",
    "LowConfidenceLimit",
    "HighConfidenceLimit",
    "Response",
    "ResponseID",
]

TOPIC_ORDER = [
    "Alcohol", "Arthritis", "Asthma", "Cancer", "Cardiovascular Disease",
    "Chronic Kidney Disease", "Chronic Obstructive Pulmonary Disease",
    "Diabetes", "Disability", "Immunization", "Mental Health",
    "Nutrition, Physical Activity, and Weight Status", "Older Adults",
    "Oral Health", "Overarching Conditions", "Reproductive Health", "Tobacco",
]


def load_raw(path: pathlib.Path | None = None) -> pd.DataFrame:
    path = path or RAW_DIR / "U.S._Chronic_Disease_Indicators__CDI_.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download from: https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi\n"
            "Place in: data/raw/"
        )
    logger.info("Loading %s ...", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows × %d cols", *df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Coerce numeric target
    df["DataValue"] = pd.to_numeric(df["DataValue"], errors="coerce")
    df["DataValueAlt"] = pd.to_numeric(df["DataValueAlt"], errors="coerce")

    # Remove rows with no measurement value
    before = len(df)
    df = df.dropna(subset=["DataValueAlt"])
    logger.info("Dropped %d rows with null DataValueAlt", before - len(df))

    # Remove rows with no geolocation (needed for spatial analysis)
    df = df.dropna(subset=["GeoLocation"])

    # Drop low-signal columns
    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)

    # Drop all-null columns
    df = df.dropna(axis=1, how="all")

    assert len(df) > 10_000, "Dataset unexpectedly small after cleaning"
    logger.info("After cleaning: %d rows", len(df))
    return df


def parse_geolocation(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Latitude/Longitude from WKT POINT string."""
    df = df.copy()
    coords = df["GeoLocation"].str.extract(r"POINT \(([^ ]+) ([^ ]+)\)")
    df["Longitude"] = pd.to_numeric(coords[0], errors="coerce")
    df["Latitude"] = pd.to_numeric(coords[1], errors="coerce")
    df = df.drop(columns=["GeoLocation"])
    return df


def impute_data_value_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing DataValueUnit using within-group mode (Topic × Question)."""
    df = df.copy()
    for (topic, question), group in df.groupby(["Topic", "Question"]):
        mode = group["DataValueUnit"].mode()
        if not mode.empty:
            mask = (
                (df["Topic"] == topic)
                & (df["Question"] == question)
                & df["DataValueUnit"].isna()
            )
            df.loc[mask, "DataValueUnit"] = mode.iloc[0]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary mortality flag from data source
    df["is_mortality"] = df["DataSource"].isin(MORTALITY_SOURCES).astype(int)

    # Disease duration in years
    df["disease_duration"] = (df["YearEnd"] - df["YearStart"]).clip(lower=0)

    # Per-capita normalization placeholder column (requires population join)
    # Kept as raw DataValueAlt until population data is available
    df["value_normalized"] = df["DataValueAlt"]

    # Risk level bins based on DataValueAlt magnitude
    df["risk_level"] = pd.cut(
        df["DataValueAlt"],
        bins=[-np.inf, 100_000, 250_000, 500_000, np.inf],
        labels=["Low", "Moderate", "High", "Very High"],
    )

    # One-hot encode Topic for modeling
    if "Topic" in df.columns:
        topic_dummies = pd.get_dummies(df["Topic"], prefix="topic")
        df = pd.concat([df, topic_dummies], axis=1)

    # Sort chronologically
    df = df.sort_values(["YearStart", "LocationAbbr"]).reset_index(drop=True)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode LocationAbbr and DataValueType for sklearn compatibility."""
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    for col in ["LocationAbbr", "DataValueType", "StratificationCategory1", "Stratification1"]:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
    return df


def save_processed(df: pd.DataFrame, name: str = "cdi_processed.parquet") -> pathlib.Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df.to_parquet(path, index=False)
    logger.info("Saved to %s (%d rows)", path, len(df))
    return path


def run_pipeline() -> pd.DataFrame:
    raw = load_raw()
    df = clean(raw)
    df = parse_geolocation(df)
    df = impute_data_value_unit(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    save_processed(df)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = run_pipeline()
    print(df.shape)
    print(df["risk_level"].value_counts())
