"""Unit tests for CDC CDI preprocessing pipeline."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from src.data.preprocessing import (
    clean,
    encode_categoricals,
    engineer_features,
    impute_data_value_unit,
    parse_geolocation,
)


def _make_raw(n: int = 300) -> pd.DataFrame:
    """Minimal synthetic CDC CDI-shaped dataset."""
    rng = np.random.default_rng(42)
    topics = [
        "Cancer", "Diabetes", "Asthma", "Cardiovascular Disease",
        "Mental Health", "Tobacco", "Alcohol",
    ]
    sources = ["NVSS, Mortality", "BRFSS", "STATE", "Death Certificate", "ACS 1-Year Estimates"]
    states = ["CA", "TX", "NY", "FL", "WA", "OH", "IL"]

    return pd.DataFrame({
        "YearStart": rng.choice(range(2010, 2022), n),
        "YearEnd": rng.choice(range(2010, 2022), n),
        "LocationAbbr": rng.choice(states, n),
        "LocationDesc": rng.choice(["California", "Texas"], n),
        "DataSource": rng.choice(sources, n),
        "Topic": rng.choice(topics, n),
        "Question": rng.choice(["Q1", "Q2", "Q3"], n),
        "DataValueUnit": rng.choice(["%", "cases", None], n),
        "DataValueType": rng.choice(["Crude Prevalence", "Age-adjusted Rate"], n),
        "DataValue": rng.uniform(0, 1000, n).astype(str),
        "DataValueAlt": rng.uniform(0, 1_000_000, n),
        "DataValueFootnoteSymbol": [None] * n,
        "DatavalueFootnote": [None] * n,
        "LowConfidenceLimit": rng.uniform(0, 500, n),
        "HighConfidenceLimit": rng.uniform(500, 1000, n),
        "StratificationCategory1": rng.choice(["Overall", "Gender", "Race/Ethnicity"], n),
        "Stratification1": rng.choice(["Overall", "Male", "Female", "Black, non-Hispanic"], n),
        "LocationID": rng.integers(1, 60, n),
        "TopicID": rng.choice(["CAN", "DIA", "AST"], n),
        "QuestionID": rng.choice(["Q001", "Q002"], n),
        "DataValueTypeID": rng.choice(["CRDPREV", "AGEADJPREV"], n),
        "StratificationCategoryID1": rng.choice(["OVERALL", "GENDER"], n),
        "StratificationID1": rng.choice(["OVR", "MALE", "FEMALE"], n),
        "GeoLocation": [
            f"POINT ({rng.uniform(-120, -70):.4f} {rng.uniform(25, 50):.4f})"
            for _ in range(n)
        ],
    })


class TestClean:
    def test_drops_footnote_columns(self):
        df = clean(_make_raw())
        assert "DataValueFootnoteSymbol" not in df.columns
        assert "DatavalueFootnote" not in df.columns

    def test_datavalue_numeric(self):
        df = clean(_make_raw())
        assert df["DataValue"].dtype == float

    def test_no_null_datavaluealt(self):
        df = clean(_make_raw())
        assert df["DataValueAlt"].isna().sum() == 0

    def test_row_count_reasonable(self):
        raw = _make_raw(300)
        df = clean(raw)
        assert len(df) > 100  # some rows may be dropped

    def test_raises_on_too_small(self):
        # Only 5 rows — should fail the size assertion
        raw = _make_raw(5)
        # Override DataValueAlt so rows aren't dropped for nulls
        raw["DataValueAlt"] = 100.0
        with pytest.raises(AssertionError):
            clean(raw)


class TestParseGeolocation:
    def test_extracts_lat_lon(self):
        raw = _make_raw(50)
        cleaned = clean(raw)
        result = parse_geolocation(cleaned)
        assert "Latitude" in result.columns
        assert "Longitude" in result.columns

    def test_removes_geolocation_col(self):
        raw = _make_raw(50)
        cleaned = clean(raw)
        result = parse_geolocation(cleaned)
        assert "GeoLocation" not in result.columns

    def test_latitude_in_valid_us_range(self):
        raw = _make_raw(100)
        cleaned = clean(raw)
        result = parse_geolocation(cleaned)
        valid = result["Latitude"].dropna()
        assert (valid >= 20).all() and (valid <= 55).all()

    def test_longitude_in_valid_us_range(self):
        raw = _make_raw(100)
        cleaned = clean(raw)
        result = parse_geolocation(cleaned)
        valid = result["Longitude"].dropna()
        assert (valid >= -130).all() and (valid <= -60).all()


class TestImputeDataValueUnit:
    def test_reduces_null_count(self):
        raw = _make_raw(200)
        cleaned = clean(raw)
        before = cleaned["DataValueUnit"].isna().sum()
        result = impute_data_value_unit(cleaned)
        after = result["DataValueUnit"].isna().sum()
        assert after <= before

    def test_preserves_non_null_values(self):
        raw = _make_raw(200)
        cleaned = clean(raw)
        non_null_before = cleaned["DataValueUnit"].dropna()
        result = impute_data_value_unit(cleaned)
        non_null_after = result.loc[non_null_before.index, "DataValueUnit"]
        assert (non_null_before == non_null_after).all()


class TestEngineerFeatures:
    def _pipeline(self, n=200):
        raw = _make_raw(n)
        return engineer_features(clean(raw))

    def test_is_mortality_is_binary(self):
        df = self._pipeline()
        assert set(df["is_mortality"].unique()).issubset({0, 1})

    def test_disease_duration_non_negative(self):
        df = self._pipeline()
        assert (df["disease_duration"] >= 0).all()

    def test_risk_level_categories(self):
        df = self._pipeline()
        expected = {"Low", "Moderate", "High", "Very High"}
        actual = set(df["risk_level"].dropna().unique())
        assert actual.issubset(expected)

    def test_topic_dummies_created(self):
        df = self._pipeline()
        topic_cols = [c for c in df.columns if c.startswith("topic_")]
        assert len(topic_cols) >= 5

    def test_topic_dummies_are_binary(self):
        df = self._pipeline()
        topic_cols = [c for c in df.columns if c.startswith("topic_")]
        for col in topic_cols:
            assert set(df[col].unique()).issubset({0, 1, True, False})


class TestEncodeCategoricals:
    def test_encoded_columns_created(self):
        raw = _make_raw(200)
        df = engineer_features(clean(raw))
        result = encode_categoricals(df)
        assert "LocationAbbr_enc" in result.columns

    def test_encoded_values_are_integers(self):
        raw = _make_raw(200)
        df = engineer_features(clean(raw))
        result = encode_categoricals(df)
        assert result["LocationAbbr_enc"].dtype in [int, np.int64, np.int32]
