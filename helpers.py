"""Reusable utilities for the Chronic Disease Indicators (CDI) analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

DEAD_SOURCES = ("Death Certificate", "NVSS, Mortality", "NVSS")


def load_cdi(path: str, dtype: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=dtype)
    df["DataValue"] = pd.to_numeric(df["DataValue"], errors="coerce")
    return df.sort_values("YearStart").reset_index(drop=True)


def extract_lat_lon(df: pd.DataFrame, geo_col: str = "GeoLocation") -> pd.DataFrame:
    coords = df[geo_col].str.extract(r"POINT \(([^ ]+) ([^ ]+)\)")
    df = df.copy()
    df["Longitude"] = pd.to_numeric(coords[0], errors="coerce")
    df["Latitude"] = pd.to_numeric(coords[1], errors="coerce")
    return df


def add_dead_status(df: pd.DataFrame, source_col: str = "DataSource") -> pd.DataFrame:
    df = df.copy()
    df["DeadStatus"] = df[source_col].isin(DEAD_SOURCES).astype(int)
    return df


def make_risk_level(values: pd.Series, bins, labels) -> pd.Series:
    return pd.cut(values, bins=bins, labels=labels)


def detect_outliers_iqr(
    df: pd.DataFrame, value_col: str, group_col: str, k: float = 3.0
) -> pd.DataFrame:
    """Flag per-group IQR outliers. Returns df with `is_outlier` and `outlier_reason` columns."""
    out = df.copy()
    out["is_outlier"] = False
    out["outlier_reason"] = ""
    for group, sub in out.groupby(group_col):
        q1, q3 = sub[value_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask = (sub[value_col] < lo) | (sub[value_col] > hi)
        out.loc[sub.index[mask], "is_outlier"] = True
        out.loc[sub.index[mask], "outlier_reason"] = f"outside [{lo:.1f}, {hi:.1f}]"
    return out


def detect_outliers_zscore(
    df: pd.DataFrame, value_col: str, group_col: str | None = None, threshold: float = 3.0
) -> pd.DataFrame:
    """Flag |z| > threshold rows; per-group if `group_col` is given, else global."""
    out = df.copy()
    out["is_outlier"] = False
    if group_col is None:
        z = (out[value_col] - out[value_col].mean()) / out[value_col].std(ddof=0)
        out["is_outlier"] = z.abs() > threshold
    else:
        for _, sub in out.groupby(group_col):
            mu, sd = sub[value_col].mean(), sub[value_col].std(ddof=0)
            if sd == 0 or np.isnan(sd):
                continue
            z = (sub[value_col] - mu) / sd
            out.loc[sub.index, "is_outlier"] = z.abs() > threshold
    return out


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor per column. VIF > 5 flags concerning multicollinearity."""
    from sklearn.linear_model import LinearRegression

    vifs = {}
    for col in X.columns:
        others = X.drop(columns=[col])
        r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
        vifs[col] = float("inf") if r2 >= 0.9999 else 1.0 / (1.0 - r2)
    return pd.Series(vifs).sort_values(ascending=False)


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    value_col: str,
    contamination: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    out = df.copy()
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    out["is_outlier"] = iso.fit_predict(out[[value_col]].fillna(0)) == -1
    return out


def evaluate_multiclass(model, X_test, y_test, labels) -> dict:
    """Return a dict of metrics: accuracy, macro PR-AUC, macro ROC-AUC, classification report, CM."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    y_test_bin = label_binarize(y_test, classes=labels)

    metrics: dict = {
        "accuracy": float((y_pred == y_test).mean()),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
    }
    if y_proba is not None and y_test_bin.shape[1] == y_proba.shape[1]:
        metrics["roc_auc_macro"] = float(
            roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        )
        metrics["pr_auc_macro"] = float(
            average_precision_score(y_test_bin, y_proba, average="macro")
        )
    return metrics


def majority_class_baseline(y_train, y_test) -> float:
    majority = pd.Series(y_train).mode().iloc[0]
    return float((pd.Series(y_test) == majority).mean())
