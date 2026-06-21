"""
Model training for CDC Chronic Disease Indicators.
1. K-Means clustering of states by disease burden profile
2. Logistic Regression for mortality (is_mortality) prediction
3. Random Forest for risk level classification with honest evaluation
"""
import logging
import pathlib

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

PROCESSED_DIR = pathlib.Path(__file__).parents[2] / "data" / "processed"
REPORTS_DIR = pathlib.Path(__file__).parents[2] / "reports"

MORTALITY_FEATURES = [
    "YearStart", "disease_duration", "value_normalized",
    "topic_Arthritis", "topic_Cancer", "topic_Cardiovascular Disease",
    "topic_Chronic Kidney Disease", "topic_Tobacco",
    "topic_Nutrition, Physical Activity, and Weight Status",
]

RISK_FEATURES = [
    "YearStart", "disease_duration", "is_mortality",
    "LocationAbbr_enc", "DataValueType_enc",
    "topic_Alcohol", "topic_Arthritis", "topic_Asthma", "topic_Cancer",
    "topic_Cardiovascular Disease", "topic_Chronic Kidney Disease",
    "topic_Chronic Obstructive Pulmonary Disease", "topic_Diabetes",
    "topic_Disability", "topic_Immunization", "topic_Mental Health",
    "topic_Nutrition, Physical Activity, and Weight Status",
    "topic_Older Adults", "topic_Oral Health", "topic_Overarching Conditions",
    "topic_Reproductive Health", "topic_Tobacco",
]


def load_processed() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "cdi_processed.parquet")


# State-level clustering

def build_state_disease_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot to [state × disease_topic] frequency matrix.
    Normalised by state total to remove size bias.
    """
    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    state_matrix = (
        df.groupby("LocationAbbr")[topic_cols]
        .sum()
        .pipe(lambda d: d.div(d.sum(axis=1), axis=0))  # row-normalize
    )
    return state_matrix


def cluster_states(
    state_matrix: pd.DataFrame, n_clusters: int = 3, random_state: int = 42
) -> dict:
    scaler = StandardScaler()
    X = scaler.fit_transform(state_matrix)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X)

    sil = silhouette_score(X, labels)
    logger.info("K-Means (k=%d): Silhouette=%.3f", n_clusters, sil)

    state_matrix = state_matrix.copy()
    state_matrix["cluster"] = labels
    return {"silhouette": sil, "n_clusters": n_clusters, "state_clusters": state_matrix}


# Mortality prediction 

def train_mortality_model(df: pd.DataFrame) -> dict:
    available = [f for f in MORTALITY_FEATURES if f in df.columns]
    X = df[available].fillna(0)
    y = df["is_mortality"]

    logger.info(
        "Mortality task: %d rows, positive rate=%.1f%%", len(y), y.mean() * 100
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y, y_prob)
    report = classification_report(y, y_pred, output_dict=True)

    logger.info("Logistic Regression  -  AUC=%.3f | Macro F1=%.3f", auc, report["macro avg"]["f1-score"])

    model.fit(X, y)
    return {
        "model": model,
        "auc": auc,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "report": report,
        "feature_names": available,
    }


# Risk level classification

def train_risk_classifier(df: pd.DataFrame) -> dict:
    """
    Random Forest risk level classifier with honest evaluation.

    Critically: we report MACRO F1, not accuracy.
    Raw accuracy is misleading when 'Low' accounts for 99%+ of records.
    """
    available = [f for f in RISK_FEATURES if f in df.columns]
    df_model = df[available + ["risk_level"]].dropna(subset=["risk_level"])

    X = df_model[available].fillna(0)
    y = df_model["risk_level"].astype(str)

    class_counts = y.value_counts()
    logger.info("Risk level distribution:\n%s", class_counts.to_string())

    # Warn if any class has fewer than 10 samples
    tiny_classes = class_counts[class_counts < 10].index.tolist()
    if tiny_classes:
        logger.warning(
            "Classes %s have <10 samples  -  metrics will be unreliable for these classes",
            tiny_classes,
        )

    model = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",  # handles imbalance honestly
            n_jobs=-1,
            random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    report = classification_report(y, y_pred_cv, output_dict=True)

    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    accuracy = report["accuracy"]

    logger.info(
        "Random Forest  -  Accuracy=%.4f | Macro F1=%.3f | Weighted F1=%.3f",
        accuracy, macro_f1, weighted_f1,
    )
    logger.info(
        "NOTE: Accuracy=%.4f is misleading due to class imbalance. "
        "Macro F1=%.3f is the honest metric.",
        accuracy, macro_f1,
    )

    model.fit(X, y)
    importances = pd.Series(
        model.named_steps["clf"].feature_importances_, index=available
    ).sort_values(ascending=False)

    return {
        "model": model,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "feature_importances": importances,
        "feature_names": available,
    }


def run() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_processed()
    logger.info("Loaded %d rows", len(df))

    # ── 1. Clustering
    state_matrix = build_state_disease_matrix(df)
    cluster_result = cluster_states(state_matrix)
    cluster_result["state_clusters"].to_csv(REPORTS_DIR / "state_clusters.csv")
    logger.info("State cluster assignments saved")

    # ── 2. Mortality model
    mortality_result = train_mortality_model(df)
    logger.info(
        "\n=== Mortality Model ===\nAUC: %.3f | Macro F1: %.3f",
        mortality_result["auc"], mortality_result["macro_f1"],
    )

    # ── 3. Risk classifier
    risk_result = train_risk_classifier(df)
    risk_result["feature_importances"].to_csv(REPORTS_DIR / "feature_importances.csv")
    logger.info(
        "\n=== Risk Classifier ===\nMacro F1: %.3f | Weighted F1: %.3f",
        risk_result["macro_f1"], risk_result["weighted_f1"],
    )

    # Save summary
    summary = {
        "kmeans_silhouette": cluster_result["silhouette"],
        "mortality_auc": mortality_result["auc"],
        "mortality_macro_f1": mortality_result["macro_f1"],
        "risk_accuracy": risk_result["accuracy"],
        "risk_macro_f1": risk_result["macro_f1"],
        "risk_weighted_f1": risk_result["weighted_f1"],
    }
    pd.Series(summary).to_csv(REPORTS_DIR / "model_summary.csv", header=["value"])
    return {"clustering": cluster_result, "mortality": mortality_result, "risk": risk_result}


if __name__ == "__main__":
    run()
