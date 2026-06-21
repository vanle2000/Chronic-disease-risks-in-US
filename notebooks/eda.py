"""Reusable EDA plots for CDC Chronic Disease Indicators analysis."""

import logging
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def disease_topic_distribution(df: pd.DataFrame) -> None:
    """Bar chart of record count by disease topic."""
    counts = df["Topic"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(counts.index, counts.values, color="#4878cf")
    ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=8)
    ax.set_xlabel("Number of Surveillance Records")
    ax.set_title("Chronic Disease Surveillance Volume by Topic\n(CDC CDI, 2001–2021)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "topic_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved topic_distribution.png")


def trend_by_topic(df: pd.DataFrame, topics: list[str] | None = None) -> None:
    """Line chart of record frequency over time for selected topics."""
    if topics is None:
        topics = df["Topic"].value_counts().head(6).index.tolist()

    subset = df[df["Topic"].isin(topics)]
    trend = (
        subset.groupby(["YearStart", "Topic"])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for topic, grp in trend.groupby("Topic"):
        ax.plot(grp["YearStart"], grp["count"], marker="o", ms=4, label=topic, lw=2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Surveillance Record Count")
    ax.set_title("Chronic Disease Trends by Topic (2001–2021)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "topic_trends.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved topic_trends.png")


def mann_kendall_trend_test(df: pd.DataFrame, topic: str) -> dict:
    """
    Mann-Kendall trend test for a single disease topic.
    Returns whether the trend is statistically significant (p < 0.05).
    Requires: pip install pymannkendall
    Falls back to Spearman correlation if library unavailable.
    """
    yearly = (
        df[df["Topic"] == topic]
        .groupby("YearStart")["DataValueAlt"]
        .mean()
        .sort_index()
    )

    try:
        import pymannkendall as mk
        result = mk.original_test(yearly.values)
        return {
            "topic": topic,
            "trend": result.trend,
            "p_value": result.p,
            "tau": result.Tau,
            "significant": result.p < 0.05,
            "method": "Mann-Kendall",
        }
    except ImportError:
        # Spearman fallback
        rho, p = stats.spearmanr(yearly.index, yearly.values)
        return {
            "topic": topic,
            "trend": "increasing" if rho > 0 else "decreasing",
            "p_value": p,
            "tau": rho,
            "significant": p < 0.05,
            "method": "Spearman (fallback)",
        }


def demographic_churn_by_stratification(df: pd.DataFrame) -> None:
    """Bar chart of average DataValueAlt by stratification group."""
    strat_col = "Stratification1"
    if strat_col not in df.columns:
        return

    focus = [
        "Overall", "Male", "Female",
        "Black, non-Hispanic", "White, non-Hispanic", "Hispanic",
    ]
    sub = df[df[strat_col].isin(focus)]
    means = sub.groupby(strat_col)["DataValueAlt"].mean().reindex(focus).dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(means.index, means.values, color="#d62728")
    ax.bar_label(bars, fmt="{:.0f}", padding=3, fontsize=9)
    ax.set_ylabel("Mean Indicator Value")
    ax.set_title("Average Disease Indicator by Demographic Group")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "demographic_breakdown.png", dpi=150)
    plt.close()
    logger.info("Saved demographic_breakdown.png")


def state_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of record count per state × year (2010–2021)."""
    sub = df[df["YearStart"] >= 2010]
    pivot = (
        sub.groupby(["LocationAbbr", "YearStart"])
        .size()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        pivot, cmap="Blues", ax=ax,
        linewidths=0.2, linecolor="white",
        cbar_kws={"label": "Record Count"},
    )
    ax.set_title("Chronic Disease Surveillance Volume: State × Year (2010–2021)")
    ax.set_xlabel("Year")
    ax.set_ylabel("State")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "state_year_heatmap.png", dpi=150)
    plt.close()
    logger.info("Saved state_year_heatmap.png")


def feature_importance_plot(importances: pd.Series, model_name: str = "Random Forest") -> None:
    """Horizontal bar chart of top-20 feature importances."""
    top = importances.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    top.sort_values().plot.barh(ax=ax, color="#4878cf")
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.set_title(f"Top 20 Feature Importances  -  {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"feature_importance_{model_name.replace(' ', '_')}.png", dpi=150)
    plt.close()
    logger.info("Saved feature importance plot")


def risk_level_distribution(df: pd.DataFrame) -> None:
    """Pie / bar chart of risk level counts  -  always show raw counts alongside %."""
    if "risk_level" not in df.columns:
        return
    counts = df["risk_level"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar  -  shows absolute scale
    counts.plot.bar(ax=axes[0], color=["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"])
    axes[0].set_title("Risk Level Counts (absolute)")
    axes[0].set_ylabel("Records")
    axes[0].tick_params(axis="x", rotation=0)
    for p in axes[0].patches:
        axes[0].annotate(
            f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontsize=9,
        )

    # Pie  -  shows proportions but labeled with n
    labels = [f"{idx}\n(n={n:,})" for idx, n in counts.items()]
    axes[1].pie(counts.values, labels=labels, autopct="%1.1f%%",
                colors=["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"])
    axes[1].set_title("Risk Level Proportions")

    plt.suptitle("Class Imbalance: Why Raw Accuracy Is Misleading", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "risk_level_imbalance.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved risk_level_imbalance.png")
