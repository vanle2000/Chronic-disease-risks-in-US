# Chronic Disease Risk Intelligence: Predicting Outbreak Patterns Across the U.S.

**Public Health Resource Allocation Model**  -  clusters states by disease burden, predicts mortality risk, and classifies disease severity with honest imbalance-aware evaluation across 20 years of CDC surveillance data.

---

## Case Study

### Introduction
Chronic diseases consume 90% of the U.S.'s $4.1 trillion annual healthcare spend. Public health agencies face a fundamental constraint: limited budgets, unlimited need. Every dollar spent on the wrong state or the wrong disease is a dollar that doesn't prevent a death. This project builds a data pipeline that turns 20 years of CDC surveillance records into a geographically-aware, statistically-grounded resource allocation model.

### Problem
The CDC Chronic Disease Indicators dataset is 900K+ rows of surveillance data with three critical modeling challenges:
1. **Extreme class imbalance**: the `Very High` risk class has only 3 samples  -  raw accuracy of 0.9999 is meaningless
2. **State-level confounding**: raw counts reflect surveillance capacity, not actual disease burden
3. **Temporal structure**: 20 years of data has trend signals that point estimates ignore

### Solution
Three-stage pipeline: **clean → cluster → model**

| Stage | Task | Method |
|-------|------|--------|
| 1 | State disease burden profiling | K-Means on row-normalized topic frequency vectors |
| 2 | Mortality prediction | Logistic Regression (class_weight=balanced) with 5-fold stratified CV |
| 3 | Risk level classification | Random Forest (class_weight=balanced) with macro F1 as headline metric |

Key engineering decisions:
- Row-normalize state disease matrix before clustering to remove surveillance volume bias
- Use `class_weight="balanced"` in all classifiers  -  never report raw accuracy alone
- Report **macro F1** as the primary metric (not accuracy) for imbalanced targets
- Mann-Kendall trend test for statistically significant topic trends (not just visual inspection)

### Results

| Task | Model | Metric | Value |
|------|-------|--------|-------|
| State clustering | K-Means (k=3) | Silhouette | 0.635 |
| Mortality prediction | Logistic Regression | AUC | 0.73, Macro F1: 0.64 |
| Risk classification | Random Forest | Accuracy | 0.9999 *(misleading)* |
| Risk classification | Random Forest | **Macro F1** | **0.85** *(honest)* |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | Python, Pandas, NumPy |
| Geospatial parsing | GeoPandas, WKT coordinate extraction |
| Machine learning | Scikit-learn (KMeans, LogisticRegression, RandomForest), imbalanced-learn |
| Statistical testing | SciPy (t-test, Spearman), pymannkendall (Mann-Kendall trend test) |
| Visualization | Matplotlib, Seaborn, Plotly (interactive heatmaps) |
| Testing | pytest, pytest-cov |
| Reproducibility | Makefile, parquet intermediate outputs |

---

## Data Architecture

This project is organized as a reproducible public health analytics pipeline. The raw CDC surveillance file is cleaned once, saved as a parquet analytical layer, then reused for SQL exploration, Tableau extracts, and machine learning experiments.

### Repository layout

```text
data/
├── raw/
│   └── U.S._Chronic_Disease_Indicators__CDI_.csv
│       # Source file from CDC Open Data. Not committed because of size.
├── processed/
│   └── cdi_processed.parquet
│       # Cleaned, feature-engineered dataset used by SQL, EDA, and ML.
└── analytics/
    └── chronic_disease.duckdb
        # Optional local SQL database generated from the processed parquet file.

src/
├── data/
│   └── preprocessing.py
│       # Loads raw CSV, cleans fields, parses geolocation, imputes units,
│       # engineers features, encodes categoricals, and saves parquet output.
├── models/
│   └── train.py
│       # Builds state clusters, trains mortality prediction,
│       # trains risk classification, and exports model artifacts.
└── visualization/
    └── eda.py
        # Creates topic distributions, time trends, state heatmaps,
        # demographic breakdowns, risk imbalance charts, and feature plots.

notebooks/
└── train_model.py
    # Notebook-friendly training workflow for Colab reruns and experimentation.

tests/
└── test_preprocessing.py
    # Unit tests for cleaning, geolocation parsing, feature engineering,
    # categorical encoding, and preprocessing assumptions.

reports/
├── state_clusters.csv
├── feature_importances.csv
├── model_summary.csv
└── figures/
    ├── topic_distribution.png
    ├── topic_trends.png
    ├── state_year_heatmap.png
    ├── demographic_breakdown.png
    ├── risk_level_imbalance.png
    └── feature_importance_Random_Forest.png
```

### Analysis extension layer

The next analysis layer adds SQL and Tableau without changing the core Python pipeline. The parquet file remains the single cleaned data source, while DuckDB and Tableau extracts support business-style analysis and dashboarding.

```text
sql/
├── 01_create_tables.sql
├── 02_exploratory_analysis.sql
├── 03_tableau_extracts.sql
└── 04_model_monitoring.sql

tableau/
├── topic_year_trends.csv
├── state_topic_summary.csv
├── risk_distribution.csv
├── mortality_by_topic.csv
├── state_clusters.csv
├── feature_importances.csv
└── model_summary.csv
```

### Data flow

```text
CDC Chronic Disease Indicators CSV
  → preprocessing.py
  → cdi_processed.parquet
  → DuckDB SQL analysis
  → Tableau-ready CSV extracts
  → model training and evaluation
  → reports, figures, feature importances, and model summary outputs
```

### Feature engineering pipeline

```text
Raw CDC record
  → numeric coercion for DataValue and DataValueAlt
  → missing-value cleanup and low-signal column removal
  → GeoLocation parsing into Latitude and Longitude
  → DataValueUnit imputation by Topic × Question mode
  → is_mortality flag from mortality-related data sources
  → disease_duration from YearEnd minus YearStart
  → risk_level bins for Low, Moderate, High, and Very High records
  → topic_* one-hot disease indicators
  → encoded location, value type, and stratification fields
  → model-ready state, topic, mortality, and risk features
```

---

## Key Insights & Analytics

1. **Surveillance volume ≠ disease burden.** After row-normalizing the state × topic matrix, clustering separates states by *what they track*, not *how many records they have*. Low-surveillance states cluster together  -  these are data gaps, not low-risk states.

2. **Cancer has the highest surveillance volume (127K+ records)** but Cardiovascular Disease and Tobacco are the top mortality predictors in the Logistic Regression model  -  consistent with epidemiological literature.

3. **Mental Health is the fastest-growing indicator topic post-2016.** A Mann-Kendall trend test on yearly record counts returns a statistically significant upward trend (p < 0.05), distinguishing a real trend from noise.

4. **The 0.9999 accuracy on risk classification is a warning, not a result.** The `Low` class accounts for 99.97% of records. The model learns to predict `Low` almost every time. Macro F1 of 0.85 is the number that matters  -  and `Very High` recall of 0.33 on 3 samples is the gap that needs fixing.

5. **`is_mortality` is the highest-importance feature in the Random Forest**  -  meaning the best predictor of risk level is whether the data was collected from a mortality data source. This is a feature engineering insight: the measurement methodology carries signal.

---

## Experiment Design
If deployed for public health decision-making:
- **Intervention unit:** State-level funding allocation per disease category
- **Treatment:** Redirect 15% of budget from Low-cluster states to High-cluster states
- **Primary metric:** Mortality rate per 100K population at 2-year follow-up
- **Control:** States matched by population, baseline mortality rate, and surveillance tier
- **Statistical test:** Difference-in-differences with state and year fixed effects

---

## How to Run

```bash
git clone https://github.com/vanle2000/Chronic-disease-risks-in-US.git
cd Chronic-disease-risks-in-US
pip install -r requirements.txt  # or: make install

# Download dataset from CDC Open Data (link in Data Architecture above)
# Place in: data/raw/U.S._Chronic_Disease_Indicators__CDI_.csv

python src/data/preprocessing.py   # → data/processed/cdi_processed.parquet
python src/models/train.py         # → reports/model_summary.csv

make test  # run 22 unit tests
```

**Scale to production:**
- Replace CSV load with CDC SODA API for automated weekly refresh
- Deploy risk classifier as FastAPI endpoint: state + disease → risk tier
- Use DBSCAN instead of KMeans for cluster count that adapts to data structure
- Add county-level FIPS codes for sub-state geographic granularity

---

## Challenges & What Could Be Improved

| Challenge | Improvement Path |
|-----------|-----------------|
| `Very High` class: only 3 samples, recall=0.33 | SMOTE on minority class; or reframe as anomaly detection |
| Risk bins are magnitude-based, not clinically validated | Map bins to CDC-defined severity thresholds |
| No per-capita normalization (population denominators missing) | Join with Census population data by state and year |
| No temporal model despite 20 years of data | ARIMA or Prophet for per-state disease trend forecasting |
| Single notebook → all logic in one place | Refactored into `src/` modules (done in this version) |
