# Chronic Disease Risk Analytics in the U.S.

**SQL and Tableau-first public health analytics project with a machine learning extension.**

This project analyzes CDC Chronic Disease Indicators data to identify disease surveillance patterns, state-level risk signals, mortality-related indicators, and opportunities for public health resource prioritization. The workflow is intentionally designed like an analytics project owned by a data analyst or early-career data scientist: start with SQL, build decision-ready Tableau dashboards, then move into notebooks for feature engineering and machine learning.

---

## Project Objective

Public health teams need to understand where chronic disease burden appears concentrated, which disease topics are changing over time, and where mortality-related indicators require deeper investigation. Before training a model, the project first answers business and analytical questions with structured SQL marts and Tableau dashboards.

The machine learning layer is treated as a second phase. It builds on the SQL analysis by converting validated analytical patterns into modeling features for clustering, mortality prediction, and risk classification.

---

## Business Questions

1. Which chronic disease topics have the largest surveillance footprint across the U.S.
2. How have disease indicators changed over time by topic and state
3. Which state-topic combinations show elevated average indicator values
4. Which topics are most associated with mortality-related records
5. Are there demographic groups with consistently higher indicator values
6. Is the dataset suitable for predictive modeling, or does imbalance create risk

---

## Project Workflow

| Phase | Focus | Primary Tools | Output |
|---|---|---|---|
| 1 | Data ingestion and cleaning | Python, Pandas | Clean parquet analytical layer |
| 2 | Exploratory analytics | SQL, DuckDB | Reusable KPI and dashboard marts |
| 3 | Business intelligence | Tableau | Executive, state-level, and modeling-readiness dashboards |
| 4 | Feature engineering | Python notebooks | Model-ready feature tables |
| 5 | Machine learning | Scikit-learn | Clustering, mortality prediction, risk classification |
| 6 | Evaluation and interpretation | Macro F1, AUC, feature importance | Model summary and feature insight outputs |

---

## Why This Design Matches a 2-Year Data Scientist Role

This project is structured to show the work expected from a data scientist with roughly two years of industry experience:

- Convert a broad public dataset into a focused analytical product
- Define business questions before modeling
- Use SQL to create reliable, reusable analytical marts
- Build Tableau dashboards for stakeholder decision-making
- Validate data quality, class balance, and feature readiness before ML
- Train baseline models with transparent evaluation
- Explain when model accuracy is misleading because of target imbalance
- Separate analytics outputs from experimental notebook work

The goal is not to make ML look bigger than the analysis. The goal is to show judgment, structure, and business thinking before modeling.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Python, Pandas, NumPy |
| SQL analytics | DuckDB SQL |
| Dashboarding | Tableau |
| Visualization support | Plotly, Matplotlib, Seaborn |
| Machine learning | Scikit-learn, imbalanced-learn |
| Model interpretation | Feature importance, SHAP-ready environment |
| Testing | pytest, pytest-cov |
| Storage | CSV, Parquet, DuckDB |

---

## Data Architecture

The project uses one cleaned analytical layer as the source of truth. SQL marts and Tableau extracts are built from that layer. Notebooks and ML models come after the SQL analysis, not before it.

```text
CDC Chronic Disease Indicators CSV
  → Python cleaning and preprocessing
  → processed parquet analytical layer
  → SQL analytics marts
  → Tableau dashboard extracts
  → notebook feature engineering
  → ML training and evaluation
  → model artifacts and interpretation outputs
```

### Repository layout

```text
data/
├── raw/
│   └── U.S._Chronic_Disease_Indicators__CDI_.csv
│       # Source file from CDC Open Data. Not committed because of size.
├── processed/
│   └── cdi_processed.parquet
│       # Cleaned analytical layer used by SQL, Tableau, notebooks, and ML.
└── analytics/
    └── chronic_disease.duckdb
        # Optional local SQL database generated from the processed parquet file.

sql/
├── 01_create_analytics_views.sql
│   # Creates dashboard KPIs, topic trends, state-topic summary,
│   # mortality analysis, demographic analysis, and ML feature base views.
└── 02_export_tableau_extracts.sql
    # Exports SQL marts into Tableau-ready CSV files.

tableau/
├── README.md
│   # Dashboard design guide and extract contract.
├── dashboard_kpis.csv
├── topic_year_trends.csv
├── state_topic_summary.csv
├── risk_distribution.csv
├── mortality_by_topic.csv
└── demographic_risk.csv

notebooks/
└── train_model.py
    # Notebook-friendly workflow for feature engineering and ML experiments.

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

tests/
└── test_preprocessing.py
    # Unit tests for preprocessing assumptions and feature engineering logic.
```

---

## SQL Analytics Layer

The SQL layer is the core of the project. It converts the cleaned parquet file into stakeholder-ready marts.

| SQL object | Purpose |
|---|---|
| `analytics.vw_dashboard_kpis` | High-level KPI cards for Tableau |
| `analytics.mart_topic_year_trends` | Topic-level record volume, average indicator value, and mortality-source rate by year |
| `analytics.mart_state_topic_summary` | State and topic-level analytical table for maps, heatmaps, and drilldowns |
| `analytics.mart_risk_distribution` | Risk-level class balance for modeling readiness |
| `analytics.mart_mortality_by_topic` | Mortality-source concentration by disease topic |
| `analytics.mart_demographic_risk` | Indicator values by topic and demographic stratification |
| `analytics.ml_feature_base` | SQL-defined feature base for downstream notebooks and ML |

Run the SQL layer after generating `data/processed/cdi_processed.parquet`:

```bash
duckdb data/analytics/chronic_disease.duckdb < sql/01_create_analytics_views.sql
duckdb data/analytics/chronic_disease.duckdb < sql/02_export_tableau_extracts.sql
```

---

## Tableau Dashboard Plan

The Tableau layer is designed as the first major deliverable, before the ML notebook.

### Dashboard 1: Executive Public Health Overview

**Purpose:** summarize national chronic disease surveillance patterns for non-technical stakeholders.

Recommended views:

- KPI cards for total records, states, topics, year range, and mortality-source rate
- Bar chart of surveillance volume by chronic disease topic
- Line chart of topic trends over time
- U.S. state map using average indicator value
- Topic and year filters for interactive exploration

### Dashboard 2: State and Topic Deep Dive

**Purpose:** help analysts identify state-topic combinations that need deeper investigation.

Recommended views:

- State-topic heatmap
- State map by selected topic
- Ranked table of highest average indicator values
- Mortality-source rate by topic
- Drilldown filters for state, topic, year, and stratification group

### Dashboard 3: Modeling Readiness and Risk Signals

**Purpose:** show whether the data is ready for ML and where modeling risk exists.

Recommended views:

- Risk-level distribution chart
- Demographic indicator comparison
- Feature availability summary
- Model metric cards after training
- Feature importance chart after training

---

## Notebook and Machine Learning Phase

The notebook phase starts only after SQL analysis and Tableau dashboards establish the business context.

Current ML tasks:

| Task | Method | Business purpose |
|---|---|---|
| State segmentation | K-Means clustering | Group states by disease surveillance profile |
| Mortality prediction | Logistic Regression | Estimate whether a record is mortality-source related |
| Risk classification | Random Forest | Classify records into risk tiers with imbalance-aware evaluation |

Model evaluation focuses on metrics that match the data problem:

- AUC for mortality prediction
- Macro F1 for imbalanced risk classification
- Weighted F1 as a secondary metric
- Feature importance for model interpretation
- Risk distribution checks before trusting classification metrics

Raw accuracy is not treated as a success metric when the target is imbalanced.

---

## Feature Engineering Roadmap

The next modeling iteration should prioritize better analytical features before trying more complex models.

| Feature area | Improvement |
|---|---|
| Population normalization | Join state-year population denominators to create per-capita indicators |
| Temporal features | Add lag values, rolling averages, and year-over-year changes |
| State context | Add region, census division, and rural-urban grouping |
| Topic severity | Replace simple risk bins with clinically or statistically justified thresholds |
| Demographics | Create stratification-level features for gender and race or ethnicity |
| Model interpretation | Add SHAP analysis after the baseline Random Forest |

---

## Key Analytical Insights to Validate

These are the claims the SQL and Tableau layers should validate before finalizing the ML narrative:

1. Surveillance volume varies heavily by disease topic, which means raw counts should not be treated as disease burden by default.
2. Some states may appear low-risk because of lower surveillance volume, not because of better public health outcomes.
3. Mortality-related records are not evenly distributed across topics.
4. Risk-level classification is affected by severe class imbalance, so macro F1 is more useful than raw accuracy.
5. Feature engineering should include population denominators before making strong burden comparisons across states.

---

## How to Run

```bash
git clone https://github.com/vanle2000/Chronic-disease-risks-in-US.git
cd Chronic-disease-risks-in-US
pip install -r requirements.txt

# Place CDC file here:
# data/raw/U.S._Chronic_Disease_Indicators__CDI_.csv

# 1. Generate the cleaned analytical layer
python src/data/preprocessing.py

# 2. Build SQL marts and Tableau extracts
duckdb data/analytics/chronic_disease.duckdb < sql/01_create_analytics_views.sql
duckdb data/analytics/chronic_disease.duckdb < sql/02_export_tableau_extracts.sql

# 3. Open Tableau and connect to the CSV files in tableau/

# 4. Continue to notebooks and ML
python notebooks/train_model.py
```

---

## Project Positioning

This project is best presented as an analytics-to-ML case study:

> I used SQL to build trusted analytical marts, Tableau to communicate chronic disease patterns to stakeholders, and Python notebooks to extend the analysis into feature engineering and baseline machine learning models.

That framing is stronger than presenting it as a model-first project because it shows business judgment, technical execution, and awareness of modeling limitations.

---

## Future Improvements

| Area | Improvement Path |
|---|---|
| SQL | Add data quality checks, null-rate profiling, and query tests |
| Tableau | Publish interactive dashboards with documented stakeholder questions |
| Feature engineering | Add Census population joins for per-capita normalization |
| Modeling | Add temporal validation instead of random cross-validation |
| Imbalance handling | Compare class weights, SMOTE, anomaly detection, and threshold tuning |
| Deployment | Package SQL marts and notebooks into a reproducible pipeline |
