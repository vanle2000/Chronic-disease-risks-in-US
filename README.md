# Population Health Intervention Prioritization Platform

DuckDB SQL, Python, Tableau, machine learning, and experimental design for prioritizing chronic disease interventions.

## Objective:

A national public-health team has limited funding for chronic disease prevention programs. The team needs to identify which state, disease topic, and demographic segments should receive intervention resources first. This project builds an analytics platform that ranks intervention opportunities, explains risk drivers, evaluates data quality, and proposes an experimental design to measure intervention impact. This decision-support analytics platform also uses for prioritizing chronic disease interventions across U.S. states, disease topics, and demographic groups.

Using CDC Chronic Disease Indicators data, I created a cleaned analytical data layer, SQL-based reporting marts, Tableau-ready extracts, and Python ML baselines. The core output is an Intervention Priority Score that ranks state-topic-demographic segments based on disease burden, mortality signal, trend growth, demographic disparity, and data reliability.

The project also includes an experimental design for measuring whether future interventions reduce disease risk using matched cohorts or difference-in-differences.

## Tools I used

Python for cleaning and notebook work

Pandas and NumPy for data processing

DuckDB SQL for analytics views and Tableau extracts

Tableau for dashboards and business storytelling

Scikit learn for predictive models

imbalanced learn for class imbalance experiments

pytest for testing

Parquet and CSV for reusable data outputs

## Metric I use:
Intervention Priority Score =
    35% disease burden
  + 25% mortality signal
  + 20% year-over-year growth
  + 10% demographic disparity
  + 10% data reliability

#### guardrail metrics: 
Data coverage rate
Missingness rate
Demographic coverage
Metric volatility
Model calibration
False-positive intervention risk
False-negative missed-risk risk


## Repository layout

Chronic-disease-risks-in-US/
├── README.md
├── EXECUTIVE_SUMMARY.md
├── CASE_STUDY.md
├── requirements.txt
├── Makefile
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── analytics/
│   └── external/
│
sql/
├── 00_create_database.sql
├── 01_staging_cdi_observations.sql
├── 02_mart_state_topic_year.sql
├── 03_mart_demographic_disparity.sql
├── 04_mart_intervention_priority.sql
├── 05_mart_data_quality.sql
├── 06_ml_feature_store.sql
└── 07_export_tableau_extracts.sql
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── validation.py
│   │   └── feature_engineering.py
│   ├── metrics/
│   │   └── intervention_priority.py
│   ├── modeling/
│   │   ├── mortality_prediction.py
│   │   ├── risk_classification.py
│   │   ├── state_segmentation.py
│   │   └── evaluation.py
│   └── experiments/
│       ├── power_analysis.py
│       └── diff_in_diff_simulation.py
│
├── notebooks/
│   ├── 01_metric_design.ipynb
│   ├── 02_sql_validation.ipynb
│   ├── 03_modeling_baselines.ipynb
│   ├── 04_experiment_design.ipynb
│   └── 05_stakeholder_readout.ipynb
│
├── tableau/
│   ├── README.md
│   ├── dashboard_specs.md
│   ├── extracts/
│   └── screenshots/
│
├── reports/
│   ├── stakeholder_recommendations.md
│   ├── experiment_design.md
│   ├── model_card.md
│   ├── data_quality_report.md
│   └── limitations.md
│
└── tests/
    ├── test_preprocessing.py
    ├── test_metrics.py
    ├── test_model_inputs.py
    └── test_sql_outputs.py
## SQL analysis

The main SQL file creates:

1. Dashboard KPI view
2. Topic and year trend view
3. State and topic summary view
4. Risk distribution view
5. Mortality by topic view
6. Demographic risk view
7. Machine learning feature base view

These views help me answer the core business questions before I start modeling.

## Tableau dashboards

I designed the Tableau section around three dashboard ideas.

### Dashboard 1: This dashboard gives a quick view of the national chronic disease picture.

It should include:

1. Total records
2. Number of states
3. Number of disease topics
4. First year and latest year in the data
5. Mortality source rate
6. Disease topic volume
7. Topic trends over time
8. State map by average indicator value

### Dashboard 2: State and topic deep dive

This dashboard helps me compare states and disease topics in more detail.

It should include:

1. State and topic heatmap
2. State map filtered by selected topic
3. Ranked state topic table
4. Mortality source rate by topic
5. Filters for state, topic, year, and demographic group

### Dashboard 3: Modeling readiness

It should include:

1. Risk level distribution
2. Demographic risk comparison
3. Feature availability checks
4. Model summary after training
5. Feature importance after training

## Notebook and machine learning phase

After SQL and Tableau, I move into notebooks.

This is where I start asking deeper modeling questions.

1. Can I group states by disease surveillance patterns
2. Can I predict whether a record is mortality related
3. Can I classify risk level without being fooled by class imbalance
4. Which features are driving the model
5. What extra data would make the model more useful

The current modeling plan includes:

1. KMeans for state segmentation
2. Logistic Regression for mortality prediction
3. Random Forest for risk classification
4. Macro F1 for imbalanced classification
5. AUC for mortality prediction
6. Feature importance for model interpretation

I do not treat raw accuracy as the main success metric because the risk target is highly imbalanced.

## Feature engineering roadmap

This is the part I am most excited to improve next.

I want to add:

1. State year population data for better per capita comparisons
2. Rolling averages by state and disease topic
3. Year over year change features
4. Region and census division features
5. Demographic stratification features
6. Better risk thresholds
7. SHAP analysis for model explanation
8. Time based validation instead of random splits

## How to run the project

Clone the repo from GitHub.

Install the Python packages from the requirements file.

Place the CDC file here:

```text
data/raw/U.S._Chronic_Disease_Indicators__CDI_.csv
```

Generate the cleaned data layer.

```bash
python src/data/preprocessing.py
```

Create the SQL views.

```bash
duckdb data/analytics/chronic_disease.duckdb < sql/01_create_analytics_views.sql
```

Export Tableau files.

```bash
duckdb data/analytics/chronic_disease.duckdb < sql/02_export_tableau_extracts.sql
```

Open Tableau and connect to the CSV files inside the Tableau folder.

Continue to notebook modeling.

```bash
python notebooks/train_model.py
```

