# Chronic Disease Risk Analytics in the U.S.

I built this project to explore chronic disease risk across the United States using SQL, Tableau, and machine learning. I want to turn a large public health dataset into clear insights that a stakeholder can actually use.

## What I am trying to answer

I use the CDC Chronic Disease Indicators data to answer practical public health questions.

1. Which chronic disease topics show the most activity across the country
2. Which states and topics need closer attention
3. How do disease indicators change over time
4. Which topics are most connected to mortality related records
5. Which demographic groups show higher average indicator values
6. Is the data ready for machine learning or does it need more feature work first

In this project, I focus on:

1. Asking useful questions based on the data analytics 
2. Creating clean SQL views
3. Building Tableau dashboards 
4. Checking data quality and class balance
5. Creating better features in notebooks
6. Training baseline machine learning models
7. Explaining why some metrics can be misleading

## Tools I used

Python for cleaning and notebook work

Pandas and NumPy for data processing

DuckDB SQL for analytics views and Tableau extracts

Tableau for dashboards and business storytelling

Scikit learn for predictive models

imbalanced learn for class imbalance experiments

pytest for testing

Parquet and CSV for reusable data outputs

## Repository layout

```text
data/
├── raw/
│   └── U.S._Chronic_Disease_Indicators__CDI_.csv
├── processed/
│   └── cdi_processed.parquet #Clean data layer used by SQL, Tableau, notebooks, and models.
└── analytics/
    └── chronic_disease.duckdb #Local DuckDB database for SQL analysis.

sql/
├── 01_create_analytics_views.sql
│   Creates KPI views, topic trends, state summaries, mortality views,
│   demographic views, risk distribution, and the ML feature base.
└── 02_export_tableau_extracts.sql
    Exports SQL results into CSV files for Tableau.

tableau/
├── README.md
│   Dashboard plan and extract guide.
├── dashboard_kpis.csv
├── topic_year_trends.csv
├── state_topic_summary.csv
├── risk_distribution.csv
├── mortality_by_topic.csv
└── demographic_risk.csv

notebooks/
└── train_model.py #For feature engineering and model training.

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
└── test_preprocessing.py # Tests for cleaning logic and feature engineering assumptions.
```

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

