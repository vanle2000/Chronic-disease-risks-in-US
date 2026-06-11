# Tableau Dashboard Design

This folder is for Tableau-ready CSV extracts generated from the SQL analytics layer.

## Dashboard 1: Executive Public Health Overview

**Audience:** program managers, policy analysts, and non-technical stakeholders.

**Business questions**

- Which chronic disease topics have the largest surveillance footprint
- Which topics are increasing over time
- Which states require deeper investigation
- How much of the dataset comes from mortality-related sources

**Recommended visuals**

| View | Data source | Purpose |
|---|---|---|
| KPI cards | `vw_dashboard_kpis` | Summarize total records, years covered, states, topics, and mortality-source share |
| Topic bar chart | `mart_topic_year_trends` | Rank chronic disease topics by surveillance volume |
| Yearly trend line | `mart_topic_year_trends` | Show how topic-level surveillance changes over time |
| State map | `mart_state_topic_summary` | Compare geographic concentration of disease indicators |
| Topic filter | all marts | Allow stakeholder-driven exploration |

## Dashboard 2: State and Topic Deep Dive

**Audience:** analytics managers and public health operations teams.

**Business questions**

- Which state-topic pairs show the highest average indicator values
- Are high-volume states also high-risk states
- Which topics are more mortality-source heavy
- Where do surveillance gaps appear

**Recommended visuals**

| View | Data source | Purpose |
|---|---|---|
| State-topic heatmap | `mart_state_topic_summary` | Compare states across chronic disease categories |
| Mortality-source rate by topic | `mart_mortality_by_topic` | Identify topics most tied to mortality records |
| Detail table | `mart_state_topic_summary` | Support drill-down investigation |
| State filter | all state-level marts | Enable regional analysis |

## Dashboard 3: Modeling Readiness and Risk Signals

**Audience:** data science reviewers and technical hiring managers.

**Business questions**

- Is the risk target balanced enough for classification
- Which features are available before modeling
- Which cuts should be validated before training
- Where does the project need better denominator data

**Recommended visuals**

| View | Data source | Purpose |
|---|---|---|
| Risk distribution chart | `mart_risk_distribution` | Show class imbalance before ML |
| Demographic risk table | `mart_demographic_risk` | Compare indicator values across stratification groups |
| Model metric cards | `model_summary.csv` | Summarize ML performance after notebook training |
| Feature importance chart | `feature_importances.csv` | Explain model drivers after training |

## Tableau extract contract

Expected extract files:

```text
tableau/
├── dashboard_kpis.csv
├── topic_year_trends.csv
├── state_topic_summary.csv
├── risk_distribution.csv
├── mortality_by_topic.csv
├── demographic_risk.csv
├── model_summary.csv
└── feature_importances.csv
```

The first six files come from SQL. The last two come from the ML notebook or training script.
