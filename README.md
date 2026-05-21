# Uncovering Patterns and Predicting Chronic Disease Risks in the U.S.

## Business Problem

Chronic diseases are the leading cause of death and disability in the United States, accounting for 7 of the top 10 causes of mortality. Public health agencies face a critical challenge: with limited budgets and personnel, where should preventive resources be deployed first?

This project addresses that question directly. Using 20 years of CDC surveillance data spanning all 50 U.S. states, I built a data-driven pipeline to **identify geographic outbreak clusters**, **predict mortality risk**, and **classify disease severity** — transforming a dense, multi-source public health dataset into actionable intelligence for resource allocation and policy intervention.

---

## Dataset

**Source:** [U.S. Chronic Disease Indicators (CDI)](https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi) — CDC, CSTE, NACDD

| Attribute | Detail |
|-----------|--------|
| Rows | 900,000+ surveillance records |
| Time span | 2001 – 2021 (20 years) |
| Geography | All 50 U.S. states + territories |
| Disease topics | 17 categories |
| Data sources | BRFSS, NVSS, Cancer Registries, CMS, YRBSS, and more |

**17 Disease Topics:** Alcohol · Arthritis · Asthma · Cancer · Cardiovascular Disease · Chronic Kidney Disease · COPD · Diabetes · Disability · Immunization · Mental Health · Nutrition & Weight · Older Adults · Oral Health · Reproductive Health · Tobacco · Overarching Conditions

---

## Methodology

### 1. Data Cleaning & Feature Engineering
- Parsed `GeoLocation` (WKT POINT format) into `Latitude` and `Longitude` for geospatial analysis
- Imputed missing `DataValueUnit` values using within-group mode (`Topic` × `Question` groupby) — preserving domain context rather than applying global imputation
- Engineered `DeadStatus` binary flag from `DataSource` field (mortality-linked sources: NVSS, Death Certificates)
- Engineered `diseaseDuration` from `YearEnd − YearStart` to capture multi-year chronic burdens
- One-hot encoded all 17 disease topics as indicator features for modeling
- Removed low-signal columns (footnote fields, redundant confidence bounds)

### 2. Exploratory Analysis

**Geographic distribution (state × year heatmap):** Identified that data density spikes post-2015 reflect expanded CDC surveillance programs, not actual disease increases — a critical distinction for modeling.

**Demographic breakdown:** Cancer and Cardiovascular Disease are disproportionately represented in male records; Mental Health and Reproductive Health show higher female indicator counts. Racial stratification reveals persistent disparities, particularly in Diabetes and Hypertension indicators for Black, non-Hispanic populations.

**Trend analysis (2010–2021):** Cancer has the highest surveillance frequency (127K+ records), while Mental Health indicators show the steepest upward trend post-2016 — consistent with known population-level shifts in mental health reporting.

### 3. Geographic Clustering (K-Means)
Applied K-Means (k=3) on state-level disease frequency vectors to segment states into surveillance intensity tiers.

- **Silhouette Score: 0.635** — well-separated clusters
- High-frequency states (dense surveillance infrastructure) vs. low-frequency states (limited reporting capacity) vs. mid-tier
- Insight: Low surveillance states are not necessarily low-risk — they may represent data gaps requiring targeted investment

### 4. Mortality Risk Prediction (Logistic Regression vs. Naive Bayes)

**Target:** `DeadStatus` (binary: mortality-linked data source = 1)

**Features:** YearStart, disease topic indicators (Arthritis, Cancer, Cardiovascular Disease, Chronic Kidney Disease, Tobacco, Nutrition), disease duration

| Model | Accuracy | ROC-AUC | Notes |
|-------|----------|---------|-------|
| Logistic Regression | 0.77 | 0.73 | Best overall balance, interpretable coefficients |
| Naive Bayes | 0.42 | — | Assumes feature independence — violated here |

**Key finding:** Cardiovascular Disease, Cancer, and Tobacco indicators have the strongest positive correlation with `DeadStatus`. Logistic Regression coefficients directly confirm what epidemiological literature has long established — validating the pipeline's integrity.

### 5. Risk Level Classification (Random Forest)

**Target:** `RiskLevel` — 4-tier classification derived from `DataValueAlt` magnitude:

| Risk Level | DataValue Range | Records |
|------------|----------------|---------|
| Low | 0 – 100,000 | Majority |
| Moderate | 100,001 – 250,000 | ~180 |
| High | 250,001 – 500,000 | ~27 |
| Very High | 500,001 – 701,437 | 3 |

**Random Forest Results:**
- **Test Accuracy: 0.9999** — Note: this figure is dominated by the heavily imbalanced `Low` class (238K records vs. 3 `Very High`). Raw accuracy is misleading here.
- **Weighted F1: 1.00** for `Low`; **F1: 0.50** for `Very High` (recall = 0.33 on 3 samples)
- The model performs strongly on well-represented classes. `Very High` risk detection requires oversampling (SMOTE) or threshold tuning — a documented limitation and area for future work.
- **Macro-average F1: 0.85** — more honest representation of multi-class performance

**Feature importance:** `DataValue` magnitude, disease duration, and `DeadStatus` are the top predictors — consistent with domain knowledge.

---

## Key Insights

1. **Cancer dominates surveillance volume** (127K+ records) but cardiovascular disease carries higher per-record mortality association in the model
2. **States cluster into three distinct tiers** by reporting density — gaps in low-surveillance states are a public health blind spot
3. **Post-2016 Mental Health surge** is the fastest-growing indicator category, suggesting emerging resource reallocation needs
4. **Tobacco and Cardiovascular Disease co-occurrence** is the strongest dual predictor of mortality-linked outcomes
5. **Class imbalance at the extremes** (`Very High` risk: n=3) is the primary constraint on reliable Very High detection — not model capability

---

## Limitations & Future Work

| Limitation | Mitigation Path |
|------------|----------------|
| `Very High` risk class has only 3 samples | SMOTE or cost-sensitive learning |
| Risk bins based on raw DataValue magnitudes (not clinical thresholds) | Consult CDC severity guidelines for bin definitions |
| No temporal modeling | ARIMA or Prophet for disease trend forecasting |
| No causal inference | Difference-in-differences for policy intervention evaluation |
| Single-notebook structure | Modularize into `src/` pipeline for reproducibility |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly (interactive maps) |
| Geospatial | GeoPandas |
| Machine learning | Scikit-learn (Logistic Regression, Naive Bayes, Random Forest, KMeans) |
| Environment | Google Colab |

---

## Repository Structure

```
.
├── CSE469FinalProject.ipynb   # Full pipeline: EDA → clustering → modeling
├── presentation.pdf           # Project presentation slides
├── README.md
└── asset/
    └── image1                 # Supporting visualizations
```

## How to Run

1. Download the dataset from [CDC Open Data](https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi)
2. Upload to Google Drive at `My Drive/DM Project/U.S._Chronic_Disease_Indicators__CDI_.csv`
3. Open `CSE469FinalProject.ipynb` in Google Colab
4. Run all cells sequentially

---

## Results Summary

| Task | Model | Key Metric |
|------|-------|-----------|
| Mortality prediction | Logistic Regression | Accuracy: 0.77, AUC: 0.73 |
| Mortality prediction | Naive Bayes | Accuracy: 0.42 |
| Risk classification | Random Forest | Weighted F1: 1.00, Macro F1: 0.85 |
| State clustering | K-Means (k=3) | Silhouette: 0.635 |
